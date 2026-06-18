#!/usr/bin/env python3
"""Manually upload a reconstruction to HESTIA /reconstructions.

Usage:
    python3 upload_reconstruction.py <dataset> [scan_id]

    dataset  — name of the dataset (e.g. power_bank_thomas_2)
    scan_id  — HESTIA scan id to link the reconstruction to (optional;
               defaults to the dataset name, or reads from input/.scan_id)

Searches:
    PGSR:  PGSR/outputs/<dataset>/mesh/  (prefers textured OBJ)
    SuGaR: SUGAR/SuGaR/outputs/**/<dataset>/**/  (skips _postprocessed)

Files are uploaded with the dataset name as prefix:
    PGSR  → <dataset>_tsdf_fusion_post_textured.obj / .mtl / .png
    SuGaR → <dataset>.obj / <dataset>.mtl / <dataset>.png
"""
from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path

import requests

# ── config ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent
ENV_FILE = REPO / ".env"

# Load .env
if ENV_FILE.is_file():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() and k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip()

API_BASE          = os.environ.get("HESTIA_API_URL", "https://api.textailes.athenarc.gr").rstrip("/")
API_KEY           = os.environ.get("HESTIA_API_KEY", "")
RECONSTRUCTIONS   = f"{API_BASE}/reconstructions"
IN_MNT            = Path(os.environ.get("IN_MNT", REPO / "SAM2/data/input"))
PGSR_RESULTS      = Path(os.environ.get("PGSR_RESULTS_ROOT",  REPO / "PGSR/outputs"))
SUGAR_RESULTS     = Path(os.environ.get("SUGAR_RESULTS_ROOT", REPO / "SUGAR/SuGaR/outputs"))

# ── mesh patching ─────────────────────────────────────────────────────────────
_MTLLIB_RE = re.compile(rb"^(mtllib\s+)(\S+)", re.MULTILINE)
_MAP_RE    = re.compile(rb"^(\s*map_\w+(?:\s+-\w+\s+\S+)*\s+)(\S+)", re.MULTILINE)

def _patch_obj(data: bytes, dataset: str) -> bytes:
    target = f"{dataset}.mtl".encode()
    return _MTLLIB_RE.sub(lambda m: m.group(1) + target, data)

def _patch_mtl(data: bytes, dataset: str) -> bytes:
    stem = dataset.encode()
    def _repl(m):
        suffix = Path(m.group(2).decode("latin-1")).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg"}:
            return m.group(1) + stem + suffix.encode("ascii")
        return m.group(0)
    return _MAP_RE.sub(_repl, data)

# ── upload ────────────────────────────────────────────────────────────────────
def upload(dataset: str, scan_id: str) -> None:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    files: list = []

    # Detect model
    model_file = IN_MNT / dataset / ".model"
    model = "sugar"
    if model_file.is_file():
        m = model_file.read_text().strip()
        if m in ("sugar", "pgsr"):
            model = m

    # Fall back: if outputs exist in PGSR dir, use pgsr
    pgsr_dir = PGSR_RESULTS / dataset
    if pgsr_dir.exists() and list(pgsr_dir.rglob("*.obj")):
        model = "pgsr"

    print(f"dataset={dataset}  scan_id={scan_id}  model={model}")

    if model == "pgsr":
        all_obj = sorted(pgsr_dir.rglob("*.obj"), key=lambda p: p.stat().st_mtime)
        if not all_obj:
            sys.exit(f"No .obj found under {pgsr_dir}")
        textured = [p for p in all_obj if "textured" in p.stem]
        obj = textured[-1] if textured else all_obj[-1]
        mtl = obj.with_suffix(".mtl")
        tex = next(
            (obj.parent / f"{obj.stem}{e}" for e in (".png", ".jpg")
             if (obj.parent / f"{obj.stem}{e}").exists()), None
        )
        def _pref(name): return name if name.startswith(f"{dataset}_") else f"{dataset}_{name}"
        print(f"  OBJ: {obj}")
        files = [("file", (_pref(obj.name), open(obj, "rb"), "application/octet-stream"))]
        if mtl.exists():
            print(f"  MTL: {mtl}")
            files.append(("file", (_pref(mtl.name), open(mtl, "rb"), "application/octet-stream")))
        if tex:
            print(f"  TEX: {tex}")
            mime = "image/jpeg" if tex.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            files.append(("file", (_pref(tex.name), open(tex, "rb"), mime)))

    else:  # sugar
        candidates = [
            p for p in SUGAR_RESULTS.rglob("*.obj")
            if dataset in p.parts and "_postprocessed" not in p.stem
        ]
        if not candidates:
            sys.exit(f"No SuGaR .obj found for {dataset} under {SUGAR_RESULTS}")
        obj = max(candidates, key=lambda p: p.stat().st_mtime)
        mtl = obj.with_suffix(".mtl")
        tex = next(
            (obj.parent / f"{obj.stem}{e}" for e in (".png", ".jpg")
             if (obj.parent / f"{obj.stem}{e}").exists()), None
        )
        print(f"  OBJ: {obj}")
        obj_data = _patch_obj(obj.read_bytes(), dataset)
        files = [("file", (f"{dataset}.obj", io.BytesIO(obj_data), "application/octet-stream"))]
        if mtl.exists():
            print(f"  MTL: {mtl}")
            mtl_data = _patch_mtl(mtl.read_bytes(), dataset)
            files.append(("file", (f"{dataset}.mtl", io.BytesIO(mtl_data), "application/octet-stream")))
        if tex:
            print(f"  TEX: {tex}")
            tex_ext = tex.suffix.lower()
            mime = "image/jpeg" if tex_ext in (".jpg", ".jpeg") else "image/png"
            files.append(("file", (f"{dataset}{tex_ext}", io.BytesIO(tex.read_bytes()), mime)))

    print(f"Uploading {len(files)} file(s) → {RECONSTRUCTIONS}")
    try:
        r = requests.post(
            RECONSTRUCTIONS, headers=headers,
            files=files, data={"scan_id": scan_id}, timeout=300,
        )
        r.raise_for_status()
        rec = r.json()
        print(f"\n✓ Uploaded successfully!")
        print(f"  object_id:         {rec.get('object_id')}")
        print(f"  public_url_model:  {rec.get('public_url_model')}")
        print(f"  public_url_glb:    {rec.get('public_url_glb')}")
        print(f"  public_url_material: {rec.get('public_url_material')}")
        print(f"  public_url_texture:  {rec.get('public_url_texture')}")
    except requests.HTTPError as e:
        sys.exit(f"Upload failed: {e}\n{e.response.text}")
    finally:
        for _, (_, fh, _) in files:
            if hasattr(fh, "close"):
                fh.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 upload_reconstruction.py <dataset> [scan_id]")

    dataset = sys.argv[1]
    # scan_id: from arg, or from .scan_id file, or fall back to dataset name
    if len(sys.argv) >= 3:
        scan_id = sys.argv[2]
    else:
        sid_file = IN_MNT / dataset / ".scan_id"
        scan_id = sid_file.read_text().strip() if sid_file.is_file() else dataset

    upload(dataset, scan_id)
