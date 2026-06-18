#!/usr/bin/env python3
"""
SAM2 worker HTTP server.

Runs inside the GPU container and exposes a tiny HTTP API the UI container
calls when the user clicks "Save". The UI passes the explicit paths it wants
the preview rendered into, so the worker is dataset-agnostic.

Endpoints
---------
GET  /healthz           liveness probe
POST /preview           run video_predict.py --preview and return file list
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List

from flask import Flask, jsonify, request


REPO_ROOT = Path(__file__).resolve().parent.parent  # /workspace
SCRIPT = REPO_ROOT / "app" / "video_predict.py"

app = Flask(__name__)


def _gather_previews(preview_dir: Path) -> List[str]:
    out: List[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        for f in sorted(preview_dir.rglob(ext)):
            out.append(str(f.relative_to(preview_dir)).replace("\\", "/"))
    return out


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True, "script": str(SCRIPT), "exists": SCRIPT.is_file()})


@app.post("/preview")
def preview():
    data = request.get_json(force=True, silent=True) or {}

    num_frames = int(data.get("num_frames", 6))
    input_dir = data.get("input_dir") or os.environ.get("INPUT", "/data/in")
    out_root = data.get("out_root") or os.environ.get("OUT", "/data/out")
    prompts_json = data.get("prompts_json")
    preview_dir = data.get("preview_dir")

    if not prompts_json or not preview_dir:
        return jsonify({"ok": False, "error": "prompts_json and preview_dir are required"}), 400

    pdir = Path(preview_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    for f in pdir.rglob("*"):
        if f.is_file():
            try:
                f.unlink()
            except Exception:
                pass

    cmd = [
        "python3", str(SCRIPT),
        "--preview",
        "--preview-num-frames", str(num_frames),
        "--preview-out", str(pdir),
    ]

    env = os.environ.copy()
    env["PROMPTS_JSON"] = str(prompts_json)
    env["AUTO_INDEX"] = "1"
    env["INPUT"] = str(input_dir).rstrip("/")
    env["OUT"] = str(out_root)
    env["QUIET"] = "0"

    print(f"[worker] cmd={' '.join(cmd)}", flush=True)
    print(f"[worker] cwd={REPO_ROOT}", flush=True)
    print(f"[worker] PROMPTS_JSON={env['PROMPTS_JSON']}", flush=True)
    print(f"[worker] INPUT={env['INPUT']}", flush=True)
    print(f"[worker] OUT={env['OUT']}", flush=True)
    print(f"[worker] preview_dir={pdir}", flush=True)

    p = subprocess.run(
        cmd,
        env=env,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"[worker] returncode={p.returncode}", flush=True)
    print(f"[worker] output:\n{p.stdout}", flush=True)

    if p.returncode != 0:
        return jsonify({"ok": False, "error": "preview failed", "log": p.stdout}), 500

    previews = _gather_previews(pdir)
    return jsonify({"ok": True, "previews": previews})


if __name__ == "__main__":
    port = int(os.environ.get("WORKER_PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=False)
