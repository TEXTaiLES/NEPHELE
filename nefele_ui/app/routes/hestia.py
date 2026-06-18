"""HESTIA integration routes — browse scans, load images, upload results."""

from __future__ import annotations

import threading
from pathlib import Path

from flask import Blueprint, current_app, redirect, render_template, request, url_for

from ..services.dataset_meta import write_scan_meta
from ..services.hestia import download_scan, list_scans, upload_reconstruction
from ..services.pipeline import request_kill
from ..services.uploads import sanitize_dataset_name, write_active_dataset
from ._helpers import cfg, json_err, json_ok

bp = Blueprint("hestia", __name__)

# Track active downloads so the UI can poll progress.
# { scan_id: {"status": "downloading"|"done"|"error", "downloaded": N, "error": str} }
_downloads: dict = {}
_downloads_lock = threading.Lock()


@bp.get("/hestia")
def page():
    return render_template("hestia.html")


@bp.get("/hestia/scans")
def scans_json():
    """Return JSON list of unique scans from HESTIA robot-images."""
    try:
        scans = list_scans()
    except Exception as e:
        return json_err(f"HESTIA API error: {e}", http=502)
    return json_ok(
        scans=[
            {
                "scan_id": s.scan_id,
                "image_count": s.image_count,
                "timestamp": s.timestamp,
                "sample_filename": s.sample_filename,
            }
            for s in scans
        ]
    )


@bp.post("/hestia/load")
def load_scan():
    """Start downloading a scan in the background; return immediately."""
    data = request.get_json(silent=True) or request.form
    scan_id = (data.get("scan_id") or "").strip()
    if not scan_id:
        return json_err("scan_id required", http=400)

    model = (data.get("model") or "sugar").strip()
    if model not in ("sugar", "pgsr"):
        model = "sugar"

    with _downloads_lock:
        if scan_id in _downloads and _downloads[scan_id]["status"] == "downloading":
            return json_ok(status="downloading", message="Already in progress")
        _downloads[scan_id] = {"status": "downloading", "downloaded": 0, "error": None}

    c = cfg()
    request_kill(c.in_mnt)
    # Use caller-supplied name if valid, else fall back to auto-name
    custom_name = sanitize_dataset_name((data.get("dataset_name") or "").strip())
    dataset_name = custom_name if custom_name else f"scan_{scan_id[:8]}"
    dest = c.in_mnt / dataset_name

    # Capture the real app object now, inside the request context,
    # before the thread starts (current_app proxy breaks in threads).
    app = current_app._get_current_object()

    def _run():
        count = 0

        def _progress(n):
            with _downloads_lock:
                _downloads[scan_id]["downloaded"] = n

        try:
            count = download_scan(scan_id, dest, on_progress=_progress)
        except Exception as e:
            with _downloads_lock:
                _downloads[scan_id] = {"status": "error", "downloaded": count, "error": str(e)}
            return

        # Persist scan id + model choice alongside the dataset so the picker
        # can build a vm_comms job for it later.
        try:
            write_scan_meta(dest, scan_id, model)
        except Exception:
            pass

        # Download succeeded — mark done regardless of what rebind does
        write_active_dataset(c.in_mnt, dataset_name)
        with _downloads_lock:
            _downloads[scan_id] = {"status": "done", "downloaded": count, "error": None}
        try:
            from .. import rebind_dataset
            rebind_dataset(app, dataset_name)
        except Exception:
            pass  # rebind is best-effort; .active_dataset file is the durable record

    threading.Thread(target=_run, daemon=True).start()
    return json_ok(status="downloading", dataset=dataset_name, scan_id=scan_id)


@bp.get("/hestia/load/status")
def load_status():
    """Poll download progress for a given scan_id."""
    scan_id = request.args.get("scan_id", "").strip()
    with _downloads_lock:
        info = _downloads.get(scan_id)
    if not info:
        return json_err("No download found for that scan_id", http=404)
    return json_ok(**info, scan_id=scan_id)


@bp.post("/hestia/upload")
def upload_result():
    """Upload a finished reconstruction OBJ to HESTIA /reconstructions."""
    data = request.get_json(silent=True) or request.form
    scan_id = (data.get("scan_id") or "").strip()
    obj_path = Path((data.get("obj_path") or "").strip())
    mtl_path_raw = (data.get("mtl_path") or "").strip()
    tex_path_raw = (data.get("texture_path") or "").strip()

    if not scan_id or not obj_path:
        return json_err("scan_id and obj_path are required", http=400)
    if not obj_path.is_file():
        return json_err(f"obj_path not found: {obj_path}", http=404)

    mtl_path = Path(mtl_path_raw) if mtl_path_raw else None
    tex_path = Path(tex_path_raw) if tex_path_raw else None

    try:
        result = upload_reconstruction(scan_id, obj_path, mtl_path, tex_path)
    except Exception as e:
        return json_err(f"Upload failed: {e}", http=502)
    return json_ok(**result)
