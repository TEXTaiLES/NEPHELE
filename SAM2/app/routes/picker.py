"""Interactive picker: points, save, preview, confirm, restart, frame serving."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, render_template, request, send_from_directory

from ..services.frames import clear_dir
from ..services.prompts import normalize_points, save_prompts
from ..services.worker_client import run_preview_masks
from ._helpers import cfg, frames, json_err, json_ok

bp = Blueprint("picker", __name__)


@bp.get("/pick")
def pick():
    return render_template("pick.html", frames=frames())


@bp.post("/save")
def save():
    c = cfg()
    fr = frames()
    if not fr:
        return json_err("No frames found", http=400)

    try:
        data = request.get_json(force=True, silent=True) or {}
        pts, labs = normalize_points(data.get("points", {}), frame_idx=0)
        save_prompts(c.prompts_json, Path(fr[0]), pts, labs, frame_idx=0, obj_id=1)

        # Clear any previous preview images before asking the worker for fresh ones.
        clear_dir(c.preview_dir, recursive=True)

        preview_files, worker_error = run_preview_masks(
            c.worker_url,
            c.worker_timeout,
            input_dir=c.input_dir,
            out_root=c.out_root,
            prompts_json=c.prompts_json,
            preview_dir=c.preview_dir,
            num_frames=6,
        )
        if worker_error and not preview_files:
            return json_err(worker_error, http=502)
        preview_urls = [f"/preview/{name}" for name in preview_files]
        return json_ok(path=str(c.prompts_json), previews=preview_urls)
    except Exception as e:
        return json_err(str(e), http=500)


@bp.post("/confirm")
def confirm():
    try:
        cfg().done_flag.touch()
        return json_ok(msg="confirmed")
    except OSError as e:
        return json_err(str(e), http=500)


@bp.post("/restart")
def restart():
    c = cfg()
    try:
        c.prompts_json.unlink(missing_ok=True)
        clear_dir(c.preview_dir)
        c.done_flag.unlink(missing_ok=True)
        c.use_existing_flag.unlink(missing_ok=True)
        return json_ok(msg="restarted")
    except OSError as e:
        return json_err(str(e), http=500)


@bp.get("/frame")
def frame():
    fr = frames()
    try:
        idx = int(request.args.get("i", "0"))
    except ValueError:
        return json_err("Invalid frame index", http=400)
    if idx < 0 or idx >= len(fr):
        return json_err("Frame index out of range", http=404)
    fp = Path(fr[idx])
    return send_from_directory(fp.parent, fp.name)
