"""Setup screen: pick a dataset name and upload images when none is configured."""

from __future__ import annotations

from flask import Blueprint, current_app, render_template, request

from ..services.dataset_meta import write_scan_meta
from ..services.frames import resolve_frames
from ..services.hestia import scan_exists, upload_robot_images
from ..services.pipeline import request_kill
from ..services.vm_comms import (
    VmCommsError,
    cancel_job,
    list_active_jobs,
)
from ..services.uploads import (
    sanitize_dataset_name,
    save_uploaded_images,
    write_active_dataset,
)
from ..services.video import VideoExtractionError, extract_frames_with_blur_filter
from ._helpers import cfg, json_err, json_ok

bp = Blueprint("setup", __name__)


def _cancel_active_jobs() -> int:
    """Best-effort: cancel any non-terminal HESTIA jobs before starting a new
    dataset. The worker_poller is single-job, so leaving an old job alive would
    block the new one until the old one finishes.

    Errors are logged but not raised — the user just asked for a new dataset,
    not for cancellation, so we never fail the new request because cleanup
    couldn't finish.
    """
    import logging as _logging
    log = _logging.getLogger(__name__)
    cancelled = 0
    try:
        jobs = list_active_jobs()
    except VmCommsError as e:
        log.warning("list_active_jobs failed before new dataset: %s", e)
        return 0
    for j in jobs:
        try:
            if cancel_job(j.job_id):
                cancelled += 1
        except VmCommsError as e:
            log.warning("cancel job %s failed: %s", j.job_id, e)
    if cancelled:
        log.info("cancelled %d active job(s) before new dataset", cancelled)
    return cancelled


@bp.get("/setup")
def setup():
    c = cfg()
    return render_template(
        "setup.html",
        current_name=c.dataset_name,
        in_mnt=str(c.in_mnt),
    )


@bp.post("/setup")
def submit():
    """Create the dataset folder, save the uploaded files, and rebind config in-process."""
    c = cfg()

    name_raw = request.form.get("name", "").strip()
    name = sanitize_dataset_name(name_raw)
    if not name:
        return json_err("Please provide a dataset name.")

    model = request.form.get("model", "sugar").strip()
    if model not in ("sugar", "pgsr"):
        model = "sugar"

    # In vm_comms mode the dataset name is also the HESTIA scan_id, so reject
    # collisions up front — HESTIA has no DELETE, so a duplicate POST would
    # silently append to whatever already lives under that scan_id.
    if c.uses_vm_comms:
        try:
            if scan_exists(name):
                return json_err(
                    f"A scan named '{name}' already exists in HESTIA. Pick a different dataset name.",
                    http=409,
                )
        except Exception as e:
            return json_err(f"HESTIA check failed: {e}", http=502)
        _cancel_active_jobs()

    request_kill(c.in_mnt)

    files = request.files.getlist("images")
    target = c.in_mnt / name
    saved, failed = save_uploaded_images(target, files)

    # Step 1b: in the decoupled (vm_comms) mode the worker has no shared disk,
    # so push the just-saved images to HESTIA robot_images using the dataset
    # name as the scan_id (the picker needs it to create a vm_comms job).
    scan_id = ""
    if c.uses_vm_comms and saved:
        try:
            result = upload_robot_images([target / n for n in saved], scan_id=name)
        except Exception as e:
            return json_err(f"HESTIA robot-images upload failed: {e}", http=502)
        scan_id = result.get("scan_id", "")

    # Persists .scan_id (empty in shared_fs mode) and .model.
    write_scan_meta(target, scan_id, model)

    write_active_dataset(c.in_mnt, name)

    # Rebind the live config so subsequent requests see the new dataset.
    from .. import rebind_dataset
    new_cfg = rebind_dataset(current_app, name)
    frames = resolve_frames(new_cfg.input_dir, new_cfg.index_suffix)

    return json_ok(dataset=name, saved=saved, failed=failed, total=len(frames))


@bp.post("/setup/video")
def submit_video():
    """Stage 1: ffmpeg + blur filter. Saves frames locally **and** pushes them to
    HESTIA in vm_comms mode so the worker can see them immediately — same
    contract as the photo-upload path (``/setup``).

    Caveat: HESTIA has no DELETE on /robot-images, so if the user later drops
    frames in the review screen or hits /setup/video/cancel, those frames stay
    orphaned in HESTIA. The local prune still controls what the pipeline sees.
    """
    from pathlib import Path as _Path

    c = cfg()

    name = sanitize_dataset_name(request.form.get("name", "").strip())
    if not name:
        return json_err("Please provide a dataset name.")

    model = request.form.get("model", "sugar").strip()
    if model not in ("sugar", "pgsr"):
        model = "sugar"

    try:
        fps = float(request.form.get("fps", "2"))
    except ValueError:
        return json_err("Invalid fps value.")
    if fps <= 0 or fps > 30:
        return json_err("fps must be between 0 and 30.")

    video = request.files.get("video")
    if not video or not video.filename:
        return json_err("Please choose a video file.")

    if c.uses_vm_comms:
        try:
            if scan_exists(name):
                return json_err(
                    f"A scan named '{name}' already exists in HESTIA. Pick a different dataset name.",
                    http=409,
                )
        except Exception as e:
            return json_err(f"HESTIA check failed: {e}", http=502)
        _cancel_active_jobs()

    request_kill(c.in_mnt)

    target = c.in_mnt / name
    if target.exists():
        return json_err(
            f"A local dataset named '{name}' already exists. Pick another name.",
            http=409,
        )
    target.mkdir(parents=True, exist_ok=True)

    suffix = _Path(video.filename).suffix or ".mp4"
    tmp_video = target / f"_source{suffix}"
    video.save(tmp_video)
    try:
        frames = extract_frames_with_blur_filter(tmp_video, target, fps=fps, max_frames=300)
    except VideoExtractionError as e:
        # Clean up the half-built dataset dir so the user can retry.
        try:
            for p in target.iterdir():
                p.unlink()
            target.rmdir()
        except OSError:
            pass
        return json_err(f"Video processing failed: {e}", http=400)
    finally:
        try:
            tmp_video.unlink()
        except OSError:
            pass

    # Mirror the photo-upload path: in vm_comms mode push frames to HESTIA
    # right after extraction so the worker can claim a job against this scan
    # without waiting for /confirm.
    scan_id = ""
    if c.uses_vm_comms and frames:
        try:
            frame_paths = [target / f["name"] for f in frames]
            result = upload_robot_images(frame_paths, scan_id=name)
        except Exception as e:
            return json_err(f"HESTIA robot-images upload failed: {e}", http=502)
        scan_id = result.get("scan_id", "") or name

    write_scan_meta(target, scan_id=scan_id, model=model)

    sharps = [f["sharpness"] for f in frames]
    return json_ok(
        dataset=name,
        frames=frames,
        stats={
            "kept": len(frames),
            "mean_sharpness": round(sum(sharps) / len(sharps), 1) if sharps else 0.0,
            "min_sharpness": round(min(sharps), 1) if sharps else 0.0,
            "max_sharpness": round(max(sharps), 1) if sharps else 0.0,
        },
    )


@bp.post("/setup/video/confirm")
def confirm_video():
    """Stage 2: delete any frames the user excluded; HESTIA already has them.

    Stage 1 (``/setup/video``) uploads all extracted frames to HESTIA. Here we
    only prune the local copy so the pipeline runs against the user's kept
    set. Dropped frames remain in HESTIA (no DELETE on /robot-images).
    """
    c = cfg()

    data = request.get_json(silent=True) or {}
    name = sanitize_dataset_name((data.get("name") or "").strip())
    if not name:
        return json_err("Missing dataset name.")
    drop = set(data.get("drop_frames") or [])

    target = c.in_mnt / name
    if not target.is_dir():
        return json_err("Pending dataset not found — did you cancel it?", http=404)

    for fname in drop:
        p = target / fname
        if p.is_file() and p.suffix.lower() == ".jpg":
            try:
                p.unlink()
            except OSError:
                pass

    remaining = sorted(p.name for p in target.glob("*.jpg"))
    if not remaining:
        return json_err("No frames left after pruning.", http=400)

    # Preserve scan_id + model that stage 1 already persisted.
    scan_id_file = target / ".scan_id"
    scan_id = scan_id_file.read_text(encoding="utf-8").strip() if scan_id_file.exists() else ""
    model_file = target / ".model"
    model = model_file.read_text(encoding="utf-8").strip() if model_file.exists() else "sugar"

    write_scan_meta(target, scan_id, model)
    write_active_dataset(c.in_mnt, name)

    from .. import rebind_dataset
    new_cfg = rebind_dataset(current_app, name)
    frames = resolve_frames(new_cfg.input_dir, new_cfg.index_suffix)

    return json_ok(dataset=name, saved=remaining, total=len(frames))


@bp.post("/setup/video/cancel")
def cancel_video():
    """Throw away the pending dataset folder created by stage 1."""
    import shutil as _shutil

    c = cfg()
    data = request.get_json(silent=True) or {}
    name = sanitize_dataset_name((data.get("name") or "").strip())
    if not name:
        return json_err("Missing dataset name.")

    target = c.in_mnt / name
    if target.is_dir():
        try:
            _shutil.rmtree(target)
        except OSError as e:
            return json_err(f"Failed to remove dataset folder: {e}", http=500)
    return json_ok(dataset=name)
