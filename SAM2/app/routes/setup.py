"""Setup screen: pick a dataset name and upload images when none is configured."""

from __future__ import annotations

from flask import Blueprint, current_app, render_template, request

from ..services.frames import resolve_frames
from ..services.pipeline import request_kill
from ..services.uploads import (
    sanitize_dataset_name,
    save_uploaded_images,
    write_active_dataset,
)
from ._helpers import cfg, json_err, json_ok

bp = Blueprint("setup", __name__)


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
    if model not in ("sugar", "pgsr", "fastpgsr"):
        model = "sugar"

    request_kill(c.in_mnt)

    files = request.files.getlist("images")
    target = c.in_mnt / name
    saved, failed = save_uploaded_images(target, files)

    (target / ".model").write_text(model)

    write_active_dataset(c.in_mnt, name)

    # Rebind the live config so subsequent requests see the new dataset.
    from .. import rebind_dataset
    new_cfg = rebind_dataset(current_app, name)
    frames = resolve_frames(new_cfg.input_dir, new_cfg.index_suffix)

    return json_ok(dataset=name, saved=saved, failed=failed, total=len(frames))
