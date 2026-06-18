"""Serve preview mask images produced by the worker."""

from __future__ import annotations

from flask import Blueprint, send_from_directory

from ._helpers import cfg, json_err

bp = Blueprint("preview", __name__)


@bp.get("/preview/<path:name>")
def preview_image(name: str):
    c = cfg()
    fp = c.preview_dir / name
    if not fp.is_file():
        return json_err("Preview image not found", http=404)
    return send_from_directory(c.preview_dir, name)
