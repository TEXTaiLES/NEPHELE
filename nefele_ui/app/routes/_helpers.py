"""Shared route helpers."""

from __future__ import annotations

from typing import Any

from flask import g, jsonify

from ..auth import get_current_user_slug
from ..config import Config, load_config
from ..services.frames import resolve_frames


def cfg() -> Config:
    if not hasattr(g, "cfg"):
        g.cfg = load_config(get_current_user_slug())
    return g.cfg


def frames() -> list[str]:
    if not hasattr(g, "frames"):
        c = cfg()
        g.frames = resolve_frames(c.input_dir, c.index_suffix) if c.is_configured else []
    return g.frames


def json_ok(**payload: Any):
    return jsonify({"ok": True, **payload})


def json_err(msg: str, http: int = 400):
    return jsonify({"ok": False, "error": msg}), http
