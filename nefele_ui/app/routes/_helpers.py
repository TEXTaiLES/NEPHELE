"""Shared route helpers."""

from __future__ import annotations

from typing import Any

from flask import current_app, jsonify

from ..config import Config


def cfg() -> Config:
    return current_app.config["APP_CONFIG"]


def frames() -> list[str]:
    return current_app.config["APP_FRAMES"]


def json_ok(**payload: Any):
    return jsonify({"ok": True, **payload})


def json_err(msg: str, http: int = 400):
    return jsonify({"ok": False, "error": msg}), http
