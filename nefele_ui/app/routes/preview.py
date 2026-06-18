"""Serve preview mask images produced by the worker.

Two flavours:

* ``/preview/<name>`` — shared_fs mode. Worker wrote the file to a disk the UI
  also mounts; we ``send_from_directory``.
* ``/hestia-preview?url=...`` — vm_comms mode. The worker uploaded the file to
  HESTIA storage, which requires a Bearer token to read. The browser can't
  send that header on ``<img src>`` requests, so the UI proxies: fetch with
  auth, stream the bytes back to the browser.
"""

from __future__ import annotations

import mimetypes
import os

from flask import Blueprint, Response, request, send_from_directory

from ..services.hestia import fetch_file
from ._helpers import cfg, json_err

bp = Blueprint("preview", __name__)


@bp.get("/preview/<path:name>")
def preview_image(name: str):
    c = cfg()
    fp = c.preview_dir / name
    if not fp.is_file():
        return json_err("Preview image not found", http=404)
    return send_from_directory(c.preview_dir, name)


@bp.get("/hestia-preview")
def hestia_preview():
    url = request.args.get("url", "")
    if not url:
        return json_err("missing url", http=400)
    api_base = os.environ.get(
        "HESTIA_API_URL", "https://api.textailes.athenarc.gr"
    ).rstrip("/")
    # SSRF guard: only fetch URLs hosted under our HESTIA storage area.
    if not url.startswith(f"{api_base}/storage/"):
        return json_err("url not on HESTIA storage", http=400)
    try:
        data = fetch_file(url)
    except Exception as e:
        return json_err(f"hestia fetch failed: {e}", http=502)
    ctype = mimetypes.guess_type(url)[0] or "application/octet-stream"
    return Response(data, mimetype=ctype)
