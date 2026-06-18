"""HTTP client for the SAM2 GPU worker container.

The worker shares /data/in and /data/out volumes with the UI, so we only need
to pass paths over the wire — image bytes never leave the host.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import requests

log = logging.getLogger(__name__)


def _summarize_worker_error(body: dict) -> str:
    """Pull a short, user-readable line out of the worker's traceback blob."""
    raw = body.get("log") or body.get("error") or ""
    if not raw:
        return "worker reported failure"
    # The worker often returns a multi-line traceback; the last non-empty line
    # tends to be the actual exception (e.g. "RuntimeError: No CUDA GPUs available").
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if line and not line.startswith(("File ", "Traceback")):
            return line[:250]
    return "worker reported failure"


def run_preview_masks(
    worker_url: str,
    timeout: float,
    *,
    input_dir: Path,
    out_root: Path,
    prompts_json: Path,
    preview_dir: Path,
    num_frames: int = 6,
) -> Tuple[List[str], Optional[str]]:
    """Ask the worker to mask ``num_frames`` frames.

    Returns ``(previews, error)``: ``previews`` is the relative filenames the
    worker produced (possibly empty); ``error`` is ``None`` on success or a
    short diagnostic message when the call failed.
    """
    url = f"{worker_url}/preview"
    payload = {
        "num_frames": num_frames,
        "input_dir": str(input_dir).rstrip("/"),
        "out_root": str(out_root),
        "prompts_json": str(prompts_json),
        "preview_dir": str(preview_dir),
    }
    log.info("POST %s payload=%s", url, payload)

    try:
        r = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as e:
        msg = f"worker unreachable at {url}: {e}"
        log.error(msg)
        return [], msg

    if r.status_code != 200:
        # The worker's body contains the original traceback; surface its tail.
        try:
            body = r.json()
        except ValueError:
            body = {"error": r.text[:500]}
        summary = _summarize_worker_error(body)
        log.error("worker HTTP %s: %s", r.status_code, summary)
        return [], f"worker HTTP {r.status_code}: {summary}"

    body = r.json()
    if not body.get("ok"):
        summary = _summarize_worker_error(body)
        log.error("worker reported failure: %s", summary)
        return [], summary

    previews = body.get("previews") or []
    log.info("worker returned %d preview files", len(previews))
    return previews, None
