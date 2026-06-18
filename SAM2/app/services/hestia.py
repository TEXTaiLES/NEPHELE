"""HESTIA API client — list robot-image scans and download them locally."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_API_BASE = os.environ.get("HESTIA_API_URL", "https://api.textailes.athenarc.gr").rstrip("/")
_API_KEY = os.environ.get("HESTIA_API_KEY", "")
# MinIO is on localhost:9001 from the API server's view.
# Override HESTIA_MINIO_URL if MinIO is exposed on a different address.
_MINIO_URL = os.environ.get("HESTIA_MINIO_URL", "").rstrip("/")

ROBOT_IMAGES_EP = f"{_API_BASE}/robot-images"
RECONSTRUCTIONS_EP = f"{_API_BASE}/reconstructions"


@dataclass
class ScanInfo:
    scan_id: str
    image_count: int
    timestamp: str
    sample_filename: str


def _headers() -> dict:
    return {"Authorization": f"Bearer {_API_KEY}"}


def _download_url(public_url: str) -> str:
    """Rewrite internal localhost:9001 MinIO Console URL to the S3 API on port 9000.

    The HESTIA API stores images in MinIO and exposes their URL as
    http://localhost:9001/... (the Console port). The actual S3-compatible
    download endpoint is on port 9000. Override with HESTIA_MINIO_URL if needed.
    """
    if _MINIO_URL:
        parsed = urlparse(public_url)
        return f"{_MINIO_URL}{parsed.path}"
    api_host = urlparse(_API_BASE).hostname or "api.textailes.athenarc.gr"
    # 9001 = MinIO Console (web UI), 9000 = MinIO S3 API (actual files)
    return public_url.replace("localhost:9001", f"{api_host}:9000")


def list_scans() -> List[ScanInfo]:
    """Fetch all robot-image records and group them into unique scans."""
    scans: Dict[str, dict] = {}
    page = 1
    while True:
        r = requests.get(
            ROBOT_IMAGES_EP,
            headers=_headers(),
            params={"page": page, "per_page": 100},
            timeout=20,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for img in batch:
            sid = img["scan_id"]
            if sid not in scans:
                scans[sid] = {
                    "image_count": 0,
                    "timestamp": img.get("timestamp", ""),
                    "sample_filename": img.get("filename", ""),
                }
            scans[sid]["image_count"] += 1
        page += 1

    return [
        ScanInfo(
            scan_id=sid,
            image_count=info["image_count"],
            timestamp=info["timestamp"],
            sample_filename=info["sample_filename"],
        )
        for sid, info in sorted(
            scans.items(), key=lambda x: x[1]["timestamp"], reverse=True
        )
    ]


def download_scan(scan_id: str, dest_dir: Path, on_progress=None) -> int:
    """Download all images for scan_id into dest_dir. Returns count saved."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    page = 1
    total = 0
    while True:
        r = requests.get(
            ROBOT_IMAGES_EP,
            headers=_headers(),
            params={"scan_id": scan_id, "page": page, "per_page": 100},
            timeout=20,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for img in batch:
            filename = img["filename"]
            url = _download_url(img["public_url"])
            save_path = dest_dir / filename
            if save_path.exists():
                total += 1
                if on_progress:
                    on_progress(total)
                continue
            try:
                ir = requests.get(url, headers=_headers(), stream=True, timeout=60)
                ir.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in ir.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                total += 1
                if on_progress:
                    on_progress(total)
                logger.info("downloaded %s", filename)
            except Exception as e:
                logger.warning("failed to download %s: %s", filename, e)
        page += 1
    return total


def upload_reconstruction(scan_id: str, obj_path: Path, mtl_path: Path | None, texture_path: Path | None) -> dict:
    """Upload a finished OBJ reconstruction to HESTIA /reconstructions."""
    files: list = []
    open_handles: list = []
    try:
        fh = open(obj_path, "rb")
        open_handles.append(fh)
        files.append(("model_file", (obj_path.name, fh, "application/octet-stream")))

        if mtl_path and mtl_path.exists():
            fh2 = open(mtl_path, "rb")
            open_handles.append(fh2)
            files.append(("material_file", (mtl_path.name, fh2, "application/octet-stream")))

        if texture_path and texture_path.exists():
            fh3 = open(texture_path, "rb")
            open_handles.append(fh3)
            files.append(("texture_file", (texture_path.name, fh3, "image/jpeg")))

        r = requests.post(
            RECONSTRUCTIONS_EP,
            headers=_headers(),
            files=files,
            data={"scan_id": scan_id},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    finally:
        for fh in open_handles:
            fh.close()
