"""Nerfstudio-style frame extraction from video.

Pipeline:
1. ffmpeg extracts frames at ``fps × OVERSAMPLE`` (canonical 3DGS command).
2. Compute Laplacian variance per frame as a blur metric (higher = sharper).
3. Divide the raw frames into ``target_count`` evenly-spaced buckets and keep
   the sharpest from each bucket. This drops motion-blurred frames while
   preserving even temporal coverage of the scene.

This is the same algorithm Nerfstudio's ``ns-process-data video`` uses.
"""

from __future__ import annotations

import base64
import io
import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2  # type: ignore
from PIL import Image

log = logging.getLogger(__name__)

OVERSAMPLE = 4
THUMB_WIDTH = 240


class VideoExtractionError(RuntimeError):
    pass


def _laplacian_variance(gray_path: Path) -> float:
    img = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def _thumbnail_b64(jpg_path: Path, width: int = THUMB_WIDTH) -> str:
    with Image.open(jpg_path) as im:
        im = im.convert("RGB")
        ratio = width / im.width
        im.thumbnail((width, int(im.height * ratio)))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=70, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_frames_with_blur_filter(
    video_path: Path,
    target_dir: Path,
    fps: float,
    max_frames: int = 300,
) -> List[dict]:
    """Extract → score → bucket-pick. Returns list of frame metadata.

    Each entry: ``{"name": "0001.jpg", "sharpness": 123.4, "thumb_b64": "..."}``
    Frames are stored in ``target_dir`` with names ``%04d.jpg`` (1-based).
    Raw oversampled frames are deleted before returning.
    """
    if not shutil.which("ffmpeg"):
        raise VideoExtractionError("ffmpeg is not installed on the server")
    target_dir.mkdir(parents=True, exist_ok=True)

    extract_fps = fps * OVERSAMPLE
    raw_pattern = target_dir / "_raw_%05d.jpg"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-qscale:v", "1",
        "-qmin", "1",
        "-vf", f"fps={extract_fps}",
        str(raw_pattern),
    ]
    log.info("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        tail = proc.stderr[-500:] if proc.stderr else ""
        raise VideoExtractionError(f"ffmpeg failed: {tail}")

    raw = sorted(target_dir.glob("_raw_*.jpg"))
    if not raw:
        raise VideoExtractionError("ffmpeg produced no frames (unreadable video?)")

    log.info("extracted %d raw frames at %.2f fps", len(raw), extract_fps)

    scores = [_laplacian_variance(p) for p in raw]

    target_count = min(max(1, len(raw) // OVERSAMPLE), max_frames)
    bucket_size = len(raw) / target_count

    selected: List[Tuple[Path, float]] = []
    for i in range(target_count):
        lo = int(i * bucket_size)
        hi = int((i + 1) * bucket_size) if i < target_count - 1 else len(raw)
        if lo >= hi:
            continue
        best_j = max(range(lo, hi), key=lambda j: scores[j])
        selected.append((raw[best_j], scores[best_j]))

    keep_paths = {p for p, _ in selected}
    for p in raw:
        if p not in keep_paths:
            try:
                p.unlink()
            except OSError:
                pass

    frames: List[dict] = []
    for i, (src, score) in enumerate(selected, start=1):
        final_name = f"{i:04d}.jpg"
        final_path = target_dir / final_name
        src.rename(final_path)
        frames.append({
            "name": final_name,
            "sharpness": round(score, 1),
            "thumb_b64": _thumbnail_b64(final_path),
        })

    log.info("kept %d frames after blur filter (from %d raw)", len(frames), len(raw))
    return frames
