"""Frame discovery on the shared input volume."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

_IMAGE_GLOBS: tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def gather_frames(dir_path: Path | str) -> List[str]:
    p = Path(dir_path)
    files: List[str] = []
    for pat in _IMAGE_GLOBS:
        files.extend(str(f) for f in p.glob(pat))
    return sorted(files)


def resolve_frames(input_dir: Path, index_suffix: str) -> List[str]:
    """Look in INPUT first, then fall back to INPUT + index_suffix."""
    frames = gather_frames(input_dir)
    if frames:
        return frames
    fallback = Path(f"{str(input_dir).rstrip('/')}{index_suffix}")
    return gather_frames(fallback)


def clear_dir(dir_path: Path, patterns: Iterable[str] = ("*",), recursive: bool = False) -> None:
    """Delete files under `dir_path` matching any of `patterns`. Silently skips failures."""
    if not dir_path.exists():
        return
    iterator = dir_path.rglob if recursive else dir_path.glob
    for pat in patterns:
        for f in iterator(pat):
            if f.is_file():
                try:
                    f.unlink()
                except OSError:
                    pass
