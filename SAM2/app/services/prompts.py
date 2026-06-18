"""Persist prompts.json for SAM2 in the indexed output directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def normalize_points(
    points_dict: Dict, frame_idx: int = 0
) -> Tuple[List[List[float]], List[int]]:
    """Accept the JS payload {points: {"0": [{x,y,l}, ...]}} and return (pts, labels)."""
    raw = points_dict.get(str(frame_idx), points_dict.get(frame_idx, []))
    pts: List[List[float]] = []
    labs: List[int] = []
    for p in raw:
        pts.append([float(p["x"]), float(p["y"])])
        labs.append(int(p["l"]))
    return pts, labs


def save_prompts(
    prompts_path: Path,
    frame_path: Path,
    pts: List[List[float]],
    labs: List[int],
    frame_idx: int = 0,
    obj_id: int = 1,
) -> None:
    """Write prompts.json. Falls back to a single POS click at the image center."""
    with Image.open(frame_path) as im:
        w, h = im.size

    if not pts:
        pts = [[w // 2, h // 2]]
        labs = [1]

    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    with prompts_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "frame_idx": int(frame_idx),
                "obj_id": int(obj_id),
                "points": pts,
                "labels": labs,
                "image_w": int(w),
                "image_h": int(h),
                "source": frame_path.name,
            },
            f,
            indent=2,
        )
