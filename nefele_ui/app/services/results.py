"""Discover SuGaR output files for the active dataset.

SuGaR writes textured meshes a few directories deep:

    <root>/<dataset>/refined_mesh/data/*.obj
    <root>/<group>/<dataset>/refined_mesh/data/*.obj   (if OUTPUT_GROUP is set)

We don't hard-code any of those intermediate directory names — instead we look
for any ``.obj`` file under ``<root>`` whose path includes the dataset name
and return the directory holding it.

Only files whose stem ends with ``_postprocessed`` are surfaced; raw and
``_before_cleanup`` variants are filtered out. Each surfaced file is also
renamed to a short, user-friendly download name (``mesh.obj`` / ``mesh.mtl``
/ ``mesh.png``) — and cross-references inside ``.obj`` / ``.mtl`` are patched
on the fly so the triplet drops straight into Blender / MeshLab.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

_TEXTURE_EXTS = {".png", ".jpg", ".jpeg"}
_PGSR_MESH_EXTS = {".ply", ".obj"}


def read_model(in_mnt: Path, dataset_name: str) -> str:
    """Return 'sugar' or 'pgsr' based on the .model file written at dataset creation."""
    f = in_mnt / dataset_name / ".model"
    if not f.is_file():
        return "sugar"
    val = f.read_text(encoding="utf-8").strip()
    return val if val in ("sugar", "pgsr") else "sugar"

# SuGaR drops both raw and `_postprocessed` variants in the same folder.
# We surface the raw originals (the un-cleaned full-detail mesh). Anything
# whose stem contains "_postprocessed" (incl. "_postprocessed_before_cleanup")
# is treated as an intermediate and skipped.
_SKIP_TOKEN = "_postprocessed"


@dataclass(frozen=True)
class ResultFile:
    name: str           # friendly display + download name (mesh.obj etc.)
    relative: str       # path relative to sugar_results_root (used by /download)
    size: int           # bytes (after patching, since text patches are tiny)
    modified: float     # unix mtime
    kind: str           # 'obj' | 'mtl' | 'texture'


def _classify(p: Path) -> Optional[str]:
    ext = p.suffix.lower()
    if ext == ".obj":
        return "obj"
    if ext == ".mtl":
        return "mtl"
    if ext in _TEXTURE_EXTS:
        return "texture"
    return None


def friendly_name(disk_name: str, dataset: str) -> str:
    """Map SuGaR's long filenames to a short, dataset-named trio.

    e.g. ``sugarfine_...gaussperface1.obj`` → ``dress_output_2.obj``.
    The trio shares one stem so the cross-references stay consistent.
    """
    stem = dataset or "mesh"
    ext = Path(disk_name).suffix.lower()
    if ext == ".obj":
        return f"{stem}.obj"
    if ext == ".mtl":
        return f"{stem}.mtl"
    if ext in _TEXTURE_EXTS:
        return f"{stem}.png"
    return Path(disk_name).name


def find_dataset_dir(root: Path, dataset_name: str) -> Optional[Path]:
    """Locate the folder that holds SuGaR's .obj output for ``dataset_name``."""
    if not root.exists():
        return None

    counts: dict[Path, int] = {}
    for obj in root.rglob("*.obj"):
        if dataset_name not in obj.parts:
            continue
        counts[obj.parent] = counts.get(obj.parent, 0) + 1

    if not counts:
        return None
    return max(counts, key=counts.get)


def list_outputs(root: Path, dataset_name: str) -> List[ResultFile]:
    """Return the postprocessed mesh trio under friendly names."""
    dataset_dir = find_dataset_dir(root, dataset_name)
    if dataset_dir is None:
        return []

    results: List[ResultFile] = []
    for p in sorted(dataset_dir.iterdir()):
        if not p.is_file():
            continue
        kind = _classify(p)
        if kind is None:
            continue
        if _SKIP_TOKEN in p.stem:
            continue
        stat = p.stat()
        results.append(
            ResultFile(
                name=friendly_name(p.name, dataset_name),
                relative=str(p.relative_to(root)),
                size=stat.st_size,
                modified=stat.st_mtime,
                kind=kind,
            )
        )
    return results


def list_pgsr_outputs(root: Path, dataset_name: str) -> List[ResultFile]:
    """Find PGSR output mesh files (.ply, .obj) for the dataset."""
    dataset_dir = root / dataset_name
    if not dataset_dir.exists():
        return []
    results: List[ResultFile] = []
    for p in sorted(dataset_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _PGSR_MESH_EXTS:
            continue
        stat = p.stat()
        results.append(ResultFile(
            name=p.name,
            relative=str(p.relative_to(root)),
            size=stat.st_size,
            modified=stat.st_mtime,
            kind="ply" if p.suffix.lower() == ".ply" else "obj",
        ))
    return sorted(results, key=lambda f: f.modified, reverse=True)


# ---- Content patching --------------------------------------------------------
#
# The OBJ references its MTL by filename (`mtllib <name>.mtl`), and the MTL
# references its texture (`map_Kd <name>.png`). When we rename to mesh.* we
# have to patch those references or the textured mesh won't load.

_MTLLIB_RE = re.compile(rb"^(mtllib\s+)(\S+)", re.MULTILINE)
_MAP_RE = re.compile(rb"^(\s*map_\w+(?:\s+-\w+\s+\S+)*\s+)(\S+)", re.MULTILINE)


def patch_obj_bytes(content: bytes, dataset: str) -> bytes:
    """Rewrite every `mtllib <something>.mtl` line to point at <dataset>.mtl."""
    target = f"{dataset or 'mesh'}.mtl".encode()
    return _MTLLIB_RE.sub(lambda m: m.group(1) + target, content)


def patch_mtl_bytes(content: bytes, dataset: str) -> bytes:
    """Rewrite every `map_*` reference to point at <dataset>.<ext>."""
    stem = (dataset or "mesh").encode()

    def repl(m: "re.Match[bytes]") -> bytes:
        old = m.group(2)
        suffix = Path(old.decode("latin-1")).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg"}:
            return m.group(1) + stem + suffix.encode("ascii")
        return m.group(0)

    return _MAP_RE.sub(repl, content)


def read_patched(disk_path: Path, dataset: str) -> bytes:
    """Return the file content with friendly cross-references applied."""
    data = disk_path.read_bytes()
    ext = disk_path.suffix.lower()
    if ext == ".obj":
        return patch_obj_bytes(data, dataset)
    if ext == ".mtl":
        return patch_mtl_bytes(data, dataset)
    return data
