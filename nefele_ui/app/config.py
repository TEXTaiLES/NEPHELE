"""Runtime configuration for the Point Picker UI.

All settings are read from environment variables so the same image can run in
different deployments (different datasets, different worker URLs, with or
without Directus auth) without code changes.

When DATASET_NAME is unset, the loader falls back to a coordination file at
``<IN_MNT>/.active_dataset`` so the choice survives container restarts and is
visible to ``run_pipeline.sh``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Filename used to coordinate the user-chosen dataset between the UI and the
# pipeline shell script. Lives at the root of the input mount.
ACTIVE_DATASET_FILE = ".active_dataset"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_active_dataset(in_mnt: Path) -> Optional[str]:
    f = in_mnt / ACTIVE_DATASET_FILE
    if not f.is_file():
        return None
    name = f.read_text(encoding="utf-8").strip()
    return name or None


@dataclass(frozen=True)
class Config:
    dataset_name: str          # empty string means "setup mode"
    in_mnt: Path               # /data/in
    out_root: Path             # /data/out
    index_suffix: str
    worker_url: str
    worker_timeout: float
    auth_enabled: bool
    debug: bool
    sugar_results_root: Path   # mount of the SuGaR obj_outputs/ directory
    pgsr_results_root: Path    # mount of the PGSR outputs/ directory
    comms_backend: str         # "shared_fs" (legacy) | "vm_comms" (HESTIA API)
    poll_interval: float       # seconds between vm_comms job polls

    @property
    def is_configured(self) -> bool:
        return bool(self.dataset_name)

    @property
    def uses_vm_comms(self) -> bool:
        """True when the UI talks to the worker through the HESTIA vm_comms API
        instead of the legacy shared-filesystem handshake."""
        return self.comms_backend == "vm_comms"

    @property
    def ds_name(self) -> str:
        return self.dataset_name

    @property
    def input_dir(self) -> Path:
        return self.in_mnt / self.dataset_name

    @property
    def indexed_dir(self) -> Path:
        return self.out_root / f"{self.dataset_name}{self.index_suffix}"

    @property
    def prompts_json(self) -> Path:
        return self.indexed_dir / "prompts.json"

    @property
    def done_flag(self) -> Path:
        return self.indexed_dir / "__picker_done.flag"

    @property
    def use_existing_flag(self) -> Path:
        return self.indexed_dir / "__use_existing.flag"

    @property
    def preview_dir(self) -> Path:
        return self.indexed_dir / "preview"

    @property
    def active_dataset_file(self) -> Path:
        return self.in_mnt / ACTIVE_DATASET_FILE

    def ensure_dirs(self) -> None:
        if self.is_configured:
            self.indexed_dir.mkdir(parents=True, exist_ok=True)
            self.preview_dir.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    """Build a Config from the current environment.

    Dataset name resolution order:
      1. ``DATASET_NAME`` env var
      2. ``<IN_MNT>/.active_dataset`` file (written by /setup)
      3. empty string -> setup mode
    """
    in_mnt = Path(os.environ.get("IN_MNT", "/data/in"))
    in_mnt.mkdir(parents=True, exist_ok=True)
    out_root = Path(os.environ.get("OUT", "/data/out"))
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_name = os.environ.get("DATASET_NAME", "").strip()
    if not dataset_name:
        dataset_name = _read_active_dataset(in_mnt) or ""

    cfg = Config(
        dataset_name=dataset_name,
        in_mnt=in_mnt,
        out_root=out_root,
        index_suffix=os.environ.get("INDEX_SUFFIX", "_indexed"),
        worker_url=os.environ.get("WORKER_URL", "http://sam2:5001").rstrip("/"),
        worker_timeout=float(os.environ.get("WORKER_TIMEOUT", "600")),
        auth_enabled=_env_bool("AUTH_ENABLED", default=False),
        debug=_env_bool("FLASK_DEBUG", default=False),
        sugar_results_root=Path(os.environ.get("SUGAR_RESULTS_ROOT", "/data/results/sugar")),
        pgsr_results_root=Path(os.environ.get("PGSR_RESULTS_ROOT", "/data/results/pgsr")),
        comms_backend=os.environ.get("COMMS_BACKEND", "shared_fs").strip().lower(),
        poll_interval=float(os.environ.get("VM_COMMS_POLL_INTERVAL", "2")),
    )
    cfg.ensure_dirs()
    return cfg
