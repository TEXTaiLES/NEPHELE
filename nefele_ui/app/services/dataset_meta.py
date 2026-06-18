"""Small per-dataset metadata files the UI keeps on its *own* disk.

These thread two values between otherwise independent HTTP requests:

* ``.scan_id``      — the HESTIA scan a dataset came from (needed to create a
                      vm_comms job and let the worker fetch the images)
* ``.vm_comms_job`` — the id of the active vm_comms job for a dataset

They live on the UI's local volume, never on a shared mount, so they remain
valid in the decoupled vm_comms world. (``.model`` is written here too but is
still read by :func:`app.services.results.read_model`.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

SCAN_ID_FILE = ".scan_id"
MODEL_FILE = ".model"
JOB_ID_FILE = ".vm_comms_job"


def write_scan_meta(dataset_dir: Path, scan_id: str, model: str) -> None:
    """Persist the scan id and model choice in the dataset's input directory."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / SCAN_ID_FILE).write_text(scan_id or "", encoding="utf-8")
    (dataset_dir / MODEL_FILE).write_text(model or "sugar", encoding="utf-8")


def write_model(dataset_dir: Path, model: str) -> None:
    """Overwrite just the .model file without touching .scan_id."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / MODEL_FILE).write_text(model or "sugar", encoding="utf-8")


def read_scan_id(dataset_dir: Path) -> str:
    """Return the scan id for a dataset, or '' if it was not loaded from HESTIA."""
    f = dataset_dir / SCAN_ID_FILE
    if not f.is_file():
        return ""
    return f.read_text(encoding="utf-8").strip()


def write_job_id(indexed_dir: Path, job_id: str) -> None:
    """Record the active vm_comms job id for a dataset."""
    indexed_dir.mkdir(parents=True, exist_ok=True)
    (indexed_dir / JOB_ID_FILE).write_text(job_id, encoding="utf-8")


def read_job_id(indexed_dir: Path) -> Optional[str]:
    """Return the active vm_comms job id, or None if there is no job yet."""
    f = indexed_dir / JOB_ID_FILE
    if not f.is_file():
        return None
    return f.read_text(encoding="utf-8").strip() or None


def clear_job_id(indexed_dir: Path) -> None:
    """Forget the active vm_comms job (used on /restart)."""
    (indexed_dir / JOB_ID_FILE).unlink(missing_ok=True)
