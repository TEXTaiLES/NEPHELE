from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

STATUS_FILENAME = "__pipeline_status.json"

# Default stage order, used when no status file exists yet so the UI can
# still draw greyed-out pills.
DEFAULT_STAGES = ("sam2", "colmap", "sugar")


@dataclass
class PipelineStatus:
    dataset: str
    stages: List[str] = field(default_factory=lambda: list(DEFAULT_STAGES))
    current: int = -1
    status: str = "idle"              # "idle" | "running" | "done" | "error"
    message: str = ""
    started_at: Optional[float] = None
    updated_at: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "stages": self.stages,
            "current": self.current,
            "status": self.status,
            "message": self.message,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }


def status_path(indexed_dir: Path) -> Path:
    return indexed_dir / STATUS_FILENAME


def request_kill(in_mnt: Path) -> None:
    """Signal pipeline_watcher.sh to kill any running pipeline on next poll."""
    (in_mnt / ".kill_pipeline").touch()


def status_from_job(job, dataset_name: str) -> PipelineStatus:
    """Map a vm_comms ``Job`` onto a :class:`PipelineStatus`.

    Lets the ``/pipeline/status`` route render identically whether progress
    came from the host-written JSON (shared_fs) or the HESTIA job row
    (vm_comms). ``job`` is a :class:`app.services.vm_comms.Job`.
    """
    from .vm_comms import (
        STATUS_CANCELLED, STATUS_DONE, STATUS_ERROR, STATUS_PREVIEWING,
        STATUS_RUNNING,
    )

    if job.status in (STATUS_RUNNING, STATUS_PREVIEWING):
        ui_status = "running"
    elif job.status == STATUS_DONE:
        ui_status = "done"
    elif job.status == STATUS_ERROR:
        ui_status = "error"
    elif job.status == STATUS_CANCELLED:
        ui_status = "cancelled"
    else:
        ui_status = "idle"

    return PipelineStatus(
        dataset=dataset_name,
        stages=list(DEFAULT_STAGES),
        current=job.stage_index,
        status=ui_status,
        message=job.message,
        error=job.error,
    )


def read_status(indexed_dir: Path, dataset_name: str) -> PipelineStatus:
    """Return the current pipeline status. Falls back to 'idle' if the file
    doesn't exist yet (pipeline hasn't been triggered)."""
    fp = status_path(indexed_dir)
    if not fp.is_file():
        return PipelineStatus(dataset=dataset_name)
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return PipelineStatus(dataset=dataset_name, status="error", error="status file unreadable")

    return PipelineStatus(
        dataset=data.get("dataset", dataset_name),
        stages=list(data.get("stages") or DEFAULT_STAGES),
        current=int(data.get("current", -1)),
        status=str(data.get("status", "idle")),
        message=str(data.get("message", "")),
        started_at=data.get("started_at"),
        updated_at=data.get("updated_at"),
        error=data.get("error"),
    )
