"""UI-side client for the HESTIA ``vm_comms`` API.

``vm_comms`` replaces the legacy shared-filesystem handshake with the SAM2
worker: instead of writing ``prompts.json`` / flag files to a disk the worker
also mounts, the UI creates a job row in HESTIA and polls it. The worker is a
separate client of the same API.

HESTIA endpoints (Flask-RESTful, NefeleResource family):
  POST   /nefele                        create job (this module, step 4)
  GET    /nefele/<job_id>               poll status (step 7)
  PATCH  /nefele/<job_id>               send decision / cancel (steps 8, 11)
  POST   /nefele/claim                  worker only — atomic claim
  POST   /nefele/<job_id>/preview       worker only — upload masks

HESTIA also publishes Kafka topics (notification-only, source of truth is
the Postgres row):
  nefele_job_created   after POST /nefele           payload {job_id, scan_id}
  nefele_job_modified  after PATCH / preview         payload {job_id, status}
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

log = logging.getLogger(__name__)

_API_BASE = os.environ.get("HESTIA_API_URL", "https://api.textailes.athenarc.gr").rstrip("/")
_API_KEY = os.environ.get("HESTIA_API_KEY", "")
VM_COMMS_EP = f"{_API_BASE}/nefele"

_HTTP_TIMEOUT = 20

STATUS_POINTS_SUBMITTED = "points_submitted"
STATUS_PREVIEWING = "previewing"
STATUS_PREVIEW_READY = "preview_ready"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"

ACTIVE_STATUSES = (
    STATUS_POINTS_SUBMITTED,
    STATUS_PREVIEWING,
    STATUS_PREVIEW_READY,
    STATUS_RUNNING,
)
TERMINAL_STATUSES = (STATUS_DONE, STATUS_ERROR, STATUS_CANCELLED)

DECISION_CONFIRM = "confirm"
DECISION_REDO = "redo"
DECISION_USE_EXISTING = "use_existing"


class VmCommsError(RuntimeError):
    """Raised when a vm_comms call fails or a polled job ends in error."""


@dataclass
class Job:
    """A single vm_comms job row. Only fields the UI reads are typed; anything
    else the server returns is kept in ``raw``."""

    job_id: str
    scan_id: str = ""
    dataset_name: str = ""
    model: str = "sugar"
    status: str = ""
    preview: List[str] = field(default_factory=list)
    stage: str = ""
    stage_index: int = -1
    message: str = ""
    error: Optional[str] = None
    raw: Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict) -> "Job":
        return cls(
            job_id=str(d.get("job_id") or d.get("id") or ""),
            scan_id=d.get("scan_id", ""),
            dataset_name=d.get("dataset_name", ""),
            model=d.get("model", "sugar"),
            status=d.get("status", ""),
            preview=list(d.get("preview") or []),
            stage=d.get("stage", ""),
            stage_index=int(d.get("stage_index", -1)),
            message=d.get("message", ""),
            error=d.get("error"),
            raw=d,
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


def _headers() -> dict:
    return {"Authorization": f"Bearer {_API_KEY}"}


def _request(method: str, url: str, **kwargs) -> dict:
    """Issue a request and return the JSON body, raising VmCommsError on failure."""
    kwargs.setdefault("timeout", _HTTP_TIMEOUT)
    try:
        r = requests.request(method, url, headers=_headers(), **kwargs)
    except requests.RequestException as e:
        raise VmCommsError(f"vm_comms unreachable at {url}: {e}") from e
    if r.status_code >= 400:
        raise VmCommsError(f"vm_comms HTTP {r.status_code} for {method} {url}: {r.text[:300]}")
    try:
        return r.json()
    except ValueError as e:
        raise VmCommsError(f"vm_comms returned non-JSON from {url}") from e


def create_job(scan_id: str, dataset_name: str, model: str, points_json: dict) -> Job:
    """POST /vm-comms — create a job from the user's prompt points.

    ``points_json`` is the same dict ``services.prompts.save_prompts`` builds.
    HESTIA resources wrap a created record as ``{message, job_id, data}`` —
    unwrap ``data`` when present.
    """
    body = {
        "scan_id": scan_id,
        "dataset_name": dataset_name,
        "model": model if model in ("sugar", "pgsr") else "sugar",
        "points_json": points_json,
    }
    log.info("vm_comms: create job dataset=%s scan=%s", dataset_name, scan_id)
    resp = _request("POST", VM_COMMS_EP, json=body)
    job = Job.from_dict(resp.get("data") or resp)
    log.info("vm_comms: job created job_id=%s status=%s", job.job_id, job.status)
    return job


def get_job(job_id: str) -> Job:
    """GET /vm-comms/{job_id} — current job row."""
    return Job.from_dict(_request("GET", f"{VM_COMMS_EP}/{job_id}"))


def find_pending_for_scan(scan_id: str) -> Optional[Job]:
    """Look for an open ``preview_ready`` job tied to this scan_id.

    Used by the picker's /save handler to avoid creating a second job while a
    previous preview is still waiting for the user's Confirm/Redo decision —
    the worker_poller can only process one job at a time and would otherwise
    queue the new one indefinitely (single-host poller).

    Returns the newest matching job, or ``None`` if nothing is pending.
    """
    if not scan_id:
        return None
    try:
        rows = _request(
            "GET", VM_COMMS_EP,
            params={"scan_id": scan_id, "status": STATUS_PREVIEW_READY},
        )
    except VmCommsError:
        return None
    if isinstance(rows, dict):
        rows = rows.get("data") or []
    if not rows:
        return None
    rows.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return Job.from_dict(rows[0])


def cancel_job(job_id: str) -> bool:
    """Cancel a non-terminal job via ``PATCH /vm-comms/{job_id}``.

    HESTIA has no dedicated /cancel endpoint; status is set to 'cancelled'
    through the normal PATCH worker path (``{"status": "cancelled"}``).
    We do a preflight GET to avoid a no-op PATCH when the job is already
    terminal.

    Returns True if the PATCH was sent, False if the job was already terminal.
    Raises VmCommsError on any HTTP failure.
    """
    try:
        job = get_job(job_id)
    except VmCommsError:
        return False
    if job.is_terminal:
        log.info("cancel: job %s already terminal (%s) — skipping", job_id, job.status)
        return False
    _request("PATCH", f"{VM_COMMS_EP}/{job_id}", json={"status": STATUS_CANCELLED})
    log.info("cancelled job %s", job_id)
    return True


def list_active_jobs(scan_id: Optional[str] = None) -> List[Job]:
    """Return every job currently in a non-terminal state.

    Issues one GET per active status because HESTIA's list endpoint filters on
    a single ``status`` value at a time. The result is small in practice (the
    worker processes one job at a time), so 4 calls is cheap and avoids
    fetching the entire jobs table.
    """
    out: Dict[str, Job] = {}
    for s in ACTIVE_STATUSES:
        params: Dict[str, str] = {"status": s}
        if scan_id:
            params["scan_id"] = scan_id
        try:
            resp = _request("GET", VM_COMMS_EP, params=params)
        except VmCommsError as e:
            log.warning("list_active_jobs(%s) failed: %s", s, e)
            continue
        rows = resp.get("data") if isinstance(resp, dict) else resp
        for row in rows or []:
            job = Job.from_dict(row)
            if job.job_id:
                out[job.job_id] = job
    return list(out.values())


def submit_instructions(
    job_id: str, decision: str, points_json: Optional[dict] = None
) -> Job:
    """PATCH /vm-comms/{job_id} — confirm / redo / use_existing.

    HESTIA mutates rows via PATCH (cf. annotation.py), so the decision is sent
    as ``{"instructions": {...}}``; the server derives the new ``status``.
    ``points_json`` may be re-sent with a ``redo`` so the user can adjust the
    points without creating a fresh job. PATCH returns only a confirmation, so
    the fresh row is fetched with a follow-up GET.
    """
    if decision not in (DECISION_CONFIRM, DECISION_REDO, DECISION_USE_EXISTING):
        raise VmCommsError(f"unknown decision: {decision}")
    instructions: dict = {"decision": decision}
    if points_json is not None:
        instructions["points_json"] = points_json
    log.info("vm_comms: job %s instruction=%s", job_id, decision)
    _request("PATCH", f"{VM_COMMS_EP}/{job_id}", json={"instructions": instructions})
    return get_job(job_id)


def wait_for(
    job_id: str,
    statuses: tuple,
    *,
    timeout: float,
    interval: float = 2.0,
) -> Job:
    """Poll GET /vm-comms/{job_id} until ``status`` is in ``statuses``.

    Returns the job as soon as it reaches a wanted status. Raises VmCommsError
    if it lands in ``error`` first or if ``timeout`` seconds elapse.
    """
    deadline = time.monotonic() + timeout
    last: Optional[Job] = None
    while time.monotonic() < deadline:
        last = get_job(job_id)
        if last.status in statuses:
            return last
        if last.status == STATUS_ERROR and STATUS_ERROR not in statuses:
            raise VmCommsError(last.error or "worker reported failure")
        time.sleep(interval)
    raise VmCommsError(
        f"timed out after {timeout:.0f}s waiting for {statuses} "
        f"(last status: {last.status if last else 'unknown'})"
    )
