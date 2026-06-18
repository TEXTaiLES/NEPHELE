#!/usr/bin/env python3
"""Cancel all active HESTIA nefele jobs (or a specific one by job_id).

Usage:
    python3 cancel_jobs.py              # cancel all active jobs
    python3 cancel_jobs.py <job_id>     # cancel one specific job
    python3 cancel_jobs.py --list       # list active jobs without cancelling
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

# ── config ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent
ENV_FILE = REPO / ".env"

if ENV_FILE.is_file():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() and k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip()

API_BASE = os.environ.get("HESTIA_API_URL", "https://api.textailes.athenarc.gr").rstrip("/")
API_KEY  = os.environ.get("HESTIA_API_KEY", "")
EP       = f"{API_BASE}/nefele"

ACTIVE_STATUSES = ("points_submitted", "previewing", "preview_ready", "running")

def _h() -> dict:
    return {"Authorization": f"Bearer {API_KEY}"}


def list_active() -> list[dict]:
    jobs: dict[str, dict] = {}
    for status in ACTIVE_STATUSES:
        r = requests.get(EP, headers=_h(), params={"status": status}, timeout=20)
        r.raise_for_status()
        data = r.json()
        rows = data.get("data") if isinstance(data, dict) else data
        for row in rows or []:
            jid = row.get("job_id") or row.get("id")
            if jid:
                jobs[jid] = row
    return list(jobs.values())


def cancel(job_id: str) -> bool:
    # preflight — skip if already terminal
    r = requests.get(f"{EP}/{job_id}", headers=_h(), timeout=20)
    if r.status_code == 404:
        print(f"  {job_id}: not found")
        return False
    r.raise_for_status()
    job = r.json()
    status = job.get("status", "")
    if status in ("done", "error", "cancelled"):
        print(f"  {job_id}: already terminal ({status}) — skipping")
        return False
    r2 = requests.patch(f"{EP}/{job_id}", headers=_h(), json={"status": "cancelled"}, timeout=20)
    r2.raise_for_status()
    print(f"  {job_id}: cancelled  (was {status}, dataset={job.get('dataset_name', '?')})")
    return True


def main() -> None:
    args = sys.argv[1:]

    if args == ["--list"]:
        jobs = list_active()
        if not jobs:
            print("No active jobs.")
            return
        print(f"{'job_id':<40} {'status':<20} dataset")
        print("-" * 80)
        for j in jobs:
            jid = j.get("job_id") or j.get("id", "?")
            print(f"{jid:<40} {j.get('status','?'):<20} {j.get('dataset_name','?')}")
        return

    if args:
        # cancel specific job(s)
        for job_id in args:
            cancel(job_id)
        return

    # cancel all active
    jobs = list_active()
    if not jobs:
        print("No active jobs to cancel.")
        return
    print(f"Found {len(jobs)} active job(s) — cancelling...")
    for j in jobs:
        jid = j.get("job_id") or j.get("id")
        cancel(jid)
    print("Done.")


if __name__ == "__main__":
    main()
