# vm_comms contract (v2 — HESTIA flask-restful)

The `vm_comms` API replaces the shared-filesystem handshake between the Nefele
UI and the SAM VM (worker). Both sides are clients of HESTIA; they never share
a disk. The UI client is `app/services/vm_comms.py`; the worker client is
`samplify_sugar/SAM2/app/worker_poller.py`.

## HESTIA platform (confirmed)

HESTIA is a **custom Flask + flask-restful API** (port 5000), backed by
Postgres + Kafka + MinIO — *not* Directus (Directus is only the archive
portal). vm_comms must be added as a new resource that follows the existing
HESTIA pattern (see `resources/robot.py`, `resources/reconstruction.py`,
`resources/annotation.py` in the HESTIA repo):

- a flask-restful `Resource` class with `method_decorators = [require_api_key]`
- auth: `Authorization: Bearer <API_SECRET_KEY>` (exact-match)
- a Postgres table `vm_comms` (add to `scripts/setup_infrastructure.py`)
- binary payloads → MinIO bucket, public URLs stored on the row
- mutation via `PATCH` (like `annotation.py`)

Kafka/Avro is **optional** for vm_comms — unlike artifacts it is mutable,
short-lived job state, so a Postgres-only resource (as `annotation.py`'s PATCH
already does) is acceptable.

## Job resource — Postgres table `vm_comms`

One row = one reconstruction job.

| column         | type            | written by | notes |
|----------------|-----------------|-----------|-------|
| `job_id`       | uuid (PK)       | server    | generated on POST |
| `scan_id`      | text            | UI        | → `robot_images.scan_id` |
| `dataset_name` | text            | UI        | |
| `model`        | text            | UI        | `sugar` \| `pgsr` |
| `points_json`  | jsonb           | UI        | SAM2 prompt points |
| `preview`      | jsonb           | worker    | list of preview-image public URLs |
| `instructions` | jsonb           | UI        | `{"decision": "...", "points_json": {...}?}` |
| `status`       | text            | both      | state machine below |
| `stage`        | text            | worker    | `sam2`\|`colmap`\|`sugar`\|`pgsr` |
| `stage_index`  | int             | worker    | 0-based |
| `message`      | text            | worker    | progress line |
| `error`        | text            | worker    | set when `status=error` |
| `created_at`   | timestamptz     | server    | |
| `updated_at`   | timestamptz     | server    | bump on every write |

## Status state machine

```
points_submitted ─claim─> previewing ─> preview_ready
                                            │
              ┌──── PATCH decision=redo ─────┤
              ▼                              │ decision=confirm|use_existing
       points_submitted                      ▼
                                          running ─> done | error
```

## Endpoints (flask-restful, HESTIA style)

Registered in `api.py`:

```python
api.add_resource(VmCommsResource,        '/vm-comms')
api.add_resource(VmCommsItemResource,    '/vm-comms/<string:job_id>')
api.add_resource(VmCommsClaimResource,   '/vm-comms/claim')
api.add_resource(VmCommsPreviewResource, '/vm-comms/<string:job_id>/preview')
```

| # | Actor  | Method & path | Body | Result |
|---|--------|---------------|------|--------|
| 4 | UI     | `POST /vm-comms` | JSON `{scan_id, dataset_name, model, points_json}` | inserts row, `status=points_submitted`; returns `{message, job_id, data: <row>}` |
| 5 | worker | `POST /vm-comms/claim` | — | **atomic** claim of one `points_submitted` job → `previewing`; returns the row, or `204` if none |
| 6 | worker | `POST /vm-comms/{job_id}/preview` | multipart, files under field `file` | uploads to MinIO, sets `preview` URLs, `status=preview_ready` |
| 7 | UI     | `GET /vm-comms/{job_id}` | — | the job row (poll for `preview_ready`) |
| 8 | UI     | `PATCH /vm-comms/{job_id}` | JSON `{instructions: {decision, points_json?}}` | `confirm`/`use_existing`→`running`, `redo`→`points_submitted` |
| 9 | worker | `GET /vm-comms/{job_id}` | — | reads `instructions` |
| 10| worker | `PATCH /vm-comms/{job_id}` | JSON `{stage, stage_index, message, status, error?}` | updates progress columns |
| 11| UI     | `GET /vm-comms/{job_id}` | — | the job row (poll for pipeline progress) |

`GET /vm-comms?status=&scan_id=&page=&per_page=` (list, paginated like
`robot-images`) is optional — handy for debugging, not required by the clients.

### PATCH semantics (steps 8 & 10)

`PATCH /vm-comms/{job_id}` applies only an allow-list of columns (mirrors
`annotation.py`'s PATCH):

- from the UI: `instructions` — and the server derives the new `status`
  (`confirm`/`use_existing` → `running`, `redo` → `points_submitted`).
- from the worker: `stage`, `stage_index`, `message`, `status`, `error`.

Every PATCH bumps `updated_at`.

### Claim semantics (step 5) — the only non-trivial endpoint

The claim **must** be atomic so two workers never get the same job. Postgres
makes this a one-liner — no extra service needed:

```sql
UPDATE vm_comms SET status = 'previewing', updated_at = now()
WHERE job_id = (
    SELECT job_id FROM vm_comms
    WHERE status = 'points_submitted'
    ORDER BY created_at
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
RETURNING *;
```

Return the row as JSON, or HTTP `204` when nothing is claimable.

### Preview upload (step 6)

`POST /vm-comms/{job_id}/preview` is multipart with the image files under the
field name **`file`** (one or more — same convention as `robot-images` and
`reconstructions`). The handler uploads each to a MinIO bucket
(e.g. `vm-comms-previews`), writes the public URLs into the `preview` jsonb
column, and sets `status = preview_ready`.

## Synchronization — polling (default) or optional Kafka

State of truth is **always** the Postgres row; reads go through the REST
endpoints. Notifications can come over Kafka *or* the clients can just poll —
both work, the contract is identical.

### Polling (default, always works)

UI polls `GET /vm-comms/{job_id}`; worker polls `POST /vm-comms/claim`.
Suggested interval 2–5 s. No Kafka required.

### Optional Kafka notifications

When HESTIA produces lightweight events to these topics (same pattern as
`annotation_modified` — fire-and-forget `send_simple_message`, not the
JDBC-sink path), the worker can skip polling entirely and the UI can push
updates to the browser.

| Topic | When produced | Payload | Consumed by |
|-------|---------------|---------|-------------|
| `nefele_job_created`  | after a successful `POST /nefele` | `{job_id, scan_id}` | worker (group `sam-worker`) — Kafka consumer-group **is** the claim mechanism |
| `nefele_job_modified` | after each `PATCH /nefele/{id}` and after `/preview` | `{job_id, status}` | UI (optional SSE bridge) and worker (during a job, to react to user `instructions`) |

Important:
- These topics carry **only notifications** — the full row is read with
  `GET /vm-comms/{job_id}`. Do NOT JDBC-sink them.
- When the worker runs in Kafka mode, the `POST /vm-comms/claim` endpoint is
  not used (the consumer-group provides the same atomicity). The endpoint can
  still exist for clients running in polling mode.

## Notes for the HESTIA developer

1. New file `resources/vm_comms.py` — copy the shape of `annotation.py`
   (GET/POST/PATCH + `require_api_key`).
2. New table `vm_comms` — add its `CREATE TABLE` to the migrations run by
   `scripts/setup_infrastructure.py`.
3. The claim route is the only piece without a direct analogue — use the SQL
   above.
4. `reconstructions` and `robot-images` are reused unchanged by these clients.
