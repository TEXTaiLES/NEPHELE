# vm_comms — HESTIA implementation guide

What the **HESTIA team** must build so the Nefele UI ↔ SAM VM `vm_comms` loop
works. This is the detailed counterpart of `vm_comms_contract.md`.

`vm_comms` is one new flask-restful resource that follows the exact pattern of
the existing HESTIA resources (`resources/robot.py`, `resources/reconstruction.py`,
`resources/annotation.py`). Three pieces:

1. a Postgres table `vm_comms`
2. a MinIO bucket for preview images
3. a resource file `resources/vm_comms.py` registered in `api.py`

### Do NOT route vm_comms through the Kafka sink — important

The existing tables (`robot_images`, `reconstructions`, ...) are **auto-created
by the JDBC sink connectors** (`auto.create: true`) and populated by
`send_avro_message` → topic → upsert. That pattern fits append-mostly records.

`vm_comms` is the opposite — mutable, high-churn job state with frequent PATCH
updates and an **atomic claim** (`FOR UPDATE SKIP LOCKED`, a pure-Postgres
operation Kafka Connect cannot do). So:

- **No `vm-comms-sink.json` connector.** Do not let Kafka Connect own this table.
- **The table needs an explicit `CREATE TABLE` migration** — unlike the other
  tables, it is *not* auto-created. The resource must be able to claim/PATCH
  rows from the first request.
- Avro/Kafka is optional — at most a lightweight `send_simple_message`
  notification. The UI and the worker poll, so it is not required.

This is exactly what `annotation.py` already does for its PATCH path:
direct Postgres SQL, no sink connector.

---

## 1. Postgres table

`vm_comms` is **not** auto-created by a connector — add this `CREATE TABLE` to
the migrations run by `scripts/setup_infrastructure.py` (`run_migrations()`):

```sql
CREATE TABLE IF NOT EXISTS vm_comms (
    job_id        UUID         PRIMARY KEY,
    scan_id       TEXT         NOT NULL,
    dataset_name  TEXT         NOT NULL,
    model         TEXT         NOT NULL DEFAULT 'sugar',   -- 'sugar' | 'pgsr'
    points_json   JSONB,                                   -- SAM2 prompt points
    preview       JSONB,                                   -- list of preview URLs
    instructions  JSONB,                                   -- {"decision": "...", ...}
    status        TEXT         NOT NULL DEFAULT 'points_submitted',
    stage         TEXT         DEFAULT '',
    stage_index   INTEGER      DEFAULT -1,
    message       TEXT         DEFAULT '',
    error         TEXT,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- the claim query filters on status; index it
CREATE INDEX IF NOT EXISTS idx_vm_comms_status ON vm_comms (status);
```

### status values (state machine)

```
points_submitted ─claim─> previewing ─> preview_ready
                                            │
              ┌──── PATCH decision=redo ─────┤
              ▼                              │ decision=confirm|use_existing
       points_submitted                      ▼
                                          running ─> done | error
```

| status             | set by | meaning |
|--------------------|--------|---------|
| `points_submitted` | UI / server | created, or sent back by `redo` |
| `previewing`       | claim | a worker claimed it |
| `preview_ready`    | preview upload | masks ready; UI shows them |
| `running`          | PATCH (instructions) | user confirmed; pipeline runs |
| `done`             | worker PATCH | finished |
| `error`            | worker PATCH | failed; see `error` |

---

## 2. MinIO bucket for previews

Add a bucket (e.g. `vm-comms-previews`) next to the existing buckets in
`services/storage.py` / `setup_minio()`:

```python
MINIO_VM_COMMS_BUCKET = os.environ.get("MINIO_VM_COMMS_BUCKET", "vm-comms-previews")
```

Preview images are stored here; their public URLs go into the `preview` column.

---

## 3. resources/vm_comms.py

Register in `api.py`:

```python
from resources.vm_comms import (
    VmCommsResource, VmCommsItemResource,
    VmCommsClaimResource, VmCommsPreviewResource,
)

api.add_resource(VmCommsResource,        '/vm-comms')
api.add_resource(VmCommsItemResource,    '/vm-comms/<string:job_id>')
api.add_resource(VmCommsClaimResource,   '/vm-comms/claim')
api.add_resource(VmCommsPreviewResource, '/vm-comms/<string:job_id>/preview')
```

### Shared helper — row → JSON dict

Same as `robot.py` does: `datetime` and `jsonb` columns need normalising.

```python
import json, io, uuid, logging
from datetime import datetime, timezone
from flask import request, jsonify
from flask_restful import Resource

from middleware.security import require_api_key
from services.database import get_db_connection
from services.storage import minio_client, build_public_url, MINIO_VM_COMMS_BUCKET

logger = logging.getLogger(__name__)

# columns the worker may write through PATCH
_WORKER_FIELDS = {'stage', 'stage_index', 'message', 'status', 'error'}


def _row_to_dict(colnames, row):
    d = {}
    for col, val in zip(colnames, row):
        if isinstance(val, datetime):
            d[col] = val.isoformat()
        else:
            d[col] = val            # psycopg2 returns JSONB already decoded
    return d


def _fetch_one(cur, job_id):
    cur.execute("SELECT * FROM vm_comms WHERE job_id = %s", (job_id,))
    row = cur.fetchone()
    if not row:
        return None
    return _row_to_dict([c[0] for c in cur.description], row)
```

### 3a. VmCommsResource — `POST /vm-comms` (step 4, create job)

```python
class VmCommsResource(Resource):
    method_decorators = [require_api_key]

    def post(self):
        data = request.get_json(silent=True) or {}
        scan_id      = data.get('scan_id')
        dataset_name = data.get('dataset_name')
        model        = data.get('model', 'sugar')
        points_json  = data.get('points_json')

        if not scan_id or not dataset_name:
            return {'error': "scan_id and dataset_name are required"}, 400
        if model not in ('sugar', 'pgsr'):
            model = 'sugar'

        job_id = str(uuid.uuid4())
        try:
            with get_db_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO vm_comms
                       (job_id, scan_id, dataset_name, model, points_json, status)
                       VALUES (%s, %s, %s, %s, %s, 'points_submitted')""",
                    (job_id, scan_id, dataset_name, model,
                     json.dumps(points_json) if points_json is not None else None),
                )
                row = _fetch_one(cur, job_id)
            return {'message': "vm_comms job created",
                    'job_id': job_id, 'data': row}, 201
        except Exception as e:
            logger.error(f"vm_comms create failed: {e}")
            return {'error': str(e)}, 500

    def get(self):
        """Optional list endpoint — handy for debugging, not used by clients."""
        status  = request.args.get('status')
        scan_id = request.args.get('scan_id')
        page     = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        offset   = (page - 1) * per_page

        sql, params = "SELECT * FROM vm_comms WHERE 1=1", []
        if status:
            sql += " AND status = %s";  params.append(status)
        if scan_id:
            sql += " AND scan_id = %s"; params.append(scan_id)
        sql += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params += [per_page, offset]

        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            cols = [c[0] for c in cur.description]
            rows = [_row_to_dict(cols, r) for r in cur.fetchall()]
        return jsonify(rows)
```

### 3b. VmCommsItemResource — `GET` & `PATCH /vm-comms/<job_id>` (steps 7/9/11, 8/10)

```python
class VmCommsItemResource(Resource):
    method_decorators = [require_api_key]

    def get(self, job_id):
        with get_db_connection() as conn, conn.cursor() as cur:
            row = _fetch_one(cur, job_id)
        if row is None:
            return {'error': f"job {job_id} not found"}, 404
        return jsonify(row)

    def patch(self, job_id):
        data = request.get_json(silent=True) or {}
        sets, params = [], []

        # --- UI path: instructions -> derive status -----------------------
        if 'instructions' in data:
            instr = data['instructions'] or {}
            decision = instr.get('decision')
            if decision not in ('confirm', 'redo', 'use_existing'):
                return {'error': f"bad decision: {decision}"}, 400
            sets.append("instructions = %s")
            params.append(json.dumps(instr))
            new_status = 'points_submitted' if decision == 'redo' else 'running'
            sets.append("status = %s")
            params.append(new_status)

        # --- worker path: progress fields ---------------------------------
        for field in ('stage', 'message', 'status', 'error'):
            if field in data and 'instructions' not in data:
                sets.append(f"{field} = %s")
                params.append(data[field])
        if 'stage_index' in data and 'instructions' not in data:
            sets.append("stage_index = %s")
            params.append(int(data['stage_index']))

        if not sets:
            return {'error': "no updatable fields provided"}, 400

        sets.append("updated_at = now()")
        params.append(job_id)
        try:
            with get_db_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    f"UPDATE vm_comms SET {', '.join(sets)} "
                    f"WHERE job_id = %s RETURNING job_id",
                    tuple(params),
                )
                if cur.fetchone() is None:
                    return {'error': f"job {job_id} not found"}, 404
            return {'message': "vm_comms job updated", 'job_id': job_id}, 200
        except Exception as e:
            logger.error(f"vm_comms patch failed: {e}")
            return {'error': str(e)}, 500
```

### 3c. VmCommsClaimResource — `POST /vm-comms/claim` (step 5, atomic claim)

The only non-trivial endpoint. `FOR UPDATE SKIP LOCKED` guarantees two workers
polling at once never grab the same job.

```python
class VmCommsClaimResource(Resource):
    method_decorators = [require_api_key]

    def post(self):
        try:
            with get_db_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE vm_comms SET status = 'previewing', updated_at = now()
                    WHERE job_id = (
                        SELECT job_id FROM vm_comms
                        WHERE status = 'points_submitted'
                        ORDER BY created_at
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    RETURNING *
                    """
                )
                row = cur.fetchone()
                if row is None:
                    return '', 204          # nothing to claim
                result = _row_to_dict([c[0] for c in cur.description], row)
            return jsonify(result)
        except Exception as e:
            logger.error(f"vm_comms claim failed: {e}")
            return {'error': str(e)}, 500
```

### 3d. VmCommsPreviewResource — `POST /vm-comms/<job_id>/preview` (step 6)

Multipart upload, same convention as `robot.py` — files under field `file`.

```python
class VmCommsPreviewResource(Resource):
    method_decorators = [require_api_key]

    def post(self, job_id):
        files = request.files.getlist('file')
        if not files or files[0].filename == '':
            return {'error': "No preview file(s) provided."}, 400

        urls = []
        try:
            for f in files:
                data = f.read()
                object_name = f"{job_id}/{f.filename}"
                minio_client.put_object(
                    MINIO_VM_COMMS_BUCKET, object_name,
                    io.BytesIO(data), len(data),
                    content_type=f.content_type or 'image/png',
                )
                urls.append(build_public_url(MINIO_VM_COMMS_BUCKET, object_name))

            with get_db_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """UPDATE vm_comms
                       SET preview = %s, status = 'preview_ready', updated_at = now()
                       WHERE job_id = %s RETURNING job_id""",
                    (json.dumps(urls), job_id),
                )
                if cur.fetchone() is None:
                    return {'error': f"job {job_id} not found"}, 404
            return {'message': "preview stored", 'preview': urls}, 200
        except Exception as e:
            logger.error(f"vm_comms preview failed: {e}")
            return {'error': str(e)}, 500
```

---

## 4. End-to-end behaviour the clients expect

| # | Client call | HESTIA must |
|---|-------------|-------------|
| 4 | `POST /vm-comms` | insert row, return `{job_id, data}` |
| 5 | `POST /vm-comms/claim` | atomically hand out one `points_submitted` job, or `204` |
| 6 | `POST /vm-comms/{id}/preview` | store images, set `preview` + `status=preview_ready` |
| 7/11 | `GET /vm-comms/{id}` | return the full row |
| 8 | `PATCH /vm-comms/{id}` `{instructions}` | save instructions, set `status` |
| 9 | `GET /vm-comms/{id}` | return the row (worker reads `instructions`) |
| 10 | `PATCH /vm-comms/{id}` `{stage,...}` | update progress columns |

Auth: every endpoint uses `@require_api_key` — `Authorization: Bearer <API_SECRET_KEY>`.

## 4b. Optional — Kafka notifications

Polling (client-side) is sufficient and the contract works **without any
Kafka changes**. If you want event-driven sync (the worker stops polling
`/claim`; UI gets push updates), add lightweight fire-and-forget notifications
to two new topics — same pattern `annotation.py` already uses for
`TOPIC_ANNOTATION_MODIFIED`.

These are **notification topics only**. Do NOT route them through a JDBC sink:
the `vm_comms` row is mutated by direct SQL in the resource (PATCH); the
topics carry just `{job_id}` as a signal.

### Step 1 — add topic constants in `services/messaging.py`

```python
TOPIC_NEFELE_JOB_CREATED  = 'nefele_job_created'
TOPIC_NEFELE_JOB_MODIFIED = 'nefele_job_modified'
```

### Step 2 — emit notifications inside the resource

After each successful write add a `send_simple_message` call — one line, can't
break the request even if Kafka is down:

```python
# In VmCommsResource.post() — right after the INSERT succeeded:
send_simple_message(TOPIC_VM_COMMS_CREATED, job_id,
                    {'job_id': job_id, 'scan_id': scan_id})

# In VmCommsItemResource.patch() — right after the UPDATE succeeded:
send_simple_message(TOPIC_VM_COMMS_UPDATED, job_id,
                    {'job_id': job_id, 'status': new_status_or_none})

# In VmCommsClaimResource.post() — after the atomic claim returned a row:
send_simple_message(TOPIC_VM_COMMS_UPDATED, job_id,
                    {'job_id': job_id, 'status': 'previewing'})

# In VmCommsPreviewResource.post() — after preview stored:
send_simple_message(TOPIC_VM_COMMS_UPDATED, job_id,
                    {'job_id': job_id, 'status': 'preview_ready'})
```

### Step 3 — clients pick mode by env

The Nefele worker (`worker_poller.py`) already supports both modes:
- `KAFKA_BROKER` unset → polls `POST /vm-comms/claim` every few seconds
- `KAFKA_BROKER` set → subscribes to `nefele_job_created` (group `sam-worker`)

The UI currently always polls; an optional Kafka→SSE bridge is a follow-up.

### What this changes server-side

- `VmCommsClaimResource` (`/vm-comms/claim`) **still exists** — it is used by
  workers running in polling mode and stays valuable as a fallback.
- No new connector, no new table, no Avro schema. Two new topic names and
  4 single-line `send_simple_message` calls.

## 5. Test checklist

1. `POST /vm-comms` → 201, row exists, `status=points_submitted`.
2. `POST /vm-comms/claim` → 200 with that row, `status=previewing`; second
   immediate call → `204`.
3. `POST /vm-comms/{id}/preview` with 2 files → `preview` has 2 URLs reachable
   from MinIO, `status=preview_ready`.
4. `PATCH /vm-comms/{id}` `{"instructions":{"decision":"confirm"}}` → `status=running`.
5. `PATCH /vm-comms/{id}` `{"status":"done","stage":"sugar"}` → columns updated.
6. `GET /vm-comms/{id}` → reflects every change above.
7. Two concurrent `claim` calls on two `points_submitted` rows → two *different*
   jobs (never the same one).
