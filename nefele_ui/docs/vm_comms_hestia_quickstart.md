# vm_comms — οδηγός βήμα-βήμα για τον HESTIA dev

Συνοδευτικά: τα `vm_comms_contract.md` (το «τι») και `vm_comms_hestia_implementation.md` (αναλυτικός κώδικας).

Το `vm_comms` είναι ένα **mutable job state** resource για τη συγχρόνιση Nefele UI ↔ SAM VM worker. Διαφέρει από τα `robot_images` / `reconstructions`: γράφεται με `PATCH` (αλλαγές μερικών στηλών), όχι με upsert ολόκληρου record — οπότε **δεν** χρησιμοποιεί JDBC sink connector.

Όλες οι αλλαγές γίνονται στο **dev** branch.

---

## Βήμα 1 — `api/services/messaging.py`

Δίπλα στα υπάρχοντα `TOPIC_*` constants:

```python
TOPIC_NEFELE_JOB_CREATED  = 'nefele_job_created'
TOPIC_NEFELE_JOB_MODIFIED = 'nefele_job_modified'
```

Τίποτα άλλο εδώ. Οι `send_simple_message` / `send_avro_message` υπάρχουν ήδη.

---

## Βήμα 2 — `api/resources/vm_comms.py` (ΝΕΟ ΑΡΧΕΙΟ)

Δημιούργησε νέο αρχείο `api/resources/vm_comms.py` με αυτό το περιεχόμενο:

```python
import io
import json
import uuid
import logging
from datetime import datetime
from flask import request, jsonify
from flask_restful import Resource

from middleware.security import require_api_key
from services.database import get_db_connection
from services.storage import (
    minio_client,
    build_public_url,
    MINIO_VM_COMMS_BUCKET,
)
from services.messaging import (
    send_simple_message,
    TOPIC_VM_COMMS_CREATED,
    TOPIC_VM_COMMS_UPDATED,
)

logger = logging.getLogger(__name__)


# --- helpers ----------------------------------------------------------------
def _row_to_dict(colnames, row):
    """psycopg2 row -> JSON-safe dict (datetime -> isoformat)."""
    d = {}
    for col, val in zip(colnames, row):
        if isinstance(val, datetime):
            d[col] = val.isoformat()
        else:
            d[col] = val
    return d


def _fetch_one(cur, job_id):
    cur.execute("SELECT * FROM vm_comms WHERE job_id = %s", (job_id,))
    row = cur.fetchone()
    if not row:
        return None
    return _row_to_dict([c[0] for c in cur.description], row)


# --- POST /vm-comms ---------------------------------------------------------
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

            send_simple_message(TOPIC_VM_COMMS_CREATED, job_id,
                                {'job_id': job_id, 'scan_id': scan_id})

            return {'message': "vm_comms job created",
                    'job_id': job_id, 'data': row}, 201
        except Exception as e:
            logger.error(f"vm_comms create failed: {e}")
            return {'error': str(e)}, 500

    def get(self):
        """List endpoint — optional, για debugging."""
        status   = request.args.get('status')
        scan_id  = request.args.get('scan_id')
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


# --- GET / PATCH /vm-comms/<job_id> -----------------------------------------
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
        new_status = None

        # UI path — instructions -> derive status
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
        else:
            # Worker path — progress fields
            for field in ('stage', 'message', 'status', 'error'):
                if field in data:
                    sets.append(f"{field} = %s")
                    params.append(data[field])
                    if field == 'status':
                        new_status = data[field]
            if 'stage_index' in data:
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

            send_simple_message(TOPIC_VM_COMMS_UPDATED, job_id,
                                {'job_id': job_id, 'status': new_status})

            return {'message': "vm_comms job updated", 'job_id': job_id}, 200
        except Exception as e:
            logger.error(f"vm_comms patch failed: {e}")
            return {'error': str(e)}, 500


# --- POST /vm-comms/claim ---------------------------------------------------
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
                    return '', 204
                result = _row_to_dict([c[0] for c in cur.description], row)

            send_simple_message(TOPIC_VM_COMMS_UPDATED, result['job_id'],
                                {'job_id': result['job_id'], 'status': 'previewing'})

            return jsonify(result)
        except Exception as e:
            logger.error(f"vm_comms claim failed: {e}")
            return {'error': str(e)}, 500


# --- POST /vm-comms/<job_id>/preview ----------------------------------------
class VmCommsPreviewResource(Resource):
    method_decorators = [require_api_key]

    def post(self, job_id):
        files = request.files.getlist('file')
        if not files or files[0].filename == '':
            return {'error': "No preview file(s) provided."}, 400

        urls = []
        try:
            for f in files:
                blob = f.read()
                object_name = f"{job_id}/{f.filename}"
                minio_client.put_object(
                    MINIO_VM_COMMS_BUCKET, object_name,
                    io.BytesIO(blob), len(blob),
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

            send_simple_message(TOPIC_VM_COMMS_UPDATED, job_id,
                                {'job_id': job_id, 'status': 'preview_ready'})

            return {'message': "preview stored", 'preview': urls}, 200
        except Exception as e:
            logger.error(f"vm_comms preview failed: {e}")
            return {'error': str(e)}, 500
```

---

## Βήμα 3 — `api/api.py`

Δίπλα στα υπάρχοντα `from resources.* import ...`:

```python
from resources.vm_comms import (
    VmCommsResource,
    VmCommsItemResource,
    VmCommsClaimResource,
    VmCommsPreviewResource,
)
```

Μετά τα υπάρχοντα `api.add_resource(...)`:

```python
api.add_resource(VmCommsResource,        '/vm-comms')
api.add_resource(VmCommsItemResource,    '/vm-comms/<string:job_id>')
api.add_resource(VmCommsClaimResource,   '/vm-comms/claim')
api.add_resource(VmCommsPreviewResource, '/vm-comms/<string:job_id>/preview')
```

---

## Βήμα 4 — `api/scripts/setup_infrastructure.py`

### 4α. Στο `run_migrations()` — πριν το `conn.commit()`:

```python
# vm_comms — mutable job state (PATCH-heavy). NOT created by a JDBC sink
# because the row is updated, not upserted. Explicit migration required.
cur.execute("""
    CREATE TABLE IF NOT EXISTS vm_comms (
        job_id        UUID         PRIMARY KEY,
        scan_id       TEXT         NOT NULL,
        dataset_name  TEXT         NOT NULL,
        model         TEXT         NOT NULL DEFAULT 'sugar',
        points_json   JSONB,
        preview       JSONB,
        instructions  JSONB,
        status        TEXT         NOT NULL DEFAULT 'points_submitted',
        stage         TEXT         DEFAULT '',
        stage_index   INTEGER      DEFAULT -1,
        message       TEXT         DEFAULT '',
        error         TEXT,
        created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
        updated_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
    )
""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_vm_comms_status ON vm_comms (status)")
logger.info("Migration: vm_comms table ensured.")
```

### 4β. Στα imports του ίδιου αρχείου, πρόσθεσε `MINIO_VM_COMMS_BUCKET`:

```python
from services.storage import (
    init_minio_bucket,
    set_public_read_policy,
    MINIO_ARTIFACT_BUCKET,
    MINIO_ROBOT_BUCKET,
    MINIO_RECONSTRUCTION_BUCKET,
    MINIO_VM_COMMS_BUCKET,        # ← νέο
)
```

### 4γ. Στο `setup_minio()` — μετά τα υπάρχοντα bucket setups:

```python
# Setup VM Comms Previews (Public)
init_minio_bucket(MINIO_VM_COMMS_BUCKET)
set_public_read_policy(MINIO_VM_COMMS_BUCKET)
```

---

## Βήμα 5 — `api/services/storage.py`

Δίπλα στα υπάρχοντα `MINIO_*_BUCKET` constants:

```python
MINIO_VM_COMMS_BUCKET = os.environ.get("MINIO_VM_COMMS_BUCKET", "vm-comms-previews")
```

---

## Βήμα 6 — ❌ ΜΗΝ προσθέσεις connector

**Δεν δημιουργείς:** `docker/connectors/vm-comms-sink.json`.

**Λόγος:** Ο πίνακας `vm_comms` ενημερώνεται με `PATCH` (μερικές στήλες κάθε φορά), όχι με `upsert` ολόκληρου record. Το JDBC sink κάνει upsert — θα έσπαζε τη λογική (overwrite με `null` τα πεδία που δεν στάλθηκαν) και θα δημιουργούσε race conditions με το atomic claim.

Αν στο μέλλον επεκτείνεις τον `register_connectors()`, **μην** βάλεις vm_comms στη λίστα.

---

## Βήμα 7 — Restart

```bash
cd /path/to/HESTIA/docker
docker compose restart api
```

Το `run_migrations()` τρέχει στο boot και δημιουργεί τον πίνακα. Το `setup_minio()` φτιάχνει το bucket. Τα νέα Kafka topics δημιουργούνται αυτόματα στην πρώτη `send_simple_message` (το `cp-kafka` έχει `auto.create.topics.enable=true` by default).

---

## Επαλήθευση

```bash
# 1. Πίνακας υπάρχει
docker exec -it postgres psql -U admin -d mydb -c "\d vm_comms"

# 2. Bucket υπάρχει στο MinIO (http://localhost:9001 → vm-comms-previews)

# 3. Create job
curl -X POST http://localhost:5000/vm-comms \
  -H "Authorization: Bearer change-me-locally" \
  -H "Content-Type: application/json" \
  -d '{"scan_id":"test","dataset_name":"test","model":"sugar","points_json":{}}'

# 4. Claim (πρέπει να επιστρέψει το παραπάνω job, status=previewing)
curl -X POST http://localhost:5000/vm-comms/claim \
  -H "Authorization: Bearer change-me-locally"

# 5. 2ο claim (πρέπει να επιστρέψει 204)
curl -X POST http://localhost:5000/vm-comms/claim \
  -H "Authorization: Bearer change-me-locally" -i | head -1
```

---

## Σύνοψη

| # | Αρχείο | Είδος αλλαγής |
|---|--------|----------------|
| 1 | `services/messaging.py` | +2 topic constants |
| 2 | `resources/vm_comms.py` | **νέο αρχείο** (4 κλάσεις) |
| 3 | `api.py` | +1 import + 4 `add_resource` |
| 4 | `scripts/setup_infrastructure.py` | +CREATE TABLE + +MinIO bucket |
| 5 | `services/storage.py` | +1 constant |
| 6 | — | ❌ ΟΧΙ connector |
| 7 | — | restart api |

Όλα τα `send_simple_message(...)` καλούνται ήδη μέσα στα handlers του βήματος 2 — δεν χρειάζεται extra patching.
