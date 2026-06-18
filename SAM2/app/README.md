# Nephele Point Picker UI

CPU-only Flask app that lets a user **upload images**, **pick SAM2 prompt points**,
and signals the pipeline (`run_pipeline.sh`) to continue. Talks to the GPU
`sam2` worker over HTTP — no torch/CUDA dependency in this image.

---

## Folder layout

```
SAM2/app/
├── __init__.py              create_app() factory + rebind_dataset() helper
├── wsgi.py                  Entry point (python -m app.wsgi)
├── config.py                @dataclass Config + load_config()
├── auth.py                  Directus refresh-token auth (opt-in)
├── routes/
│   ├── _helpers.py          cfg(), frames(), json_ok / json_err
│   ├── welcome.py           GET /welcome  (landing screen)
│   ├── setup.py             GET/POST /setup  (dataset name + drag-drop)
│   ├── home.py              GET /, POST /use_existing, POST /create_new
│   ├── picker.py            GET /pick, POST /save, POST /confirm, /restart, GET /frame
│   ├── preview.py           GET /preview/<name>
│   └── results.py           GET /results, /results/files, /results/file/<rel>, /results/zip
├── services/
│   ├── frames.py            gather_frames, resolve_frames, clear_dir
│   ├── prompts.py           normalize_points, save_prompts
│   ├── uploads.py           sanitize_dataset_name, save_uploaded_images, write_active_dataset
│   ├── results.py           find_dataset_dir, list_outputs (SuGaR .obj/.mtl/textures)
│   └── worker_client.py     HTTP client → http://sam2:5001/preview
├── templates/
│   ├── base.html
│   ├── welcome.html
│   ├── setup.html
│   ├── home.html
│   ├── pick.html
│   ├── done.html
│   └── results.html
├── static/
│   ├── css/  (app.css, picker.css, welcome.css, setup.css, results.css)
│   ├── js/   (picker.js, setup.js, results.js)
│   └── img/  (nephele-logo.png)
├── worker_server.py         GPU worker HTTP API (runs in sam2 container)
└── video_predict.py         GPU mask propagation logic
```

The two `*.bak` files (`point_picker_flask.py.bak`, `point_picker_flask_login.py.bak`)
are the pre-refactor single-file versions kept for reference only.

---

## Request flow

```
Browser                Flask (ui container)            Worker (sam2 container)
   │                          │                                  │
   ├─ GET / ─────────────────▶│
   │                          ├─ no dataset? → 302 /welcome
   │                          ├─ no frames?  → 302 /setup
   │                          └─ ok          → render home.html
   │
   ├─ POST /setup ───────────▶│ save_uploaded_images() — Pillow converts to JPG
   │   (name + files)         │ write_active_dataset()  /data/in/.active_dataset
   │                          │ rebind_dataset(app, name) — in-process config swap
   │
   ├─ POST /save ────────────▶│ normalize_points → save_prompts
   │   {points}               │ clear_dir(preview)
   │                          │ run_preview_masks() ──── POST /preview ─────▶│
   │                          │                          {prompts_json,      │  (SAM2 inference,
   │                          │                           preview_dir}        │   writes PNGs to
   │                          │◀──────────── 200 {previews:[...]} ───────────┤   shared volume)
   │◀── 200 {previews:[...]} ─┤
   │
   ├─ GET /preview/x.png ────▶│ send_from_directory(preview_dir, ...)
```

---

## Configuration (environment variables)

| Name | Default | Purpose |
|---|---|---|
| `DATASET_NAME` | *(empty)* | Active dataset. Empty → setup mode. |
| `IN_MNT` | `/data/in` | Container path for the input mount. Dataset folder is `<IN_MNT>/<DATASET_NAME>`. |
| `OUT` | `/data/out` | Container path for the output mount. Indexed dir is `<OUT>/<DATASET_NAME><INDEX_SUFFIX>`. |
| `INDEX_SUFFIX` | `_indexed` | Suffix appended to dataset name for the output folder. |
| `WORKER_URL` | `http://sam2:5001` | URL of the GPU worker. |
| `WORKER_TIMEOUT` | `600` | HTTP timeout (seconds) when calling the worker. |
| `SUGAR_RESULTS_ROOT` | `/data/results/sugar` | Mount point for `SUGAR/SuGaR/obj_outputs/` — read by `/results`. |
| `AUTH_ENABLED` | `0` | `1` to enable Directus refresh-token middleware (see [auth.py](auth.py)). |
| `FLASK_DEBUG` | `0` | `1` enables Flask debug + DEBUG log level. |
| `PORT` | `5000` | Bind port inside the container (compose maps it to `WEB_PORT`). |
| `WEB_PORT` *(host)* | `8092` | Host port set by `.env`. Used by compose to publish container `:5000`. |

Dataset name resolution order at startup:
1. `DATASET_NAME` env var
2. Contents of `<IN_MNT>/.active_dataset`
3. Empty → app boots in **setup mode**

`.active_dataset` is also how `run_pipeline.sh` learns the user's choice when
called with no argument.

---

## Routes summary

| Path | Method | Action |
|---|---|---|
| `/` | GET | If unconfigured → `/welcome`. If 0 frames → `/setup`. Else home page. |
| `/welcome` | GET | Branded landing screen with "Get started" link to `/setup`. |
| `/setup` | GET | Dataset name input + drag-drop uploader. |
| `/setup` | POST | Save uploads (PIL → JPG), write `.active_dataset`, rebind config. Returns `{ok, saved, failed, total}`. |
| `/use_existing` | POST | Touch `__use_existing.flag` + `__picker_done.flag`. Renders `done.html`. |
| `/create_new` | POST | Delete previous `prompts.json` / `__use_existing.flag`. 302 → `/pick`. |
| `/pick` | GET | Canvas UI for marking POS/NEG points. |
| `/save` | POST | Write `prompts.json`, clear previews, call worker for preview. |
| `/confirm` | POST | Touch `__picker_done.flag` (pipeline proceeds). |
| `/restart` | POST | Clear prompts + previews + flags. |
| `/frame?i=N` | GET | Serve source image N to the canvas. |
| `/preview/<name>` | GET | Serve a worker-generated mask PNG. |
| `/results` | GET | Results page for the active dataset (polls for SuGaR output). |
| `/results/files` | GET | JSON `{ok, ready, files: [{name, relative, size, modified, kind}]}`. |
| `/results/file/<rel>` | GET | Download a single output file as attachment. Path-traversal-safe. |
| `/results/zip` | GET | Bundle all current SuGaR outputs into one `.zip`. |

---

## Running the UI

The UI container (`ui` in [docker-compose.yml](../../docker-compose.yml)) bind-mounts
`./SAM2:/workspace`, so code edits take effect after a process restart — you
don't need to rebuild the image for Python / template / CSS / JS changes.

### Option 1 — Through the pipeline (typical)

```bash
# Explicit dataset name (matches old behavior)
./run_pipeline.sh dress

# No name → UI boots in setup mode; pipeline waits for the user to upload
./run_pipeline.sh
```

When the user submits `/setup`, the script reads the chosen name from
`SAM2/data/input/.active_dataset` and continues normally. The coordination
file is cleaned up at the end of the run.

### Option 2 — UI only (just the container, no pipeline)

```bash
# unset DATASET_NAME so the UI starts in setup mode
unset DATASET_NAME
docker compose up -d --force-recreate ui
# open http://localhost:8092/
```

Or with a pre-existing dataset:

```bash
DATASET_NAME=dress docker compose up -d --force-recreate ui
```

### Option 3 — With Directus auth turned on

Add to `.env`:
```bash
AUTH_ENABLED=1
```
Then recreate the UI:
```bash
docker compose up -d --force-recreate ui
```
Every non-`/static/` request will be gated by the Directus refresh-token check
in [auth.py](auth.py).

### Option 4 — Debug mode (verbose logs, auto-reload)

```bash
FLASK_DEBUG=1 DATASET_NAME=dress docker compose up -d --force-recreate ui
docker compose logs -f ui
```

### Option 5 — Direct exec without docker compose

```bash
docker run --rm -it \
  -v "$(pwd)/SAM2":/workspace \
  -v "$(pwd)/SAM2/data/input":/data/in \
  -v "$(pwd)/SAM2/data/output":/data/out \
  -p 8092:5000 \
  -e DATASET_NAME="" \
  -e IN_MNT=/data/in \
  -e OUT=/data/out \
  -e WORKER_URL=http://sam2:5001 \
  -w /workspace \
  sam2-ui:local \
  python3 -u -m app.wsgi
```

---

## Auto-triggered pipeline

Picking points + clicking **Looks good, continue** writes `__picker_done.flag`
into the dataset's indexed dir. The host-side watcher [pipeline_watcher.sh](../../pipeline_watcher.sh)
picks that up and spawns `run_pipeline.sh <dataset>` automatically — no
terminal babysitting required.

**Start the watcher once per host** (will run forever):

```bash
# in tmux, screen, or systemd; works fine with nohup too:
nohup ./pipeline_watcher.sh > /dev/null 2>&1 &
```

It scans `SAM2/data/output/*_indexed/` every 3 seconds and marks each flag as
`__picker_done.processed` after triggering, so the same dataset never starts
twice.

`run_pipeline.sh` writes `__pipeline_status.json` next to the picker flag at
every stage transition. The UI's [GET /pipeline/status](routes/results.py)
returns the latest snapshot:

```json
{
  "dataset": "dress_with_output",
  "stages": ["sam2", "colmap", "sugar"],
  "current": 1,
  "status": "running",
  "message": "Running COLMAP structure-from-motion",
  "started_at": 1778621508,
  "updated_at": 1778621600,
  "error": null
}
```

The [/results](templates/results.html) page polls this every 5s and updates
three stage pills (SAM2 / COLMAP / SuGaR) — pending → running (pulsing blue)
→ done (green) → error (red).

### Future: UI on a different server than the GPU containers

The current design assumes UI and pipeline scripts share a filesystem so the
flag + status JSON work as a coordination channel. When you split them:

1. Replace [pipeline_watcher.sh](../../pipeline_watcher.sh) with a tiny HTTP
   service on the GPU host that exposes `POST /trigger` and `GET /status`.
2. Add a `PIPELINE_API_URL` env var to the UI; have `/confirm` POST to it
   and have `/pipeline/status` proxy the GET.
3. Keep `__pipeline_status.json` as the on-disk format — the HTTP service
   just serializes it over the wire.

The stage-pill UI, status schema, and Flask routes don't change.

---

## Results page

After the user clicks **Looks good, continue** in the picker, the "All set"
overlay now offers an **Open results page** button. That page (`/results`):

- Polls `/results/files` every 5 seconds until SuGaR finishes writing.
- Renders each output as a row with kind pill (OBJ / MTL / TEXTURE), name,
  size, modified time, and a per-file Download button.
- Offers a **Download all (.zip)** button that streams a bundled archive.

The UI scans `SUGAR_RESULTS_ROOT` (default `/data/results/sugar`, bind-mounted
from `./SUGAR/SuGaR/obj_outputs`) for either:

- `<root>/<dataset>/` — flat layout, or
- `<root>/<group>/<dataset>/` — when SuGaR is invoked with `OUTPUT_GROUP`

The first directory containing at least one `.obj` wins; only `.obj`, `.mtl`,
and texture images (`.png`/`.jpg`) are surfaced.

---

## How upload conversion works

The setup screen accepts any image format the browser recognizes
(`accept="image/*"`). Server-side, [services/uploads.py](services/uploads.py)
does the following per file:

1. `Image.open(file.stream)` — Pillow decodes (JPG/PNG/WebP/BMP/TIFF/GIF/HEIC...).
2. `ImageOps.exif_transpose(img)` — applies EXIF rotation to pixel data.
3. If alpha channel present, composites onto a white background.
4. Saves as `<stem>.jpg` at JPEG quality 95. Collisions get `_1`, `_2`, ...
5. Files that Pillow can't decode go into the `failed` list returned to the UI.

This makes the post-upload PNG→JPG step in `run_pipeline.sh:152` a no-op for
UI-uploaded data, but it still runs as a safety net for files dropped in
manually.

---

## Smoke testing without containers

Pure logic (config, frames, uploads, prompts) can be exercised on the host:

```bash
cd /opt/samplify_sugar/SAM2
python3 -m py_compile $(find app -name '*.py')
```

For a quick functional check that doesn't need Flask:

```bash
python3 -c "
import sys, os, importlib.util
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); sys.modules[n] = m
    s.loader.exec_module(m); return m

os.environ.update(IN_MNT='/tmp/_in', OUT='/tmp/_out', DATASET_NAME='demo')
os.makedirs('/tmp/_in/demo', exist_ok=True)
config = load('app/config.py', 'config')
print(config.load_config())
"
```

Inside the container:

```bash
# 1. UI smoke test (no worker needed for /welcome and /setup)
docker compose up -d --force-recreate ui
curl -i http://localhost:8092/welcome
curl -I http://localhost:8092/static/css/welcome.css
curl -I http://localhost:8092/static/img/nephele-logo.png

# 2. End-to-end (needs sam2 worker)
docker compose up -d --force-recreate ui sam2
# open http://localhost:8092/ in browser
```

---

## Common operations

**Reset to setup screen:**
```bash
rm -f SAM2/data/input/.active_dataset
unset DATASET_NAME
docker compose up -d --force-recreate ui
```

**Tail logs:**
```bash
docker compose logs -f ui
```

**Hot-reload (templates / CSS / JS only):** just refresh the browser. They're
served from the bind-mounted `./SAM2/app/` directory.

**Hot-reload (Python):** Python is loaded once at process start; restart the
container:
```bash
docker compose restart ui
```

**Re-build the image (after Dockerfile_ui or pip dep changes):**
```bash
docker compose build ui
docker compose up -d --force-recreate ui
```

---

## Adding features

- **New route:** add a blueprint under `routes/`, register it in
  [`routes/__init__.py`](routes/__init__.py) and [`__init__.py`](__init__.py).
- **New service / shared logic:** add a module under `services/`. Keep it pure
  Python (no Flask imports) so it can be unit-tested.
- **New page:** template under `templates/` extending `base.html`. Static
  assets under `static/css/` and `static/js/`. Reference via `url_for()`.
- **New config knob:** add a field to `Config` in [`config.py`](config.py) and
  read it from `os.environ` inside `load_config()`.
