# Nefele UI

The Point Picker web UI, extracted from the `samplify_sugar` repo into a
standalone project so it can be built and hosted on its own server.

It is a CPU-only Flask app (no torch/CUDA). It lets a user upload images or
browse robot scans, place SAM2 prompt points, preview masks, watch pipeline
progress, and download the final reconstruction.

## Layout

```
nefele_ui/
  app/                 Flask application package
    __init__.py        create_app() factory
    wsgi.py            entry point (python -m app.wsgi)
    config.py          env-driven runtime config
    auth.py            optional Directus auth
    routes/            blueprints (home, picker, preview, results, setup,
                       welcome, hestia)
    services/          frames, prompts, uploads, pipeline, results,
                       worker_client, hestia
    templates/  static/
  Dockerfile           self-contained CPU image
  docker-compose.yml   single `ui` service
  requirements.txt
  .env / .env.example
  data/                local in/out/results mounts
```

## Run

```bash
cp .env.example .env   # then edit HESTIA_API_KEY etc.
docker compose up --build
```

UI is served on `http://localhost:${WEB_PORT}` (default 8092).

## Known coupling — not yet fully portable

This is step 1 (code extraction). The HESTIA-backed flows (scan browsing,
reconstruction upload) already work purely over HTTP. But these features still
assume a **shared filesystem** with the SAM2 worker / pipeline host:

- mask preview — `services/worker_client.py` passes filesystem paths to the
  worker and expects it to read the same disk
- pipeline status — `services/pipeline.py` reads `__pipeline_status.json`
  written by `run_pipeline.sh` on the host
- local results — `routes/results.py` serves meshes from mounted output dirs

Running on a truly separate server requires the planned `vm_comms` API
refactor (route preview / instructions / status through the HESTIA DB instead
of shared volumes).
