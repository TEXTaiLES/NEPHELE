# Deploying the vm_comms worker poller on the SAM VM

The poller is a **host process** on the SAM VM (the same machine that runs the
`sam2`/`colmap`/`sugar`/`pgsr` containers and `run_pipeline.sh`). It is *not*
a container — it needs the host's `docker` CLI to drive the pipeline.

## One-time install

```bash
# 1. Python deps
sudo apt-get install -y python3-pip
sudo pip3 install requests
# Only if you'll run in Kafka mode:
sudo pip3 install confluent_kafka

# 2. Copy the env template and edit it
cp deploy/worker_poller.env.example deploy/worker_poller.env
chmod 600 deploy/worker_poller.env
$EDITOR deploy/worker_poller.env   # set HESTIA_API_URL and HESTIA_API_KEY

# 3. Install the systemd unit
# The shipped unit hard-codes /opt/samplify_sugar. If the checkout lives
# elsewhere, rewrite the paths inline as you install:
sudo sed "s|/opt/samplify_sugar|$PWD|g; s|User=youruser|User=$USER|; s|Group=youruser|Group=$USER|" \
    deploy/worker_poller.service \
    | sudo tee /etc/systemd/system/worker_poller.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now worker_poller
```

## Daily use

```bash
# Status / logs
systemctl status worker_poller
journalctl -u worker_poller -f

# Restart after editing the env file
sudo systemctl restart worker_poller

# Stop
sudo systemctl stop worker_poller
```

## Polling vs Kafka mode

Both supported, picked by env:

- `KAFKA_BROKER` **unset** → polling (`POST /vm-comms/claim` every
  `VM_COMMS_POLL_INTERVAL` seconds).
- `KAFKA_BROKER` set → Kafka consumer of `nefele_job_created`
  (group `sam-worker` — the group **is** the claim mechanism).

If you set `KAFKA_BROKER` but `confluent_kafka` is not installed, the poller
logs a warning and falls back to polling automatically.

## Prerequisites on the SAM VM

- Docker installed and the host user in the `docker` group:
  `sudo usermod -aG docker $USER`
- A samplify_sugar checkout anywhere on disk — the install snippet above
  rewrites `/opt/samplify_sugar` → `$PWD` so the unit matches your
  layout. Default `User=youruser` is also rewritten to `$USER`.
- The pipeline containers ready: `cd samplify_sugar && docker compose up -d sam2 colmap sugar pgsr`
  (note: there is no `ui` service here anymore — the UI lives in the
  `nefele_ui` project on the HESTIA host).

## What the poller drives

For each vm_comms job it:

1. `download_scan()` — downloads scan images from HESTIA `/robot-images` into
   the local `IN_MNT/<dataset>` dir (which is bind-mounted into `sam2` as
   `/data/in/<dataset>`).
2. `render_preview()` — `docker compose exec sam2 python3 app/video_predict.py
   --preview ...` (runs SAM2 in the GPU container, container-side paths).
3. Uploads previews to HESTIA `POST /vm-comms/<id>/preview`.
4. Waits for the user's `instructions` via polling `GET /vm-comms/<id>`
   (or via Kafka `nefele_job_modified` consumer — TODO if needed).
5. On confirm: `bash run_pipeline.sh <dataset>` with `NO_UI=1`, streams the
   pipeline status file to HESTIA via `PATCH /vm-comms/<id>`.
6. On done: `POST /reconstructions` with the final OBJ/MTL/PNG.
