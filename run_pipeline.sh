#!/usr/bin/env bash

set -euo pipefail

# ====== ARGS / DATASET ======

# This script is the pipeline executor only — it expects the dataset to be
# prepared elsewhere (the UI lives in the separate `nefele_ui` project and
# talks to HESTIA over HTTP; the local pipeline_watcher.sh and the SAM VM's
# worker_poller.py both call this script with the dataset as $1 after the
# user has chosen points).
#
# Required:
# $1 the dataset name
# prompts.json on disk at OUT/<dataset>_indexed/prompts.json

DATASET_NAME="${1:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# PIPELINE_MODE and other knobs are set by worker_poller.py (which loads .env
# before spawning this script). For manual runs without worker_poller, export
# them first:  source .env && bash run_pipeline.sh <dataset>

COORD_FILE="$SCRIPT_DIR/SAM2/data/input/.active_dataset"

if [[ -z "$DATASET_NAME" ]]; then
	echo "[!] Usage: $0 <dataset_name>" >&2
	echo "[!] The ui service is no longer part of this compose; the dataset must" >&2
	echo "[!] be created via nefele_ui (HESTIA-backed) before calling this script." >&2
	exit 1
fi

export DATASET_NAME

# Prevent concurrent runs of the same dataset from batch/manual invocations.
LOCK_DIR="$SCRIPT_DIR/logs/.locks"
mkdir -p "$LOCK_DIR"
LOCK_FILE="$LOCK_DIR/${DATASET_NAME//\//__}.lock"
exec 7>"$LOCK_FILE"

if ! flock -n 7; then
	echo "[!] Dataset is already running: $DATASET_NAME"
	echo "[!] Lock file: $LOCK_FILE"
	exit 1
fi

# ====== GPU ======

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# ====== PATHS ======

nephele_PATH="${nephele_PATH:-$SCRIPT_DIR}"
export nephele_PATH

SAM2_PATH="${SAM2_PATH:-${nephele_PATH}/SAM2}"
SUGAR_PATH="${SUGAR_PATH:-${nephele_PATH}/SUGAR/SuGaR}"
COLMAP_OUT_PATH="${COLMAP_OUT_PATH:-${nephele_PATH}/colmap}"

cd "$nephele_PATH"

# Where SAM2 expects input/output INSIDE the container:
IN_MNT_HOST="$SAM2_PATH/data/input"
OUT_MNT_HOST="$SAM2_PATH/data/output"
IN_MNT_CONT="/data/in"
OUT_MNT_CONT="/data/out"

# If you want INPUT to be dataset-specific, put images in: $IN_MNT_HOST/$DATASET_NAME
INPUT_SUBDIR="${INPUT_SUBDIR:-$DATASET_NAME}"
INPUT_CONT="$IN_MNT_CONT/$INPUT_SUBDIR"

# ====== MODEL CHOICE (written by Flask /setup or /hestia/load) ======

_MODEL_FILE="$IN_MNT_HOST/$DATASET_NAME/.model"
PIPELINE_MODEL="sugar"

if [[ -f "$_MODEL_FILE" ]]; then
	_m="$(tr -d '[:space:]' < "$_MODEL_FILE")"
	[[ "$_m" == "pgsr" || "$_m" == "sugar" ]] && PIPELINE_MODEL="$_m"
fi

echo "[*] Model: $PIPELINE_MODEL"

# ====== PIPELINE MODE ======
# PIPELINE_MODE=full  (default) — original settings, maximum quality, ~90-120 min
# PIPELINE_MODE=fast             — "better fast" config, full resolution, ~40-50 min
#
# Individual knobs (take precedence over PIPELINE_MODE if set explicitly):
#   PGSR_ITERATIONS         — total training iterations     (full=30000, fast=7000)
#   PGSR_RESOLUTION         — image scale factor            (full=1,     fast=2)
#   PGSR_DENSIFY_UNTIL      — stop densifying at iter       (full=15000, fast=5000)
#   PGSR_GEOM_FROM          — start geometry losses at iter (full=7000,  fast=2500)

PIPELINE_MODE="${PIPELINE_MODE:-full}"

if [[ "$PIPELINE_MODE" == "fast" ]]; then
	: "${PGSR_ITERATIONS:=7000}"
	: "${PGSR_RESOLUTION:=2}"
	: "${PGSR_DENSIFY_UNTIL:=5000}"
	: "${PGSR_GEOM_FROM:=2500}"
	echo "[*] PIPELINE_MODE=fast — iter=$PGSR_ITERATIONS res=$PGSR_RESOLUTION densify_until=$PGSR_DENSIFY_UNTIL geom_from=$PGSR_GEOM_FROM"
else
	: "${PGSR_ITERATIONS:=30000}"
	: "${PGSR_RESOLUTION:=1}"
	: "${PGSR_DENSIFY_UNTIL:=15000}"
	: "${PGSR_GEOM_FROM:=7000}"
	echo "[*] PIPELINE_MODE=full — iter=$PGSR_ITERATIONS res=$PGSR_RESOLUTION (original quality)"
fi

export PGSR_ITERATIONS PGSR_RESOLUTION PGSR_DENSIFY_UNTIL PGSR_GEOM_FROM

# ====== LOGGING ======

LOG_SCOPE="${LOG_SCOPE:-${DATASET_NAME%%/*}}"
LOG_BASENAME="${DATASET_NAME##*/}"

if [[ -n "$LOG_SCOPE" && "$LOG_SCOPE" != "$LOG_BASENAME" ]]; then
	LOGDIR="$nephele_PATH/logs/$LOG_SCOPE"
else
	LOGDIR="$nephele_PATH/logs"
fi

LOGFILE="$LOGDIR/${LOG_BASENAME}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOGDIR"
exec >>"$LOGFILE" 2>&1

# ---- Internal logs (inside containers) ----
# Propagation/preview run in the `sam2` container.

SAM2_PROP_LOG_CONT="/workspace/logs/propagate_${DATASET_NAME}.log"

TAIL_PIDS=()

start_sam2_log_tails() {
	# Tail propagation log from the SAM2 worker container
	$DOCKER_BIN compose exec -T sam2 bash -lc "
		mkdir -p /workspace/logs
		touch '$SAM2_PROP_LOG_CONT'
		stdbuf -oL -eL tail -n 0 -F '$SAM2_PROP_LOG_CONT'
	" 2>/dev/null | stdbuf -oL -eL sed 's/^/[SAM2:prop] /' >>"$LOGFILE" &
	TAIL_PIDS+=($!)
}

stop_sam2_log_tails() {
	for pid in "${TAIL_PIDS[@]:-}"; do
		kill "$pid" >/dev/null 2>&1 || true
	done
	TAIL_PIDS=()
}

on_error() {
	stop_sam2_log_tails || true
	echo "STATUS: ERROR"
	echo "LOG: $LOGFILE"
	exit 1
}

trap on_error ERR
trap stop_sam2_log_tails EXIT

on_error() {
	echo "STATUS: ERROR"
	echo "LOG: $LOGFILE"
	exit 1
}

trap on_error ERR

echo "======================================"
echo " SAM2 + COLMAP + SuGaR pipeline"
echo " Dataset: $DATASET_NAME"
echo " SAM2_PATH: $SAM2_PATH"
echo " SUGAR_PATH: $SUGAR_PATH"
echo " Host IN: $IN_MNT_HOST"
echo " Host OUT: $OUT_MNT_HOST"
echo " Container INPUT: $INPUT_CONT"
echo " Log: $LOGFILE"
echo "======================================"

# ====== SANITY CHECKS ======

[[ -d "$SAM2_PATH" ]] || { echo "SAM2 path not found: $SAM2_PATH"; exit 1; }
[[ -d "$SUGAR_PATH" ]] || { echo "SuGaR path not found: $SUGAR_PATH"; exit 1; }
command -v docker >/dev/null || { echo "Docker not found in PATH."; exit 1; }

HOST_UID=$(id -u)
HOST_GID=$(id -g)
DOCKER_BIN="${DOCKER_BIN:-docker}" # no sudo

# Export UID/GID so docker-compose can use them (compose variable interpolation)
export HOST_UID HOST_GID

# ====== Ensure mount folders exist (owned by you) ======

mkdir -p "$IN_MNT_HOST" "$OUT_MNT_HOST" "$IN_MNT_HOST/$INPUT_SUBDIR"
chmod -R u+rwX,g+rwX "$IN_MNT_HOST" "$OUT_MNT_HOST" || true

# ====== Image presence (sam2:local); build if missing with host UID/GID ======

if ! $DOCKER_BIN image inspect sam2:local >/dev/null 2>&1; then
	echo "Docker image 'sam2:local' not found. Building..."
	$DOCKER_BIN build \
		--build-arg UID="$HOST_UID" \
		--build-arg GID="$HOST_GID" \
		-t sam2:local "$SAM2_PATH"
fi

# ====== GUI or HEADLESS ======

GUI="${GUI:-1}" # default GUI on
FRAME_IDX="${FRAME_IDX:-0}"
OBJ_ID="${OBJ_ID:-1}"

DOCKER_GUI_FLAGS=()

if [[ "$GUI" == "1" ]]; then
	if command -v xhost >/dev/null 2>&1; then
		xhost +local:docker >/dev/null 2>&1 || true
	fi
	: "${DISPLAY:=${DISPLAY:-:0}}"
	DOCKER_GUI_FLAGS=( -e DISPLAY="$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix )
else
	echo "[i] GUI=0 → running without X display."
fi

# ====== Pre-SAM2: Convert any PNG images in input folder to JPG ======
# SAM2 requires JPG (calls load_video_frames_from_jpg_images internally)

_INPUT_DIR="$SAM2_PATH/data/input/$DATASET_NAME"

_PNG_COUNT=$(find "$_INPUT_DIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)

if [[ "$_PNG_COUNT" -gt 0 ]]; then
	echo "[*] Converting $_PNG_COUNT PNG image(s) to JPG in $_INPUT_DIR (SAM2 requires JPG)..."
	python3 -c "
import glob, os
try:
    from PIL import Image
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'Pillow'])
    from PIL import Image

src_dir = '$_INPUT_DIR'

for src in sorted(glob.glob(os.path.join(src_dir, '*.png'))):
	dst = os.path.splitext(src)[0] + '.jpg'
	Image.open(src).convert('RGB').save(dst, 'JPEG', quality=95)
	os.remove(src)
	print(f' Converted to JPG: {len(glob.glob(os.path.join(src_dir, \"*.jpg\")))} files')
"
fi

# ====== RUN SAM2 (picker + propagation) ======

echo "[*] Running SAM2 for dataset: $DATASET_NAME"
echo "[*] INPUT (container): $INPUT_CONT"
echo "[*] OUT (container): $OUT_MNT_CONT"

mkdir -p "$SAM2_PATH/data/input/$DATASET_NAME" "$SAM2_PATH/data/output"
chmod -R u+rwX,g+rwX "$SAM2_PATH/data/input" "$SAM2_PATH/data/output" || true

SAM2_CNAME="${SAM2_CONTAINER_NAME:-sam2}"

# Ensure the SAM2 worker container is running (auto-start if stopped). The
# container is dataset-agnostic — the caller (nefele_ui worker_poller, or the
# local pipeline_watcher.sh) passes paths per request.

if [[ "$($DOCKER_BIN inspect -f '{{.State.Running}}' "$SAM2_CNAME" 2>/dev/null || echo false)" != "true" ]]; then
	echo "[*] sam2 worker is not running — starting it"
	$DOCKER_BIN compose up -d sam2
	sleep 2
fi

# GPU pre-flight: a running sam2 container is not enough — it may have been
# started before the nvidia runtime was wired in. If nvidia-smi fails inside
# the container, force-recreate it so it picks up GPU access from compose.

check_sam2_gpu() {
	$DOCKER_BIN compose exec -T "$SAM2_CNAME" \
		bash -lc 'nvidia-smi --query-gpu=index --format=csv,noheader' \
		>/dev/null 2>&1
}

if ! check_sam2_gpu; then
	echo "[!] sam2 worker cannot see a GPU — recreating it (GPU_ID=${GPU_ID:-3})"
	$DOCKER_BIN compose up -d --force-recreate sam2
	sleep 3
	if ! check_sam2_gpu; then
		echo "[!] sam2 still cannot see a GPU after recreate. Check nvidia runtime / GPU_ID." >&2
		$DOCKER_BIN compose logs --tail=20 "$SAM2_CNAME" >&2 || true
		exit 1
	fi
	echo "[*] sam2 worker GPU access verified."
fi

# The UI runs in the separate nefele_ui project — no port-coordination needed here.

# ====== INDEXED / FLAGS ======

INDEX_SUFFIX="${INDEX_SUFFIX:-_indexed}"
INDEXED_NAME="${INPUT_SUBDIR}${INDEX_SUFFIX}"
INDEXED_DIR="$OUT_MNT_HOST/${INDEXED_NAME}"

mkdir -p "$INDEXED_DIR"
chmod 775 "$INDEXED_DIR" || true

PROMPTS_HOST="${PROMPTS_HOST:-$INDEXED_DIR/prompts.json}"
DONE_FLAG="${DONE_FLAG:-$INDEXED_DIR/__picker_done.flag}"
USE_EXISTING_FLAG="${USE_EXISTING_FLAG:-$INDEXED_DIR/__use_existing.flag}"
PIPELINE_STATUS="${PIPELINE_STATUS:-$INDEXED_DIR/__pipeline_status.json}"

# Write a single-line status JSON read by the UI's /pipeline/status endpoint.
# Args: <current_index> <status: running|done|error> <message>

write_status() {
	local current="$1" state="$2" msg="${3:-}"
	python3 - "$DATASET_NAME" "$current" "$state" "$msg" "$PIPELINE_STATUS" "$PIPELINE_MODEL" <<'PY' || true
import json, sys, time, os

dataset, current, state, msg, path, model = sys.argv[1:7]

now = time.time()

prev = {}

try:
	with open(path, "r", encoding="utf-8") as f:
		prev = json.load(f)
except Exception:
	pass

stages = ["sam2", "colmap", "pgsr"] if model == "pgsr" else ["sam2", "colmap", "sugar"]

out = {
	"dataset": dataset,
	"stages": stages,
	"model": model,
	"current": int(current),
	"status": state,
	"message": msg,
	"started_at": prev.get("started_at") or now,
	"updated_at": now,
	"error": msg if state == "error" else None,
}

os.makedirs(os.path.dirname(path), exist_ok=True)

tmp = path + ".tmp"

with open(tmp, "w", encoding="utf-8") as f:
	json.dump(out, f)

os.replace(tmp, path)

PY
}

# On any failure, mark the status as "error" so the UI stops spinning.

on_pipeline_error() {
	local rc=$?
	write_status "${PIPELINE_CURRENT_STAGE:-1}" error "pipeline aborted (exit $rc)"
}

trap on_pipeline_error ERR

# The picker work happens upstream — either the local pipeline_watcher.sh saw
# `__picker_done.flag` (shared_fs) or the SAM VM's worker_poller.py wrote
# prompts.json after the user confirmed in nefele_ui (vm_comms). Either way,
# prompts.json must already be on disk at this point.

if [[ ! -f "$PROMPTS_HOST" ]]; then
	echo "[!] prompts.json not found at: $PROMPTS_HOST" >&2
	echo "[!] The caller must save it before invoking this script." >&2
	exit 1
fi

echo "[*] Using saved prompts: $PROMPTS_HOST"

# Clean up the watcher's flag files now that we've taken ownership of the run.

rm -f "$DONE_FLAG" "$USE_EXISTING_FLAG"

# ====== PROPAGATION (use existing prompts) ======

PIPELINE_CURRENT_STAGE=0

write_status 0 running "Propagating SAM2 masks across frames"

echo "[*] Running SAM2 propagation using saved prompts..."

# Run propagation inside the already-running sam2 container so we reuse the image/container

$DOCKER_BIN compose exec -T sam2 bash -lc "

	export DATASET_NAME=\"$DATASET_NAME\"

	export INPUT=\"/data/in/$DATASET_NAME\"

	export OUT=\"/data/out\"

	export OUT_NAME=\"${DATASET_NAME}_indexed\"

	export PROMPTS_JSON=\"/data/out/${DATASET_NAME}_indexed/prompts.json\"

	export INDEX_SUFFIX=\"$INDEX_SUFFIX\"

	export QUIET=0

	export MPLBACKEND=Agg

	export HF_HOME=/data/out/.cache/huggingface

	umask 0002

	python3 -u /workspace/app/video_predict.py

"

echo "[*] SAM2 finished successfully (until here)."

# Free GPU memory before COLMAP/SuGaR

echo "[*] Releasing SAM2 GPU memory..."

$DOCKER_BIN compose exec -T sam2 bash -c "pkill -f video_predict.py" >/dev/null 2>&1 || true

# Keep $COORD_FILE in place so /results stays accessible after the pipeline
# finishes. The start-of-script logic (line ~19) wipes it when the user
# launches a new no-arg run, which is the only time we actually need it gone.

sleep 2

# ====== COLMAP STAGE ======

PIPELINE_CURRENT_STAGE=1

write_status 1 running "Running COLMAP structure-from-motion"

$DOCKER_BIN pull colmap/colmap

if [ -f "$COLMAP_OUT_PATH/run_colmap.sh" ]; then
	chmod +x "$COLMAP_OUT_PATH/run_colmap.sh"
else
	echo "[*] run_colmap.sh not found in $COLMAP_OUT_PATH (skipping copy)"
fi

cd "$COLMAP_OUT_PATH"

# Ensure COLMAP dirs exist and are writable by you

mkdir -p "$COLMAP_OUT_PATH/input" "$COLMAP_OUT_PATH/output"
chmod -R u+rwX,g+rwX "$COLMAP_OUT_PATH/input" || true

install -d -m 775 \
	"$COLMAP_OUT_PATH/input/$DATASET_NAME" \
	"$COLMAP_OUT_PATH/input/${DATASET_NAME}_indexed"

# ---- paths ----

IMAGES_SRC="$SAM2_PATH/data/input/${DATASET_NAME}_indexed"
MASKS_SRC="$SAM2_PATH/data/output/${DATASET_NAME}_indexed"
IMAGES_DST="$COLMAP_OUT_PATH/input/${DATASET_NAME}"
MASKS_DST="$COLMAP_OUT_PATH/input/${DATASET_NAME}_indexed"
OUT_DST="$COLMAP_OUT_PATH/output/${DATASET_NAME}"

# ---- Auto-detect known poses BEFORE image copy ----

KNOWN_POSES="${KNOWN_POSES:-}"

_DATASET_COLMAP="$SAM2_PATH/data/input/$DATASET_NAME/colmap"

if [[ -z "$KNOWN_POSES" && -f "$_DATASET_COLMAP/cameras.txt" && -f "$_DATASET_COLMAP/images.txt" ]]; then
	KNOWN_POSES="$_DATASET_COLMAP"
	echo "[*] Auto-detected known poses: $KNOWN_POSES"
fi

# ---- ensure dest dirs ----

mkdir -p "$IMAGES_DST" "$MASKS_DST" "$OUT_DST"

if [[ -n "$KNOWN_POSES" ]]; then
	# Known poses: copy original images from dataset folder preserving original names
	# PNG → JPG, keep stems (frame_001.png → frame_001.jpg)

	echo "[*] Known poses mode: copying original images with preserved names..."

	python3 - <<PYEOF

import os, glob
try:
    from PIL import Image
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'Pillow'])
    from PIL import Image

src_dir = "$SAM2_PATH/data/input/$DATASET_NAME"
dst_dir = "$IMAGES_DST"
mask_src = "$SAM2_PATH/data/output/${DATASET_NAME}_indexed"
mask_dst = "$MASKS_DST"
masked_src = "$SAM2_PATH/data/output/${DATASET_NAME}_indexed_masked"

# 1. Copy original images → COLMAP input, converting to JPG, preserving stems

srcs = sorted(f for f in glob.glob(os.path.join(src_dir, '*'))
	if os.path.splitext(f)[1].lower() in ('.png', '.jpg', '.jpeg')
	and os.path.isfile(f))

for src in srcs:
	stem = os.path.splitext(os.path.basename(src))[0]
	dst = os.path.join(dst_dir, stem + '.jpg')
	Image.open(src).convert('RGB').save(dst, 'JPEG', quality=95)

print(f" Copied {len(srcs)} images with original names to COLMAP input")

# 2. Rename SAM2-indexed masks/masked images to original stems (sorted order mapping)

def rename_dir_to_original(indexed_dir, orig_stems):
	if not os.path.isdir(indexed_dir):
		return 0

	indexed = sorted(f for f in os.listdir(indexed_dir)
		if os.path.splitext(f)[1].lower() in ('.png', '.jpg', '.jpeg'))

	if len(indexed) != len(orig_stems):
		print(f" [!] {indexed_dir}: {len(indexed)} files vs {len(orig_stems)} originals — skipping rename")
		return 0

	for idx_name, orig_stem in zip(indexed, orig_stems):
		src = os.path.join(indexed_dir, idx_name)
		dst = os.path.join(indexed_dir, orig_stem + '.jpg')
		if src != dst:
			os.rename(src, dst)

	return len(indexed)

orig_stems = [os.path.splitext(os.path.basename(s))[0] for s in srcs]

n = rename_dir_to_original(mask_src, orig_stems)
if n: print(f" Renamed {n} mask files to original names")

n = rename_dir_to_original(masked_src, orig_stems)
if n: print(f" Renamed {n} masked images to original names")

# 3. Copy renamed masks to COLMAP masks dir

import shutil

for f in os.listdir(mask_src) if os.path.isdir(mask_src) else []:
	if os.path.splitext(f)[1].lower() in ('.png', '.jpg', '.jpeg'):
		shutil.copy2(os.path.join(mask_src, f), os.path.join(mask_dst, f))

PYEOF

else
	# Normal SAM2 flow: rsync numbered indexed images

	rsync -a --delete \
		--include '*/' --include '*.jpg' --include '*.jpeg' --include '*.png' --exclude '*' \
		"${IMAGES_SRC}/" "${IMAGES_DST}/"

	rsync -a --delete \
		--include '*/' \
		--exclude 'preview/**' \
		--include '*.jpg' --include '*.jpeg' --include '*.png' --exclude '*' \
		"${MASKS_SRC}/" "${MASKS_DST}/"

fi

echo "Copied images: $(find "$IMAGES_DST" -maxdepth 1 -type f | wc -l)"
echo "Copied masks : $(find "$MASKS_DST" -maxdepth 1 -type f | wc -l)"

# ---- run COLMAP ----

if [[ -n "$KNOWN_POSES" ]]; then
	echo "[*] KNOWN_POSES: $KNOWN_POSES — using point_triangulator instead of mapper"
fi

bash "$COLMAP_OUT_PATH/run_colmap.sh" \
	"$IMAGES_DST" \
	"$MASKS_DST" \
	"$OUT_DST" \
	"${COLMAP_MATCHER:-sequential}" \
	"$KNOWN_POSES"

# --- stage SuGaR overrides into the submodule ---

# The SUGAR/SuGaR submodule is a fresh upstream clone of Anttwo/SuGaR. Our
# project-specific modifications (Dockerfile, modified train.py, custom
# coarse_mesh.py, runner script, convert_to_rgba.py utility) live in
# sugar_overrides/ and get copied into the submodule before each pipeline run.

SUGAR_OVR="$nephele_PATH/sugar_overrides"

if [ -d "$SUGAR_OVR" ]; then
	echo "[*] Staging SuGaR overrides from sugar_overrides/ → $SUGAR_PATH"

	[ -f "$SUGAR_OVR/run_sugar_pipeline_with_sam.sh" ] && \
		cp -f "$SUGAR_OVR/run_sugar_pipeline_with_sam.sh" "$SUGAR_PATH" && \
		chmod +x "$SUGAR_PATH/run_sugar_pipeline_with_sam.sh"

	[ -f "$SUGAR_OVR/Dockerfile_final" ] && \
		cp -f "$SUGAR_OVR/Dockerfile_final" "$SUGAR_PATH"

	[ -f "$SUGAR_OVR/train.py" ] && \
		cp -f "$SUGAR_OVR/train.py" "$SUGAR_PATH/gaussian_splatting/"

	[ -f "$SUGAR_OVR/coarse_mesh.py" ] && \
		cp -f "$SUGAR_OVR/coarse_mesh.py" "$SUGAR_PATH/sugar_extractors/coarse_mesh.py"

	[ -f "$SUGAR_OVR/convert_to_rgba.py" ] && \
		cp -f "$SUGAR_OVR/convert_to_rgba.py" "$SUGAR_PATH"

else
	echo "[*] sugar_overrides/ not found in $nephele_PATH (skipping override stage)"
fi

# --- stage PGSR overrides into the submodule ---

# Same pattern as above: PGSR/ is a fresh clone of zju3dv/PGSR. Our project
# patches (Dockerfile, modified train.py/render.py/scene/*, bake_texture.py)
# live in pgsr_overrides/ and are copied in just before the PGSR run.

PGSR_OVR="$nephele_PATH/pgsr_overrides"

if [ -d "$PGSR_OVR" ]; then
	echo "[*] Staging PGSR overrides from pgsr_overrides/ → $nephele_PATH/PGSR"
	cp -rf "$PGSR_OVR/." "$nephele_PATH/PGSR/"
else
	echo "[*] pgsr_overrides/ not found in $nephele_PATH (skipping override stage)"
fi

PIPELINE_CURRENT_STAGE=2

if [[ "$PIPELINE_MODEL" == "pgsr" ]]; then
	# --- run PGSR ---

	write_status 2 running "Training PGSR gaussian splat + exporting mesh"

	echo "[*] Running PGSR pipeline for dataset: $DATASET_NAME..."

	cd "$nephele_PATH"

	DATASET_NAME="$DATASET_NAME" \
		nephele_PATH="$nephele_PATH" \
		COLMAP_OUT_PATH="$COLMAP_OUT_PATH" \
		bash "$nephele_PATH/run_pgsr_pipeline_with_sam.sh" "$DATASET_NAME"

else
	# --- run SUGAR ---

	write_status 2 running "Training SuGaR gaussian splat + building mesh"

	echo "[*] Running SuGaR pipeline for dataset: $DATASET_NAME..."

	cd "$SUGAR_PATH"

	DATASET_NAME="$DATASET_NAME" \
		SUGAR_PATH="$SUGAR_PATH" \
		nephele_PATH="$nephele_PATH" \
		bash ./run_sugar_pipeline_with_sam.sh "$DATASET_NAME"

fi

write_status 2 done "All stages completed"

echo "[*] Pipeline completed successfully!"

echo "Pipeline completed. Check log: $LOGFILE"

# Mirror the final marker to a controlling TTY only when one exists. When
# this script is invoked by worker_poller via subprocess.Popen there is no
# TTY, and `tee /dev/tty` would fail and propagate exit 1 even though every
# stage above succeeded — which made the poller publish status=error to
# HESTIA and skip upload_reconstruction.

echo "[*] Pipeline completed"

{ echo "[*] Pipeline completed" > /dev/tty; } 2>/dev/null || true
