#!/usr/bin/env bash
set -euo pipefail

# ====== ARGS / DATASET ======
DATASET_NAME="${1:?Usage: $0 DATASET_NAME}"
export DATASET_NAME

# ====== PATHS ======
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

# ====== LOGGING ======
LOGDIR="$nephele_PATH/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec >>"$LOGFILE" 2>&1

# ---- SAM2 internal logs (inside container) ----
SAM2_PICKER_LOG_CONT="/workspace/logs/picker_${DATASET_NAME}.log"
SAM2_PROP_LOG_CONT="/workspace/logs/propagate_${DATASET_NAME}.log"  
TAIL_PIDS=()
start_sam2_log_tails() {
  # Tail picker log from inside container into our host LOGFILE
  $DOCKER_BIN compose exec -T sam2 bash -lc "
    mkdir -p /workspace/logs
    touch '$SAM2_PICKER_LOG_CONT'
    stdbuf -oL -eL tail -n 0 -F '$SAM2_PICKER_LOG_CONT'
  " 2>/dev/null | stdbuf -oL -eL sed 's/^/[SAM2:picker] /' >>"$LOGFILE" &
  TAIL_PIDS+=($!)

  # Tail propagation log too (we'll write to it with tee)
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
echo " Dataset:          $DATASET_NAME"
echo " SAM2_PATH:        $SAM2_PATH"
echo " SUGAR_PATH:       $SUGAR_PATH"
echo " Host IN:          $IN_MNT_HOST"
echo " Host OUT:         $OUT_MNT_HOST"
echo " Container INPUT:  $INPUT_CONT"
echo " Log:              $LOGFILE"
echo "======================================"

# ====== SANITY CHECKS ======
[[ -d "$SAM2_PATH" ]]   || { echo "SAM2 path not found: $SAM2_PATH"; exit 1; }
[[ -d "$SUGAR_PATH" ]]  || { echo "SuGaR path not found: $SUGAR_PATH"; exit 1; }
command -v docker >/dev/null || { echo "Docker not found in PATH."; exit 1; }

HOST_UID=$(id -u)
HOST_GID=$(id -g)
DOCKER_BIN="${DOCKER_BIN:-docker}"   # no sudo
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
GUI="${GUI:-1}"          # default GUI on
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
from PIL import Image
src_dir = '$_INPUT_DIR'
for src in sorted(glob.glob(os.path.join(src_dir, '*.png'))):
    dst = os.path.splitext(src)[0] + '.jpg'
    Image.open(src).convert('RGB').save(dst, 'JPEG', quality=95)
    os.remove(src)
print(f'  Converted to JPG: {len(glob.glob(os.path.join(src_dir, \"*.jpg\")))} files')
"
fi

# ====== RUN SAM2 (picker + propagation) ======

echo "[*] Running SAM2 for dataset: $DATASET_NAME"
echo "[*] INPUT (container): $INPUT_CONT"
echo "[*] OUT   (container): $OUT_MNT_CONT"

mkdir -p "$SAM2_PATH/data/input/$DATASET_NAME" "$SAM2_PATH/data/output"
chmod -R u+rwX,g+rwX "$SAM2_PATH/data/input" "$SAM2_PATH/data/output" || true

# Auto-pick a free port starting at 8092
WEB_PORT="${WEB_PORT:-8092}"
if ss -ltn | awk '{print $4}' | grep -q ":${WEB_PORT}\$"; then
  for p in $(seq 8092 8110); do
    if ! ss -ltn | awk '{print $4}' | grep -q ":${p}\$"; then
      WEB_PORT=$p; break
    fi
  done
fi

# Ensure sam2 container is running (auto-start if stopped)
if $DOCKER_BIN compose ps -q sam2 >/dev/null 2>&1; then
  if [[ "$($DOCKER_BIN inspect -f '{{.State.Running}}' sam2 2>/dev/null || echo false)" != "true" ]]; then
    echo "[*] sam2 container is stopped — starting it automatically"
    # Start with DATASET_NAME="" so the container runs "tail -f /dev/null" (stays alive)
    # rather than launching the Flask picker and exiting immediately
    DATASET_NAME="" $DOCKER_BIN compose up -d sam2
    sleep 2
  fi
fi

# If a sam2 container is running and already maps container port 5000,
# prefer that host port so we don't pick a different port than the container exposes.
if $DOCKER_BIN compose ps -q sam2 >/dev/null 2>&1; then
  if [[ "$($DOCKER_BIN inspect -f '{{.State.Running}}' sam2 2>/dev/null || echo false)" == "true" ]]; then
    MAPPED="$($DOCKER_BIN compose port sam2 5000 2>/dev/null || true)"
    if [[ -n "$MAPPED" ]]; then
      # MAPPED looks like 0.0.0.0:8092 or [::]:8092 — extract the port after the last ':'
      HOST_PORT="${MAPPED##*:}"
      if [[ -n "$HOST_PORT" ]]; then
        echo "[*] sam2 is already running and maps container:5000 -> host:${HOST_PORT}; using that port"
        WEB_PORT="$HOST_PORT"
      fi
    fi
  fi
fi

echo "[*] Using WEB_PORT=$WEB_PORT"

# Export chosen WEB_PORT so docker compose uses the same host port mapping
export WEB_PORT

PICKER_NAME="sam2picker_${DATASET_NAME}_${WEB_PORT}"
$DOCKER_BIN rm -f "$PICKER_NAME" >/dev/null 2>&1 || true

# ====== INDEXED / FLAGS ======
INDEX_SUFFIX="${INDEX_SUFFIX:-_indexed}"
INDEXED_NAME="${INPUT_SUBDIR}${INDEX_SUFFIX}"
INDEXED_DIR="$OUT_MNT_HOST/${INDEXED_NAME}"
mkdir -p "$INDEXED_DIR"
chmod 775 "$INDEXED_DIR" || true

PROMPTS_HOST="${PROMPTS_HOST:-$INDEXED_DIR/prompts.json}"
DONE_FLAG="${DONE_FLAG:-$INDEXED_DIR/__picker_done.flag}"
USE_EXISTING_FLAG="${USE_EXISTING_FLAG:-$INDEXED_DIR/__use_existing.flag}"
rm -f "$DONE_FLAG" "$USE_EXISTING_FLAG"

echo "[*] Starting Flask point picker for '$DATASET_NAME' on http://localhost:${WEB_PORT}/ ..."

PICKER_SERVICE="sam2"

# Check whether the service/container already exists
SERVICE_ID="$($DOCKER_BIN compose ps -q "$PICKER_SERVICE" 2>/dev/null || true)"
if [[ -n "$SERVICE_ID" ]]; then
  # Container exists — check whether it's running
  if [[ "$($DOCKER_BIN inspect -f '{{.State.Running}}' "$PICKER_SERVICE" 2>/dev/null || echo false)" == "true" ]]; then
    echo "[*] $PICKER_SERVICE already running; picker should be at http://localhost:$WEB_PORT/"
    # Kill any existing picker (may be from a different dataset) and start fresh
    $DOCKER_BIN compose exec -T "$PICKER_SERVICE" bash -c 'pkill -f point_picker_flask.py' >/dev/null 2>&1 || true
    echo "[*] Starting picker inside running $PICKER_SERVICE container"
    $DOCKER_BIN compose exec -T "$PICKER_SERVICE" bash -c "
      export DATASET_NAME=\"$DATASET_NAME\"
      export INPUT=\"/data/in/$DATASET_NAME\"
      export OUT=\"/data/out\"
      export INDEX_SUFFIX=\"$INDEX_SUFFIX\"
      export HF_HOME=/data/out/.cache/huggingface
      umask 0002
      nohup python3 -u /workspace/app/point_picker_flask.py > /workspace/logs/picker_${DATASET_NAME}.log 2>&1 &
    "
  else
    echo "[*] $PICKER_SERVICE exists but is stopped — starting it"
    $DOCKER_BIN compose start "$PICKER_SERVICE"
    echo "[*] Started $PICKER_SERVICE; starting picker"
    $DOCKER_BIN compose exec -T "$PICKER_SERVICE" bash -lc "
      export DATASET_NAME=\"$DATASET_NAME\"
      export INPUT=\"/data/in/$DATASET_NAME\"
      export OUT=\"/data/out\"
      export INDEX_SUFFIX=\"$INDEX_SUFFIX\"
      export HF_HOME=/data/out/.cache/huggingface
      umask 0002
      nohup python3 -u /workspace/app/point_picker_flask.py > /workspace/logs/picker_${DATASET_NAME}.log 2>&1 &
    "
  fi
else
  echo "[*] $PICKER_SERVICE not found — creating and starting it"
  $DOCKER_BIN compose up -d "$PICKER_SERVICE"
  echo "[*] Created and started $PICKER_SERVICE; starting picker"
  $DOCKER_BIN compose exec -T "$PICKER_SERVICE" bash -lc "
    export DATASET_NAME=\"$DATASET_NAME\"
    export INPUT=\"/data/in/$DATASET_NAME\"
    export OUT=\"/data/out\"
    export INDEX_SUFFIX=\"$INDEX_SUFFIX\"
    export HF_HOME=/data/out/.cache/huggingface
    umask 0002
    nohup python3 -u /workspace/app/point_picker_flask.py > /workspace/logs/picker_${DATASET_NAME}.log 2>&1 &
  "
fi

echo "[*] Attaching SAM2 logs into: $LOGFILE"
start_sam2_log_tails

echo "[*] Open to select points: http://localhost:${WEB_PORT}/" | tee /dev/tty


echo "[*] Waiting for decision/save → $DONE_FLAG"
while :; do
  if [[ -f "$DONE_FLAG" ]]; then
    echo "[*] Picker signaled DONE_FLAG. Proceeding..."
    break
  fi
  sleep 1
done

$DOCKER_BIN stop "$PICKER_NAME" >/dev/null 2>&1 || true
$DOCKER_BIN rm -f "$PICKER_NAME" >/dev/null 2>&1 || true

# Use Existing vs Create New
if [[ -f "$USE_EXISTING_FLAG" ]]; then
  if [[ ! -f "$PROMPTS_HOST" ]]; then
    echo "[!] You chose 'Use existing' but prompts.json not found at: $PROMPTS_HOST"
    exit 1
  fi
  echo "[*] Using existing prompts: $PROMPTS_HOST"
else
  if [[ ! -f "$PROMPTS_HOST" ]]; then
    echo "[!] No prompts.json saved. Aborting."
    exit 1
  fi
  echo "[*] New prompts saved at: $PROMPTS_HOST"
fi

rm -f "$DONE_FLAG" "$USE_EXISTING_FLAG"

echo "[*] Running SAM2 propagation using saved prompts..."
# Run propagation inside the already-running sam2 container so we reuse the image/container
$DOCKER_BIN compose exec -T sam2 bash -lc "
  export DATASET_NAME=\"$DATASET_NAME\"
  export INPUT=\"/data/in/$DATASET_NAME\"
  export OUT=\"/data/out\"
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
$DOCKER_BIN compose exec -T sam2 bash -c "pkill -f video_predict.py; pkill -f point_picker_flask.py" >/dev/null 2>&1 || true
sleep 2

# ====== COLMAP STAGE ======
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
from PIL import Image

src_dir  = "$SAM2_PATH/data/input/$DATASET_NAME"
dst_dir  = "$IMAGES_DST"
mask_src = "$SAM2_PATH/data/output/${DATASET_NAME}_indexed"
mask_dst = "$MASKS_DST"
masked_src = "$SAM2_PATH/data/output/${DATASET_NAME}_indexed_masked"

# 1. Copy original images → COLMAP input, converting to JPG, preserving stems
srcs = sorted(f for f in glob.glob(os.path.join(src_dir, '*'))
              if os.path.splitext(f)[1].lower() in ('.png', '.jpg', '.jpeg')
              and os.path.isfile(f))
for src in srcs:
    stem = os.path.splitext(os.path.basename(src))[0]
    dst  = os.path.join(dst_dir, stem + '.jpg')
    Image.open(src).convert('RGB').save(dst, 'JPEG', quality=95)
print(f"  Copied {len(srcs)} images with original names to COLMAP input")

# 2. Rename SAM2-indexed masks/masked images to original stems (sorted order mapping)
def rename_dir_to_original(indexed_dir, orig_stems):
    if not os.path.isdir(indexed_dir):
        return 0
    indexed = sorted(f for f in os.listdir(indexed_dir)
                     if os.path.splitext(f)[1].lower() in ('.png', '.jpg', '.jpeg'))
    if len(indexed) != len(orig_stems):
        print(f"  [!] {indexed_dir}: {len(indexed)} files vs {len(orig_stems)} originals — skipping rename")
        return 0
    for idx_name, orig_stem in zip(indexed, orig_stems):
        src = os.path.join(indexed_dir, idx_name)
        dst = os.path.join(indexed_dir, orig_stem + '.jpg')
        if src != dst:
            os.rename(src, dst)
    return len(indexed)

orig_stems = [os.path.splitext(os.path.basename(s))[0] for s in srcs]
n = rename_dir_to_original(mask_src, orig_stems)
if n: print(f"  Renamed {n} mask files  to original names")
n = rename_dir_to_original(masked_src, orig_stems)
if n: print(f"  Renamed {n} masked images to original names")

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
  sequential \
  "$KNOWN_POSES"

# --- optionally stage helper files ---
if [ -f "$nephele_PATH/run_sugar_pipeline_with_sam.sh" ]; then
  echo "[*] Copying run_sugar_pipeline_with_sam.sh to $SUGAR_PATH"
  cp -f "$nephele_PATH/run_sugar_pipeline_with_sam.sh" "$SUGAR_PATH"
  chmod +x "$SUGAR_PATH/run_sugar_pipeline_with_sam.sh"
else
  echo "[*] run_sugar_pipeline_with_sam.sh not found in $nephele_PATH (skipping copy)"
fi

if [ -f "$nephele_PATH/Dockerfile_final" ]; then
  echo "[*] Copying Dockerfile and helpers to $SUGAR_PATH"
  cp -f "$nephele_PATH/Dockerfile_final" "$SUGAR_PATH"
  cp -f "$nephele_PATH/train.py" "$SUGAR_PATH/gaussian_splatting/"
  cp -f "$nephele_PATH/coarse_mesh.py" "$SUGAR_PATH/sugar_extractors/coarse_mesh.py"
else
  echo "[*] Dockerfile/train.py/coarse_mesh.py not found in $nephele_PATH (skipping copy)"
fi

# --- run SUGAR (pass DATASET_NAME as env) ---
echo "[*] Running SuGaR pipeline for dataset: $DATASET_NAME..."
cd "$SUGAR_PATH"

DATASET_NAME="$DATASET_NAME" \
SUGAR_PATH="$SUGAR_PATH" \
nephele_PATH="$nephele_PATH" \
bash ./run_sugar_pipeline_with_sam.sh "$DATASET_NAME"

echo "[*] Pipeline completed successfully!"
echo "Pipeline completed. Check log: $LOGFILE"
echo "[*] Pipeline completed" | tee /dev/tty
