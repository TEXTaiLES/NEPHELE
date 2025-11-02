#!/usr/bin/env bash
set -euo pipefail

# ====== ARGS / DATASET ======
DATASET_NAME="${DATASET_NAME:-${1:-}}"
: "${DATASET_NAME:?Usage: $0 DATASET_NAME  (or export DATASET_NAME first)}"

# ====== PATHS ======
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMplify_SuGaR_PATH="${SAMplify_SuGaR_PATH:-$SCRIPT_DIR}"
SAM2_PATH="${SAM2_PATH:-${SAMplify_SuGaR_PATH}/SAM2}"
SUGAR_PATH="${SUGAR_PATH:-${SAMplify_SuGaR_PATH}/SUGAR/SuGaR}"
COLMAP_OUT_PATH="${COLMAP_OUT_PATH:-${SAMplify_SuGaR_PATH}/colmap}"

# Where SAM2 expects input/output INSIDE the container:
IN_MNT_HOST="$SAM2_PATH/data/input"
OUT_MNT_HOST="$SAM2_PATH/data/output"
IN_MNT_CONT="/data/in"
OUT_MNT_CONT="/data/out"

# If you want INPUT to be dataset-specific, put images in: $IN_MNT_HOST/$DATASET_NAME
INPUT_SUBDIR="${INPUT_SUBDIR:-$DATASET_NAME}"
INPUT_CONT="$IN_MNT_CONT/$INPUT_SUBDIR"

# ====== LOGGING ======
LOGDIR="$SAMplify_SuGaR_PATH/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec >>"$LOGFILE" 2>&1

on_error() {
  echo "STATUS: ERROR"
  echo "LOG: $LOGFILE"
  exit 1
}
trap on_error ERR

echo "======================================"
echo " SAM2 stage runner"
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

# ====== RUN SAM2 (picker + propagation) ======
cd "$SAM2_PATH"
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
echo "[*] Using WEB_PORT=$WEB_PORT"

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

$DOCKER_BIN run -d --name "$PICKER_NAME" --gpus all \
  --user "${HOST_UID}:${HOST_GID}" \
  --workdir /workspace \
  -v "$SAM2_PATH":/workspace \
  -v "$IN_MNT_HOST":/data/in \
  -v "$OUT_MNT_HOST":/data/out \
  -p 127.0.0.1:${WEB_PORT}:5000 \
  -e DATASET_NAME="$DATASET_NAME" \
  -e INPUT="/data/in/$DATASET_NAME" \
  -e OUT="/data/out" \
  -e INDEX_SUFFIX="$INDEX_SUFFIX" \
  "${DOCKER_GUI_FLAGS[@]}" \
  sam2:local \
  bash -lc 'umask 0002; python3 app/point_picker_flask.py'
exec 3>&1
echo "[*] Open to select points: http://localhost:${WEB_PORT}/" | tee /dev/tty


echo "[*] Waiting for decision/save → $DONE_FLAG"

baseline_mtime=0
[[ -f "$PROMPTS_HOST" ]] && baseline_mtime=$(stat -c %Y "$PROMPTS_HOST" 2>/dev/null || echo 0)

while :; do
  if [[ -f "$DONE_FLAG" ]]; then
    echo "[*] Picker signaled DONE_FLAG. Proceeding..."
    break
  fi
  if [[ -f "$PROMPTS_HOST" ]]; then
    cur_mtime=$(stat -c %Y "$PROMPTS_HOST" 2>/dev/null || echo 0)
    if [[ "$cur_mtime" -gt "$baseline_mtime" ]]; then
      echo "[*] prompts.json updated (mtime=$cur_mtime). Proceeding..."
      break
    fi
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
$DOCKER_BIN run --rm --gpus all \
  --user "${HOST_UID}:${HOST_GID}" \
  --workdir /workspace \
  -v "$SAM2_PATH":/workspace \
  -v "$IN_MNT_HOST":/data/in \
  -v "$OUT_MNT_HOST":/data/out \
  -e DATASET_NAME="$DATASET_NAME" \
  -e INPUT="/data/in/$DATASET_NAME" \
  -e OUT="/data/out" \
  -e INDEX_SUFFIX="$INDEX_SUFFIX" \
  -e QUIET=0 \
  -e MPLBACKEND=Agg \
  -e HF_HOME="/data/out/.cache/huggingface" \
  sam2:local \
  bash -lc 'umask 0002; python3 -u /workspace/app/video_predict.py'
echo "[*] SAM2 finished successfully (until here)."
$DOCKER_BIN pull colmap/colmap

if [ -f "$COLMAP_OUT_PATH/run_colmap.sh" ]; then
  chmod +x "$COLMAP_OUT_PATH/run_colmap.sh"
else
  echo "[*] run_colmap.sh not found in $SAMplify_SuGaR_PATH (skipping copy)"
fi
cd "$COLMAP_OUT_PATH"

# Ensure COLMAP dirs exist and are writable by you
mkdir -p "$COLMAP_OUT_PATH/input" "$COLMAP_OUT_PATH/output"
chmod -R u+rwX,g+rwX "$COLMAP_OUT_PATH/input" || true

install -d -m 775 \
  "$COLMAP_OUT_PATH/input/$DATASET_NAME" \
  "$COLMAP_OUT_PATH/input/${DATASET_NAME}_indexed"

# ---- paths ----
IMAGES_SRC="$SAM2_PATH/data/input/${DATASET_NAME}"
MASKS_SRC="$SAM2_PATH/data/output/${DATASET_NAME}_indexed"
IMAGES_DST="$COLMAP_OUT_PATH/input/${DATASET_NAME}"
MASKS_DST="$COLMAP_OUT_PATH/input/${DATASET_NAME}_indexed"
OUT_DST="$COLMAP_OUT_PATH/output/${DATASET_NAME}"

# ---- ensure dest dirs ----
mkdir -p "$IMAGES_DST" "$MASKS_DST" "$OUT_DST"

# ---- copy only images (jpg/jpeg/png) ----
rsync -a --delete \
  --include '*/' --include '*.jpg' --include '*.jpeg' --include '*.png' --exclude '*' \
  "${IMAGES_SRC}/" "${IMAGES_DST}/"

rsync -a --delete \
  --include '*/' --include '*.jpg' --include '*.jpeg' --include '*.png' --exclude '*' \
  "${MASKS_SRC}/" "${MASKS_DST}/"

echo "Copied images: $(find "$IMAGES_DST" -maxdepth 1 -type f | wc -l)"
echo "Copied masks : $(find "$MASKS_DST" -maxdepth 1 -type f | wc -l)"

# ---- run COLMAP ----
bash "$COLMAP_OUT_PATH/run_colmap.sh" \
  "$IMAGES_DST" \
  "$MASKS_DST" \
  "$OUT_DST" \
  exhaustive

# --- optionally stage helper files ---
if [ -f "$SAMplify_SuGaR_PATH/run_sugar_pipeline_with_sam.sh" ]; then
  echo "[*] Copying run_sugar_pipeline_with_sam.sh to $SUGAR_PATH"
  cp -f "$SAMplify_SuGaR_PATH/run_sugar_pipeline_with_sam.sh" "$SUGAR_PATH"
  chmod +x "$SUGAR_PATH/run_sugar_pipeline_with_sam.sh"
else
  echo "[*] run_sugar_pipeline_with_sam.sh not found in $SAMplify_SuGaR__PATH (skipping copy)"
fi

if [ -f "$SAMplify_SuGaR_PATH/Dockerfile_final" ]; then
  echo "[*] Copying Dockerfile and helpers to $SUGAR_PATH"
  cp -f "$SAMplify_SuGaR_PATH/Dockerfile_final" "$SUGAR_PATH"
  cp -f "$SAMplify_SuGaR_PATH/train.py" "$SUGAR_PATH/gaussian_splatting/"
  cp -f "$SAMplify_SuGaR_PATH/coarse_mesh.py" "$SUGAR_PATH/sugar_extractors/coarse_mesh.py"
else
  echo "[*] Dockerfile/train.py/coarse_mesh.py not found in $SAMplify_SuGaR_PATH (skipping copy)"
fi

# --- run SUGAR (pass DATASET_NAME as env) ---
echo "[*] Running Sugar pipeline for dataset: $DATASET_NAME..."
cd "$SUGAR_PATH"
DATASET_NAME="$DATASET_NAME" \
SUGAR_PATH="$SUGAR_PATH" \
SAMplify_SuGaR__PATH="$SAMplify_SuGaR_PATH"

bash ./run_sugar_pipeline_with_sam.sh "$DATASET_NAME"

echo "[*] Pipeline completed successfully!"
echo "Pipeline completed. Check log: $LOGFILE"
echo "[*] Pipeline completed" | tee /dev/tty
