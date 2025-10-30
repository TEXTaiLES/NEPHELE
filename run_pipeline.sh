#!/usr/bin/env bash
set -euo pipefail

# ====== ARGS / DATASET ======
DATASET_NAME="${DATASET_NAME:-${1:-}}"
: "${DATASET_NAME:?Usage: $0 DATASET_NAME  (or export DATASET_NAME first)}"

# ====== PATHS ======

# --- resolve repo paths early ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMplify_SuGaR_PATH="${SAMplify_SuGaR_PATH:-$SCRIPT_DIR}"
SAM2_PATH="${SAM2_PATH:-${SAMplify_SuGaR_PATH}/SAM2}"
SUGAR_PATH="${SUGAR_PATH:-${SAMplify_SuGaR_PATH}/SUGAR/SuGaR}"
COLMAP_OUT_PATH="${COLMAP_OUT_PATH:-${SAMplify_SuGaR_PATH}/colmap}"

export DATASET_NAME SUGAR_PATH SAMplify_SuGaR_PATH SAM2_PATH COLMAP_OUT_PATH
#

# Where SAM2 expects input/output INSIDE the container:
IN_MNT_HOST="$SAM2_PATH/data/input"
OUT_MNT_HOST="$SAM2_PATH/data/output"
IN_MNT_CONT="/data/in"
OUT_MNT_CONT="/data/out"

# If you want INPUT to be dataset-specific, put images in: $IN_MNT_HOST/$DATASET_NAME
INPUT_SUBDIR="${INPUT_SUBDIR:-$DATASET_NAME}"    # override with INPUT_SUBDIR=foo if you like
INPUT_CONT="$IN_MNT_CONT/$INPUT_SUBDIR"          # what we pass to the container as INPUT

# ====== LOGGING (SILENT: only to file, nothing on terminal) ======
LOGDIR="$SAMplify_SuGaR_PATH/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).log"

# Send ALL stdout+stderr ONLY to the logfile
exec 1>>"$LOGFILE" 2>&1

on_error() {
  echo "STATUS: ERROR"
  echo "LOG: $LOGFILE"
  exit 1
}
trap on_error ERR
# (No restore of FDs on EXIT; remains silent)

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

# Image presence (sam2:local); skip build here, just check:
if ! docker image inspect sam2:local >/dev/null 2>&1; then
  echo "Docker image 'sam2:local' not found. Build it first (e.g., docker build -t sam2:local $SAM2_PATH)"
  exit 1
fi

# Ensure mount folders exist
mkdir -p "$IN_MNT_HOST" "$OUT_MNT_HOST"
# Also ensure dataset subfolder exists (so INPUT points to a real path)
mkdir -p "$IN_MNT_HOST/$INPUT_SUBDIR"

# ====== GUI or HEADLESS ======
# Set GUI=1 to use your X display, else GUI=0 for headless (xvfb inside the container if supported)
GUI="${GUI:-1}"          # default GUI on
FRAME_IDX="${FRAME_IDX:-0}"
OBJ_ID="${OBJ_ID:-1}"

# X perms for GUI=1
DOCKER_GUI_FLAGS=()
if [[ "$GUI" == "1" ]]; then
  # Allow local docker container to access X
  if command -v xhost >/dev/null 2>&1; then
    xhost +local:docker >/dev/null 2>&1 || true
  fi
  : "${DISPLAY:=${DISPLAY:-:0}}"
  DOCKER_GUI_FLAGS=( -e DISPLAY="$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix )
else
  echo "[i] GUI=0 → running without X display."
fi

# ====== RUN SAM2 (ONLY UNTIL HERE) ======
cd "$SAM2_PATH"
echo "[*] Running SAM2 for dataset: $DATASET_NAME"
echo "[*] INPUT (container): $INPUT_CONT"
echo "[*] OUT   (container): $OUT_MNT_CONT"
IN="$SAM2_PATH/data/input"
OUT="$SAM2_PATH/data/output"

sudo chown -R "$USER:$USER" "$IN" "$OUT"
chmod -R u+rwX "$IN" "$OUT"
mkdir -p "$IN/$DATASET_NAME" "$OUT"

# NOTE: removed -it to avoid attaching TTY; stays silent
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -v "$PWD/data/input:/data/in" \
  -v "$PWD/data/output:/data/out" \
  -e DISPLAY="$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e INPUT="/data/in/$DATASET_NAME" \
  -e OUT="/data/out" \
  -e GUI=1 -e FRAME_IDX=0 -e OBJ_ID=1 \
  sam2:local \
  bash -lc 'python3 /workspace/app/video_predict.py'

echo "[*] SAM2 finished successfully (until here)."
docker pull colmap/colmap

if [ -f "$COLMAP_OUT_PATH/run_colmap.sh" ]; then
  chmod +x "$COLMAP_OUT_PATH/run_colmap.sh"
else
  echo "[*] run_colmap.sh not found in $SAMplify_SuGaR_PATH (skipping copy)"
fi
cd "$COLMAP_OUT_PATH"

sudo chown -R "$USER:$USER" "$COLMAP_OUT_PATH/input"
chmod -R u+rwX,g+rwX "$COLMAP_OUT_PATH/input"

sudo install -d -m 775 -o "$USER" -g "$USER" \
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
  echo "[*] Copying Dockerfile to $SUGAR_PATH"
  cp -f "$SAMplify_SuGaR_PATH/Dockerfile_final" "$SUGAR_PATH"
  cp -f "$SAMplify_SuGaR_PATH/train.py" "$SUGAR_PATH/gaussian_splatting/"
  cp -f "$SAMplify_SuGaR_PATH/coarse_mesh.py" "$SUGAR_PATH/sugar_extractors/coarse_mesh.py"
else
  echo "[*] Dockerfile not found in $SAMplify_SuGaR_PATH (skipping copy)"
  echo "[*] train.py not found in $SAMplify_SuGaR_PATH (skipping copy)"
fi

# --- run SUGAR (pass DATASET_NAME as env) ---
echo "[*] Running Sugar pipeline for dataset: $DATASET_NAME..."
cd "$SUGAR_PATH"
DATASET_NAME="$DATASET_NAME" \
SUGAR_PATH="$SUGAR_PATH" \
SAMplify_SuGaR__PATH="$SAMplify_SuGaR_PATH"

# Uncomment to run the sugar pipeline
bash ./run_sugar_pipeline_with_sam.sh "$DATASET_NAME"

echo "[*] Pipeline completed successfully!"
echo "LOG: $LOGFILE"

