#!/usr/bin/env bash

set -Eeuo pipefail

# --- safe defaults ---

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SAMplify_SuGaR_PATH:=$SCRIPT_DIR}"

: "${SAM2_PATH:=$SAMplify_SuGaR_PATH/SAM2}"

: "${SUGAR_PATH:=$SAMplify_SuGaR_PATH/SUGAR/SuGaR}"

: "${COLMAP_OUT_PATH:=$SAMplify_SuGaR_PATH/colmap}"

DATASET_NAME="${DATASET_NAME:-${1:-}}"

: "${DATASET_NAME:?Usage: $0 DATASET_NAME}"

REFINEMENT_TIME="${REFINEMENT_TIME:-short}"

# --- logging ---

LOGDIR="$SUGAR_PATH/logs"

mkdir -p "$LOGDIR"

LOGFILE="$LOGDIR/${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).log"

# keep fds & traps...

exec 3>&1 4>&2

restore_fds() { exec 1>&3 2>&4; }

on_error(){ restore_fds; echo "STATUS: ERROR" >&2; echo "LOG: $LOGFILE" >&2; exit 1; }

trap on_error ERR

trap restore_fds EXIT

exec >"$LOGFILE" 2>&1

restore_fds() { exec 1>&3 2>&4; }

on_error() {

restore_fds

echo "STATUS: ERROR" >&2

echo "LOG: $LOGFILE" >&2

exit 1

}

trap on_error ERR

trap restore_fds EXIT

# From here on, all output goes to the log

exec >"$LOGFILE" 2>&1

# -------------------------------

# 0. Parse args

# -------------------------------

if [ -z "$DATASET_NAME" ]; then

echo "Usage: $0 DATASET_NAME"

exit 1

fi

echo "======================================"

echo " Running SuGaR pipeline with SAM2 outputs"

echo " Dataset: $DATASET_NAME"

echo " Refinement time: $REFINEMENT_TIME"

echo "======================================"

# -------------------------------

# 1. Prepare directories

# -------------------------------

echo "[*] STEP 1: Preparing directories..."

# Where COLMAP writes its results (change if you keep them elsewhere)

COLMAP_OUT_PATH="${COLMAP_OUT_PATH:-${SAMplify_SuGaR_PATH}/colmap}"

COLMAP_SPARSE_DIR="${COLMAP_SPARSE_DIR:-${COLMAP_OUT_PATH}/output/${DATASET_NAME}/}"

# Create the folder if it doesn't exist

mkdir -p "$SUGAR_PATH/data/${DATASET_NAME}"

mkdir -p "$SUGAR_PATH/outputs/${DATASET_NAME}"

# Set folder permissions

sudo chown -R "$USER:$USER" "$SUGAR_PATH/data" "$SUGAR_PATH/outputs" "$SUGAR_PATH/cache"

chmod -R u+rwX,g+rwX "$SUGAR_PATH/data" "$SUGAR_PATH/outputs" "$SUGAR_PATH/cache"

mkdir -p "$SUGAR_PATH/cache"

mkdir -p "$SUGAR_PATH/data/${DATASET_NAME}_masked/images_sugar"

mkdir -p "$SUGAR_PATH/data/${DATASET_NAME}_masked/distorted"

mkdir -p "$SUGAR_PATH/data/${DATASET_NAME}_masked/input"

# Make sure our mounted trees are ours & writable

sudo chown -R "$USER:$USER" "$SUGAR_PATH/data" "$SUGAR_PATH/outputs" "$SUGAR_PATH/cache" || true

chmod -R u+rwX,g+rwX "$SUGAR_PATH/data" "$SUGAR_PATH/outputs" "$SUGAR_PATH/cache"

echo "ok"

# Copy SAM2 images to SuGaR input directory for COLMAP convertion and Copy SAM2 masked images to SuGaR directories

for img in "$SAM2_PATH/data/output/${DATASET_NAME}_indexed_masked/"*; do

cp "$img" "$SUGAR_PATH/data/${DATASET_NAME}_masked/input"

cp "$img" "$SUGAR_PATH/data/${DATASET_NAME}_masked/images_sugar"

done

echo "[*] Sync COLMAP sparse -> SuGaR distorted..."

DEST_DISTORTED="$SUGAR_PATH/data/${DATASET_NAME}_masked/distorted"

# ensure destination exists & is writable

mkdir -p "$DEST_DISTORTED/sparse/0"

sudo chown -R "$USER:$USER" "$DEST_DISTORTED" || true

chmod -R u+rwX,g+rwX "$DEST_DISTORTED"

# copy sparse model (cameras.bin, images.bin, points3D.bin, etc.)

rsync -a --delete "${COLMAP_SPARSE_DIR}/" "${DEST_DISTORTED}/"

echo " copied: $(ls -1 "${DEST_DISTORTED}/sparse/0" 2>/dev/null | wc -l) files"

echo "[*] STEP 1: Directories prepared"

# -------------------------------

# 2. Build Docker image (if needed)

# -------------------------------

echo "[*] STEP 2: Building Docker image sugar-final (if not already built)..."

cd "$SUGAR_PATH"

docker build -t sugar-final -f Dockerfile_final .

#-------------------------------

# # 3. Run conversion script

# # -------------------------------

echo "[*] STEP 4: Running convert.py for dataset=${DATASET_NAME}..."

docker run --gpus all -it --rm \

--user "$(id -u):$(id -g)" \

-v "$SUGAR_PATH/data/${DATASET_NAME}_masked:/app/data" \

-v "$SUGAR_PATH/data/${DATASET_NAME}_output:/app/output" \

-v "$SUGAR_PATH/cache:/app/.cache" \

-e HOME=/tmp \

-e XDG_CACHE_HOME=/app/.cache \

-e TORCH_EXTENSIONS_DIR=/app/.cache/torch_extensions \

sugar-final bash -lc '

set -e

umask 002

python /app/gaussian_splatting/convert.py -s /app/data --skip_matching

'

echo "======================================"

echo " DONE! Dataset conversion completed."

# -------------------------------

# 5. Run SuGaR pipeline with SAM2 masks

# -------------------------------

echo "[*] STEP 3: Running SuGaR on dataset=$DATASET_NAME..."

sudo docker run -it --rm --gpus all \

--user "$(id -u):$(id -g)" \

-v "$SUGAR_PATH/data/${DATASET_NAME}_masked:/app/data" \

-v "$SUGAR_PATH/outputs/${DATASET_NAME}:/app/output" \

-v "$SUGAR_PATH/cache:/app/.cache" \

-e XDG_CACHE_HOME=/app/.cache \

-e TORCH_EXTENSIONS_DIR=/app/.cache/torch_extensions \

-e HOME=/app \

sugar-final \

/app/run_with_xvfb.sh python train_full_pipeline.py \

-s /app/data \

-r dn_consistency \

--refinement_time $REFINEMENT_TIME \

--export_obj True \

--postprocess_mesh True \

--postprocess_density_threshold 0.1

echo "======================================"

echo " DONE! SuGaR training completed."

# -------------------------------

# 6. Extract textured mesh

# -------------------------------

echo "[*] STEP 4: Extracting textured mesh..."

sudo docker run -it --rm --gpus all \

--user "$(id -u):$(id -g)" \

-v "$SUGAR_PATH/data/${DATASET_NAME}_masked:/app/data" \

-v "$SUGAR_PATH/outputs/$DATASET_NAME:/app/output" \

-v "$SUGAR_PATH/cache:/app/.cache" \

-e XDG_CACHE_HOME=/app/.cache \

-e TORCH_EXTENSIONS_DIR=/app/.cache/torch_extensions \

-e HOME=/app \

sugar-final bash -lc "

set -e

REF=\$(find /app/output/refined/ -type f -name '2000.pt' | head -n1)

if [ -z \"\$REF\" ]; then

echo '[!] No refined checkpoint found, skipping mesh extraction.'

exit 0

fi

echo 'Using refined checkpoint: '\$REF

cp -r /app/sugar_utils /tmp/sugar_utils

sed -i 's/RasterizeGLContext()/RasterizeCudaContext()/g' /tmp/sugar_utils/mesh_rasterization.py

ln -sfn /app/output /tmp/output

cd /tmp

PYTHONPATH=/tmp:/app:/app/gaussian_splatting:\$PYTHONPATH \

python -m extract_refined_mesh_with_texture \

-s /app/data \

-c /app/output/vanilla_gs/data \

-m \"\$REF\" \

-o /app/output/refined_mesh/data \

--square_size 8

"

restore_fds

echo "STATUS: MESH DONE"
