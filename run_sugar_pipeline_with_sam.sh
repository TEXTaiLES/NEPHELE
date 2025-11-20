#!/usr/bin/env bash
set -Eeuo pipefail


# ===== BASIC SAFE DEFAULTS (robust paths) =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Where the SuGaR repo lives: this script is inside it
SUGAR_PATH="${SUGAR_PATH:-$SCRIPT_DIR}"

# Monorepo root is two levels up from SuGaR (…/SUGAR/SuGaR -> …/)
MONO_ROOT="${nephele_PATH:-$(cd "$SUGAR_PATH/../.." && pwd)}"

# Other components relative to repo root (can be overridden from env)
SAM2_PATH="${SAM2_PATH:-$MONO_ROOT/SAM2}"
COLMAP_OUT_PATH="${COLMAP_OUT_PATH:-$MONO_ROOT/colmap}"

DOCKER_BIN="${DOCKER_BIN:-docker}"
UMASK="${UMASK:-0002}"
umask "$UMASK"


HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

DATASET_NAME="${DATASET_NAME:-${1:-}}"
: "${DATASET_NAME:?Usage: $0 DATASET_NAME}"
REFINEMENT_TIME="${REFINEMENT_TIME:-short}"

# ========= LOGGING (μόνο σε αρχείο) =========
LOGDIR="$SUGAR_PATH/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).log"

# Κράτα FD για αποκατάσταση
exec 3>&1 4>&2
restore_fds() { exec 1>&3 2>&4; }
on_error(){ restore_fds; echo "STATUS: ERROR" >&2; echo "LOG: $LOGFILE" >&2; exit 1; }
trap on_error ERR
trap restore_fds EXIT

# Από εδώ και πέρα, όλα πάνε στο log
exec >"$LOGFILE" 2>&1

echo "======================================"
echo " Running SuGaR pipeline with SAM2 outputs"
echo " Dataset: $DATASET_NAME"
echo " Refinement time: $REFINEMENT_TIME"
echo " Log: $LOGFILE"
echo "======================================"

# ========= HELPERS =========
owner_uid() { stat -c %u "$1" 2>/dev/null || echo -1; }
is_owner()   { [[ "$(owner_uid "$1")" == "$HOST_UID" ]]; }

ensure_dir() {
  local d="$1"
  [[ -d "$d" ]] || mkdir -p "$d"
  # setgid + group-writable ΜΟΝΟ αν είσαι owner (για να μη βγάζει errors)
  if is_owner "$d"; then
    chmod 2775 "$d" 2>/dev/null || true   # drwxrwsr-x
  fi
}

write_test() {
  local d="$1"
  local tmp="$d/.__w_test__"
  if ! ( : > "$tmp" ) 2>/dev/null; then
    echo "⚠️  No write access to $d. Fix host perms/ACLs."
  else
    rm -f "$tmp" || true
  fi
}

# ========= 0. Paths & inputs =========
COLMAP_SPARSE_DIR="${COLMAP_SPARSE_DIR:-${COLMAP_OUT_PATH}/output/${DATASET_NAME}}"

# Consistent SuGaR working dirs
SUGAR_DATA_ROOT="$SUGAR_PATH/data/${DATASET_NAME}_masked"
SUGAR_OUT_ROOT="$SUGAR_PATH/outputs/${DATASET_NAME}"
SUGAR_CACHE="$SUGAR_PATH/cache"

ensure_dir "$SUGAR_PATH/data"
ensure_dir "$SUGAR_PATH/outputs"
ensure_dir "$SUGAR_CACHE"
ensure_dir "$SUGAR_DATA_ROOT/images_sugar"
ensure_dir "$SUGAR_DATA_ROOT/distorted/sparse/0"
ensure_dir "$SUGAR_DATA_ROOT/input"
ensure_dir "$SUGAR_OUT_ROOT"

write_test "$SUGAR_OUT_ROOT"
write_test "$SUGAR_CACHE"

# ========= 1. Copy masked images from SAM2 =========
echo "[*] STEP 1: Copy SAM2 masked images → SuGaR input"

SAM2_MASK_DIR="$SAM2_PATH/data/output/${DATASET_NAME}_indexed_masked"
if [[ -d "$SAM2_MASK_DIR" ]]; then
  shopt -s nullglob
  for img in "$SAM2_MASK_DIR/"*; do
    cp -f "$img" "$SUGAR_DATA_ROOT/input/" || true
    cp -f "$img" "$SUGAR_DATA_ROOT/images_sugar/" || true
  done
  shopt -u nullglob
else
  echo "[!] SAM2 masked dir not found: $SAM2_MASK_DIR (continuing)"
fi

# ========= 1b. Sync COLMAP sparse model =========
echo "[*] STEP 1b: Sync COLMAP sparse → SuGaR distorted"
DEST_DISTORTED="$SUGAR_DATA_ROOT/distorted"
ensure_dir "$DEST_DISTORTED/sparse/0"
write_test "$DEST_DISTORTED"

if [[ -d "$COLMAP_SPARSE_DIR/sparse/0" ]]; then
  rsync -a --delete "${COLMAP_SPARSE_DIR}/" "${DEST_DISTORTED}/" || true
else
  echo "[!] Expected COLMAP sparse at: $COLMAP_SPARSE_DIR/sparse/0 (continuing)"
fi
echo "  files in distorted/sparse/0: $(ls -1 "${DEST_DISTORTED}/sparse/0" 2>/dev/null | wc -l)"

echo "[*] STEP 1: Directories prepared"

# ========= 2. Build sugar-final image if missing =========
echo "[*] STEP 2: Build sugar-final (if needed)"
cd "$SUGAR_PATH"
if ! $DOCKER_BIN image inspect sugar-final >/dev/null 2>&1; then
  $DOCKER_BIN build \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  -t sugar-final -f Dockerfile_final .
else
  echo "[i] sugar-final image already present."
fi


# ========= Common container env/volumes =========
VOL_DATA="-v $SUGAR_DATA_ROOT:/app/data"
VOL_OUT="-v $SUGAR_OUT_ROOT:/app/output"
VOL_CACHE="-v $SUGAR_CACHE:/app/.cache"

ENV_COMMON=(
  -e HOME=/app
  -e XDG_CACHE_HOME=/app/.cache
  -e TORCH_EXTENSIONS_DIR=/app/.cache/torch_extensions
  -e HF_HOME=/app/.cache/hf
  -e TRANSFORMERS_CACHE=/app/.cache/hf
  -e HUGGINGFACE_HUB_CACHE=/app/.cache/hf
  -e MPLBACKEND=Agg
)

DOCKER_COMMON=(
  --gpus all --rm --ipc=host
  --user "${HOST_UID}:${HOST_GID}"
  $VOL_DATA $VOL_OUT $VOL_CACHE
  "${ENV_COMMON[@]}"
)

# ========= 3. Convert (COLMAP format → SuGaR) =========
echo "[*] STEP 3: convert.py"
$DOCKER_BIN run "${DOCKER_COMMON[@]}" \
   sugar-final bash -lc '
    set -e
    umask 0002
    mkdir -p /app/.cache/hf /app/output
    # --skip_matching γιατί έρχεται ήδη από COLMAP
    python /app/gaussian_splatting/convert.py -s /app/data 
  '

echo "======================================"
echo " DONE! Dataset conversion completed."

# ========= 4. SuGaR training =========
echo "[*] STEP 4: SuGaR training"
$DOCKER_BIN run "${DOCKER_COMMON[@]}" \
  sugar-final \
  /app/run_with_xvfb.sh python train_full_pipeline.py \
    -s /app/data \
    -r dn_consistency \
    --refinement_time "$REFINEMENT_TIME" \
    --export_obj True \
    --postprocess_mesh True \
    --postprocess_density_threshold 0.1

echo "======================================"
echo " DONE! SuGaR training completed."

# ========= 5. Extract refined textured mesh =========
echo "[*] STEP 5: Extract refined textured mesh"
$DOCKER_BIN run "${DOCKER_COMMON[@]}" \
  sugar-final bash -lc '
    set -e
    umask 0002
    REF=$(find /app/output/refined/ -type f -name "2000.pt" | head -n1 || true)
    if [ -z "$REF" ]; then
      echo "[!] No refined checkpoint found, skipping mesh extraction."
      exit 0
    fi
    echo "Using refined checkpoint: $REF"
    cp -r /app/sugar_utils /tmp/sugar_utils
    sed -i "s/RasterizeGLContext()/RasterizeCudaContext()/g" /tmp/sugar_utils/mesh_rasterization.py
    ln -sfn /app/output /tmp/output
    cd /tmp
    PYTHONPATH=/tmp:/app:/app/gaussian_splatting:$PYTHONPATH \
      python -m extract_refined_mesh_with_texture \
        -s /app/data \
        -c /app/output/vanilla_gs/data \
        -m "$REF" \
        -o /app/output/refined_mesh/data \
        --square_size 8
  '

restore_fds
echo "STATUS: MESH DONE"
echo "LOG: $LOGFILE"
