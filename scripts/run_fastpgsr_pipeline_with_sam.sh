#!/usr/bin/env bash

# Run Fast-PGSR (FastGS-accelerated PGSR) using SAM2 masked images + COLMAP.
# Usage: DATASET_NAME=<name> bash run_fastpgsr_pipeline_with_sam.sh
# or: bash run_fastpgsr_pipeline_with_sam.sh <dataset_name>
#
# This is a copy of run_pgsr_pipeline_with_sam.sh targeting the isolated
# fastpgsr:local image (conda env `fast-pgsr`) instead of pgsr:local. FastGS's
# render.py writes the same mesh/tsdf_fusion*.ply shape as stock PGSR, so the
# RGBA-build and ply->obj (preferring _post) conversion steps are unchanged.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

MONO_ROOT="${nephele_PATH:-$REPO_ROOT}"

SAM2_PATH="${SAM2_PATH:-$MONO_ROOT/SAM2}"

COLMAP_OUT_PATH="${COLMAP_OUT_PATH:-$MONO_ROOT/colmap}"

FASTPGSR_PATH="${FASTPGSR_PATH:-$MONO_ROOT/FASTPGSR}"

DOCKER_BIN="${DOCKER_BIN:-docker}"

umask 0002

HOST_UID="$(id -u)"

HOST_GID="$(id -g)"

DATASET_NAME="${DATASET_NAME:-${1:-}}"

: "${DATASET_NAME:?Usage: $0 DATASET_NAME}"

ITERATIONS="${PGSR_ITERATIONS:-30000}"

MAX_DEPTH="${PGSR_MAX_DEPTH:-5.0}"

VOXEL_SIZE="${PGSR_VOXEL_SIZE:-0.01}"

# PGSR_RESOLUTION: 1=full res (default), 2=half, 4=quarter.
RESOLUTION="${PGSR_RESOLUTION:-1}"

# Densification window: stop adding Gaussians after this iteration.
DENSIFY_UNTIL="${PGSR_DENSIFY_UNTIL:-15000}"

# Geometry loss schedule: start single-view + multi-view losses at this iteration.
GEOM_FROM="${PGSR_GEOM_FROM:-7000}"

# ====== PATHS ======

COLMAP_SPARSE_DIR="${COLMAP_SPARSE_DIR:-${COLMAP_OUT_PATH}/output/${DATASET_NAME}}"

SAM2_MASKED_DIR="$SAM2_PATH/data/output/${DATASET_NAME}_indexed_masked"

SAM2_MASK_DIR="$SAM2_PATH/data/output/${DATASET_NAME}_indexed"

FASTPGSR_DATA_ROOT="$FASTPGSR_PATH/data/${DATASET_NAME}"

FASTPGSR_OUT_ROOT="$FASTPGSR_PATH/outputs/${DATASET_NAME}"

# ====== LOGGING ======

mkdir -p "$MONO_ROOT/logs"

LOGFILE="$MONO_ROOT/logs/fastpgsr_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).log"

# tee so output goes to the dedicated Fast-PGSR log AND to the parent pipeline log

exec > >(tee -a "$LOGFILE") 2>&1

echo "======================================";

echo " Fast-PGSR pipeline"

echo " Dataset: $DATASET_NAME"

echo " Iterations: $ITERATIONS  densify_until: $DENSIFY_UNTIL  geom_from: $GEOM_FROM"

echo " Log: $LOGFILE"

echo "======================================"

# ====== HELPERS ======

run_in_fastpgsr() {

local script="$1"

local DOCKER_COMMON=(

--rm --gpus all

-u "${HOST_UID}:${HOST_GID}"

-v "$FASTPGSR_DATA_ROOT:/app/data"

-v "$FASTPGSR_OUT_ROOT:/app/output"

-e HOME=/tmp

--workdir /app

)

set +e

$DOCKER_BIN compose -f "$MONO_ROOT/docker-compose.yml" run \
--rm --no-deps \
-v "$FASTPGSR_DATA_ROOT:/app/data" \
-v "$FASTPGSR_OUT_ROOT:/app/output" \
-e HOME=/tmp \
--workdir /app \
fastpgsr bash -lc "$script"

local RC=$?

set -e

if [[ $RC -ne 0 ]]; then

# Fallback: docker run directly with image name

echo "[i] compose run failed (rc=$RC), falling back to docker run with image fastpgsr:local"

$DOCKER_BIN run --gpus all "${DOCKER_COMMON[@]}" fastpgsr:local bash -lc "$script"

fi

}

# ====== STEP 1: Prepare data directory ======

echo "[*] STEP 1: Prepare Fast-PGSR data directory"

mkdir -p "$FASTPGSR_DATA_ROOT/images" "$FASTPGSR_DATA_ROOT/sparse" "$FASTPGSR_OUT_ROOT"

# Build RGBA PNGs: combine _indexed_masked (RGB) + _indexed (grayscale mask) → RGBA PNG
# FastGS reads the alpha channel and uses it to mask the photometric loss.

echo " Building RGBA PNGs from masked RGB + mask..."

$DOCKER_BIN run --rm \
-v "$SAM2_MASKED_DIR:/rgb:ro" \
-v "$SAM2_MASK_DIR:/mask:ro" \
-v "$FASTPGSR_DATA_ROOT/images:/out" \
-u "${HOST_UID}:${HOST_GID}" \
fastpgsr:local bash -lc "

python3 - << 'RGBA_PY'
import os, sys
import numpy as np
from PIL import Image

rgb_dir  = '/rgb'
mask_dir = '/mask'
out_dir  = '/out'

fnames = sorted(f for f in os.listdir(rgb_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png')))

if not fnames:
    print(' [!] No images found in', rgb_dir); sys.exit(1)

count = 0
for fname in fnames:
    rgb  = Image.open(os.path.join(rgb_dir, fname)).convert('RGB')
    stem = os.path.splitext(fname)[0]
    mpath = None
    for ext in ('.jpg', '.jpeg', '.png'):
        cand = os.path.join(mask_dir, stem + ext)
        if os.path.exists(cand):
            mpath = cand; break
    if mpath:
        gray  = np.array(Image.open(mpath).convert('L'))
        alpha = Image.fromarray(np.where(gray > 0, 255, 0).astype(np.uint8))
    else:
        alpha = Image.new('L', rgb.size, 255)
    rgba = rgb.copy()
    rgba.putalpha(alpha)
    # Keep the original filename so it matches what COLMAP recorded in images.bin.
    # PIL writes PNG format regardless of the .jpg extension; PIL.Image.open reads by
    # file header, so both sides handle this correctly.
    rgba.save(os.path.join(out_dir, fname), format='PNG')
    count += 1

print(f' Built {count} RGBA PNGs → /out')
RGBA_PY

"

MASKED_COUNT=$(ls "$FASTPGSR_DATA_ROOT/images/" 2>/dev/null | wc -l)

echo " $MASKED_COUNT RGBA images in $FASTPGSR_DATA_ROOT/images/"

if [[ $MASKED_COUNT -eq 0 ]]; then

echo "[!] WARNING: No RGBA images built — check SAM2 output dirs"

fi

# Copy COLMAP sparse reconstruction (flat into sparse/ — the fastpgsr dataset
# reader override reads sparse/ directly, no sparse/0/ subdir).

if [[ -d "$COLMAP_SPARSE_DIR/sparse/0" ]]; then

cp -f "$COLMAP_SPARSE_DIR/sparse/0/"* "$FASTPGSR_DATA_ROOT/sparse/" 2>/dev/null || true

echo " Copied COLMAP sparse → $FASTPGSR_DATA_ROOT/sparse/"

else

echo "[!] WARNING: COLMAP sparse not found at $COLMAP_SPARSE_DIR/sparse/0"

fi

# ====== STEP 2: Train Fast-PGSR ======

echo "[*] STEP 2: Train Fast-PGSR (${ITERATIONS} iterations)"

run_in_fastpgsr "

set -e

umask 0002

source /opt/conda/etc/profile.d/conda.sh

conda activate fast-pgsr

mkdir -p /app/output

python /app/train.py \
-s /app/data \
-m /app/output \
--iterations ${ITERATIONS} \
--test_iterations -1 \
--save_iterations ${ITERATIONS} \
-r ${RESOLUTION} \
--densify_until_iter ${DENSIFY_UNTIL} \
--single_view_weight_from_iter ${GEOM_FROM} \
--multi_view_weight_from_iter ${GEOM_FROM}

"

echo "[*] Training done."

# ====== STEP 3: Export mesh ======

echo "[*] STEP 3: Export mesh (max_depth=${MAX_DEPTH}, voxel_size=${VOXEL_SIZE})"

run_in_fastpgsr "

set -e

umask 0002

source /opt/conda/etc/profile.d/conda.sh

conda activate fast-pgsr

python /app/render.py \
-m /app/output \
--max_depth ${MAX_DEPTH} \
--voxel_size ${VOXEL_SIZE} \
--num_cluster 1 \
--skip_test

"

echo "[*] Mesh export done."

# ====== STEP 4: Ensure an OBJ exists ======
# FastGS render.py (like stock PGSR) only writes tsdf_fusion(.post).ply. The
# worker's upload step wants an .obj, so convert the newest .ply (preferring
# the _post variant) inside the container, which has open3d available.

if ! ls "$FASTPGSR_OUT_ROOT/mesh/"*.obj >/dev/null 2>&1; then

echo "[*] STEP 4: No .obj in mesh/ — converting .ply → .obj"

run_in_fastpgsr "

set -e

umask 0002

source /opt/conda/etc/profile.d/conda.sh

conda activate fast-pgsr

python - <<'PY'
import glob, os
import open3d as o3d

mesh_dir = '/app/output/mesh'
plys = sorted(glob.glob(os.path.join(mesh_dir, '*.ply')), key=os.path.getmtime)
if not plys:
    raise SystemExit('[!] no .ply found in ' + mesh_dir)
post = [p for p in plys if '_post' in os.path.basename(p)]
src = (post or plys)[-1]
dst = os.path.splitext(src)[0] + '.obj'
mesh = o3d.io.read_triangle_mesh(src)
o3d.io.write_triangle_mesh(dst, mesh, write_triangle_uvs=True,
                           write_vertex_colors=True, write_vertex_normals=True)
print(f'[*] converted {src} -> {dst}')
PY

"

else

echo "[*] STEP 4: .obj already present in mesh/ — skipping conversion"

fi

echo ""

echo "======================================"

echo " Fast-PGSR pipeline complete!"

echo " Output: $FASTPGSR_OUT_ROOT"

echo "======================================"
