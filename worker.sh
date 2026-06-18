#!/usr/bin/env bash

set -euo pipefail

WORKER_ID="${1:?Usage: $0 WORKER_ID GPU_ID LIST_FILE}"

GPU_ID="${2:?}"

LIST_FILE="${3:?}"

NEP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$NEP"

# Isolate each worker with its own compose project + GPU override

export COMPOSE_PROJECT_NAME="sugar_w${WORKER_ID}"

export COMPOSE_FILE="$NEP/docker-compose.yml:$NEP/docker-compose.gpu${GPU_ID}.yml"

export SAM2_CONTAINER_NAME="sam2_w${WORKER_ID}"

export COLMAP_CONTAINER_NAME="colmap_w${WORKER_ID}"

export SUGAR_CONTAINER_NAME="sugar_w${WORKER_ID}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

export WEB_PORT="${WEB_PORT:-$((8095 + WORKER_ID))}"

echo "=== Worker $WORKER_ID starting (GPU $GPU_ID, port $WEB_PORT) ==="

echo "=== Containers: sam2_w${WORKER_ID}, colmap_w${WORKER_ID}, sugar_w${WORKER_ID} ==="

# Start the sam2 container for this worker

DATASET_NAME="" docker compose up -d sam2

sleep 3

while IFS= read -r DATASET; do

[[ -z "$DATASET" ]] && continue

# Double-check not already done

OUT_DIR="$NEP/SUGAR/SuGaR/outputs/${DATASET}"

if ls "$OUT_DIR"/refined_mesh/data/*_postprocessed.obj &>/dev/null 2>&1; then

echo "[W${WORKER_ID}] Skipping $DATASET (already completed)"

continue

fi

echo "[W${WORKER_ID}] === Running: $DATASET ==="

./run_pipeline.sh "$DATASET" </dev/null || { echo "[W${WORKER_ID}] Pipeline FAILED for $DATASET"; continue; }

echo "[W${WORKER_ID}] === Finished: $DATASET ==="

done < "$LIST_FILE"

# Cleanup

docker compose down || true

echo "[W${WORKER_ID}] === ALL DONE ==="
