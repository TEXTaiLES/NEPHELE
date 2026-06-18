#!/usr/bin/env bash
set -euo pipefail

# Default to collaborative perms: files 664, dirs 775
UMASK="${UMASK:-0002}"
umask "$UMASK"

# Make sure mounts exist inside container namespace (harmless if they’re bind-mounts)
mkdir -p "${IN_MNT:-/data/in}" "${OUT_MNT:-/data/out}"

# Ensure HF cache exists and is user-writable
mkdir -p "${HF_HOME:-/home/user/.cache/huggingface}"

# Pre-create dataset-specific output dirs if provided (avoids race/perm issues)
if [[ -n "${DATASET_NAME:-}" ]]; then
  mkdir -p "/data/out/${DATASET_NAME}_indexed" \
           "/data/out/${DATASET_NAME}_indexed_masked" \
           "/data/out/.cache/hf" \
           "/data/out/checkpoints"
fi

exec "$@"
