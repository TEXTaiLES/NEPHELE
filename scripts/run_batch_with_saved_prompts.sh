#!/usr/bin/env bash
set -euo pipefail

# Batch-run run_pipeline.sh for datasets under SAM2/data/input/all_faces_sculpted.
# Work is split evenly across two workers:
#   Worker 0 -> GPU 0
#   Worker 1 -> GPU 3
# Each dataset must already have prompts.json in SAM2/data/output/<DATASET>_indexed/.
# Usage:
#   ./run_batch_with_saved_prompts.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEP="$(dirname "$SCRIPT_DIR")"
SUBDIR="all_faces_sculpted"
LOG_DIR="$NEP/logs/$SUBDIR"
mkdir -p "$LOG_DIR"

# Prevent accidental duplicate launches.
LOCK_FILE="$NEP/logs/.run_batch_with_saved_prompts.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another batch instance is already running."
  echo "Lock file: $LOCK_FILE"
  exit 1
fi

cd "$NEP"

REMAINING=()
for ds_path in "SAM2/data/input/$SUBDIR"/*; do
  [[ -d "$ds_path" ]] || continue
  DATASET="$SUBDIR/$(basename "$ds_path")"
  IDX_DIR="$NEP/SAM2/data/output/${DATASET}_indexed"
  OUT_DIR="$NEP/SUGAR/SuGaR/outputs/${DATASET}"

  [[ -f "$IDX_DIR/prompts.json" ]] || continue
  if ls "$OUT_DIR"/refined_mesh/data/*_postprocessed.obj &>/dev/null 2>&1; then
    continue
  fi

  REMAINING+=("$DATASET")
done

TOTAL=${#REMAINING[@]}
echo "=== $TOTAL datasets queued under $SUBDIR ==="

if [[ $TOTAL -eq 0 ]]; then
  echo "Nothing to do."
  exit 0
fi

GROUP0=()
GROUP1=()
for i in "${!REMAINING[@]}"; do
  if (( i % 2 == 0 )); then
    GROUP0+=("${REMAINING[$i]}")
  else
    GROUP1+=("${REMAINING[$i]}")
  fi
done

LIST0="/tmp/sugar_all_faces_sculpted_gpu0_${$}.txt"
LIST1="/tmp/sugar_all_faces_sculpted_gpu3_${$}.txt"
trap 'rm -f "$LIST0" "$LIST1"' EXIT

printf '%s\n' "${GROUP0[@]}" > "$LIST0"
printf '%s\n' "${GROUP1[@]}" > "$LIST1"

echo "Worker 0 on GPU 0: ${#GROUP0[@]} datasets"
echo "Worker 1 on GPU 3: ${#GROUP1[@]} datasets"

"$SCRIPT_DIR/worker.sh" 0 0 "$LIST0" >"$LOG_DIR/worker0_gpu0.log" 2>&1 &
PID0=$!
"$SCRIPT_DIR/worker.sh" 1 3 "$LIST1" >"$LOG_DIR/worker1_gpu3.log" 2>&1 &
PID1=$!

RC0=0
RC1=0
wait "$PID0" || RC0=$?
wait "$PID1" || RC1=$?

if [[ $RC0 -ne 0 || $RC1 -ne 0 ]]; then
  echo "One or both workers failed: GPU0=$RC0 GPU3=$RC1"
  exit 1
fi

echo "All datasets processed successfully."
