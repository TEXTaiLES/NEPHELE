#!/usr/bin/env bash

# Pipeline watcher daemon.
#
# Runs in the background on the GPU/compute host. Polls the SAM2 output tree
# for `__picker_done.flag` files and spawns `run_pipeline.sh <dataset>` for
# each unprocessed one. Lets the UI auto-trigger the rest of the pipeline
# (COLMAP → SuGaR) without anyone needing to keep a terminal open.
#
# Future cross-server architecture: when the UI moves to a different host,
# replace this file-watch with an HTTP trigger endpoint hosted on the GPU
# server (POST /trigger). The run_pipeline.sh call stays the same.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SAM2_OUT="$SCRIPT_DIR/SAM2/data/output"

INTERVAL="${PIPELINE_WATCH_INTERVAL:-3}" # seconds between scans

mkdir -p "$SCRIPT_DIR/logs"

LOGFILE="$SCRIPT_DIR/logs/pipeline_watcher.log"

log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOGFILE"; }

# Each dataset's flag gets paired with __picker_done.processed once consumed
# so we don't re-run the pipeline on every poll.

process_dataset() {

local dataset="$1"

local flag="$2"

local processed="${flag%.flag}.processed"

log "triggering pipeline for: $dataset"

touch "$processed"

# Run in a subshell so errors here can't kill the watcher itself.

(

cd "$SCRIPT_DIR"

bash ./run_pipeline.sh "$dataset" \
>> "$SCRIPT_DIR/logs/pipeline_watcher_${dataset}.log" 2>&1 \
|| log "pipeline for '$dataset' exited non-zero (see logs)"

) &

}

scan() {

shopt -s nullglob

for flag in "$SAM2_OUT"/*_indexed/__picker_done.flag; do

local indexed_dir

indexed_dir="$(dirname "$flag")"

local dataset

dataset="$(basename "$indexed_dir")"

dataset="${dataset%_indexed}"

local processed="${flag%.flag}.processed"

[[ -f "$processed" ]] && continue

process_dataset "$dataset" "$flag"

done

}

log "watcher started; scanning $SAM2_OUT every ${INTERVAL}s"

trap 'log "watcher stopping"; exit 0' INT TERM

while :; do

scan

sleep "$INTERVAL"

done
