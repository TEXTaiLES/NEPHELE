#!/usr/bin/env bash
set -euo pipefail

IMAGES=${1:-"$nephele_PATH/colmap/input/${DATASET_NAME}"}
MASKS=${2:-"$nephele_PATH/colmap/input/${DATASET_NAME}_indexed"}
OUT=${3:-"$nephele_PATH/colmap/output/${DATASET_NAME}"}
MATCHER=${4:-exhaustive}  # or "sequential"


mkdir -p "$OUT"

docker run --gpus all -it --rm \
  --user "$(id -u):$(id -g)" \
  -v "$IMAGES:/images:ro" \
  -v "$MASKS:/masks:ro" \
  -v "$OUT:/output" \
  colmap/colmap \
  bash -lc '
    set -e

    # --- Prepare masks in the naming COLMAP expects: <image_filename>.png ---
    mkdir -p /tmp/masks_colmap
    shopt -s nullglob
    for img in /images/*.{jpg,JPG,jpeg,JPEG,png,PNG}; do
      [ -e "$img" ] || continue
      base="$(basename "$img")"              # e.g. image_123.jpg
      stem="${base%.*}"                      # e.g. image_123

      # Priority: exact COLMAP name -> same name -> stem.png -> stem.jpg
      if   [ -f "/masks/${base}.png" ]; then
        ln -sf "/masks/${base}.png" "/tmp/masks_colmap/${base}.png"
      elif [ -f "/masks/${base}" ]; then
        ln -sf "/masks/${base}"      "/tmp/masks_colmap/${base}.png"
      elif [ -f "/masks/${stem}.png" ]; then
        ln -sf "/masks/${stem}.png"  "/tmp/masks_colmap/${base}.png"
      elif [ -f "/masks/${stem}.jpg" ] || [ -f "/masks/${stem}.JPG" ]; then
        src="/masks/${stem}.jpg"
        [ -f "$src" ] || src="/masks/${stem}.JPG"
        ln -sf "$src" "/tmp/masks_colmap/${base}.png"
      fi
    done
    echo "Prepared $(ls -1 /tmp/masks_colmap | wc -l) mask links for COLMAP."

    # --- COLMAP pipeline ---
    colmap feature_extractor \
      --database_path /output/database.db \
      --image_path /images \
      --ImageReader.mask_path /tmp/masks_colmap \
      --SiftExtraction.max_num_features 10000 \
      --ImageReader.single_camera 1

    if [ "'"$MATCHER"'" = "exhaustive" ]; then
      colmap exhaustive_matcher --database_path /output/database.db
    else
      colmap sequential_matcher --database_path /output/database.db \
        --SequentialMatching.loop_detection 0
    fi

    mkdir -p /output/sparse
    colmap mapper \
      --database_path /output/database.db \
      --image_path /images \
      --output_path /output/sparse

  '

# chmod +x run_colmap.sh
# ./run_colmap.sh \
#   /home/ilias/test/colmap/input/vazo \
#  /home/ilias/test/colmap/input/vazo_indexed \
#   /home/ilias/test/colmap/output/vazo_run1\
#   exhaustive
