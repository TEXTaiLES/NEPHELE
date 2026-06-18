#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_colmap.sh <images> <masks> <output> [matcher] [known_poses_dir]
#
# Optional env vars:
#   DO_BA=1   — run bundle adjustment after triangulation (only with known poses)

IMAGES=${1:-"$nephele_PATH/colmap/input/${DATASET_NAME}"}
MASKS=${2:-"$nephele_PATH/colmap/input/${DATASET_NAME}_indexed"}
OUT=${3:-"$nephele_PATH/colmap/output/${DATASET_NAME}"}
MATCHER=${4:-sequential}   # sequential | exhaustive
KNOWN_POSES=${5:-""}       # optional: dir with cameras.txt + images.txt
DO_BA=${DO_BA:-1}

mkdir -p "$OUT"

# Prevent concurrent COLMAP runs writing to the same output directory.
LOCK_FILE="$OUT/.colmap_run.lock"
exec 8>"$LOCK_FILE"
if ! flock -n 8; then
  echo "[!] Another COLMAP run is already using: $OUT"
  echo "[!] Lock file: $LOCK_FILE"
  exit 1
fi

# Clean potentially stale sqlite files from interrupted previous runs.
rm -f "$OUT/database.db" "$OUT/database.db-shm" "$OUT/database.db-wal" "$OUT/database.db-journal"

CMD_SCRIPT=$(cat <<'INNER'
set -e
export QT_QPA_PLATFORM=offscreen

# Fail fast if output mount is not writable from inside the container.
touch /output/.colmap_write_test && rm -f /output/.colmap_write_test

# Detect CPU-only SIFT flags
if colmap feature_extractor --help 2>&1 | grep -q -- 'SiftExtraction.use_gpu'; then
  CPU_EXTRACT="--SiftExtraction.use_gpu 0"
  CPU_MATCH="--SiftMatching.use_gpu 0"
else
  CPU_EXTRACT=""
  CPU_MATCH=""
fi

rm -f /output/database.db /output/database.db-shm /output/database.db-wal /output/database.db-journal

if [ -n "__KNOWN_POSES__" ] && [ -d "__KNOWN_POSES__" ]; then
  # ── KNOWN POSES: images.txt defines canonical filenames → triangulate only ─
  echo "[i] Known poses provided → point_triangulator"

  # Parse cameras.txt
  while read -r line; do
    [[ "$line" == \#* || -z "$line" ]] && continue
    read -ra f <<< "$line"
    CAM_MODEL="${f[1]}"
    CAM_PARAMS="${f[4]}"
    for ((i=5; i<${#f[@]}; i++)); do CAM_PARAMS="${CAM_PARAMS},${f[i]}"; done
    break
  done < __KNOWN_POSES__/cameras.txt
  echo "[i] Camera: $CAM_MODEL  params: $CAM_PARAMS"

  # Collect canonical names from images.txt (pose lines have exactly 10 fields)
  mapfile -t KP_NAMES < <(awk '!/^#/ && NF==10 {print $NF}' __KNOWN_POSES__/images.txt)
  # Collect actual images in /images sorted
  mapfile -t ACTUAL_IMGS < <(ls /images | sort)
  N_KP=${#KP_NAMES[@]}
  N_ACTUAL=${#ACTUAL_IMGS[@]}
  echo "[i] Known poses: $N_KP images  /images: $N_ACTUAL files"

  # Patch images.txt to use the actual filenames from /images (by sorted position).
  # Keep --image_path /images so COLMAP registers them by their real names.
  # No temp image dir needed — COLMAP resolves symlinks, so only real paths work.
  PATCHED_POSES=/tmp/known_poses_patched
  rm -rf "$PATCHED_POSES"
  mkdir -p "$PATCHED_POSES"
  cp __KNOWN_POSES__/cameras.txt "$PATCHED_POSES/"
  cp __KNOWN_POSES__/points3D.txt "$PATCHED_POSES/" 2>/dev/null || touch "$PATCHED_POSES/points3D.txt"

  # Build a sed script: frame_001.png → 000000.jpg, frame_002.png → 000001.jpg, …
  > /tmp/_name_patch.sed
  for i in "${!KP_NAMES[@]}"; do
    kp_esc="${KP_NAMES[$i]//./\\.}"          # escape dots for sed
    printf 's/ %s$/ %s/\n' "$kp_esc" "${ACTUAL_IMGS[$i]}" >> /tmp/_name_patch.sed
  done
  sed -f /tmp/_name_patch.sed __KNOWN_POSES__/images.txt > "$PATCHED_POSES/images.txt"
  echo "[i] Patched images.txt: ${KP_NAMES[0]} → ${ACTUAL_IMGS[0]}  … (${N_KP} entries)"

  # Masks: standard lookup by actual filename
  MASK_PATH=/tmp/masks_kp
  mkdir -p "$MASK_PATH"
  shopt -s nullglob
  for img in /images/*.{jpg,JPG,jpeg,JPEG,png,PNG}; do
    base="$(basename "$img")"
    stem="${base%.*}"
    for mtry in "/masks/${base}.png" "/masks/${base}" "/masks/${stem}.png"; do
      [ -f "$mtry" ] && ln -sf "$mtry" "$MASK_PATH/${base}.png" && break
    done
  done
  echo "[i] Masks prepared: $(ls -1 "$MASK_PATH" | wc -l)"

  MASK_FLAG=""
  [ "$(ls -A "$MASK_PATH" 2>/dev/null)" ] && MASK_FLAG="--ImageReader.mask_path $MASK_PATH"

  colmap feature_extractor \
    --database_path /output/database.db \
    --image_path /images \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model "$CAM_MODEL" \
    --ImageReader.camera_params "$CAM_PARAMS" \
    --SiftExtraction.max_num_features 50000 \
    --SiftExtraction.peak_threshold 0.002 \
    --SiftExtraction.first_octave -1 \
    $MASK_FLAG \
    $CPU_EXTRACT

  # Exhaustive matching: far better triangulation geometry than sequential
  # (sequential gives tiny baselines between adjacent frames → degenerate triangulation)
  colmap exhaustive_matcher \
    --database_path /output/database.db \
    --SiftMatching.max_ratio 0.95 \
    --SiftMatching.min_num_inliers 5 \
    $CPU_MATCH

  # Remap images.txt IDs to match database image IDs (DB assigns IDs by thread-pool
  # insertion order, not filename order — mismatch causes Reconstruction::Load() crash)
  python3 - <<'PYEOF'
import sqlite3, os
db = '/output/database.db'
txt = '/tmp/known_poses_patched/images.txt'
conn = sqlite3.connect(db)
db_name_to_id = dict(conn.execute("SELECT name, image_id FROM images"))
conn.close()
lines = open(txt).readlines()
out = []
remapped = 0
for line in lines:
    stripped = line.rstrip('\n')
    if stripped.startswith('#') or not stripped.strip():
        out.append(line)
        continue
    parts = stripped.split(' ')
    if len(parts) == 10:
        name = parts[-1]
        if name in db_name_to_id:
            parts[0] = str(db_name_to_id[name])
            remapped += 1
    out.append(' '.join(parts) + '\n')
open(txt, 'w').writelines(out)
print(f"[i] Remapped {remapped} image IDs in images.txt to match database")
PYEOF

  rm -rf /output/sparse/0 && mkdir -p /output/sparse/0
  colmap point_triangulator \
    --database_path /output/database.db \
    --image_path /images \
    --input_path "$PATCHED_POSES" \
    --output_path /output/sparse/0 \
    --Mapper.min_num_matches 4 \
    --Mapper.tri_min_angle 1.0 \
    --Mapper.tri_ignore_two_view_tracks 1 \
    || echo "[!] point_triangulator failed — continuing with camera-only model"

  if [ "__DO_BA__" = "1" ]; then
    echo "[i] Bundle adjustment (intrinsics fixed)..."
    colmap bundle_adjuster \
      --input_path /output/sparse/0 \
      --output_path /output/sparse/0 \
      --BundleAdjustment.refine_focal_length 0 \
      --BundleAdjustment.refine_principal_point 0 \
      --BundleAdjustment.refine_extra_params 0 \
      || echo "[!] bundle_adjuster failed — continuing without BA"
  fi

else
  # ── STANDARD: full reconstruction ─────────────────────────────────────────
  echo "[i] No known poses → full mapper"

  # Prepare masks: COLMAP expects <image_filename>.png
  mkdir -p /tmp/masks_colmap
  shopt -s nullglob
  for img in /images/*.{jpg,JPG,jpeg,JPEG,png,PNG}; do
    base="$(basename "$img")"
    stem="${base%.*}"
    for try in "/masks/${base}.png" "/masks/${base}" "/masks/${stem}.png"; do
      [ -f "$try" ] && ln -sf "$try" "/tmp/masks_colmap/${base}.png" && break
    done
  done
  echo "[i] Masks prepared: $(ls -1 /tmp/masks_colmap | wc -l)"

  colmap feature_extractor \
    --database_path /output/database.db \
    --image_path /images \
    --ImageReader.mask_path /tmp/masks_colmap \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model PINHOLE \
    --SiftExtraction.max_num_features 50000 \
    --SiftExtraction.peak_threshold 0.006 \
    $CPU_EXTRACT

  if [ "__MATCHER__" = "exhaustive" ]; then
    colmap exhaustive_matcher \
      --database_path /output/database.db \
      --SiftMatching.max_ratio 0.9 \
      $CPU_MATCH
  else
    colmap sequential_matcher \
      --database_path /output/database.db \
      --SequentialMatching.loop_detection 1 \
      --SiftMatching.max_ratio 0.9 \
      $CPU_MATCH
  fi

  rm -rf /output/sparse && mkdir -p /output/sparse
  colmap mapper \
    --database_path /output/database.db \
    --image_path /images \
    --output_path /output/sparse \
    --Mapper.min_num_matches 6 \
    --Mapper.init_min_num_inliers 20 \
    --Mapper.abs_pose_min_num_inliers 15

  # Keep only the best model (most registered images) as sparse/0
  best=-1; best_dir=""
  for d in /output/sparse/*; do
    [ -d "$d" ] || continue
    n="$(colmap model_analyzer --path "$d" 2>&1 | awk -F': ' '/Registered images/{print $2}' | tail -1)"
    n="${n:-0}"
    echo "[model] $(basename "$d") registered=$n"
    if [ "$n" -gt "$best" ]; then best="$n"; best_dir="$d"; fi
  done

  if [ -z "$best_dir" ] || [ "$best" -le 0 ]; then
    echo "[error] No valid sparse model found!"
    exit 1
  fi

  tmp="/output/__best_tmp__"
  rm -rf "$tmp"
  mv "$best_dir" "$tmp"
  rm -rf /output/sparse
  mkdir -p /output/sparse
  mv "$tmp" /output/sparse/0
  echo "[ok] Best model: $best registered images → sparse/0"
fi

colmap model_analyzer --path /output/sparse/0 | grep -E "Registered images|Points|Observations" || true
echo "[✓] Done. Output: /output/sparse/0"
INNER
)

# Inject variables into the container script
CMD_SCRIPT="${CMD_SCRIPT//__MATCHER__/$MATCHER}"
CMD_SCRIPT="${CMD_SCRIPT//__DO_BA__/$DO_BA}"
KNOWN_POSES_CONT=""
POSES_VOLUME_FLAG=""
if [[ -n "$KNOWN_POSES" ]]; then
  KNOWN_POSES_CONT="/known_poses"
  POSES_VOLUME_FLAG="-v ${KNOWN_POSES}:/known_poses:ro"
fi
CMD_SCRIPT="${CMD_SCRIPT//__KNOWN_POSES__/$KNOWN_POSES_CONT}"

# Convert JPEG masks → PNG on the host (COLMAP requires PNG-format masks).
# Output: <stem>.<ext>.png  e.g. 000000.jpg → 000000.jpg.png
# The inside-Docker mask lookup tries /masks/<image_base>.png first, which matches.
MASKS_MNT="$MASKS"
if ls "$MASKS"/*.jpg "$MASKS"/*.JPG "$MASKS"/*.jpeg "$MASKS"/*.JPEG 2>/dev/null | head -1 >/dev/null 2>&1; then
  MASKS_MNT="${OUT}/_masks_png"
  mkdir -p "$MASKS_MNT"
  python3 - <<PYEOF
import os, glob
try:
    from PIL import Image
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'Pillow'])
    from PIL import Image
src, dst = "$MASKS", "$MASKS_MNT"
converted = 0
for pattern in ("*.jpg","*.JPG","*.jpeg","*.JPEG"):
    for f in sorted(glob.glob(os.path.join(src, pattern))):
        base = os.path.basename(f)
        out = os.path.join(dst, base + ".png")
        if not os.path.exists(out):
            Image.open(f).convert("L").save(out, "PNG")
            converted += 1
print(f"[i] Converted {converted} masks to PNG → {dst}")
PYEOF
  echo "[i] Masks PNG dir: $MASKS_MNT"
fi

# Run via docker compose, fall back to docker run --gpus all
COMPOSE_FILE="${COMPOSE_FILE:-${nephele_PATH:-.}/docker-compose.yml}"
CF_FLAGS=()
IFS=':' read -ra _CF_PARTS <<< "$COMPOSE_FILE"
_cf_ok=true
for _cf in "${_CF_PARTS[@]}"; do
  if [ -f "$_cf" ]; then
    CF_FLAGS+=(-f "$_cf")
  else
    _cf_ok=false; break
  fi
done
set +e
if $_cf_ok; then
  docker compose "${CF_FLAGS[@]}" run --rm \
    --user "$(id -u):$(id -g)" \
    -e QT_QPA_PLATFORM=offscreen \
    -e HOME=/output \
    -e SQLITE_TMPDIR=/output \
    -e TMPDIR=/output \
    -v "$IMAGES:/images:ro" \
    -v "$MASKS_MNT:/masks:ro" \
    -v "$OUT:/output" \
    $POSES_VOLUME_FLAG \
    colmap bash -lc "$CMD_SCRIPT"
  RC=$?
else
  RC=1
fi
set -e

if [ $RC -ne 0 ]; then
  echo "[!] Compose failed (rc=$RC) — falling back to docker run --gpus all"
  docker run --gpus all --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/output \
    -e SQLITE_TMPDIR=/output \
    -e TMPDIR=/output \
    -v "$IMAGES:/images:ro" \
    -v "$MASKS_MNT:/masks:ro" \
    -v "$OUT:/output" \
    $POSES_VOLUME_FLAG \
    colmap/colmap bash -lc "$CMD_SCRIPT"
fi
