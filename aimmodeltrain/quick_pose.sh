#!/usr/bin/env bash
set -e
# One-liner style runner: minimal args, smart defaults
# Usage: bash quick_pose.sh <images_dir> [output_json]

if [ $# -lt 1 ]; then
  echo "Usage: bash quick_pose.sh <images_dir> [output_json]" >&2
  exit 1
fi

IMAGES_DIR="$1"
OUTPUT_JSON="${2:-prelabel_pose_conf.json}"

python3 offline_preannotate_yolo11s_pose_to_coco.py \
  --images-dir "$IMAGES_DIR" \
  --output "$OUTPUT_JSON"
