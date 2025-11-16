#!/usr/bin/env bash
# 使用 ffmpeg 将输入文件夹中的图片居中裁剪为 640x640，并输出到目标文件夹
# 用法：
#   bash crop_center_640.sh <输入文件夹> <输出文件夹>
# 说明：
# - 支持的图片扩展名：png, jpg, jpeg, bmp, webp, tif, tiff
# - 若图片尺寸小于 640x640，会先在中心进行填充（pad），再裁剪，保证输出为 640x640
# - 文件名与扩展名保持不变（目录结构扁平化到输出文件夹）

set -euo pipefail

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "错误：未检测到 ffmpeg，请先安装 ffmpeg。" >&2
  exit 1
fi

if [ $# -ne 2 ]; then
  echo "用法：bash $0 <输入文件夹> <输出文件夹>" >&2
  exit 1
fi

IN_DIR="$1"
OUT_DIR="$2"

if [ ! -d "$IN_DIR" ]; then
  echo "错误：输入文件夹不存在：$IN_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# 处理函数（单文件）
process_one() {
  local in_file="$1"
  local base
  base=$(basename "$in_file")
  local out_file="$OUT_DIR/$base"

  # ffmpeg 过滤器说明：
  # 1) pad 将较小边补到至少 640，使得后续 crop 始终可用；位置在中心 (ow-iw)/2, (oh-ih)/2
  # 2) crop 640x640，居中 (iw-640)/2, (ih-640)/2
  # 这样对任何尺寸的图片都能输出标准 640x640
  ffmpeg -hide_banner -loglevel error -y -i "$in_file" \
    -vf "pad='max(iw,640)':'max(ih,640)':'(ow-iw)/2':'(oh-ih)/2',crop=640:640:'(iw-640)/2':'(ih-640)/2'" \
    "$out_file"

  echo "已生成：$out_file"
}

# 遍历输入文件夹（含子目录）
# 使用 null 分隔以处理空格文件名
find "$IN_DIR" -type f \( \
  -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o \
  -iname "*.bmp" -o -iname "*.webp" -o -iname "*.tif" -o -iname "*.tiff" \
\) -print0 | while IFS= read -r -d '' f; do
  process_one "$f"
done

echo "完成：输出目录 -> $OUT_DIR"
