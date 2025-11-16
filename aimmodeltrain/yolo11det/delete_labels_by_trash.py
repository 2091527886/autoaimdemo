#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 trash 目录中的图片文件名，删除 imagestest/label 下对应的 .txt 标签

用法示例：
  预演（不删除）：
    python3 delete_labels_by_trash.py -n
  实际删除：
    python3 delete_labels_by_trash.py
  指定目录/扩展名：
    python3 delete_labels_by_trash.py --trash-dir trash --label-dir imagestest/label \
        --image-exts .jpg .jpeg .png
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Iterable


def iter_files(dir_path: Path) -> Iterable[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    # 仅遍历第一层（不递归）
    return (p for p in dir_path.iterdir() if p.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="根据 trash 中图片同名删除 label 下 .txt 文件",
    )
    parser.add_argument("--trash-dir", default="trash", help="图片所在目录，默认 trash")
    parser.add_argument("--label-dir", default="imagestest/label", help="标签目录，默认 imagestest/label")
    parser.add_argument(
        "--image-exts",
        nargs="*",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"],
        help="要匹配的图片扩展名（含点），默认常见格式",
    )
    parser.add_argument("-n", "--dry-run", action="store_true", help="预演，不实际删除")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默，仅输出汇总")

    args = parser.parse_args(argv)

    trash_dir = Path(args.trash_dir)
    label_dir = Path(args.label_dir)
    image_exts = {e.lower() for e in args.image_exts}

    if not trash_dir.is_dir():
        print(f"错误：未找到目录 {trash_dir}", file=sys.stderr)
        return 1
    if not label_dir.is_dir():
        print(f"错误：未找到目录 {label_dir}", file=sys.stderr)
        return 1

    candidates = 0
    removed = 0
    missing = 0

    for f in iter_files(trash_dir):
        # 仅处理指定扩展名
        if f.suffix.lower() not in image_exts:
            continue
        candidates += 1
        stem = f.stem
        label_file = label_dir / f"{stem}.txt"
        if label_file.exists() and label_file.is_file():
            if args.dry_run:
                if not args.quiet:
                    print(f"[DRY-RUN] 将删除：{label_file}")
            else:
                try:
                    label_file.unlink()
                    if not args.quiet:
                        print(f"已删除：{label_file}")
                except Exception as e:
                    print(f"删除失败：{label_file}，错误：{e}", file=sys.stderr)
                    continue
            removed += 1
        else:
            missing += 1

    if args.dry_run:
        print(f"预演完成：候选图片数={candidates}，将删除的标签数={removed}，未找到对应标签={missing}")
    else:
        print(f"完成：候选图片数={candidates}，已删除标签数={removed}，未找到对应标签={missing}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
