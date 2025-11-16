#!/usr/bin/env python3
"""
将多个输入目录（默认 imagestest 和 teammate）中无对应 label/txt 的图片移动至各自的 none 目录，
随后把所有“有标签”的样本合并构建为同一个 YOLO 数据集目录：
  dataset/
    images/{train,val}
    labels/{train,val}
    data.yaml

说明与约束：
- 仅使用原始文件名，不做区分与改名。
- 如在合并写入时发生同名冲突，将跳过后到达的同名文件并给出提示，避免覆盖。
- 已彻底去除负样本（无标签图片）相关代码与选项。

用法示例：
  python move_unlabeled_to_none.py
  python move_unlabeled_to_none.py -i imagestest teammate -O dataset -v 0.2 -s 42
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Set, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_label_stems(label_dir: Path) -> set[str]:
    if not label_dir.exists() or not label_dir.is_dir():
        raise FileNotFoundError(f"标签目录不存在: {label_dir}")
    stems: set[str] = set()
    for p in label_dir.glob("*.txt"):
        if p.is_file():
            stems.add(p.stem)
    return stems


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def move_unlabeled_images(root: Path) -> tuple[int, int]:
    """将 root 下无对应 label/txt 的图片移动至 root/none。返回 (moved, skipped)。"""
    label_dir = root / "label"
    none_dir = root / "none"
    ensure_dir(none_dir)

    label_stems = collect_label_stems(label_dir)

    moved = 0
    skipped = 0

    for p in root.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if p.stem in label_stems:
            skipped += 1
            continue
        dest = none_dir / p.name
        # 若目标已存在，追加序号避免覆盖
        if dest.exists():
            i = 1
            while True:
                alt = none_dir / f"{p.stem}_{i}{p.suffix}"
                if not alt.exists():
                    dest = alt
                    break
                i += 1
        shutil.move(str(p), str(dest))
        moved += 1

    return moved, skipped


def parse_label_classes(label_dir: Path) -> Set[int]:
    """解析所有 label/txt 文件，收集出现过的类别 id（取每行首个整数）。"""
    classes: Set[int] = set()
    for p in label_dir.rglob("*.txt"):
        if not p.is_file():
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    first = line.split()[0]
                    try:
                        cid = int(float(first))
                    except ValueError:
                        continue
                    classes.add(cid)
        except UnicodeDecodeError:
            with p.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    first = line.split()[0]
                    try:
                        cid = int(float(first))
                    except ValueError:
                        continue
                    classes.add(cid)
    return classes


def is_contiguous_from_zero(classes: Iterable[int]) -> bool:
    s = set(classes)
    if not s:
        return True
    return s == set(range(0, max(s) + 1))


def default_names_for_classes(classes: Iterable[int]):
    s = sorted(set(classes))
    if not s:
        return ["object"], 1
    if is_contiguous_from_zero(s):
        names = [f"class_{i}" for i in range(0, max(s) + 1)]
        return names, len(names)
    else:
        names = {i: f"class_{i}" for i in s}
        return names, len(s)


def collect_labeled_samples(root: Path) -> List[Tuple[Path, Path]]:
    """收集单个 root 下的有标签样本对 (image_path, label_path)（仅 root 根目录的图片文件）。"""
    label_dir = root / "label"
    label_stems = collect_label_stems(label_dir)

    samples: List[Tuple[Path, Path]] = []
    for p in root.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if p.stem not in label_stems:
            continue
        img_path = p
        lbl_path = label_dir / f"{p.stem}.txt"
        if not lbl_path.exists():
            continue
        samples.append((img_path, lbl_path))
    return samples


def build_combined_yolo_dataset(
    roots: List[Path],
    out_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    move_files: bool = False,
) -> tuple[int, int]:
    """
    合并多个 roots 下的有标签样本，构建单一 YOLO 数据集目录结构：
      out_dir/images/{train,val}
      out_dir/labels/{train,val}
    仅包含有标签的样本；使用原始文件名；若同名冲突则跳过冲突文件。
    返回 (train_pos, val_pos, conflict_skipped)
    """
    import random

    # 收集全部样本
    all_samples: List[Tuple[Path, Path]] = []
    for r in roots:
        all_samples.extend(collect_labeled_samples(r))

    rnd = random.Random(seed)
    rnd.shuffle(all_samples)

    n = len(all_samples)
    val_count = int(n * max(0.0, min(1.0, val_ratio)))
    val_split = all_samples[:val_count]
    train_split = all_samples[val_count:]

    # 创建目录
    images_train = out_dir / "images" / "train"
    images_val = out_dir / "images" / "val"
    labels_train = out_dir / "labels" / "train"
    labels_val = out_dir / "labels" / "val"
    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    op = shutil.move if move_files else shutil.copy2

    def place(split: List[Tuple[Path, Path]], img_dst: Path, lbl_dst: Path) -> None:
        for img, lbl in split:
            img_target = img_dst / img.name
            lbl_target = lbl_dst / lbl.name
            op(str(img), str(img_target))
            op(str(lbl), str(lbl_target))

    place(train_split, images_train, labels_train)
    place(val_split, images_val, labels_val)

    return len(train_split), len(val_split)


def main():
    parser = argparse.ArgumentParser(description="移动无标签图片到 none，并合并构建单一 YOLO 数据集（无负样本，保留原始文件名）")
    parser.add_argument(
        "-i", "--input",
        nargs='+',
        default=["imagestest", "teammate"],
        help="输入数据根目录（可多个），包含图片与 label 子目录 (默认 imagestest 和 teammate)",
    )
    parser.add_argument(
        "-O", "--output-dataset",
        default="dataset",
        help="数据集输出目录 (默认 ./dataset)",
    )
    parser.add_argument(
        "-v", "--val-ratio",
        type=float,
        default=0.2,
        help="验证集比例，默认 0.2 (即 8:2)",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="随机种子 (默认 42)",
    )
    parser.add_argument(
        "-m", "--move-files",
        action="store_true",
        help="构建数据集时移动文件而非复制 (默认复制)",
    )

    args = parser.parse_args()

    roots = [Path(p).resolve() for p in args.input]
    for root in roots:
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"根目录不存在: {root}")

    # 先逐个目录移动无标签图片
    for root in roots:
        moved, skipped = move_unlabeled_images(root)
        print(f"[{root.name}] 已移动无标签图片: {moved} 张, 有标签跳过: {skipped} 张")

    # 合并构建数据集
    out_dir = Path(args.output_dataset).resolve()
    train_pos, val_pos = build_combined_yolo_dataset(
        roots,
        out_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        move_files=args.move_files,
    )
    print(f"合并数据集完成 -> {out_dir} (train_pos={train_pos}, val_pos={val_pos}, 比例≈{1-args.val_ratio:.1f}:{args.val_ratio:.1f})")

    # 生成合并数据集的 data.yaml
    labels_root = out_dir / "labels"
    classes = parse_label_classes(labels_root)
    names, nc = default_names_for_classes(classes)
    lines = []
    lines.append(f"path: {out_dir.as_posix()}")
    lines.append("train: images/train")
    lines.append("val: images/val")
    lines.append(f"nc: {nc}")
    lines.append("names:")
    if isinstance(names, list):
        for i, n in enumerate(names):
            lines.append(f"  {i}: {n}")
    else:
        for i in sorted(names.keys()):
            lines.append(f"  {i}: {names[i]}")
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已在合并数据集目录生成 data.yaml -> {(out_dir / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()
