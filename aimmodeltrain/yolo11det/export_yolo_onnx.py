#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅导出 YOLO 为 ONNX（不做量化）。

用法示例：
  python export_yolo_onnx.py -i best.pt --imgsz 640 --opset 13

依赖：
  pip install ultralytics onnx opencv-python
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Export YOLO to ONNX (no quantization)")
    p.add_argument("-i", "--input", type=str, default="best.pt", help="输入权重路径（.pt 或 .onnx），默认 best.pt")
    p.add_argument("--imgsz", type=int, default=640, help="导出的输入尺寸，默认 640")
    p.add_argument("--opset", type=int, default=13, help="ONNX opset，默认 13")
    # 与原脚本参数保持兼容：默认开启端到端 NMS，传 --nms 则关闭
    p.add_argument("--nms", action="store_false", help="关闭端到端 NMS（默认开启）")
    p.add_argument("--project", type=str, default="runs/export", help="导出目录（与 Ultralytics 保持一致）")
    p.add_argument("--name", type=str, default="onnx_e2e", help="导出子目录名称")
    return p.parse_args()


def export_onnx_if_needed(inp: str, imgsz: int, opset: int, nms: bool, project: str, name: str) -> Path:
    p = Path(inp)
    if p.suffix.lower() == ".onnx":
        print(f"[INFO] Using existing ONNX: {p}")
        return p
    print(f"[1/1] Exporting ONNX from {inp} ...")
    model = YOLO(inp)
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        batch=1,
        opset=opset,
        simplify=True,
        dynamic=False,
        nms=True if nms is None else nms,
        project=project,
        name=name,
    )
    cand = []
    if isinstance(exported, (list, tuple)):
        cand.extend(map(str, exported))
    elif isinstance(exported, (str, Path)):
        cand.append(str(exported))
    # 常见默认输出
    cand.append("best.onnx")
    cand.append(str(Path(project) / name / "best.onnx"))
    for c in cand:
        if c and Path(c).is_file():
            print(f"[OK] ONNX exported: {c}")
            return Path(c)
    raise SystemExit("[ERR] Failed to locate exported ONNX.")


def main():
    args = parse_args()

    onnx_path = export_onnx_if_needed(
        inp=args.input,
        imgsz=args.imgsz,
        opset=args.opset,
        nms=args.nms,
        project=args.project,
        name=args.name,
    )

    print("\nDone.")
    print(f"ONNX: {onnx_path}")


if __name__ == "__main__":
    sys.exit(main())
