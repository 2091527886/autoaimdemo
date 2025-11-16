#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A minimal yet solid training script for Ultralytics YOLOv11s.
- Fixed random seed for reproducibility
- Configurable hyperparameters via CLI args
- Trains and validates on the given data.yaml
- Prints the path to best.pt when finished

Usage examples:
  python train_yolo11s.py --data data.yaml --epochs 100 --imgsz 640 --device 0
  python train_yolo11s.py --data data.yaml --device cpu

Requires:
  pip install -U ultralytics
  # Optional: install a matching torch/torchvision for your CUDA
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:
    torch = None

from ultralytics import YOLO


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv11s with Ultralytics")
    p.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    p.add_argument("--model", type=str, default="yolo11s.pt", help="Model weights or cfg")
    p.add_argument("--epochs", type=int, default=100, help="Training epochs")
    p.add_argument("--imgsz", type=int, default=640, help="Training image size")
    p.add_argument("--batch", type=int, default=-1, help="Batch size (-1 for auto)")
    p.add_argument("--device", type=str, default="0", help="CUDA device id like '0', or 'cpu'")
    p.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    p.add_argument("--project", type=str, default="runs", help="Project dir for outputs")
    p.add_argument("--name", type=str, default="yolo11s_exp", help="Experiment name")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--save_dir", type=str, default=None, help="Override save_dir (project/name)")
    p.add_argument("--exist_ok", action="store_true", help="Overwrite existing experiment dir")
    p.add_argument("--patience", type=int, default=50, help="Early stopping patience (epochs)")
    p.add_argument("--resume", action="store_true", help="Resume training if checkpoints exist")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Resolve output directory
    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        project = str(save_dir.parent) if save_dir.parent.as_posix() != "." else "runs"
        name = save_dir.name
    else:
        project = args.project
        name = args.name

    # Load model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=project,
        name=name,
        exist_ok=args.exist_ok,
        patience=args.patience,
        resume=args.resume,
        verbose=True,
    )

    # Validate on val split
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=project,
        name=name,
        split="val",
        verbose=True,
    )

    # Locate best.pt
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(project) / "detect" / name
    best_path = save_dir / "weights" / "best.pt"

    print("\nTraining finished.")
    print(f"Best weights: {best_path if best_path.exists() else 'not found (check runs)'}")
    print(f"Artifacts saved to: {save_dir}")


if __name__ == "__main__":
    sys.exit(main())
