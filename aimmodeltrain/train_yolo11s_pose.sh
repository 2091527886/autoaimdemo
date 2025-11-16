#!/usr/bin/env bash
yolo pose train model=yolo11s-pose.pt data=posetrain/dataset/data.yaml imgsz=640 epochs=900 batch=9 device=0 project=posetrain/runs name=yolo11s-pose
