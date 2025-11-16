#!/usr/bin/env bash
yolo task=detect mode=train model=yolo11s.pt data=data.yaml epochs=100 imgsz=640 batch=3 device=0 project=runs name=yolo11s_exp