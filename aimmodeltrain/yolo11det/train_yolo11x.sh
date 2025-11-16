#!/usr/bin/env bash
yolo task=detect mode=train model=yolo11x.pt data=dataset/data.yaml epochs=800 imgsz=640 batch=9 device=0 project=runs name=yolo11x_exp
