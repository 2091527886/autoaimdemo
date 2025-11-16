#!/usr/bin/env bash
yolo task=detect mode=train model=yolo11n.pt data=dataset/data.yaml epochs=1000 imgsz=640 batch=12 device=0 project=runs name=yolo11n_exp
