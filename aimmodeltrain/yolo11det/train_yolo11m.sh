#!/usr/bin/env bash
yolo task=detect mode=train model=yolo11m.pt data=dataset/data.yaml epochs=500 imgsz=640 batch=9 device=0 project=runs name=yolo11m_exp
