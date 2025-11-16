#!/usr/bin/env python3
"""
Offline pre-annotation for pose using Ultralytics YOLOv11s-Pose and export to COCO Keypoints 1.0

Usage example:
  python tools/offline_preannotate_yolo11s_pose_to_coco.py \
      --images-dir path/to/images \
      --model yolo11s-pose.pt \
      --output coco_keypoints.json \
      --conf 0.35 --kpt-thr 0.5

Notes:
- Keypoints and skeleton are fixed in the script (pfperson, 17 points, custom skeleton).
- Resulting JSON can be imported into CVAT: Create a task -> Import -> COCO Keypoints 1.0.
- For better reproducibility, pin ultralytics and torch versions in your environment.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from ultralytics import YOLO
except Exception as e:
    print("[ERROR] Failed to import ultralytics. Please install: pip install ultralytics", file=sys.stderr)
    raise


# Default COCO 17 keypoints and skeleton edges
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Fixed keypoints order (from pose.conf) hard-coded as requested
FIXED_KEYPOINTS = [
    "Nose","LeftEye","RightEye","LeftEar","RightEar",
    "LeftShoulder","RightShoulder","LeftElbow","RightElbow",
    "LeftWrist","RightWrist","LeftHip","RightHip",
    "LeftKnee","RightKnee","LeftAnkle","RightAnkle",
]

# COCO skeleton connections (1-indexed indices per COCO spec). We'll convert to 0-index internally if needed.
COCO_SKELETON = [[1,2],[2,4],[1,3],[3,5],[6,7],[6,8],[8,10],[7,9],[9,11],[6,12],[12,14],[14,16],[7,13],[13,15],[15,17],[12,13]]


def list_images(images_dir: Path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")) -> List[Path]:
    files = []
    for root, _, fnames in os.walk(images_dir):
        for f in fnames:
            if f.lower().endswith(exts):
                files.append(Path(root) / f)
    files.sort()
    return files


def coco_image_entry(img_id: int, file_path: Path) -> Tuple[dict, Tuple[int, int]]:
    img = cv2.imread(str(file_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {file_path}")
    h, w = img.shape[:2]
    return {
        "id": img_id,
        "file_name": os.path.relpath(str(file_path), start=str(file_path.parents[0])),
        "height": h,
        "width": w,
    }, (w, h)


def keypoints_xyv_from_ultralytics(person_xy: np.ndarray, person_conf: np.ndarray, kpt_thr: float, kpt_names: List[str]) -> Tuple[List[float], int]:
    """
    Convert Ultralytics keypoints to COCO format [x1, y1, v1, ..., xk, yk, vk]
    v = 0 (not labeled) if conf < thr, else 2 (labeled & visible). Count visible.
    """
    k = len(kpt_names)
    if person_xy.shape[0] < k:
        # pad with zeros if model outputs fewer kpts than expected
        pad = np.zeros((k - person_xy.shape[0], 2), dtype=person_xy.dtype)
        person_xy = np.vstack([person_xy, pad])
        if person_conf is not None:
            padc = np.zeros((k - person_conf.shape[0],), dtype=person_conf.dtype)
            person_conf = np.concatenate([person_conf, padc])

    keypoints = []
    visible_count = 0
    for i in range(k):
        x = float(person_xy[i, 0])
        y = float(person_xy[i, 1])
        c = float(person_conf[i]) if person_conf is not None else 1.0
        v = 2 if c >= kpt_thr else 0
        visible_count += 1 if v == 2 else 0
        keypoints.extend([x, y, v])
    return keypoints, visible_count


def to_coco(
    images: List[Path],
    model_path: Path,
    output_json: Path,
    conf: float,
    kpt_thr: float,
    category_name: str,
    dataset_name: str,
    keypoints_names: List[str],
    skeleton_edges: List[List[int]],
):
    model = YOLO(str(model_path))

    coco = {
        "info": {
            "description": dataset_name,
            "version": "1.0",
            "year": 2025,
            "contributor": "offline_preannotate_yolo11s_pose_to_coco.py",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": category_name,
                "supercategory": category_name,
                "keypoints": keypoints_names,
                "skeleton": skeleton_edges,
            }
        ],
    }

    ann_id = 1
    img_id = 1

    # Cache image sizes to avoid re-reading
    img_sizes = {}

    for img_path in tqdm(images, desc="Inferencing"):
        image_entry, (w, h) = coco_image_entry(img_id, img_path)
        coco["images"].append(image_entry)
        img_sizes[img_id] = (w, h)

        results = model.predict(str(img_path), conf=conf, verbose=False)
        for r in results:
            if r.keypoints is None:
                continue
            boxes_xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else None
            boxes_conf = r.boxes.conf.cpu().numpy().tolist() if (r.boxes is not None and r.boxes.conf is not None) else None
            kpts_xy = r.keypoints.xy
            kpts_conf = r.keypoints.conf

            for i, person_xy in enumerate(kpts_xy):
                p_xy = person_xy.cpu().numpy()  # (K,2)
                p_conf = kpts_conf[i].cpu().numpy() if kpts_conf is not None else None
                keypoints, visible_count = keypoints_xyv_from_ultralytics(p_xy, p_conf, kpt_thr, keypoints_names)

                # bbox from boxes
                if boxes_xyxy is not None and i < len(boxes_xyxy):
                    x1, y1, x2, y2 = boxes_xyxy[i].tolist()
                    x, y = float(max(0, x1)), float(max(0, y1))
                    w_box, h_box = float(max(0, x2 - x1)), float(max(0, y2 - y1))
                else:
                    # fallback: bbox from keypoints extents
                    xs = keypoints[0::3]
                    ys = keypoints[1::3]
                    xs_vis = [xs[j] for j in range(len(xs)) if keypoints[2 + 3*j] == 2]
                    ys_vis = [ys[j] for j in range(len(ys)) if keypoints[2 + 3*j] == 2]
                    if xs_vis and ys_vis:
                        x, y = float(min(xs_vis)), float(min(ys_vis))
                        w_box = float(max(xs_vis) - x)
                        h_box = float(max(ys_vis) - y)
                    else:
                        # no visible keypoints, skip annotation
                        continue

                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "keypoints": keypoints,
                    "num_keypoints": int(visible_count),
                    "bbox": [x, y, w_box, h_box],
                    "area": float(w_box * h_box),
                    "iscrowd": 0,
                    "score": float(boxes_conf[i]) if boxes_conf and i < len(boxes_conf) else 1.0,
                }
                coco["annotations"].append(ann)
                ann_id += 1

        img_id += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(f"[OK] COCO keypoints saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Offline pre-annotation to COCO Keypoints 1.0 (fixed schema) using YOLOv11s Pose")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory of images to annotate")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to Ultralytics YOLO pose model (.pt). Default: best.pt in repo root")
    parser.add_argument("--output", type=Path, required=True, help="Output COCO JSON path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold for persons")
    parser.add_argument("--kpt-thr", type=float, default=0.50, help="Keypoint visibility threshold")
    parser.add_argument("--category-name", type=str, default="pfperson", help="COCO category name")
    parser.add_argument("--dataset-name", type=str, default="YOLOv11s Pose Pre-Annotations", help="COCO info.description")
    
    args = parser.parse_args()

    if not args.images_dir.exists():
        raise FileNotFoundError(f"images-dir not found: {args.images_dir}")

    images = list_images(args.images_dir)
    if not images:
        raise RuntimeError(f"No images found in: {args.images_dir}")

    # Hard-coded schema (as requested)
    category_name = args.category_name or "pfperson"
    keypoints = FIXED_KEYPOINTS
    skeleton = COCO_SKELETON  # fixed skeleton; replace here if you want a custom list

    # Use model path as-is, no fallbacks
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Place best.pt in repo root or pass --model explicitly.")

    to_coco(
        images=images,
        model_path=model_path,
        output_json=args.output,
        conf=args.conf,
        kpt_thr=args.kpt_thr,
        category_name=category_name,
        dataset_name=args.dataset_name,
        keypoints_names=keypoints,
        skeleton_edges=skeleton,
    )


if __name__ == "__main__":
    main()
