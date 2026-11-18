import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "autolabel"
MODEL_PATH = ROOT / "best.pt"
LABEL_CONFIG = ROOT / "raw.txt"
OUTPUT_DIR = ROOT / "labelout"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
KEYPOINT_CONFIDENCE = 0.5
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def load_labels():
    with LABEL_CONFIG.open(encoding="utf-8") as file:
        labels = json.load(file)
    if not labels or any(label.get("type") != "skeleton" for label in labels):
        raise ValueError("raw.txt must contain skeleton labels")
    names = [label["name"] for label in labels]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate class names in raw.txt")
    for label in labels:
        points = tuple(point["name"] for point in label.get("sublabels", []))
        if points != KEYPOINT_NAMES:
            raise ValueError("Keypoint order in raw.txt does not match the expected pose model")
    return names


def find_images():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")
    images = sorted(p for p in IMAGE_DIR.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No supported images in {IMAGE_DIR}")
    label_paths = set()
    for image in images:
        label_path = image.relative_to(IMAGE_DIR).with_suffix(".txt").as_posix().casefold()
        if label_path in label_paths:
            raise ValueError(f"Multiple images map to the same label file: {label_path}")
        label_paths.add(label_path)
    return images


def write_result(result, class_ids, label_file):
    boxes, keypoints = result.boxes, result.keypoints
    if boxes is None or len(boxes) == 0:
        label_file.write_text("", encoding="utf-8")
        return 0
    if keypoints is None or len(keypoints) != len(boxes):
        raise ValueError(f"Missing pose keypoints: {result.path}")
    lines = []
    for box, points in zip(boxes, keypoints):
        model_id = int(box.cls.item())
        if model_id not in class_ids:
            raise ValueError(f"Unknown model class ID: {model_id}")
        coords, confidence = points.xyn[0].cpu().tolist(), points.conf
        if len(coords) != len(KEYPOINT_NAMES) or confidence is None:
            raise ValueError(f"Invalid keypoint data: {result.path}")
        values = [str(class_ids[model_id])]
        values.extend(f"{value:.6f}" for value in box.xywhn[0].cpu().tolist())
        for (x, y), conf in zip(coords, confidence[0].cpu().tolist()):
            if conf < KEYPOINT_CONFIDENCE or (x == 0 and y == 0):
                values.extend(("0", "0", "0"))
            else:
                values.extend((f"{x:.6f}", f"{y:.6f}", "2"))
        lines.append(" ".join(values))
    label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output already exists; move or remove it first: {OUTPUT_DIR}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    images = find_images()
    names = load_labels()
    model = YOLO(str(MODEL_PATH))
    if model.task != "pose" or list(getattr(model.model, "kpt_shape", [])) != [len(KEYPOINT_NAMES), 3]:
        raise ValueError("best.pt must be a pose model with 17 three-channel keypoints")
    model_names = model.names
    if len(model_names) != len(names) or set(model_names.values()) != set(names):
        raise ValueError(f"Model classes {model_names} do not match raw.txt classes {names}")
    class_ids = {model_id: names.index(name) for model_id, name in model_names.items()}
    with tempfile.TemporaryDirectory(prefix=".autolabel-", dir=ROOT) as temp:
        dataset = Path(temp) / "labelout"
        image_root, label_root = dataset / "images" / "train", dataset / "labels" / "train"
        image_root.mkdir(parents=True)
        label_root.mkdir(parents=True)
        image_list, detection_count = [], 0
        for index, image in enumerate(images, 1):
            relative = image.relative_to(IMAGE_DIR)
            target_image, target_label = image_root / relative, (label_root / relative).with_suffix(".txt")
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            result = model.predict(source=str(image), verbose=False)[0]
            detection_count += write_result(result, class_ids, target_label)
            shutil.copy2(image, target_image)
            image_list.append((Path("images") / "train" / relative).as_posix())
            print(f"[{index}/{len(images)}] {relative}")
        (dataset / "train.txt").write_text("\n".join(image_list) + "\n", encoding="utf-8")
        config = {"path": "./", "train": "train.txt", "kpt_shape": [len(KEYPOINT_NAMES), 3], "names": dict(enumerate(names))}
        (dataset / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        archive = dataset / "labelout.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for path in sorted(dataset.rglob("*")):
                if path.is_file() and path != archive:
                    output.write(path, path.relative_to(dataset).as_posix())
        dataset.rename(OUTPUT_DIR)
    print(f"Done: {len(images)} images, {detection_count} detections")
    print(f"CVAT import: {OUTPUT_DIR / archive.name}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Autolabel failed: {error}", file=sys.stderr)
        sys.exit(1)
