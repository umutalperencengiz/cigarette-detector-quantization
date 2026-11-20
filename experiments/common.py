"""Shared helpers: letterbox preprocessing, YOLO output decoding and COCO data."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "val" / "images"
LABELS = ROOT / "data" / "val" / "labels"

# The dataset lives outside the repo; point DATA_ROOT at it.
DATA_ROOT = Path(r"C:\Users\umuta\OneDrive\Masaüstü\BitirmeProjesi\BannedReactions\smokerDetector")
VAL_IMAGES = DATA_ROOT / "valid" / "images"
VAL_LABELS = DATA_ROOT / "valid" / "labels"
TRAIN_IMAGES = DATA_ROOT / "train" / "images"

INPUT_MEAN = 114.0


def imread(path: Path) -> np.ndarray:
    """cv2.imread that survives non-ASCII Windows paths."""
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not decode {path}")
    return image


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, dict]:
    """Resize with preserved aspect ratio, pad with 114, return RGB CHW float32."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    dx, dy = (size - new_w) // 2, (size - new_h) // 2

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), INPUT_MEAN, dtype=np.uint8)
    canvas[dy : dy + new_h, dx : dx + new_w] = resized

    rgb = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    meta = {"scale": scale, "dx": dx, "dy": dy, "width": width, "height": height, "size": size}
    return np.ascontiguousarray(rgb[None]), meta


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def decode(output: np.ndarray, meta: dict, conf: float, iou: float) -> tuple[np.ndarray, np.ndarray]:
    """[1, 4+classes, anchors] -> boxes in original image pixels + scores."""
    prediction = output[0]
    anchors = prediction.shape[1]
    channels = prediction.shape[0]

    scores = prediction[4:].max(axis=0) if channels > 5 else prediction[4]
    mask = scores >= conf
    if not mask.any():
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

    xywh = prediction[:4, mask].T
    scores = scores[mask]

    scale, dx, dy = meta["scale"], meta["dx"], meta["dy"]
    boxes = np.empty_like(xywh)
    boxes[:, 0] = (xywh[:, 0] - xywh[:, 2] / 2 - dx) / scale
    boxes[:, 1] = (xywh[:, 1] - xywh[:, 3] / 2 - dy) / scale
    boxes[:, 2] = (xywh[:, 0] + xywh[:, 2] / 2 - dx) / scale
    boxes[:, 3] = (xywh[:, 1] + xywh[:, 3] / 2 - dy) / scale

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, meta["width"])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, meta["height"])

    keep = nms(boxes, scores, iou)
    return boxes[keep], scores[keep]


def load_ground_truth(image_paths: list[Path]) -> tuple[list[dict], list[dict]]:
    images, annotations = [], []
    ann_id = 1
    for image_id, path in enumerate(image_paths, start=1):
        image = imread(path)
        height, width = image.shape[:2]
        images.append({"id": image_id, "file_name": path.name, "width": width, "height": height})

        label_path = VAL_LABELS / f"{path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = (float(v) for v in parts[:5])
            x, y = (cx - bw / 2) * width, (cy - bh / 2) * height
            w, h = bw * width, bh * height
            if w <= 0 or h <= 0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return images, annotations


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
