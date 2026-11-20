"""Evaluate one ONNX variant on the full 1,370-image validation split.

    python experiments/evaluate_coco.py --model models/yolo11n_fp32_640.onnx --size 640

Metrics come from pycocotools (so APs/APm/APl are available), latency is measured
around the ONNX session only, and the ground truth is cached in results/val_gt.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common import VAL_IMAGES, decode, imread, letterbox, load_ground_truth, write_json

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GT_CACHE = RESULTS / "val_gt.json"

CONF = 0.001
NMS_IOU = 0.7
MAX_DETECTIONS = 100
AP_NAMES = ["ap", "ap50", "ap75", "aps", "apm", "apl", "ar1", "ar10", "ar100", "ars", "arm", "arl"]


def val_images() -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in VAL_IMAGES.iterdir() if p.suffix.lower() in suffixes)


def ground_truth(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    if GT_CACHE.exists():
        cached = json.loads(GT_CACHE.read_text(encoding="utf-8"))
        return cached["images"], cached["annotations"]

    images, annotations = load_ground_truth(paths)
    write_json(GT_CACHE, {"images": images, "annotations": annotations})
    return images, annotations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--threads", type=int, default=0, help="0 = ORT default (all cores)")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N images")
    parser.add_argument("--out", default="")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    model_path = (ROOT / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    paths = val_images()
    if args.limit:
        paths = paths[: args.limit]

    images, annotations = ground_truth(val_images())
    by_name = {image["file_name"]: image["id"] for image in images}

    options = ort.SessionOptions()
    if args.threads:
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = args.threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    warmup, _ = letterbox(imread(paths[0]), args.size)
    for _ in range(5):
        session.run(None, {input_name: warmup})

    results: list[dict] = []
    latencies: list[float] = []
    pipelines: list[float] = []

    for index, path in enumerate(paths, start=1):
        pipeline_started = time.perf_counter()
        image = imread(path)
        tensor, meta = letterbox(image, args.size)

        started = time.perf_counter()
        output = session.run(None, {input_name: tensor})[0]
        latencies.append((time.perf_counter() - started) * 1000)

        boxes, scores = decode(output, meta, CONF, NMS_IOU)
        pipelines.append((time.perf_counter() - pipeline_started) * 1000)
        if len(boxes) > MAX_DETECTIONS:
            order = np.argsort(-scores)[:MAX_DETECTIONS]
            boxes, scores = boxes[order], scores[order]

        image_id = by_name[path.name]
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            width, height = float(x2 - x1), float(y2 - y1)
            if width <= 0 or height <= 0:
                continue
            results.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), width, height],
                    "score": float(score),
                }
            )

        if index % 200 == 0:
            print(f"  {index}/{len(paths)} images · {np.mean(latencies):.1f} ms/img", flush=True)

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "cigarette"}],
    }
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(results) if results else COCO()
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = [by_name[p.name] for p in paths]
    evaluator.params.maxDets = [1, 10, MAX_DETECTIONS]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = {name: (None if np.isnan(v) else round(float(v) * 100, 3)) for name, v in zip(AP_NAMES, evaluator.stats)}
    latency = np.asarray(latencies)
    pipeline = np.asarray(pipelines)

    payload = {
        "variant": args.tag or model_path.stem,
        "model": model_path.name,
        "model_mb": round(model_path.stat().st_size / 1e6, 2),
        "size": args.size,
        "threads": args.threads or "default",
        "conf": CONF,
        "nms_iou": NMS_IOU,
        "images": len(paths),
        "instances": len(annotations),
        "predictions": len(results),
        "metrics": stats,
        "latency_ms": {
            "mean": round(float(latency.mean()), 2),
            "p50": round(float(np.percentile(latency, 50)), 2),
            "p95": round(float(np.percentile(latency, 95)), 2),
            "min": round(float(latency.min()), 2),
        },
        "pipeline_ms": {
            "mean": round(float(pipeline.mean()), 2),
            "p50": round(float(np.percentile(pipeline, 50)), 2),
            "p95": round(float(np.percentile(pipeline, 95)), 2),
        },
        "throughput_fps": round(1000 / float(latency.mean()), 2),
        "pipeline_fps": round(1000 / float(pipeline.mean()), 2),
        "hardware": {
            "cpu": platform.processor(),
            "cores": os.cpu_count(),
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
        },
    }

    out = Path(args.out) if args.out else RESULTS / f"{payload['variant']}.json"
    write_json(out, payload)
    print(json.dumps({k: payload[k] for k in ("variant", "model_mb", "metrics", "latency_ms")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
