"""Thread-scaling benchmark: how much does CPU inference gain from more threads?

    python experiments/bench_threads.py

Runs a fixed set of validation frames through each model with
intra_op_num_threads set to 1 / 2 / 4 / 8 / 16 (16 = all logical cores on this
laptop). Only `session.run` is timed, so the numbers are pure inference cost.
Writes results/threads.json.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from common import VAL_IMAGES, imread, letterbox, write_json

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "threads.json"

MODELS = [
    ("yolo11n_fp32_640.onnx", "fp32", 640),
    ("yolo11n_fp16_640.onnx", "fp16", 640),
    # Dynamic INT8 is measured in the full suite; it is omitted here because its
    # ConvInteger path is an order of magnitude slower and adds nothing to the
    # thread-scaling story.
    ("yolo11n_int8_static_640.onnx", "int8_static", 640),
]

THREADS = [1, 2, 4, 8, os.cpu_count() or 16]
FRAMES = 100
REPEATS = 2


def frames(size: int) -> list[np.ndarray]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in VAL_IMAGES.iterdir() if p.suffix.lower() in suffixes)
    # Spread the sample across the split instead of taking the first N frames.
    step = max(1, len(paths) // FRAMES)
    chosen = paths[::step][:FRAMES]
    return [letterbox(imread(p), size)[0] for p in chosen]


def benchmark(model: Path, size: int, threads: int, tensors: list[np.ndarray]) -> dict:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    for _ in range(5):
        session.run(None, {input_name: tensors[0]})

    timings: list[float] = []
    for _ in range(REPEATS):
        for tensor in tensors:
            started = time.perf_counter()
            session.run(None, {input_name: tensor})
            timings.append((time.perf_counter() - started) * 1000)

    values = np.asarray(timings)
    return {
        "threads": threads,
        "mean_ms": round(float(values.mean()), 2),
        "p50_ms": round(float(np.percentile(values, 50)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "fps": round(1000 / float(values.mean()), 2),
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    payload: dict = {"hardware": {"cpu": platform.processor(), "logical_cores": os.cpu_count()}, "frames": FRAMES, "repeats": REPEATS, "models": {}}

    if OUT.exists():
        existing = json.loads(OUT.read_text(encoding="utf-8"))
        if existing.get("frames") == FRAMES and existing.get("hardware", {}).get("logical_cores") == os.cpu_count():
            payload = existing

    for filename, tag, size in MODELS:
        model = ROOT / "models" / filename
        if not model.exists():
            print(f"missing {filename}, skipping", flush=True)
            continue

        tensors = frames(size)
        rows = payload["models"].setdefault(tag, {"model": filename, "size": size, "runs": []})
        done = {row["threads"] for row in rows["runs"]}

        for threads in THREADS:
            if threads in done:
                print(f"skip {tag} @ {threads} threads", flush=True)
                continue
            print(f"{tag} @ {threads} threads ...", end=" ", flush=True)
            row = benchmark(model, size, threads, tensors)
            print(f"{row['mean_ms']:.2f} ms/img ({row['fps']:.1f} FPS)", flush=True)
            rows["runs"].append(row)
            rows["runs"].sort(key=lambda r: r["threads"])
            write_json(OUT, payload)

    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
