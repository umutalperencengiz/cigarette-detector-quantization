"""Run the full validation suite for every exported variant.

    python experiments/run_suite.py

Each run evaluates all 1,370 validation images and writes results/<tag>.json.
Existing result files are skipped, so the suite can be resumed.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

VARIANTS = [
    ("yolo11n_fp32_640.onnx", 640, "fp32_640"),
    ("yolo11n_fp16_640.onnx", 640, "fp16_640"),
    ("yolo11n_int8_dynamic_640.onnx", 640, "int8_dynamic_640"),
    ("yolo11n_int8_static_640.onnx", 640, "int8_static_640"),
    ("yolo11n_fp32_512.onnx", 512, "fp32_512"),
    ("yolo11n_fp32_416.onnx", 416, "fp32_416"),
    ("yolo11n_fp32_320.onnx", 320, "fp32_320"),
]


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    for model, size, tag in VARIANTS:
        out = RESULTS / f"{tag}.json"
        if out.exists():
            print(f"skip {tag} (exists)", flush=True)
            continue

        started = time.time()
        print(f"\n=== {tag} ({size}px) ===", flush=True)
        command = [
            sys.executable,
            str(ROOT / "experiments" / "evaluate_coco.py"),
            "--model",
            str(ROOT / "models" / model),
            "--size",
            str(size),
            "--out",
            str(out),
            "--tag",
            tag,
        ]
        subprocess.run(command, check=True, cwd=ROOT)
        print(f"finished {tag} in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
