"""Inspect a quantized graph against the FP32 graph on one frame.

    python experiments/debug_quant.py models/yolo11n_int8_static_640.onnx

Prints score statistics and the top detections for both graphs so a broken
quantization shows up immediately instead of as AP = 0 after a full run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from common import decode, imread, letterbox

ROOT = Path(__file__).resolve().parents[1]


def run(model: Path, image_path: Path) -> tuple[np.ndarray, dict]:
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    tensor, meta = letterbox(imread(image_path), 640)
    return session.run(None, {session.get_inputs()[0].name: tensor})[0], meta


def report(name: str, output: np.ndarray, meta: dict, conf: float) -> None:
    scores = output[0, 4]
    print(f"{name}")
    print(f"  output shape   {output.shape}")
    print(f"  class scores   min {scores.min():.5f}  max {scores.max():.5f}  mean {scores.mean():.5f}  >0.5: {(scores > 0.5).sum()}")
    boxes, kept = decode(output, meta, conf, 0.7)
    print(f"  detections@{conf}: {len(kept)}")
    for box, score in list(zip(boxes, kept))[:5]:
        print(f"    {score:.4f}  [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "yolo11n_int8_static_640.onnx"
    image = sorted((ROOT / "static" / "samples").glob("frame-1.jpg"))[0]

    fp32, meta = run(ROOT / "models" / "yolo11n_fp32_640.onnx", image)
    report("FP32", fp32, meta, 0.25)

    quant, _ = run(target, image)
    report(target.name, quant, meta, 0.25)
    report(target.name, quant, meta, 0.001)

    diff = np.abs(quant - fp32)
    print(f"\n  abs diff vs FP32: mean {diff.mean():.5f}  max {diff.max():.4f}")


if __name__ == "__main__":
    main()
