"""Export the shipped YOLO11n checkpoint to the variants used in the study.

    python experiments/export_variants.py

Produces, under models/:
    yolo11n_fp32_{640,512,416,320}.onnx   input-size sweep
    yolo11n_fp16_640.onnx                 float16 weights, fp32 I/O
    yolo11n_int8_dynamic_640.onnx         dynamic weight quantization
    yolo11n_int8_static_640.onnx          QDQ, calibrated on 256 training frames
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from ultralytics import YOLO

from common import TRAIN_IMAGES, letterbox, imread

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
CHECKPOINT = MODELS / "yolo11n_fp32_640.pt"
SIZES = [640, 512, 416, 320]
CALIBRATION_FRAMES = 256


def export_size(size: int) -> Path:
    target = MODELS / f"yolo11n_fp32_{size}.onnx"
    if target.exists():
        print(f"skip {target.name} (exists)")
        return target

    # Ultralytics writes <checkpoint-stem>.onnx, so export from a size-specific copy
    # to avoid overwriting the 640 graph when the next size is exported.
    staging = MODELS / f"_tmp_yolo11n_{size}.pt"
    shutil.copy2(CHECKPOINT, staging)
    model = YOLO(str(staging))
    exported = Path(
        model.export(format="onnx", imgsz=size, opset=13, dynamic=False, simplify=False)
    )
    shutil.move(str(exported), target)
    staging.unlink(missing_ok=True)
    print(f"exported {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def to_fp16(source: Path) -> Path:
    target = MODELS / "yolo11n_fp16_640.onnx"
    if target.exists():
        return target
    model = onnx.load(str(source))
    model = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model, str(target))
    print(f"converted {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def to_int8_dynamic(source: Path) -> Path:
    target = MODELS / "yolo11n_int8_dynamic_640.onnx"
    if target.exists():
        return target
    quantize_dynamic(
        model_input=str(source),
        model_output=str(target),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    print(f"quantized {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


class FrameReader(CalibrationDataReader):
    def __init__(self, frames: list[Path], size: int) -> None:
        self.frames = frames
        self.size = size
        self.index = 0

    def get_next(self):
        if self.index >= len(self.frames):
            return None
        frame = self.frames[self.index]
        self.index += 1
        tensor, _ = letterbox(imread(frame), self.size)
        return {"images": tensor}


def output_node(path: Path) -> str:
    """Name of the node that produces the graph output."""
    model = onnx.load(str(path))
    target = model.graph.output[0].name
    for node in model.graph.node:
        if target in node.output:
            return node.name
    raise RuntimeError(f"no node produces {target} in {path.name}")


def to_int8_static(source: Path, size: int = 640) -> Path:
    target = MODELS / "yolo11n_int8_static_640.onnx"
    if target.exists():
        return target

    frames = sorted(TRAIN_IMAGES.glob("*.jpg"))
    random.Random(0).shuffle(frames)
    reader = FrameReader(frames[:CALIBRATION_FRAMES], size)

    # Two exclusions, both measured rather than guessed:
    #
    # 1. The final node concatenates box coordinates (0..640 px) with class
    #    probabilities (0..1). One scale over that tensor steps by ~2.7, which
    #    rounds every probability below 0.5 to zero: the graph returns nothing
    #    (AP 0.00).
    # 2. The Detect head's box branch regresses pixel offsets through a DFL
    #    projection, and its class branch needs fine score resolution. Quantizing
    #    it costs ~12 mAP; keeping /model.23 in float costs 1 mAP and 1.2 MB.
    #    Mixed precision is the right call at this model size.
    exclude = [output_node(source)] + [n.name for n in onnx.load(str(source)).graph.node if n.name.startswith("/model.23")]
    print(f"  excluding {len(exclude)} nodes (output node + Detect head)")

    quantize_static(
        model_input=str(source),
        model_output=str(target),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        calibrate_method=CalibrationMethod.MinMax,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        nodes_to_exclude=exclude,
    )
    print(f"quantized {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def sanity_check(path: Path, size: int, reference: Path | None = None) -> None:
    """Run one frame and refuse to ship a graph whose output collapsed."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    frame = sorted(TRAIN_IMAGES.glob("*.jpg"))[0]
    tensor, _ = letterbox(imread(frame), size)
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]

    scores = output[0, 4]
    if not np.isfinite(output).all():
        raise SystemExit(f"{path.name}: non-finite output")

    if reference is not None:
        ref_session = ort.InferenceSession(str(reference), providers=["CPUExecutionProvider"])
        ref = ref_session.run(None, {ref_session.get_inputs()[0].name: tensor})[0]
        ref_peak, peak = float(ref[0, 4].max()), float(scores.max())
        if ref_peak > 0.05 and peak < 0.5 * ref_peak:
            raise SystemExit(
                f"{path.name}: output collapsed (peak score {peak:.4f} vs {ref_peak:.4f} in "
                f"{reference.name}) — check which node the quantizer folded into the output range"
            )
        print(f"  runs ok: {path.name} (peak score {peak:.3f} vs {ref_peak:.3f})")
        return

    print(f"  runs ok: {path.name} (peak score {float(scores.max()):.3f})")


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    if not CHECKPOINT.exists():
        raise SystemExit(f"missing checkpoint: {CHECKPOINT}")

    exported = {size: export_size(size) for size in SIZES}
    variants = {
        640: [
            to_fp16(exported[640]),
            to_int8_dynamic(exported[640]),
            to_int8_static(exported[640]),
        ],
    }

    for size, paths in variants.items():
        for path in paths:
            # Compare against the FP32 graph: quantized exports can fail quietly.
            sanity_check(path, size, reference=exported[size])

    print("\nmodel sizes:")
    for path in sorted(MODELS.glob("*.onnx")):
        print(f"  {path.name:34s} {path.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
