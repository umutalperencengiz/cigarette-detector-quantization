"""Screen static-quantization recipes, one subprocess per recipe.

    python experiments/quant_experiments.py           # build + score every recipe
    python experiments/quant_experiments.py --only percentile

Each recipe gets a fresh process: ONNX Runtime keeps large arenas alive per
session, and running calibration and evaluation in the same interpreter can
fail with allocation errors after a few rounds.

Scoring uses the full 1,370-image split, because a 300-image subset is noisy
enough (±4 mAP) to hide the differences this script exists to find. Results go
to results/_exp/, which the site builder ignores.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import onnx
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

from common import TRAIN_IMAGES
from export_variants import FrameReader

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "_exp"
RESULTS = ROOT / "results" / "_exp"
SOURCE = ROOT / "models" / "yolo11n_fp32_640.onnx"
FRAMES = 256

RECIPES = {
    "minmax": {},
    # Percentile/Entropy calibrators keep per-frame activation values, so 256
    # frames exhausts the arena; 64 is enough to estimate a quantile.
    "percentile": {
        "calibrate_method": CalibrationMethod.Percentile,
        "extra_options": {"Percentile": 99.99},
        "frames": 64,
    },
    "entropy": {"calibrate_method": CalibrationMethod.Entropy, "frames": 64},
    "minmax_reduce_range": {"reduce_range": True},
    "minmax_per_tensor": {"per_channel": False},
    "minmax_head_fp32": {"exclude_head": True},
    "percentile_head_fp32": {
        "calibrate_method": CalibrationMethod.Percentile,
        "extra_options": {"Percentile": 99.99},
        "exclude_head": True,
        "frames": 64,
    },
    "qoperator": {"quant_format": QuantFormat.QOperator},
}


def head_nodes(path: Path) -> list[str]:
    model = onnx.load(str(path))
    return [node.name for node in model.graph.node if node.name.startswith("/model.23")]


def output_node(path: Path) -> str:
    model = onnx.load(str(path))
    target = model.graph.output[0].name
    return next(node.name for node in model.graph.node if target in node.output)


def build(name: str) -> Path:
    target = MODELS / f"{name}.onnx"
    if target.exists():
        return target

    options = dict(RECIPES[name])
    frames = sorted(TRAIN_IMAGES.glob("*.jpg"))
    random.Random(0).shuffle(frames)
    count = options.pop("frames", FRAMES)

    exclude = [output_node(SOURCE)]
    if options.pop("exclude_head", False):
        exclude += head_nodes(SOURCE)

    quantize_static(
        model_input=str(SOURCE),
        model_output=str(target),
        calibration_data_reader=FrameReader(frames[:count], 640),
        quant_format=options.pop("quant_format", QuantFormat.QDQ),
        calibrate_method=options.pop("calibrate_method", CalibrationMethod.MinMax),
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=options.pop("per_channel", True),
        reduce_range=options.pop("reduce_range", False),
        nodes_to_exclude=exclude,
        extra_options=options.pop("extra_options", {}),
    )
    return target


def evaluate(name: str, model: Path) -> dict:
    out = RESULTS / f"{name}.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "evaluate_coco.py"),
            "--model",
            str(model),
            "--size",
            "640",
            "--out",
            str(out),
            "--tag",
            name,
        ],
        check=True,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
    )
    return json.loads(out.read_text(encoding="utf-8"))


def run_one(name: str) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    model = build(name)
    data = evaluate(name, model)
    m = data["metrics"]
    print(
        f"{name:22s} AP {m['ap']:6.2f}  AP50 {m['ap50']:6.2f}  AP75 {m['ap75']:6.2f}  "
        f"APs {m['aps']:6.2f}  {data['model_mb']:.2f} MB  {data['latency_ms']['mean']:6.2f} ms",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    if args.only:
        run_one(args.only)
        return

    for name in RECIPES:
        print(f"--- {name} ---", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--only", name], cwd=ROOT)

    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append((path.stem, data))
    rows.sort(key=lambda row: -row[1]["metrics"]["ap"])

    print("\nranking by AP:")
    for name, data in rows:
        m = data["metrics"]
        print(f"  {name:22s} AP {m['ap']:6.2f}  APs {m['aps']:6.2f}  {data['latency_ms']['mean']:6.2f} ms")


if __name__ == "__main__":
    main()
