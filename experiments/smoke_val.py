import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(r"C:\Users\umuta\Projects\small-object-detection-edge")
model = YOLO(str(ROOT / "models" / "yolo11n_fp32_640.pt"))
print("data:", (ROOT / "data.local.yaml").read_text(encoding="utf-8"))

result = model.val(
    data=str(ROOT / "data.local.yaml"),
    imgsz=640,
    batch=8,
    fraction=0.02,
    plots=False,
    verbose=False,
    project=str(ROOT / "runs"),
    name="smoke",
)

box = result.box
print("mAP50-95", round(float(box.map), 4), "mAP50", round(float(box.map50), 4), "mAP75", round(float(box.map75), 4))
print("speed", {k: round(v, 1) for k, v in result.speed.items()})
