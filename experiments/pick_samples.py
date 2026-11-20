"""Pick validation frames for the web demo, then render the hero figure.

    python experiments/pick_samples.py
    python experiments/pick_samples.py --annotate

Candidates must contain exactly one labelled cigarette and be on the small side.
They are then scored with the FP32 graph: only frames where the detector is
confident (peak score >= 0.55) make useful demo material, because the A/B panel
compares two precisions at the same threshold. The hero figure is drawn by the
INT8 graph, so the page shows what the quantized model really outputs.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

from common import VAL_IMAGES, VAL_LABELS, decode, imread, letterbox, write_json

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "static" / "samples"
IMG = ROOT / "docs" / "img"
PICKS = SAMPLES / "picks.json"

CANDIDATES = 140
MIN_PEAK = 0.55


def boxes_of(image: Path) -> list[list[float]]:
    label = VAL_LABELS / f"{image.stem}.txt"
    if not label.exists():
        return []
    out = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        out.append([float(v) for v in parts[1:5]])
    return out


def session(model: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])


def score_frame(sess, image: Path) -> tuple[float, int]:
    tensor, meta = letterbox(imread(image), 640)
    output = sess.run(None, {sess.get_inputs()[0].name: tensor})[0]
    _, scores = decode(output, meta, 0.05, 0.7)
    return (float(scores.max()) if len(scores) else 0.0), len(scores)


def candidates() -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in VAL_IMAGES.iterdir() if p.suffix.lower() in suffixes)

    scored = []
    for image in images:
        boxes = boxes_of(image)
        if len(boxes) != 1:
            continue
        _, _, w, h = boxes[0]
        scored.append((w * h, image))
    scored.sort(key=lambda row: row[0])

    # Small objects only: the first 40% by labelled area.
    pool = [image for _, image in scored[: int(len(scored) * 0.4)]]
    random.Random(7).shuffle(pool)
    return pool[:CANDIDATES]


def pick() -> list[Path]:
    sess = session(ROOT / "models" / "yolo11n_fp32_640.onnx")
    found = []
    for image in candidates():
        peak, count = score_frame(sess, image)
        if peak >= MIN_PEAK and count <= 2:
            found.append((peak, image))

    found.sort(key=lambda row: -row[0])
    chosen: list[Path] = []
    used_prefixes: set[str] = set()
    for peak, image in found:
        prefix = image.name.split("_")[0]
        if prefix in used_prefixes:
            continue
        used_prefixes.add(prefix)
        chosen.append(image)
        if len(chosen) == 3:
            break

    if len(chosen) < 3:
        chosen = [image for _, image in found[:3]]
    return chosen


def save_sample(image: Path, target: Path) -> None:
    """Copy a frame at web-friendly size (max 1280 px wide, quality 86)."""
    import cv2

    data = imread(image)
    height, width = data.shape[:2]
    if width > 1280:
        scale = 1280 / width
        data = cv2.resize(data, (1280, round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", data, [cv2.IMWRITE_JPEG_QUALITY, 86])
    if not ok:
        raise RuntimeError(f"could not encode {image.name}")
    target.write_bytes(encoded.tobytes())


def annotate(image_path: Path, model_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    sess = session(model_path)
    image = imread(image_path)
    tensor, meta = letterbox(image, 640)
    output = sess.run(None, {sess.get_inputs()[0].name: tensor})[0]
    boxes, scores = decode(output, meta, 0.25, 0.7)

    canvas = Image.fromarray(image[:, :, ::-1]).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", max(13, canvas.width // 60))
    except OSError:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), score in zip(boxes, scores):
        colour = (53, 208, 165) if score >= 0.5 else (255, 180, 84)
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=max(2, canvas.width // 320))
        text = f"cigarette {score:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        draw.rectangle([x1, max(0, y1 - (bottom - top) - 8), x1 + (right - left) + 12, y1], fill=colour)
        draw.text((x1 + 6, max(0, y1 - (bottom - top) - 6)), text, fill=(4, 18, 14), font=font)

    IMG.mkdir(parents=True, exist_ok=True)
    target = IMG / "hero.png"
    canvas.save(target, optimize=True)
    print(f"wrote {target.relative_to(ROOT)} ({target.stat().st_size / 1e3:.0f} KB)")

    write_json(
        ROOT / "docs" / "hero.json",
        {
            "source": image_path.name,
            "model": model_path.name,
            "detections": [
                {"score": round(float(s), 3), "box": [round(float(v), 1) for v in b]} for b, s in zip(boxes, scores)
            ],
        },
    )
    print(f"hero: {len(boxes)} detection(s), peak {float(scores.max()) if len(scores) else 0:.2f}")


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    picks = pick()
    entries = []
    for index, image in enumerate(picks, start=1):
        target = SAMPLES / f"frame-{index}.jpg"
        save_sample(image, target)
        entries.append(
            {"file": f"static/samples/frame-{index}.jpg", "source": image.name, "kb": round(target.stat().st_size / 1024, 1)}
        )
        print(f"{target.name} <- {image.name} ({entries[-1]['kb']} KB)")

    write_json(PICKS, {"frames": entries})

    if "--annotate" in sys.argv:
        annotate(picks[0], ROOT / "models" / "yolo11n_int8_static_640.onnx")


if __name__ == "__main__":
    main()
