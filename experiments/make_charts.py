"""Render the charts used on the results page.

    python experiments/make_charts.py

Reads results/*.json (written by evaluate_coco.py / run_suite.py) and
results/threads.json, then writes SVG files into docs/charts/. SVG keeps the
page dependency-free and sharp at any zoom.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHARTS = ROOT / "docs" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

BG = "#0d121a"
TEXT = "#e7eef7"
MUTED = "#8b9bb0"
DIM = "#5c6b7f"
LINE = "#1d2734"
ACCENT = "#35d0a5"
BLUE = "#3b9bff"
AMBER = "#ffb454"
DANGER = "#ff6b6b"
VIOLET = "#a78bfa"

PRECISIONS = ["fp32_640", "fp16_640", "int8_dynamic_640", "int8_static_640"]
PRECISION_LABELS = {
    "fp32_640": "FP32",
    "fp16_640": "FP16",
    "int8_dynamic_640": "INT8 dynamic",
    "int8_static_640": "INT8 static (QDQ)",
}
PRECISION_COLORS = {
    "fp32_640": BLUE,
    "fp16_640": ACCENT,
    "int8_dynamic_640": AMBER,
    "int8_static_640": VIOLET,
}
SIZE_TAGS = ["fp32_320", "fp32_416", "fp32_512", "fp32_640"]


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.edgecolor": LINE,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": "#182130",
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "semibold",
            "legend.frameon": False,
            "legend.labelcolor": MUTED,
            # Text as outlines: the SVG is loaded through <img>, which cannot use
            # web fonts, and "DejaVu Sans" is not installed on every visitor's
            # machine. Outlines keep every chart identically typeset.
            "svg.fonttype": "path",
        }
    )


def load() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(RESULTS.glob("*.json")):
        if path.name in {"val_gt.json", "threads.json", "summary.json"}:
            continue
        if path.name.startswith("smoke"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("images", 0) < 500:
            continue
        out[data.get("variant") or path.stem] = data
    return out


def finish(fig, name: str) -> None:
    target = CHARTS / name
    fig.savefig(target, format="svg", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {target.relative_to(ROOT)}")


def clean(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    ax.grid(True, axis="y", alpha=0.55)
    ax.set_axisbelow(True)


def pareto(data: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    clean(ax)

    for tag, run in data.items():
        ap = run["metrics"]["ap"]
        ms = run["latency_ms"]["mean"]
        mb = run["model_mb"]
        color = PRECISION_COLORS.get(tag, BLUE)
        marker = "o" if tag in PRECISION_COLORS else "s"
        ax.scatter(ms, ap, s=28 + mb * 9, color=color, alpha=0.9, edgecolor=BG, linewidth=1.2, marker=marker, zorder=3)

        label = PRECISION_LABELS.get(tag) or f"FP32 {run['size']}px"
        offsets = {
            "fp32_640": ((7, 5), "left"),
            "fp16_640": ((7, -14), "left"),
            "int8_static_640": ((-9, 6), "right"),
            "fp32_512": ((-9, -14), "right"),
            "fp32_416": ((8, 4), "left"),
            "fp32_320": ((8, 4), "left"),
            "int8_dynamic_640": ((-9, 4), "right"),
        }
        offset, align = offsets.get(tag, ((7, 4), "left"))
        ax.annotate(
            label,
            (ms, ap),
            textcoords="offset points",
            xytext=offset,
            ha=align,
            fontsize=8.6,
            color=TEXT,
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_xlabel("CPU inference latency, ONNX Runtime (ms/image, log scale)")
    ax.set_ylabel("mAP@50–95 (%)")
    ax.set_title("Accuracy vs latency · bubble area = model size", loc="left", pad=10)
    ax.set_ylim(bottom=min(r["metrics"]["ap"] for r in data.values()) - 3)

    handles = [
        Line2D([], [], marker="o", linestyle="", color=MUTED, markersize=6, label="precision variant"),
        Line2D([], [], marker="s", linestyle="", color=MUTED, markersize=6, label="input-size sweep"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5)
    finish(fig, "pareto.svg")


def precision(data: dict[str, dict]) -> None:
    tags = [t for t in PRECISIONS if t in data]
    metrics = ["ap", "ap50", "ap75", "aps", "apm"]
    labels = ["mAP\n50–95", "mAP50", "mAP75", "AP small", "AP medium"]

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    clean(ax)
    width = 0.2
    x = np.arange(len(metrics))

    for index, tag in enumerate(tags):
        values = [data[tag]["metrics"][m] or 0 for m in metrics]
        pos = x + (index - (len(tags) - 1) / 2) * width
        bars = ax.bar(pos, values, width * 0.88, color=PRECISION_COLORS[tag], label=PRECISION_LABELS[tag], zorder=3)
        for bar, value in zip(bars, values):
            # Adjacent bars are nearly the same height in the accuracy groups, so
            # stagger the value labels vertically instead of letting them collide.
            lift = 3 if index % 2 == 0 else 13
            ax.annotate(
                f"{value:.1f}",
                (bar.get_x() + bar.get_width() / 2, value),
                textcoords="offset points",
                xytext=(0, lift),
                ha="center",
                fontsize=7.2,
                color=MUTED,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("score (%)")
    ax.set_title("What each precision actually costs, 640px input", loc="left", pad=10)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", ncol=2, fontsize=8.3, labelcolor=MUTED)
    finish(fig, "precision.svg")


def size_sweep(data: dict[str, dict]) -> None:
    tags = [t for t in SIZE_TAGS if t in data]
    sizes = [data[t]["size"] for t in tags]
    ap = [data[t]["metrics"]["ap"] for t in tags]
    aps = [data[t]["metrics"]["aps"] for t in tags]
    ms = [data[t]["latency_ms"]["mean"] for t in tags]

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.2, 3.6), gridspec_kw={"wspace": 0.32})

    clean(left)
    left.plot(sizes, ap, color=ACCENT, marker="o", linewidth=2, label="mAP@50–95", zorder=3)
    left.plot(sizes, aps, color=AMBER, marker="s", linewidth=2, label="AP small", zorder=3)
    for size, a, s in zip(sizes, ap, aps):
        left.annotate(f"{a:.1f}", (size, a), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=ACCENT)
        left.annotate(f"{s:.1f}", (size, s), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=AMBER)
    left.set_xticks(sizes)
    left.set_xticklabels([f"{s}px" for s in sizes])
    left.set_ylim(0, 65)
    left.set_ylabel("score (%)")
    left.set_title("Accuracy vs input size", loc="left", pad=8)
    left.legend(loc="lower right", fontsize=8.3)

    clean(right)
    bars = right.bar([str(s) for s in sizes], ms, color=BLUE, width=0.55, zorder=3)
    for bar, value in zip(bars, ms):
        right.annotate(
            f"{value:.1f} ms",
            (bar.get_x() + bar.get_width() / 2, value),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8.2,
            color=TEXT,
        )
    right.set_xlabel("input size")
    right.set_ylabel("latency (ms/image)")
    right.set_ylim(0, max(ms) * 1.22)
    right.set_title("Latency vs input size", loc="left", pad=8)

    finish(fig, "size-sweep.svg")


def threads() -> None:
    path = RESULTS / "threads.json"
    if not path.exists():
        print("no threads.json yet, skipping thread chart")
        return
    data = json.loads(path.read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    clean(ax)
    for tag, color in (
        ("fp32", PRECISION_COLORS["fp32_640"]),
        ("fp16", PRECISION_COLORS["fp16_640"]),
        ("int8_static", PRECISION_COLORS["int8_static_640"]),
    ):
        block = data["models"].get(tag)
        if not block:
            continue
        runs = sorted(block["runs"], key=lambda r: r["threads"])
        xs = [r["threads"] for r in runs]
        ys = [r["mean_ms"] for r in runs]
        ax.plot(xs, ys, color=color, marker="o", linewidth=1.9, label=PRECISION_LABELS[f"{tag}_640"], zorder=3)
        ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, -3), fontsize=8, color=color)

    cores = data["hardware"]["logical_cores"]
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, cores])
    ax.set_xticklabels(["1", "2", "4", "8", f"{cores} (all)"])
    ax.set_xlabel("intra-op threads")
    ax.set_ylabel("latency (ms/image)")
    ax.set_title("Thread scaling · single-session CPU inference", loc="left", pad=10)
    ax.legend(loc="upper right", fontsize=8.2)
    finish(fig, "threads.svg")


def main() -> None:
    style()
    data = load()
    if not data:
        raise SystemExit("no full-validation results found in results/ — run run_suite.py first")
    print(f"loaded {len(data)} runs: {', '.join(sorted(data))}")
    pareto(data)
    precision(data)
    size_sweep(data)
    threads()


if __name__ == "__main__":
    main()
