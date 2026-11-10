#!/usr/bin/env python3
"""Point to why Fixed 180 loses: IMU/gear timeline plus annotated event maps.

The unlabeled native panels stay untouched. This figure is the comparison:
arrows on the dash window, and edge F1 of Ours drawn on top of Fixed 180.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

from scripts.plot_eveyes_crawl_then_dash_imu import (
    CONSTRUCTED,
    EVAL_JSON,
    METRICS,
    _motion,
    _rows,
    _segments,
)

PANEL_DIR = Path("results/ami_next/figures/crawl_then_dash")
INK = "#1F2933"
MUTED = "#6B7280"
NAVY = "#1B4F8A"
TEAL = "#0F766E"
WARM = "#B45309"
MAGENTA = "#9D174D"
GOLD = "#F5D76E"
CRAWL = "#DBEAFE"
TRANSIT = "#FEF3C7"
DASH = "#FCE7F3"
BG = np.asarray([3.0, 7.0, 18.0], dtype=np.float32)


def _arrow_targets() -> list[tuple[float, float]]:
    """Matched callouts on facade / overpass, not on the car."""
    return [(168.0, 175.0), (455.0, 95.0), (835.0, 145.0)]


def _callout(axis, xy, text, *, xytext, ha="left") -> None:
    axis.annotate(
        text,
        xy=xy,
        xytext=xytext,
        xycoords="data",
        textcoords="data",
        color="white",
        fontsize=8,
        fontweight="bold",
        ha=ha,
        va="center",
        arrowprops={"arrowstyle": "-|>", "color": GOLD, "lw": 1.6, "shrinkA": 0, "shrinkB": 2},
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#111827", "edgecolor": GOLD, "linewidth": 0.8},
        zorder=6,
    )


def _panel(axis, image: np.ndarray, title: str, f1: float, points: list[tuple[float, float]], texts: list[str], offsets: list[tuple[float, float, str]]) -> None:
    axis.imshow(image, extent=(0, image.shape[1], image.shape[0], 0), interpolation="nearest")
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(0.8)
    axis.set_title(f"{title}   edge F1 {f1:.3f}", color=INK, fontsize=10, pad=6, loc="left")
    for (x, y), text, (dx, dy, ha) in zip(points, texts, offsets):
        axis.add_patch(Circle((x, y), 16, fill=False, edgecolor=GOLD, linewidth=1.6, zorder=5))
        _callout(axis, (x, y), text, xytext=(x + dx, y + dy), ha=ha)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PANEL_DIR)
    args = parser.parse_args()

    fixed = np.asarray(imageio.imread(PANEL_DIR / "b3_fixed180.png")[..., :3])
    ours_img = np.asarray(imageio.imread(PANEL_DIR / "b4_rpm360.png")[..., :3])
    points = _arrow_targets()
    if len(points) < 2:
        points = [(220.0, 90.0), (780.0, 160.0), (470.0, 210.0)]

    payload = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    hero = metrics["row_b"]
    hero_t = float(hero["t"])
    ours = _rows(payload, "ours")
    fixed180 = _rows(payload, "ami_fixed_180")
    ours_t0 = np.asarray([float(row["t0"]) for row in ours])
    ours_t1 = np.asarray([float(row["t1"]) for row in ours])
    ours_rpm = np.asarray([float(row["rpm"]) for row in ours])
    ours_f1 = np.asarray([float(row["edge_f1"]) for row in ours])
    f1_180 = np.asarray([float(row["edge_f1"]) for row in fixed180])
    mid = 0.5 * (ours_t0 + ours_t1)
    times, rate, fps, _ = _motion()
    segments = _segments(times, rate, fps)
    fill = {"crawl": CRAWL, "transit": TRANSIT, "dash": DASH}
    imu = np.genfromtxt(CONSTRUCTED / "imu.csv", delimiter=",", names=True)
    imu_t = np.asarray(imu["timestamp_s"])
    gyro = np.asarray(imu["gyro_norm_rad_s"])
    stride = max(1, int(round(len(imu_t) / 2500)))
    gear_t = np.concatenate([ours_t0[:1], ours_t1])
    gear_rpm = np.concatenate([ours_rpm[:1], ours_rpm])
    dash_win = np.asarray([float(row["t0"]) >= segments[-1][0] for row in ours])
    n_dash = int(dash_win.sum())
    n_win = int(np.sum(dash_win & (ours_f1 > f1_180)))
    delta = float(hero["ours_minus_180_edge_f1"])

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(11.4, 8.9))
    grid = GridSpec(4, 2, height_ratios=[2.45, 0.72, 0.78, 1.25], hspace=0.30, wspace=0.05, left=0.07, right=0.99, top=0.94, bottom=0.07)

    ax180 = fig.add_subplot(grid[0, 0])
    axours = fig.add_subplot(grid[0, 1])
    offsets_180 = [(18, 85, "left"), (95, -8, "left"), (-15, 95, "right")]
    offsets_ours = [(18, 85, "left"), (95, -8, "left"), (-15, 95, "right")]
    texts_180 = ["sparse wall", "missing overpass", "weak facade"]
    texts_ours = ["wall recovered", "overpass filled", "facade recovered"]
    _panel(ax180, fixed, "Fixed 180", float(hero["edge_f1"]["180"]), points, texts_180, offsets_180)
    _panel(axours, ours_img, "Ours (360 RPM)", float(hero["edge_f1"]["ours"]), points, texts_ours, offsets_ours)

    aximu = fig.add_subplot(grid[1, :])
    axgear = fig.add_subplot(grid[2, :], sharex=aximu)
    axf1 = fig.add_subplot(grid[3, :], sharex=aximu)
    axes = (aximu, axgear, axf1)
    for axis in axes:
        for left, right, name in segments:
            axis.axvspan(left, right, color=fill[name], alpha=0.55, linewidth=0, zorder=0)
        axis.axvline(hero_t, color=MAGENTA, linewidth=0.9, linestyle="--", zorder=4)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(colors=INK, length=3, width=0.7)
        axis.set_xlim(0.0, float(times[-1]))

    aximu.plot(imu_t[::stride], 1000.0 * gyro[::stride], color=TEAL, linewidth=1.25, zorder=3)
    aximu.set_ylabel("gyro |ω|\n(mrad/s)")
    aximu.set_ylim(0.0, 4.2)
    aximu.set_yticks([0, 2, 4])
    for left, right, name in segments:
        aximu.text(0.5 * (left + right), 3.95, name, ha="center", va="top", color=INK, fontsize=8)

    axgear.hlines([180.0], 0.0, float(times[-1]), colors=MUTED, linestyles="--", linewidth=1.1, zorder=2, label="Fixed 180")
    axgear.step(gear_t, gear_rpm, where="post", color=MAGENTA, linewidth=1.8, zorder=3, label="Ours")
    axgear.set_ylabel("RPM")
    axgear.set_ylim(120.0, 410.0)
    axgear.set_yticks([180, 360])
    axgear.legend(frameon=False, loc="center left", fontsize=8)
    dash_start = segments[-1][0]
    axgear.annotate(
        "upshift 180 → 360",
        xy=(dash_start + 0.02, 360.0),
        xytext=(dash_start - 0.72, 250.0),
        color=MAGENTA,
        fontsize=8,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": MAGENTA, "lw": 1.3},
        zorder=6,
    )

    delta_f1 = ours_f1 - f1_180
    axf1.axhline(0.0, color=MUTED, linewidth=0.8, zorder=1)
    axf1.fill_between(mid, 0.0, delta_f1, where=delta_f1 >= 0, color=TEAL, alpha=0.35, linewidth=0, interpolate=True, zorder=2)
    axf1.fill_between(mid, 0.0, delta_f1, where=delta_f1 < 0, color=MUTED, alpha=0.25, linewidth=0, interpolate=True, zorder=2)
    axf1.plot(mid, delta_f1, color=MAGENTA, linewidth=1.35, zorder=3)
    axf1.set_ylabel("edge F1\nOurs − 180")
    axf1.set_xlabel("time (s)")
    axf1.set_ylim(-0.08, 0.12)
    axf1.annotate(
        f"dash window: Ours {hero['edge_f1']['ours']:.3f} vs Fixed 180 {hero['edge_f1']['180']:.3f}\n"
        f"({delta:+.3f})   Ours wins {n_win}/{n_dash} dash windows",
        xy=(hero_t, delta),
        xytext=(2.42, 0.082),
        color=INK,
        fontsize=8,
        arrowprops={"arrowstyle": "-|>", "color": MAGENTA, "lw": 1.4},
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": MAGENTA, "linewidth": 0.8},
        zorder=6,
    )

    fig.savefig(args.output_dir / "imu_gear_vs_180.png", dpi=220)
    fig.savefig(args.output_dir / "imu_gear_vs_180.pdf")
    plt.close(fig)
    print(json.dumps({"figure": str(args.output_dir / "imu_gear_vs_180.png"), "arrow_xy": points, "dash_wins": f"{n_win}/{n_dash}", "delta": delta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
