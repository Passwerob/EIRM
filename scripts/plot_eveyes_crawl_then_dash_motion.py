#!/usr/bin/env python3
"""Four-panel dash row plus a motion timeline for crawl-then-dash.

The event panels stay unlabelled 960x540 EIRM maps. The motion figure is a
separate chart: scene speed, sensor clock, prism sweep, IMU, and Ours RPM.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v3 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

READY = Path("data/eveyes_comotion_v2_ready/test/Scene_005/crawl_then_dash")
EVAL_JSON = Path("results/ami_next/eveyes_crawl_then_dash.json")
PANEL_DIR = Path("results/ami_next/figures/crawl_then_dash")
SCENE = "test/Scene_005/crawl_then_dash"
GUTTER = 14
PAPER_BG = (255, 255, 255)
INK = "#1F2933"
MUTED = "#6B7280"
NAVY = "#1B4F8A"
WARM = "#B45309"
TEAL = "#0F766E"
MAGENTA = "#9D174D"
CRAWL = "#DBEAFE"
TRANSIT = "#FEF3C7"
DASH = "#FCE7F3"


def _grid(panels: list[np.ndarray]) -> np.ndarray:
    height, width = panels[0].shape[:2]
    canvas = np.full((height, len(panels) * width + (len(panels) - 1) * GUTTER, 3), PAPER_BG, dtype=np.uint8)
    for index, panel in enumerate(panels):
        left = index * (width + GUTTER)
        canvas[:, left : left + width] = panel
    return canvas


def _rows(payload: dict, method: str) -> list[dict]:
    selected = [row for row in payload["rows"] if row["scene"] == SCENE and row["method"] == method]
    selected.sort(key=lambda row: int(row["window"]))
    return selected


def _segments(times: np.ndarray, rate: np.ndarray, fps: np.ndarray) -> list[tuple[float, float, str]]:
    labels = np.where(fps > 100, "dash", np.where(rate > 0.9, "transit", "crawl"))
    marks = []
    start = 0
    for index in range(1, len(labels)):
        if labels[index] != labels[index - 1]:
            marks.append((float(times[start]), float(times[index]), str(labels[index - 1])))
            start = index
    marks.append((float(times[start]), float(times[-1]), str(labels[-1])))
    return marks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PANEL_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    names = ("b1_rgb", "b2_rpm0", "b3_fixed180", "b4_rpm360")
    panels = [np.asarray(imageio.imread(PANEL_DIR / f"{name}.png")[..., :3]) for name in names]
    grid = _grid(panels)
    imageio.imwrite(args.output_dir / "grid_four_dash.png", grid)

    state = list(csv.DictReader((READY / "motion_state.csv").open(encoding="utf-8")))
    times = np.asarray([float(row["timestamp_s"]) for row in state], dtype=np.float64)
    speed = np.asarray([float(row["speed_m_s"]) for row in state], dtype=np.float64)
    rate = np.asarray([float(row["playback_rate"]) for row in state], dtype=np.float64)
    fps = np.asarray([float(row["sensor_fps"]) for row in state], dtype=np.float64)
    sweep = 180.0 / 60.0 * 360.0 / fps
    imu = np.genfromtxt(READY / "imu.csv", delimiter=",", names=True)
    imu_t = np.asarray(imu["timestamp_s"], dtype=np.float64)
    gyro = np.asarray(imu["gyro_norm_rad_s"], dtype=np.float64)
    stride = max(1, int(round(len(imu_t) / 2000)))
    payload = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    ours = _rows(payload, "ours")
    ours_t = np.asarray([0.5 * (float(row["t0"]) + float(row["t1"])) for row in ours], dtype=np.float64)
    ours_rpm = np.asarray([float(row["rpm"]) for row in ours], dtype=np.float64)
    hero = json.loads((PANEL_DIR / "metrics.json").read_text(encoding="utf-8"))["row_b"]
    hero_t = float(hero["t"])
    segments = _segments(times, rate, fps)
    fill = {"crawl": CRAWL, "transit": TRANSIT, "dash": DASH}

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(4, 1, figsize=(10.2, 7.2), sharex=True, gridspec_kw={"hspace": 0.12})
    for left, right, name in segments:
        for axis in axes:
            axis.axvspan(left, right, color=fill[name], alpha=0.55, linewidth=0, zorder=0)
    for axis in axes:
        axis.axvline(hero_t, color=MAGENTA, linewidth=1.0, linestyle="--", zorder=4)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(colors=INK, length=3, width=0.7)
        axis.yaxis.label.set_color(INK)
        axis.set_xlim(0.0, float(times[-1]))

    axes[0].plot(times, speed, color=NAVY, linewidth=1.6, zorder=3)
    axes[0].set_ylabel("ego speed (m/s)")
    axes[0].set_ylim(0.0, max(20.0, float(speed.max()) * 1.08))
    for left, right, name in segments:
        axes[0].text(0.5 * (left + right), 18.6, name, ha="center", va="top", color=INK, fontsize=8)

    axes[1].plot(times, rate, color=WARM, linewidth=1.6, zorder=3)
    axes[1].set_ylabel("playback rate")
    axes[1].set_ylim(0.0, 9.2)

    axes[2].plot(times, fps, color=TEAL, linewidth=1.6, zorder=3)
    axes[2].set_ylabel("sensor clock (Hz)")
    axes[2].set_ylim(0.0, 560.0)
    twin = axes[2].twinx()
    twin.plot(times, sweep, color=MUTED, linewidth=1.1, linestyle=":", zorder=3)
    twin.set_ylabel("180 RPM sweep (deg/win)", color=MUTED)
    twin.set_ylim(0.0, 22.0)
    twin.spines["top"].set_visible(False)
    twin.tick_params(colors=MUTED, length=3, width=0.7)
    twin.yaxis.label.set_color(MUTED)

    axes[3].step(ours_t, ours_rpm, where="post", color=MAGENTA, linewidth=1.7, zorder=3)
    axes[3].set_ylabel("Ours RPM")
    axes[3].set_ylim(-20.0, 400.0)
    axes[3].set_yticks([0, 180, 360])
    axes[3].set_xlabel("time (s)")
    gyro_axis = axes[3].twinx()
    gyro_axis.plot(imu_t[::stride], 1000.0 * gyro[::stride], color=MUTED, linewidth=0.7, alpha=0.9, zorder=2)
    gyro_axis.set_ylabel("gyro (mrad/s)", color=MUTED)
    gyro_axis.set_ylim(0.0, 4.2)
    gyro_axis.spines["top"].set_visible(False)
    gyro_axis.tick_params(colors=MUTED, length=3, width=0.7)
    gyro_axis.yaxis.label.set_color(MUTED)
    axes[3].text(hero_t + 0.04, 330.0, "figure window", color=MAGENTA, fontsize=8, va="bottom")

    fig.align_ylabels(axes)
    fig.savefig(args.output_dir / "motion.png", dpi=220, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(args.output_dir / "motion.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(json.dumps({"four_panel": str(args.output_dir / "grid_four_dash.png"), "motion": str(args.output_dir / "motion.png")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
