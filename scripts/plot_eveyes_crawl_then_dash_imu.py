#!/usr/bin/env python3
"""IMU construction and Ours gear-shifting on the crawl-then-dash sequence.

Top: how imu.csv is made (pose-derived body rate, road-vibration residual, sum).
Bottom: the windowed IMU feature the controller sees, and the resulting RPM.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.build_eveyes_constructed_motion import (
    IMU_RATE_HZ,
    SOURCE_FPS,
    _body_rates,
    _read_imu,
    _read_pose,
    _sample_pose,
)
from ami_simulator.pipeline import _read_timestamp_file
from scripts.evaluate_eveyes_v2_downstream import _imu_fraction

READY = Path("data/eveyes_comotion_v2_ready/test/Scene_005")
CONSTRUCTED = READY / "crawl_then_dash"
SOURCE = READY / "drive_stop_drive"
EVAL_JSON = Path("results/ami_next/eveyes_crawl_then_dash.json")
METRICS = Path("results/ami_next/figures/crawl_then_dash/metrics.json")
SCENE = "test/Scene_005/crawl_then_dash"
INK = "#1F2933"
MUTED = "#6B7280"
NAVY = "#1B4F8A"
TEAL = "#0F766E"
WARM = "#B45309"
MAGENTA = "#9D174D"
CRAWL = "#DBEAFE"
TRANSIT = "#FEF3C7"
DASH = "#FCE7F3"


def _motion() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.DictReader((CONSTRUCTED / "motion_state.csv").open(encoding="utf-8")))
    times = np.asarray([float(row["timestamp_s"]) for row in rows], dtype=np.float64)
    rate = np.asarray([float(row["playback_rate"]) for row in rows], dtype=np.float64)
    fps = np.asarray([float(row["sensor_fps"]) for row in rows], dtype=np.float64)
    path = np.asarray([float(row["source_path_frame"]) for row in rows], dtype=np.float64)
    return times, rate, fps, path


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


def _rows(payload: dict, method: str) -> list[dict]:
    selected = [row for row in payload["rows"] if row["scene"] == SCENE and row["method"] == method]
    selected.sort(key=lambda row: int(row["window"]))
    return selected


def _reconstruct() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times, _, _, path = _motion()
    imu_t = np.arange(0.0, float(times[-1]) + 0.5 / IMU_RATE_HZ, 1.0 / IMU_RATE_HZ)
    imu_path = np.interp(imu_t, times, path)
    ego_p, ego_q = _read_pose(SOURCE / "ego_pose.csv")
    sampled_p, sampled_q = _sample_pose(ego_p, ego_q, imu_path)
    pose_gyro, _ = _body_rates(sampled_p, sampled_q, 1.0 / IMU_RATE_HZ)
    source_t, source_gyro, _ = _read_imu(SOURCE / "imu.csv")
    source_path = source_t * SOURCE_FPS
    base_p, base_q = _sample_pose(ego_p, ego_q, source_path)
    base_gyro, _ = _body_rates(base_p, base_q, float(np.median(np.diff(source_t))))
    residual = source_gyro - base_gyro
    vibration = np.stack([np.interp(imu_path, source_path, residual[:, axis]) for axis in range(3)], axis=1)
    written_t, written_gyro, _ = _read_imu(CONSTRUCTED / "imu.csv")
    return imu_t, np.linalg.norm(pose_gyro, axis=1), np.linalg.norm(vibration, axis=1), written_t, np.linalg.norm(written_gyro, axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/ami_next/figures/crawl_then_dash"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    times, rate, fps, _ = _motion()
    imu_t, pose_norm, vibe_norm, written_t, written_norm = _reconstruct()
    payload = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    bounds = tuple(payload["imu_motion_bounds"])
    timestamps = _read_timestamp_file(CONSTRUCTED / "timestamps.csv")
    imu_feature = _imu_fraction(CONSTRUCTED / "imu.csv", timestamps, bounds)
    ours = _rows(payload, "ours")
    ours_t0 = np.asarray([float(row["t0"]) for row in ours], dtype=np.float64)
    ours_t1 = np.asarray([float(row["t1"]) for row in ours], dtype=np.float64)
    ours_rpm = np.asarray([float(row["rpm"]) for row in ours], dtype=np.float64)
    gear_t = np.concatenate([ours_t0[:1], ours_t1])
    gear_rpm = np.concatenate([ours_rpm[:1], ours_rpm])
    switches = np.where(np.diff(ours_rpm) != 0.0)[0]
    hero_t = float(json.loads(METRICS.read_text(encoding="utf-8"))["row_b"]["t"])
    segments = _segments(times, rate, fps)
    fill = {"crawl": CRAWL, "transit": TRANSIT, "dash": DASH}
    stride = max(1, int(round(len(imu_t) / 2500)))

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
    fig, axes = plt.subplots(3, 1, figsize=(10.4, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.25, 0.9, 1.05], "hspace": 0.08})
    for axis in axes:
        for left, right, name in segments:
            axis.axvspan(left, right, color=fill[name], alpha=0.55, linewidth=0, zorder=0)
        axis.axvline(hero_t, color=MAGENTA, linewidth=0.9, linestyle="--", zorder=4)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(colors=INK, length=3, width=0.7)
        axis.yaxis.label.set_color(INK)
        axis.set_xlim(0.0, float(times[-1]))

    pose_line, = axes[0].plot(imu_t[::stride], 1000.0 * pose_norm[::stride], color=NAVY, linewidth=1.15, zorder=3)
    vibe_line, = axes[0].plot(imu_t[::stride], 1000.0 * vibe_norm[::stride], color=MUTED, linewidth=0.7, zorder=2)
    written_line, = axes[0].plot(written_t[::stride], 1000.0 * written_norm[::stride], color=TEAL, linewidth=1.35, zorder=4)
    axes[0].set_ylabel("gyro |ω| (mrad/s)")
    axes[0].set_ylim(0.0, 4.0)
    for left, right, name in segments:
        axes[0].text(0.5 * (left + right), 3.82, name, ha="center", va="top", color=INK, fontsize=8)
    fig.legend(
        [vibe_line, pose_line, written_line],
        ["road vibration", "pose-derived", "written IMU"],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        fontsize=8,
    )

    feature_t = np.empty(2 * len(imu_feature))
    feature_y = np.empty(2 * len(imu_feature))
    feature_t[0::2] = ours_t0
    feature_t[1::2] = ours_t1
    feature_y[0::2] = imu_feature
    feature_y[1::2] = imu_feature
    axes[1].fill_between(feature_t, 0.0, feature_y, color=WARM, alpha=0.28, linewidth=0, zorder=2)
    axes[1].plot(feature_t, feature_y, color=WARM, linewidth=1.2, zorder=3)
    axes[1].set_ylabel("IMU feature")
    axes[1].set_ylim(0.0, 1.08)
    axes[1].set_yticks([0.0, 0.5, 1.0])

    axes[2].hlines([180.0, 360.0], 0.0, float(times[-1]), colors=MUTED, linestyles=":", linewidth=0.7, zorder=1)
    axes[2].step(gear_t, gear_rpm, where="post", color=MAGENTA, linewidth=1.8, zorder=3)
    axes[2].scatter(ours_t1[switches], ours_rpm[switches + 1], s=18, color=MAGENTA, zorder=5, linewidths=0)
    axes[2].set_ylabel("Ours RPM")
    axes[2].set_ylim(120.0, 400.0)
    axes[2].set_yticks([180, 360])
    axes[2].set_xlabel("time (s)")
    axes[2].text(hero_t - 0.05, 372.0, "figure window", color=MAGENTA, fontsize=8, ha="right", va="bottom")

    fig.align_ylabels(axes)
    fig.savefig(args.output_dir / "imu_gear.png", dpi=220, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(args.output_dir / "imu_gear.pdf", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    stats = {
        "pose_gyro_median_mrad_s": float(1000.0 * np.median(pose_norm)),
        "vibration_median_mrad_s": float(1000.0 * np.median(vibe_norm)),
        "written_median_mrad_s": float(1000.0 * np.median(written_norm)),
        "imu_feature_mean": float(imu_feature.mean()),
        "n_gear_switches": int(len(switches)),
        "frac_360": float(np.mean(ours_rpm >= 359)),
        "frac_180": float(np.mean(np.abs(ours_rpm - 180.0) < 1.0)),
    }
    (args.output_dir / "imu_gear.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(args.output_dir / "imu_gear.png"), **stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
