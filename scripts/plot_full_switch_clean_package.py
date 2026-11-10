#!/usr/bin/env python3
"""Create clean, screenshot-ready FULL_SWITCH paper figures.

Outputs separate event, motion/IMU, and 360-vs-180 figures. Event panels use
the shared red/blue polarity rendering so zero-event pixels remain black.
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
import torch

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import EventStream
from scripts.evaluate_constructed_by_segment import _segments
from scripts.evaluate_dl3dv_downstream import _ami_windows, _dvs_windows
from scripts.evaluate_eveyes_v2_downstream import _intrinsics
from scripts.render_eveyes_camera_nfb_event_video import _events_to_map


GEARS = (
    ("ami_passive", 0.0, "0 RPM", "#4d4d4d"),
    ("ami_fixed_180", 180.0, "180 RPM", "#1f77b4"),
    ("ami_fixed_360", 360.0, "360 RPM", "#d95f02"),
)
FPS_BANDS = ("#e8eef7", "#eaf4ea", "#fff2df", "#f0eaf7")


def _runs(scene: Path, n_windows: int) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    rows = list(csv.DictReader((scene / "motion_state.csv").open(encoding="utf-8")))
    fps = np.asarray([float(row["sensor_fps"]) for row in rows], dtype=np.float64)
    labels = _segments(scene, n_windows)
    result: list[tuple[int, int, float]] = []
    start = 0
    while start < n_windows:
        label = labels[start]
        end = start + 1
        while end < n_windows and labels[end] == label:
            end += 1
        result.append((start, end, float(np.median(fps[start:end]))))
        start = end
    return labels, result


def _shade(ax, runs: list[tuple[int, int, float]], *, xscale: float = 1.0) -> None:
    for index, (start, end, _) in enumerate(runs):
        ax.axvspan(start * xscale, end * xscale, color=FPS_BANDS[index % len(FPS_BANDS)], alpha=0.7, lw=0, zorder=0)
        if index:
            ax.axvline(start * xscale, color="#8b8b8b", lw=0.65, ls="--", zorder=1)


def _event_counts(streams: list[object], start: int, end: int, height: int, width: int) -> np.ndarray:
    counts = np.zeros((height, width), dtype=np.float32)
    for stream in streams[start:end]:
        if not len(stream):
            continue
        x = np.rint(np.asarray(stream.x)).astype(np.int64)
        y = np.rint(np.asarray(stream.y)).astype(np.int64)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        np.add.at(counts, (y[valid], x[valid]), 1.0)
    return counts


def _event_polarity_map(streams: list[object], start: int, end: int, height: int, width: int) -> np.ndarray:
    selected = [stream for stream in streams[start:end] if len(stream)]
    if not selected:
        return np.zeros((height, width, 3), dtype=np.uint8)
    events = EventStream(
        np.concatenate([np.asarray(stream.x) for stream in selected]),
        np.concatenate([np.asarray(stream.y) for stream in selected]),
        np.concatenate([np.asarray(stream.t) for stream in selected]),
        np.concatenate([np.asarray(stream.p) for stream in selected]),
    )
    return _events_to_map(events, height, width, storm=False)


def _load_csv(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _mean_segment(rows: list[dict], segment: str, key: str) -> float:
    selected = [row for row in rows if row["segment"] == segment]
    return float(np.mean([float(row[key]) for row in selected]))


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _clean_event_figure(output: Path, maps_by_segment: dict[int, dict[str, np.ndarray]], payload: dict, runs: list[tuple[int, int, float]], segment_indices: list[int], vmax: float) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.3), dpi=300, constrained_layout=True)
    for row_index, segment_index in enumerate(segment_indices):
        for col_index, (method, _, label, color) in enumerate(GEARS):
            ax = axes[row_index, col_index]
            ax.imshow(maps_by_segment[segment_index][method], interpolation="nearest")
            rows = payload["per_gear"][method]
            event_rows = [item for item in rows if item["segment"] == f"s{segment_index}"]
            count = sum(int(item["event_count"]) for item in event_rows)
            occupancy = float(np.mean([float(item["unique_pixel_ratio"]) for item in event_rows]))
            ax.set_title(label, fontsize=11, color=color, fontweight="semibold", pad=5)
            ax.set_xlabel(f"{count / 1e6:.2f}M events | occ. {occupancy:.3f}", fontsize=8.5, color=color, labelpad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(1.4)
        axes[row_index, 0].set_ylabel(f"s{segment_index}: {runs[segment_index][2]:g} FPS", fontsize=9, labelpad=8)
    fig.suptitle("CAMERA_PASSBY_COMPLETE: normal 360/180 comparison and 180 storm response", fontsize=13, fontweight="semibold")
    fig.text(0.5, 0.018, "Positive events: red  ·  negative events: blue  ·  black: no events",
             ha="center", va="bottom", fontsize=8.5, color="#374151")
    _save(fig, output)


def _motion_figure(output: Path, scene: Path, runs: list[tuple[int, int, float]]) -> None:
    motion = _load_csv(scene / "motion_state.csv")
    imu = _load_csv(scene / "imu.csv")
    frame_t = np.asarray([row["timestamp_s"] for row in motion])
    speed = np.asarray([row["speed_m_s"] for row in motion])
    longitudinal = np.asarray([row["longitudinal_accel_m_s2"] for row in motion])
    imu_t = np.asarray([row["timestamp_s"] for row in imu])
    gyro = np.stack([[row["gyro_x_rad_s"], row["gyro_y_rad_s"], row["gyro_z_rad_s"]] for row in imu])
    gyro_norm = np.asarray([row["gyro_norm_rad_s"] for row in imu])
    accel = np.stack([[row["accel_x_m_s2"], row["accel_y_m_s2"], row["accel_z_m_s2"]] for row in imu])
    fps = np.asarray([row["sensor_fps"] for row in motion])
    prism_sweep = 1080.0 / np.maximum(fps, 1e-9)
    time_runs = [
        (float(frame_t[start]), float(frame_t[min(end, len(frame_t) - 1)]), sensor_fps)
        for start, end, sensor_fps in runs
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4), dpi=300, sharex=False)
    ax = axes[0, 0]
    ax.plot(frame_t, speed, color="#111827", lw=1.8, label="speed")
    ax.set_ylabel("Speed (m/s)")
    ax.set_xlabel("Sequence time (s)")
    ax.set_title("(a) Translational motion", loc="left", fontweight="semibold")
    ax.grid(alpha=0.25, lw=0.5)
    ax2 = ax.twinx()
    ax2.plot(frame_t, longitudinal, color="#d95f02", lw=1.2, label="longitudinal acceleration")
    ax2.set_ylabel("Longitudinal accel. (m/s²)", color="#d95f02")
    ax2.tick_params(axis="y", colors="#d95f02")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    ax = axes[0, 1]
    for idx, color in enumerate(("#1f77b4", "#2ca02c", "#d95f02")):
        ax.plot(imu_t, gyro[:, idx], color=color, lw=0.9, label=("x", "y", "z")[idx])
    ax.plot(imu_t, gyro_norm, color="#111827", lw=1.5, label="norm")
    ax.set_title("(b) Gyroscope", loc="left", fontweight="semibold")
    ax.set_ylabel("Angular rate (rad/s)")
    ax.set_xlabel("Sequence time (s)")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper right")
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)

    ax = axes[1, 0]
    for idx, color in enumerate(("#1f77b4", "#2ca02c", "#d95f02")):
        ax.plot(imu_t, accel[:, idx], color=color, lw=0.9, label=("x", "y", "z")[idx])
    ax.axhline(9.81, color="#111827", lw=0.9, ls="--", label="gravity")
    ax.set_title("(c) Accelerometer", loc="left", fontweight="semibold")
    ax.set_ylabel("Specific force (m/s²)")
    ax.set_xlabel("Sequence time (s)")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper right")
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)

    ax = axes[1, 1]
    ax.step(frame_t, fps, where="post", color="#1f77b4", lw=1.8, label="sensor FPS")
    ax.set_yscale("log")
    ax.set_yticks([7.5, 60, 480], ["7.5", "60", "480"])
    ax.set_ylabel("Sensor clock (FPS)")
    ax.set_xlabel("Sequence time (s)")
    ax2 = ax.twinx()
    ax2.step(frame_t, prism_sweep, where="post", color="#d95f02", lw=1.5, label="180 RPM sweep")
    ax2.set_ylabel("180 RPM sweep (deg/window)", color="#d95f02")
    ax2.tick_params(axis="y", colors="#d95f02")
    ax.set_title("(d) Clock-to-prism coupling", loc="left", fontweight="semibold")
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    for axis in axes.ravel():
        _shade(axis, time_runs)
    fig.suptitle("CAMERA_PASSBY_COMPLETE motion and reconstructed IMU", fontsize=13, fontweight="semibold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, output)


def _comparison_figure(output: Path, payload: dict, runs: list[tuple[int, int, float]]) -> None:
    segments = [f"s{i}" for i in range(len(runs))]
    x = np.arange(len(segments))
    rows_180 = payload["per_gear"]["ami_fixed_180"]
    rows_360 = payload["per_gear"]["ami_fixed_360"]
    rows_0 = payload["per_gear"]["ami_passive"]
    edge_180 = np.asarray([_mean_segment(rows_180, segment, "edge_f1") for segment in segments])
    edge_360 = np.asarray([_mean_segment(rows_360, segment, "edge_f1") for segment in segments])
    obj_180 = np.asarray([_mean_segment(rows_180, segment, "object_edge_f1") for segment in segments])
    obj_0 = np.asarray([_mean_segment(rows_0, segment, "object_edge_f1") for segment in segments])
    event_180 = np.asarray([_mean_segment(rows_180, segment, "event_count") for segment in segments])
    event_360 = np.asarray([_mean_segment(rows_360, segment, "event_count") for segment in segments])
    delta_edge = edge_360 - edge_180
    delta_obj = obj_0 - obj_180

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.3), dpi=300, constrained_layout=True)
    width = 0.34
    ax = axes[0]
    ax.bar(x - width / 2, edge_180, width, color="#1f77b4", label="180 RPM")
    ax.bar(x + width / 2, edge_360, width, color="#d95f02", label="360 RPM")
    ax.set_xticks(x, [f"{fps:g}" for _, _, fps in runs])
    ax.set_xlabel("Sensor clock (FPS)")
    ax.set_ylabel("Image-wide edge F1")
    ax.set_ylim(0.2, 0.66)
    ax.set_title("(a) 360 vs 180", loc="left", fontweight="semibold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.axhline(0.0, color="#111827", lw=0.8)
    colors = ["#d95f02" if value >= 0 else "#9ca3af" for value in delta_edge]
    ax.bar(x - width / 2, delta_edge, width, color=colors, label="360 - 180 image F1")
    ax.bar(x + width / 2, delta_obj, width, color="#2ca02c", alpha=0.85, label="0 - 180 object F1")
    ax.set_xticks(x, [f"{fps:g}" for _, _, fps in runs])
    ax.set_xlabel("Sensor clock (FPS)")
    ax.set_ylabel("F1 difference")
    ax.set_ylim(-0.27, 0.13)
    ax.set_title("(b) Advantage over fixed 180", loc="left", fontweight="semibold")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for index, value in enumerate(delta_edge):
        ax.text(index - width / 2, value + (0.008 if value >= 0 else -0.012), f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)

    ax = axes[2]
    ax.plot(x, event_180 / 1e6, marker="o", lw=2.0, color="#1f77b4", label="180 RPM")
    ax.plot(x, event_360 / 1e6, marker="o", lw=2.0, color="#d95f02", label="360 RPM")
    ax.set_xticks(x, [f"{fps:g}" for _, _, fps in runs])
    ax.set_xlabel("Sensor clock (FPS)")
    ax.set_ylabel("Mean events per window (M)")
    ax.set_title("(c) Event bandwidth", loc="left", fontweight="semibold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("CAMERA_PASSBY_COMPLETE: 360 wins normal clock; 0 suppresses the storm", fontsize=13, fontweight="semibold")
    _save(fig, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("data/eveyes_comotion_v2_ready/test/Scene_005/full_switch"))
    parser.add_argument("--evaluation", type=Path, default=Path("results/ami_next/constructed_eval_full_switch.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ami_next/figures/full_switch_clean"))
    parser.add_argument("--event-segment", type=int, default=None, help="Contiguous storm segment index; normal segment is the first segment.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    payload = json.loads(args.evaluation.read_text(encoding="utf-8"))
    frames, timestamps = load_rgb_sequence(args.scene / "images", args.scene / "timestamps.csv", timestamp_unit="s")
    luminance = rgb_to_linear_luminance(frames)
    intrinsics = _intrinsics(args.scene)
    prism = RotatingWedgePrism(intrinsics, 0.5)
    labels, runs = _runs(args.scene, len(frames) - 1)
    valid = np.ones((len(frames) - 1, intrinsics.height, intrinsics.width), dtype=bool)
    event_index = args.event_segment
    if event_index is None:
        event_index = next((index for index, (_, _, fps) in enumerate(runs) if fps <= 10.0), len(runs) - 1)
    if event_index < 0 or event_index >= len(runs):
        raise SystemExit(f"event segment {event_index} is outside 0..{len(runs) - 1}")
    normal_index = 0
    event_start, event_end, _ = runs[event_index]

    streams: dict[str, list[object]] = {}
    for method, rpm, _, _ in GEARS:
        if rpm == 0.0:
            streams[method], _ = _dvs_windows(luminance, timestamps)
        else:
            streams[method], _, _ = _ami_windows(luminance, timestamps, valid, prism, rpm, device=args.device)
        print(f"computed {method}", flush=True)
    maps_by_segment = {}
    for segment_index in (normal_index, event_index):
        start, end, _ = runs[segment_index]
        maps_by_segment[segment_index] = {method: _event_polarity_map(streams[method], start, end, intrinsics.height, intrinsics.width) for method, _, _, _ in GEARS}
    vmax = max(float(np.max(image)) for maps in maps_by_segment.values() for image in maps.values())

    event_path = args.output_dir / "fig_full_switch_events_clean"
    motion_path = args.output_dir / "fig_full_switch_imu_motion"
    comparison_path = args.output_dir / "fig_full_switch_360_vs_180"
    _clean_event_figure(event_path, maps_by_segment, payload, runs, [normal_index, event_index], vmax)
    _motion_figure(motion_path, args.scene, runs)
    _comparison_figure(comparison_path, payload, runs)

    stats = {
        "scene": str(args.scene),
        "evaluation": str(args.evaluation),
        "event_figure": str(event_path),
        "motion_figure": str(motion_path),
        "comparison_figure": str(comparison_path),
        "normal_segment": {"index": normal_index, "start_window": runs[normal_index][0], "end_window": runs[normal_index][1], "sensor_fps": runs[normal_index][2]},
        "event_segment": {"index": event_index, "start_window": event_start, "end_window": event_end, "sensor_fps": runs[event_index][2]},
        "event_map_palette": {"background": "#030712", "positive": "#ff0000", "negative": "#0000ff"},
        "two_sided_ok": bool(payload["two_sided_ok"]),
        "best_360_segment": payload["best_360_segment"],
        "best_0_segment": payload["best_0_segment"],
    }
    (args.output_dir / "figures_manifest.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
