#!/usr/bin/env python3
"""Render a paper figure for the constructed FULL_SWITCH EVEYES sequence.

The figure combines the sensor-clock schedule, official AMI scores, and a
polarity event triptych from the 7.5 FPS storm segment. The event maps are
recomputed from the native 960x540 sequence; no score rows are reused for
visualization.
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
SEGMENT_COLORS = ("#e8eef7", "#eaf4ea", "#fff2df", "#f0eaf7")


def _read_schedule(scene: Path, n_windows: int) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    rows = list(csv.DictReader((scene / "motion_state.csv").open(encoding="utf-8")))
    fps = np.asarray([float(row["sensor_fps"]) for row in rows], dtype=np.float64)
    labels = _segments(scene, n_windows)
    runs: list[tuple[int, int, float]] = []
    start = 0
    while start < n_windows:
        label = labels[start]
        end = start + 1
        while end < n_windows and labels[end] == label:
            end += 1
        runs.append((start, end, float(np.median(fps[start:end]))))
        start = end
    return labels, runs


def _aggregate(values: list[dict], segment: str, key: str) -> float:
    selected = [row for row in values if row["segment"] == segment]
    return float(np.mean([float(row[key]) for row in selected]))


def _event_count_maps(streams: list[object], start: int, end: int, height: int, width: int) -> np.ndarray:
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


def _add_segment_bands(ax, runs: list[tuple[int, int, float]], labels: list[str] | None = None) -> None:
    for index, (start, end, fps) in enumerate(runs):
        ax.axvspan(start, end, color=SEGMENT_COLORS[index % len(SEGMENT_COLORS)], alpha=0.75, lw=0)
        if index:
            ax.axvline(start, color="#7f7f7f", lw=0.65, ls="--")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("data/eveyes_comotion_v2_ready/test/Scene_005/full_switch"))
    parser.add_argument("--evaluation", type=Path, default=Path("results/ami_next/constructed_eval_full_switch.json"))
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/figures/fig_full_switch_paper"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    payload = json.loads(args.evaluation.read_text(encoding="utf-8"))
    frames, timestamps = load_rgb_sequence(args.scene / "images", args.scene / "timestamps.csv", timestamp_unit="s")
    luminance = rgb_to_linear_luminance(frames)
    intrinsics = _intrinsics(args.scene)
    prism = RotatingWedgePrism(intrinsics, 0.5)
    labels, runs = _read_schedule(args.scene, len(frames) - 1)
    storm_index = next(index for index, (_, _, fps) in enumerate(runs) if fps <= 10.0)
    storm_start, storm_end, storm_fps = runs[storm_index]
    height, width = intrinsics.height, intrinsics.width
    valid = np.ones((len(frames) - 1, height, width), dtype=bool)

    streams_by_gear: dict[str, list[object]] = {}
    for method, rpm, _, _ in GEARS:
        if rpm == 0.0:
            streams_by_gear[method], _ = _dvs_windows(luminance, timestamps)
        else:
            streams_by_gear[method], _, _ = _ami_windows(luminance, timestamps, valid, prism, rpm, device=args.device)
        print(f"computed {method}", flush=True)

    maps = {
        method: _event_polarity_map(streams, storm_start, storm_end, height, width)
        for method, streams in streams_by_gear.items()
    }

    fig = plt.figure(figsize=(12.0, 7.2), dpi=300)
    grid = fig.add_gridspec(2, 6, height_ratios=(1.04, 1.42), hspace=0.48, wspace=0.42)
    ax_schedule = fig.add_subplot(grid[0, :2])
    ax_score = fig.add_subplot(grid[0, 2:4])
    ax_quality = fig.add_subplot(grid[0, 4:])

    x = np.arange(len(labels)) + 0.5
    fps_per_window = np.zeros(len(labels), dtype=float)
    for start, end, fps in runs:
        fps_per_window[start:end] = fps
    ax_schedule.step(x, fps_per_window, where="mid", color="#111827", lw=2.0)
    ax_schedule.set_yscale("log")
    ax_schedule.set_yticks([7.5, 60, 480])
    ax_schedule.set_yticklabels(["7.5", "60", "480"])
    ax_schedule.set_ylabel("Sensor clock (FPS)")
    ax_schedule.text(
        0.02,
        0.08,
        "s0: 480   s1: 60   s2: 7.5   s3: 60 FPS",
        transform=ax_schedule.transAxes,
        fontsize=7.2,
        color="#111827",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.86, "pad": 2.5},
    )
    ax_schedule.set_xlabel("Contiguous segment / evaluation window")
    ax_schedule.set_xlim(0, len(labels))
    ax_schedule.grid(axis="y", alpha=0.25, lw=0.5)
    ax_schedule.spines["top"].set_visible(False)
    ax_schedule.spines["right"].set_visible(False)
    _add_segment_bands(ax_schedule, runs)
    ax_schedule.set_title("(a) Clock schedule", loc="left", fontsize=10, fontweight="semibold")

    segment_names = [f"s{i}" for i in range(len(runs))]
    xseg = np.arange(len(segment_names))
    for method, _, label, color in GEARS:
        rows = payload["per_gear"][method]
        y = [_aggregate(rows, segment, "edge_f1") for segment in segment_names]
        ax_score.plot(xseg, y, marker="o", lw=1.8, ms=4.5, label=label, color=color)
    ax_score.set_xticks(xseg, [f"{fps:g}" for _, _, fps in runs])
    ax_score.set_xlabel("Sensor clock (FPS)")
    ax_score.set_ylabel("Image-wide edge F1")
    ax_score.set_ylim(0.18, 0.66)
    ax_score.grid(axis="y", alpha=0.25, lw=0.5)
    ax_score.spines["top"].set_visible(False)
    ax_score.spines["right"].set_visible(False)
    ax_score.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax_score.set_title("(b) Image-wide edge quality", loc="left", fontsize=10, fontweight="semibold")

    for method, _, label, color in GEARS:
        rows = payload["per_gear"][method]
        y = [_aggregate(rows, segment, "object_edge_f1") for segment in segment_names]
        ax_quality.plot(xseg, y, marker="o", lw=1.8, ms=4.5, label=label, color=color)
    ax_quality.set_xticks(xseg, [f"{fps:g}" for _, _, fps in runs])
    ax_quality.set_xlabel("Sensor clock (FPS)")
    ax_quality.set_ylabel("Object-edge F1")
    ax_quality.set_ylim(0.10, 0.28)
    ax_quality.grid(axis="y", alpha=0.25, lw=0.5)
    ax_quality.spines["top"].set_visible(False)
    ax_quality.spines["right"].set_visible(False)
    ax_quality.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax_quality.set_title("(c) Target quality", loc="left", fontsize=10, fontweight="semibold")

    panel_axes = []
    for col, (method, _, label, color) in enumerate(GEARS):
        ax = fig.add_subplot(grid[1, col * 2 : col * 2 + 2])
        panel_axes.append(ax)
        ax.imshow(maps[method], interpolation="nearest")
        rows = payload["per_gear"][method]
        storm_rows = [row for row in rows if row["segment"] == f"s{storm_index}"]
        event_count = sum(int(row["event_count"]) for row in storm_rows)
        occupancy = float(np.mean([float(row["unique_pixel_ratio"]) for row in storm_rows]))
        ax.set_title(label, color=color, fontsize=10, fontweight="semibold", pad=5)
        ax.set_xlabel(f"{event_count / 1e6:.2f}M events  ·  occupancy {occupancy:.3f}", fontsize=8, color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.4)
        ax.text(0.03, 0.96, f"{storm_fps:g} FPS storm segment", transform=ax.transAxes, color="white", fontsize=7.5, va="top", bbox={"facecolor": color, "edgecolor": "none", "pad": 2.5, "alpha": 0.9})

    fig.suptitle("FULL_SWITCH: sensor-clock changes expose distinct optimal prism speeds", fontsize=13, fontweight="semibold", y=0.985)
    fig.text(0.5, 0.015, "Panels (d–f) show the same 7.5 FPS interval; red and blue encode positive and negative events.", ha="center", fontsize=8, color="#374151")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")

    triptych, trip_axes = plt.subplots(1, 3, figsize=(11.2, 3.8), dpi=300, constrained_layout=True)
    for ax, (method, _, label, color) in zip(trip_axes, GEARS):
        ax.imshow(maps[method], interpolation="nearest")
        rows = payload["per_gear"][method]
        storm_rows = [row for row in rows if row["segment"] == f"s{storm_index}"]
        events = sum(int(row["event_count"]) for row in storm_rows)
        occ = float(np.mean([float(row["unique_pixel_ratio"]) for row in storm_rows]))
        ax.set_title(label, color=color, fontsize=10, fontweight="semibold")
        ax.set_xlabel(f"{events / 1e6:.2f}M events; occ. {occ:.3f}", fontsize=8, color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.4)
    triptych.suptitle(f"FULL_SWITCH storm segment ({storm_fps:g} FPS): fixed 180 over-fires", fontsize=11, fontweight="semibold")
    triptych.text(0.5, 0.012, "Positive events: red  ·  negative events: blue  ·  black: no events",
                  ha="center", va="bottom", fontsize=8, color="#374151")
    triptych.savefig(args.output.with_name(args.output.name + "_storm_triptych.png"), dpi=300, bbox_inches="tight")
    triptych.savefig(args.output.with_name(args.output.name + "_storm_triptych.pdf"), bbox_inches="tight")
    plt.close(fig)
    plt.close(triptych)

    stats = {
        "figure_png": str(args.output.with_suffix(".png")),
        "figure_pdf": str(args.output.with_suffix(".pdf")),
        "storm_triptych_png": str(args.output.with_name(args.output.name + "_storm_triptych.png")),
        "storm_triptych_pdf": str(args.output.with_name(args.output.name + "_storm_triptych.pdf")),
        "scene": str(args.scene),
        "storm_segment": {"index": storm_index, "start_window": storm_start, "end_window": storm_end, "sensor_fps": storm_fps},
        "event_map_palette": {"background": "#030712", "positive": "#ff0000", "negative": "#0000ff"},
        "two_sided_ok": bool(payload["two_sided_ok"]),
        "best_360": payload["best_360_segment"],
        "best_0": payload["best_0_segment"],
    }
    args.output.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
