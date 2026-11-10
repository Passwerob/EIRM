#!/usr/bin/env python3
"""Clear single-window camera-passby view for the 0/180/360 storm comparison."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ami_simulator.event_visualization import eirm_counts_from_events, eirm_from_counts, eirm_peak
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from scripts.evaluate_dl3dv_downstream import _ami_windows, _dvs_windows
from scripts.evaluate_eveyes_v2_downstream import _intrinsics

METHODS = (("ami_passive", 0.0, "0 RPM", "#374151"), ("ami_fixed_180", 180.0, "180 RPM", "#1677b8"), ("ami_fixed_360", 360.0, "360 RPM", "#d45b0a"))


def _render_eirm(positive: np.ndarray, negative: np.ndarray, peak: tuple[float, float]) -> np.ndarray:
    """Render with the repository-wide EIRM palette and shared polarity scale."""
    return eirm_from_counts(positive, negative, lam=3.0, gamma=0.65, peak=peak).astype(np.float32) / 255.0


def _window_stats(payload: dict, method: str, window: int) -> dict:
    rows = [row for row in payload["per_gear"][method] if row["window"] == window]
    return rows[0] if rows else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=Path("data/eveyes_comotion_v2_ready/test/Scene_005/camera_passby_complete"))
    parser.add_argument("--evaluation", type=Path, default=Path("results/ami_next/constructed_eval_camera_passby_complete.json"))
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/figures/camera_passby_complete_clean/fig_camera_passby_0rpm_clear"))
    parser.add_argument("--segment", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    payload = json.loads(args.evaluation.read_text(encoding="utf-8"))
    frames, timestamps = load_rgb_sequence(args.scene / "images", args.scene / "timestamps.csv", timestamp_unit="s")
    motion = list(csv.DictReader((args.scene / "motion_state.csv").open(encoding="utf-8")))
    fps = np.asarray([float(row["sensor_fps"]) for row in motion])
    boundaries = [0]
    for i in range(1, len(fps)):
        if fps[i] != fps[i - 1]:
            boundaries.append(i)
    boundaries.append(len(fps) - 1)
    if args.segment >= len(boundaries) - 1:
        raise SystemExit(f"segment must be 0..{len(boundaries)-2}")
    start, end = boundaries[args.segment], boundaries[args.segment + 1]
    n_windows = len(timestamps) - 1
    # Select the event-count median window in the requested segment.
    candidate_windows = list(range(start, min(end, n_windows)))
    counts_by_method = {}
    for method, rpm, _, _ in METHODS:
        values = [float(row["event_count"]) for row in payload["per_gear"][method] if int(row["window"]) in candidate_windows]
        counts_by_method[method] = values
    median_counts = np.median(np.asarray([sum(counts_by_method[method][j] for method, _, _, _ in METHODS) for j in range(len(candidate_windows))]))
    window = candidate_windows[int(np.argmin([abs(sum(counts_by_method[method][j] for method, _, _, _ in METHODS) - median_counts) for j in range(len(candidate_windows))]))]

    luminance = rgb_to_linear_luminance(frames)
    intrinsics = _intrinsics(args.scene)
    prism = RotatingWedgePrism(intrinsics, 0.5)
    valid = np.ones((len(frames) - 1, intrinsics.height, intrinsics.width), dtype=bool)
    streams = {}
    maps = {}
    for method, rpm, label, color in METHODS:
        if rpm == 0.0:
            streams[method], _ = _dvs_windows(luminance, timestamps)
        else:
            streams[method], _, _ = _ami_windows(luminance, timestamps, valid, prism, rpm, device=args.device)
        maps[method] = eirm_counts_from_events(streams[method][window], intrinsics.height, intrinsics.width)
    peak = (
        max(eirm_peak(pos, lam=3.0) for pos, _ in maps.values()),
        max(eirm_peak(neg, lam=3.0) for _, neg in maps.values()),
    )
    rendered = {method: _render_eirm(pos, neg, peak) for method, (pos, neg) in maps.items()}
    rgb_index = min(window, len(frames) - 1)
    figure = plt.figure(figsize=(16.0, 5.0), dpi=220, facecolor="white")
    grid = figure.add_gridspec(1, 4, width_ratios=(1.1, 1, 1, 1), wspace=0.025)
    ax = figure.add_subplot(grid[0, 0]); ax.imshow(np.clip(frames[rgb_index], 0, 255).astype(np.uint8)); ax.set_title("RGB reference", fontsize=12, fontweight="semibold")
    ax.set_xticks([]); ax.set_yticks([])
    for method, rpm, label, color in METHODS:
        col = {"ami_passive": 1, "ami_fixed_180": 2, "ami_fixed_360": 3}[method]
        ax = figure.add_subplot(grid[0, col]); ax.imshow(rendered[method]); stat = _window_stats(payload, method, window)
        ax.set_title(label, color=color, fontsize=12, fontweight="semibold")
        ax.set_xlabel(f"{int(stat.get('event_count', 0)) / 1e6:.2f}M events\nocc. {float(stat.get('unique_pixel_ratio', 0.0)):.3f}", color=color, fontsize=9, labelpad=5)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_color(color); spine.set_linewidth(1.5)
    figure.suptitle(f"Camera pass-by event response · {fps[window]:g} FPS", fontsize=14, fontweight="semibold", y=0.98)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    metadata = {"scene": str(args.scene), "segment": args.segment, "window": window, "sensor_fps": float(fps[window]), "selection": "median total event count among windows in the selected segment", "display": "repository EIRM palette: background #030712, positive #37DCFF, negative #E148FF; shared per-polarity log1p(3*count) scale; gamma=0.65", "black_lines_explanation": "The previous figure aggregated multiple windows with a dark underflow display. This version keeps the established EIRM palette and uses one representative window with a shared scale; event coordinates and counts are unchanged.", "files": {"png": str(args.output.with_suffix('.png')), "pdf": str(args.output.with_suffix('.pdf'))}}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
