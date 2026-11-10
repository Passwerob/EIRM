#!/usr/bin/env python3
"""Sweep prism RPM and report useful-event quality instead of event count alone."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ami_simulator.event_quality import active_pixel_ids, event_quality_metrics, event_quality_score
from ami_simulator.events import ThresholdEventSimulator
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.rotation import ConstantRotation
from ami_simulator.types import CameraIntrinsics, EventStream
from ami_simulator.warp import PrismWarper


def _continuous_samples(luminance: np.ndarray, timestamps: np.ndarray, rotation: ConstantRotation, max_phase_step_rad: float) -> tuple[np.ndarray, np.ndarray]:
    values = [luminance[0]]
    times = [float(timestamps[0])]
    for index, (start, end) in enumerate(zip(timestamps[:-1], timestamps[1:])):
        subdivisions = max(1, int(np.ceil(abs(rotation.omega_rad_s * float(end - start)) / max_phase_step_rad)))
        for step in range(1, subdivisions + 1):
            alpha = step / subdivisions
            values.append(luminance[index] * (1.0 - alpha) + luminance[index + 1] * alpha)
            times.append(float(start + alpha * (end - start)))
    return np.stack(values).astype(np.float32), np.asarray(times, dtype=np.float64)


def _window_events(events: EventStream, start: float, end: float) -> EventStream:
    left = int(np.searchsorted(events.t, start, side="left"))
    right = int(np.searchsorted(events.t, end, side="right"))
    return EventStream(events.x[left:right], events.y[left:right], events.t[left:right], events.p[left:right])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--timestamp-unit", choices=["s", "ms", "us", "ns"], default="s")
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--rpm", type=float, nargs="+", default=[180, 360, 540, 720, 900, 1080, 1440])
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--prism-angle-deg", type=float, default=0.5)
    parser.add_argument("--contrast-threshold", type=float, default=0.2)
    parser.add_argument("--max-phase-step-deg", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(value < 0.0 for value in args.rpm) or args.max_frames < 2 or args.max_phase_step_deg <= 0.0:
        parser.error("RPM must be non-negative, max-frames must be >= 2, and max-phase-step-deg must be positive")

    frames, timestamps = load_rgb_sequence(args.rgb_dir, args.timestamps, timestamp_unit=args.timestamp_unit, max_frames=args.max_frames)
    luminance = rgb_to_linear_luminance(frames)
    height, width = luminance.shape[1:]
    intrinsics = CameraIntrinsics(width, height, args.fx, args.fy, args.cx, args.cy)
    prism = RotatingWedgePrism(intrinsics, args.prism_angle_deg)
    simulator = ThresholdEventSimulator(args.contrast_threshold)
    output_rows: list[dict[str, float | int]] = []
    phase_step_rad = np.deg2rad(args.max_phase_step_deg)

    for rpm in args.rpm:
        rotation = ConstantRotation(float(rpm))
        sampled_luminance, sampled_times = _continuous_samples(luminance, timestamps, rotation, phase_step_rad)
        warper = PrismWarper(prism, rotation, height=height, width=width, reference_time=float(timestamps[0]), padding_mode="border")
        with torch.inference_mode():
            sensor, _ = warper.warp(torch.from_numpy(sampled_luminance)[:, None], sampled_times)
        events = simulator.simulate(sensor[:, 0].numpy(), sampled_times)
        previous_active: np.ndarray | None = None
        windows: list[dict[str, float | int]] = []
        for index, (start, end) in enumerate(zip(timestamps[:-1], timestamps[1:])):
            window = _window_events(events, float(start), float(end))
            metrics = event_quality_metrics(window, reference_intensity=luminance[index], previous_active_pixels=previous_active, image_shape=(height, width))
            metrics["quality_score"] = event_quality_score(metrics)
            metrics["window_index"] = index
            windows.append(metrics)
            previous_active = active_pixel_ids(window, image_shape=(height, width))
        numeric_keys = ["event_count", "event_density", "unique_pixel_ratio", "edge_coverage", "structural_ratio", "temporal_overlap", "polarity_imbalance", "quality_score"]
        row = {"rpm": float(rpm), **{f"mean_{key}": float(np.mean([item[key] for item in windows])) for key in numeric_keys}, "internal_samples": int(len(sampled_times))}
        output_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {"resolution": {"width": width, "height": height}, "frames": len(timestamps), "window_count": len(timestamps) - 1, "max_phase_step_deg": args.max_phase_step_deg, "rows": output_rows, "output_csv": str(args.output.with_suffix('.csv'))}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
