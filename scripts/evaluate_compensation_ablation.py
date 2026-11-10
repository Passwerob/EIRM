#!/usr/bin/env python3
"""Compare sensor-plane, interval-level, and per-event prism compensation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ami_simulator.event_quality import compute_reference_edge_mask, event_quality_metrics
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import EventStream
from scripts.evaluate_eveyes_v2_downstream import _intrinsics
from scripts.evaluate_phase_continuous_control import _simulate_action


def _constant_compensation(events: EventStream, prism: RotatingWedgePrism, theta: float) -> EventStream:
    if not len(events):
        return EventStream.empty()
    points = torch.from_numpy(np.stack([events.x, events.y], axis=1).astype(np.float32))
    reference = prism.inverse_points(points, float(theta)).detach().cpu().numpy()
    return EventStream(np.rint(reference[:, 0]), np.rint(reference[:, 1]), events.t, events.p)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=Path("data/eveyes_comotion_v2_ready/test/Scene_005/camera_only"))
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/compensation_ablation_camera_only.json"))
    parser.add_argument("--max-windows", type=int, default=120)
    parser.add_argument("--rpm", type=float, default=180.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    frames, timestamps = load_rgb_sequence(args.scene / "images", args.scene / "timestamps.csv", timestamp_unit="s", max_frames=args.max_windows + 1)
    luminance = rgb_to_linear_luminance(frames)
    intrinsics = _intrinsics(args.scene)
    prism = RotatingWedgePrism(intrinsics, 0.5)
    valid = np.ones((len(frames), intrinsics.height, intrinsics.width), dtype=bool)
    phase = 0.0
    previous_active = {"sensor": None, "interval": None, "event": None}
    sensor_reference = None
    previous_rpm = float(args.rpm)
    rows = []
    for index in range(len(timestamps) - 1):
        result = _simulate_action(
            luminance,
            timestamps,
            valid,
            prism,
            index=index,
            rpm=args.rpm,
            phase=phase,
            previous_active=previous_active["event"],
            previous_rpm=previous_rpm,
            target_density=0.2,
            objective={"event_efficiency_weight": 0.0, "overlap_weight": 0.0, "density_weight": 0.0, "invalid_fraction_weight": 0.0, "rpm_effort_weight": 0.0, "rpm_change_weight": 0.0},
            contrast_threshold=0.2,
            max_phase_step_deg=30.0,
            sensor_reference_log=sensor_reference,
            device=args.device,
        )
        start, end = float(timestamps[index]), float(timestamps[index + 1])
        omega = float(args.rpm) * 2.0 * np.pi / 60.0
        theta_mid = phase + omega * (end - start) * 0.5
        streams = {
            "no_compensation": result["sensor_events"],
            "interval_compensation": _constant_compensation(result["sensor_events"], prism, theta_mid),
            "per_event_compensation": result["reference_events"],
        }
        edge = compute_reference_edge_mask(luminance[index], edge_quantile=0.75)
        for method, events in streams.items():
            metrics = event_quality_metrics(
                events,
                reference_edge_mask=edge,
                previous_active_pixels=previous_active["sensor" if method == "no_compensation" else "interval" if method == "interval_compensation" else "event"],
                image_shape=luminance.shape[1:],
            )
            rows.append({"window": index, "method": method, **{key: float(value) for key, value in metrics.items() if isinstance(value, (float, int, np.floating, np.integer))}})
            active = events.y.astype(np.int64) * intrinsics.width + events.x.astype(np.int64)
            previous_active["sensor" if method == "no_compensation" else "interval" if method == "interval_compensation" else "event"] = np.unique(active)
        phase = float(result["phase_end"])
        sensor_reference = np.asarray(result["sensor_reference_log"])
        previous_rpm = args.rpm
    summary = {}
    for method in ("no_compensation", "interval_compensation", "per_event_compensation"):
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            "edge_f1": float(np.mean([row["edge_f1"] for row in selected])),
            "edge_precision": float(np.mean([row["edge_precision"] for row in selected])),
            "edge_recall": float(np.mean([row["edge_coverage"] for row in selected])),
            "event_count": float(np.mean([row["event_count"] for row in selected])),
            "unique_pixel_ratio": float(np.mean([row["unique_pixel_ratio"] for row in selected])),
            "temporal_overlap": float(np.mean([row["temporal_overlap"] for row in selected])),
        }
    report = {"protocol": "ami_compensation_ablation_v1", "scene": str(args.scene), "windows": len(timestamps) - 1, "rpm": args.rpm, "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
