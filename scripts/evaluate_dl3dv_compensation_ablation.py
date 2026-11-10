#!/usr/bin/env python3
"""DL3DV timestamp compensation ablation on identical AMI event streams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ami_simulator.event_quality import compute_reference_edge_mask, event_quality_metrics
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import EventStream
from scripts.evaluate_dl3dv_downstream import _intrinsics
from scripts.evaluate_phase_continuous_control import _simulate_action


def _constant_compensation(events: EventStream, prism: RotatingWedgePrism, theta: float) -> EventStream:
    if not len(events):
        return EventStream.empty()
    points = torch.from_numpy(np.stack([events.x, events.y], axis=1).astype(np.float32))
    reference = prism.inverse_points(points, float(theta)).detach().cpu().numpy()
    return EventStream(np.rint(reference[:, 0]), np.rint(reference[:, 1]), events.t, events.p)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/dl3dv_large5"))
    parser.add_argument("--scenes", nargs="*", default=("screen-000083", "screen-000087", "screen-000089", "screen-000095", "screen-000108"))
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--eval-width", type=int, default=160)
    parser.add_argument("--rpm", type=float, default=180.0)
    parser.add_argument("--prism-angle-deg", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/dl3dv_compensation_ablation.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    rows = []
    for scene_name in args.scenes:
        scene = args.dataset_root / scene_name
        frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=args.max_frames)
        timestamps = timestamps[: len(frames)]
        intrinsics = _intrinsics(scene)
        if args.eval_width != intrinsics.width:
            height = int(round(intrinsics.height * args.eval_width / intrinsics.width))
            frames = np.stack([cv2.resize(frame, (args.eval_width, height), interpolation=cv2.INTER_AREA) for frame in frames])
            intrinsics = intrinsics.resized(args.eval_width, height)
        luminance = rgb_to_linear_luminance(frames)
        valid = np.ones((len(frames), intrinsics.height, intrinsics.width), dtype=bool)
        prism = RotatingWedgePrism(intrinsics, args.prism_angle_deg)
        phase = 0.0
        previous_active = {"no_compensation": None, "interval_compensation": None, "per_event_compensation": None}
        previous_reference = None
        for index in range(len(timestamps) - 1):
            result = _simulate_action(
                luminance, timestamps, valid, prism, index=index, rpm=args.rpm, phase=phase,
                previous_active=previous_active["per_event_compensation"], previous_rpm=args.rpm,
                target_density=0.2,
                objective={"event_efficiency_weight": 0.0, "overlap_weight": 0.0, "density_weight": 0.0, "invalid_fraction_weight": 0.0, "rpm_effort_weight": 0.0, "rpm_change_weight": 0.0},
                contrast_threshold=0.2, max_phase_step_deg=30.0, sensor_reference_log=previous_reference, device=args.device,
            )
            start, end = float(timestamps[index]), float(timestamps[index + 1])
            theta_mid = phase + float(args.rpm) * 2.0 * np.pi / 60.0 * (end - start) * 0.5
            streams = {
                "no_compensation": result["sensor_events"],
                "interval_compensation": _constant_compensation(result["sensor_events"], prism, theta_mid),
                "per_event_compensation": result["reference_events"],
            }
            edge = compute_reference_edge_mask(luminance[index], edge_quantile=0.75)
            for method, events in streams.items():
                metrics = event_quality_metrics(events, reference_edge_mask=edge, previous_active_pixels=previous_active[method], image_shape=luminance.shape[1:])
                rows.append({"scene": scene_name, "window": index, "method": method, **{key: float(value) for key, value in metrics.items() if isinstance(value, (float, int, np.floating, np.integer))}})
                active = events.y.astype(np.int64) * intrinsics.width + events.x.astype(np.int64)
                previous_active[method] = np.unique(active)
            phase = float(result["phase_end"])
            previous_reference = np.asarray(result["sensor_reference_log"])
        print(f"evaluated {scene_name}", flush=True)
    summary = {}
    for method in ("no_compensation", "interval_compensation", "per_event_compensation"):
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {key: float(np.mean([row[key] for row in selected])) for key in ("edge_f1", "edge_precision", "edge_coverage", "event_count", "unique_pixel_ratio", "temporal_overlap")}
    report = {"protocol": "ami_dl3dv_compensation_ablation_v1", "dataset": "DL3DV-large5", "scenes": list(args.scenes), "max_frames": args.max_frames, "eval_width": args.eval_width, "rpm": args.rpm, "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
