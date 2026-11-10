#!/usr/bin/env python3
"""Evaluate fixed versus calibrated adaptive policies over multiple time clips."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

from ami_active_sensing.control.trajectory import TrajectoryLimits, make_trajectory
from ami_active_sensing.data import discover_eveyes_scenes, load_scene_imu, load_scene_rgb
from ami_active_sensing.evaluation.adaptive_rollout import rollout_adaptive_controller
from ami_active_sensing.evaluation.trajectory_sweep import SweepConfig, evaluate_trajectory
from run_scene_motion_sweep import trajectory_grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("NEW/results/eveyes_policy_stability.json"))
    parser.add_argument("--frame-count", type=int, default=31)
    parser.add_argument("--clip-count", type=int, default=7)
    parser.add_argument("--clip-stride", type=int, default=60)
    parser.add_argument("--window-count", type=int, default=3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--useful-speed-cap", type=float, default=8.0)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.frame_count, args.clip_count, args.clip_stride, args.window_count, args.width, args.height) < 2 or args.useful_speed_cap <= 0.0:
        raise SystemExit("clip and image dimensions must be positive")
    torch.set_num_threads(args.torch_threads)
    rows: list[dict[str, object]] = []
    for scene in discover_eveyes_scenes(args.dataset, split="test"):
        max_start = len(scene.rgb_paths) - args.frame_count
        starts = [min(index * args.clip_stride, max_start) for index in range(args.clip_count)]
        starts = list(dict.fromkeys(starts))
        for start in starts:
            indices = np.arange(start, start + args.frame_count, dtype=np.int64)
            frames = load_scene_rgb(scene, indices)
            frames = np.stack([cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA) for frame in frames])
            timestamps = scene.timestamps[indices]
            edge_indices = np.linspace(0, args.frame_count - 1, args.window_count + 1, dtype=np.int64)
            edges = timestamps[edge_indices]
            window_duration = float(np.mean(np.diff(edges)))
            config = SweepConfig(prism_angle_deg=1.0, contrast_threshold=0.15, event_window_seconds=window_duration, time_surface_decay_seconds=window_duration / 3.0, temporal_bins=5)
            grid = trajectory_grid(float(edges[-1] - edges[0]))
            candidates = [replace(parameters, duration_s=window_duration) for parameters in grid.values()]
            limits = TrajectoryLimits(40.0, 500.0, min_duration_s=window_duration, max_duration_s=window_duration)
            aggregates: dict[str, dict[str, object]] = {}
            for name in ("constant_0", "constant_6", "constant_30"):
                parameters = replace(grid[name], duration_s=float(edges[-1] - edges[0]))
                report = evaluate_trajectory(name, frames, timestamps, scene.intrinsics.resized(args.width, args.height), make_trajectory(parameters, start_time=float(timestamps[0])), edges, config=config)
                aggregates[name] = report["aggregate"]
            adaptive = rollout_adaptive_controller(frames, timestamps, scene.intrinsics.resized(args.width, args.height), edges, candidates, limits, config=config, initial_index=1, imu_stream=load_scene_imu(scene), useful_speed_cap_rad_s=args.useful_speed_cap)
            aggregates["adaptive"] = adaptive["aggregate"]
            baseline = aggregates["constant_6"]
            for policy, aggregate in aggregates.items():
                rows.append({
                    "sequence_id": scene.sequence_id,
                    "motion_regime": scene.motion_regime,
                    "clip_start_frame": int(start),
                    "policy": policy,
                    "event_count": int(aggregate["event_count"]),
                    "mean_event_density": float(aggregate["mean_event_density"]),
                    "mean_edge_coverage": float(aggregate["mean_edge_coverage"]),
                    "mean_edge_f1": float(aggregate["mean_edge_f1"]),
                    "mean_event_efficiency": float(aggregate["mean_event_efficiency"]),
                    "mean_reference_valid_fraction": float(aggregate["mean_reference_valid_fraction"]),
                    "adaptive_minus_constant6_density": float(aggregate["mean_event_density"] - baseline["mean_event_density"]) if policy == "adaptive" else None,
                    "adaptive_minus_constant6_efficiency": float(aggregate["mean_event_efficiency"] - baseline["mean_event_efficiency"]) if policy == "adaptive" else None,
                    "selected_indices": [int(item["selected_index"]) for item in adaptive["windows"]] if policy == "adaptive" else None,
                })
            print(json.dumps({"sequence_id": scene.sequence_id, "clip_start_frame": start, "adaptive_selected": [int(item["selected_index"]) for item in adaptive["windows"]]}), flush=True)
    adaptive_rows = [row for row in rows if row["policy"] == "adaptive"]
    result = {
        "status": "synthetic_rendered_policy_stability_not_task_evidence",
        "provenance": "Blender-rendered RGB with trajectory-derived deterministic IMU",
        "dataset": str(args.dataset),
        "configuration": {"frame_count": args.frame_count, "clip_count": args.clip_count, "clip_stride": args.clip_stride, "window_count": args.window_count, "width": args.width, "height": args.height, "useful_speed_cap_rad_s": args.useful_speed_cap},
        "aggregate": {
            "clip_count": len(adaptive_rows),
            "adaptive_minus_constant6_density_mean": float(np.mean([float(row["adaptive_minus_constant6_density"]) for row in adaptive_rows])),
            "adaptive_minus_constant6_density_median": float(np.median([float(row["adaptive_minus_constant6_density"]) for row in adaptive_rows])),
            "adaptive_minus_constant6_efficiency_mean": float(np.mean([float(row["adaptive_minus_constant6_efficiency"]) for row in adaptive_rows])),
            "adaptive_minus_constant6_efficiency_positive_clips": int(sum(float(row["adaptive_minus_constant6_efficiency"]) > 0.0 for row in adaptive_rows)),
        },
        "rows": rows,
        "note": "This is a synthetic rendered stability check; no real-world or correspondence-based task claim is made.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["aggregate"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
