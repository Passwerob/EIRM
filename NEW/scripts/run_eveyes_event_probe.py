#!/usr/bin/env python3
"""Run a small event-quality-only prism probe on one EveYES RGB sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ami_active_sensing.data import discover_eveyes_scenes, load_scene_imu, load_scene_rgb
from ami_active_sensing.control.trajectory import TrajectoryParameters, make_trajectory
from ami_active_sensing.evaluation.trajectory_sweep import SweepConfig, evaluate_trajectory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scene-id", default="Scene_005")
    parser.add_argument("--motion-regime", default="CAM_MOVING_OBJ_MOVING")
    parser.add_argument("--output", type=Path, default=Path("NEW/results/eveyes_event_probe.json"))
    parser.add_argument("--frame-count", type=int, default=31)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=72)
    args = parser.parse_args()
    if min(args.frame_count, args.width, args.height) < 2:
        raise SystemExit("frame-count, width, and height must be at least two")
    scenes = discover_eveyes_scenes(args.dataset, split="test")
    matches = [scene for scene in scenes if scene.scene_id == args.scene_id and scene.motion_regime == args.motion_regime]
    if len(matches) != 1:
        raise SystemExit(f"Expected one matching test sequence, found {len(matches)}")
    scene = matches[0]
    count = min(args.frame_count, len(scene.rgb_paths))
    indices = np.arange(count, dtype=np.int64)
    frames = load_scene_rgb(scene, indices)
    frames = np.stack([cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA) for frame in frames])
    intrinsics = scene.intrinsics.resized(args.width, args.height)
    timestamps = scene.timestamps[indices]
    midpoint = max(1, count // 2)
    edges = np.asarray([timestamps[0], timestamps[midpoint], timestamps[-1]], dtype=np.float64)
    config = SweepConfig(
        prism_angle_deg=1.0,
        contrast_threshold=0.15,
        event_window_seconds=float(edges[1] - edges[0]),
        time_surface_decay_seconds=0.03,
        temporal_bins=5,
    )
    duration = float(timestamps[-1] - timestamps[0])
    policies = {
        "no_prism": 0.0,
        "fixed_slow": 6.0,
        "fixed_fast": 20.0,
    }
    records: dict[str, dict[str, float | int]] = {}
    for name, speed in policies.items():
        parameters = TrajectoryParameters("constant", angular_speed_rad_s=speed, duration_s=duration)
        report = evaluate_trajectory(name, frames, timestamps, intrinsics, make_trajectory(parameters, start_time=float(timestamps[0])), edges, config=config)
        aggregate = report["aggregate"]
        records[name] = {key: aggregate[key] for key in ("event_count", "mean_event_density", "mean_edge_coverage", "mean_event_efficiency", "mean_temporal_diversity", "mean_reference_valid_fraction")}
    result = {
        "status": "synthetic_rendered_rgb_event_quality_probe_not_task_evidence",
        "provenance": "Blender-rendered RGB; IMU and motion are synthetic",
        "sequence_id": scene.sequence_id,
        "dataset": str(args.dataset),
        "frame_count": count,
        "resolution": [args.width, args.height],
        "duration_s": duration,
        "imu_sample_count": len(load_scene_imu(scene).t),
        "event_quality": records,
        "note": "No correspondence/depth is supplied; ORB matching is intentionally not used as a task claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sequence_id": scene.sequence_id, "event_quality": records}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
