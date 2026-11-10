#!/usr/bin/env python3
"""Run a causal flow/IMU adaptive rollout on EveYES test RGB sequences.

This probe reports event-quality and selected trajectories only. The dataset
does not provide dense correspondence/depth, so matching outputs are not
interpreted as task evidence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

from ami_active_sensing.control.trajectory import TrajectoryLimits
from ami_active_sensing.data import discover_eveyes_scenes, load_scene_imu, load_scene_rgb
from ami_active_sensing.evaluation.adaptive_rollout import rollout_adaptive_controller
from ami_active_sensing.evaluation.trajectory_sweep import SweepConfig
from run_scene_motion_sweep import trajectory_grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("NEW/results/eveyes_adaptive_probe.json"))
    parser.add_argument("--frame-count", type=int, default=61)
    parser.add_argument("--window-count", type=int, default=5)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--initial-index", type=int, default=1)
    parser.add_argument("--no-flow", action="store_true")
    parser.add_argument("--no-imu", action="store_true")
    parser.add_argument("--useful-speed-cap", type=float, default=8.0, help="Train-sweep calibrated useful-objective speed cap in rad/s.")
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.frame_count, args.window_count, args.width, args.height) < 2:
        raise SystemExit("frame-count, window-count, width, and height must be at least two")
    if args.frame_count - 1 < args.window_count:
        raise SystemExit("frame-count must provide at least one RGB interval per window")
    torch.set_num_threads(args.torch_threads)
    scenes = discover_eveyes_scenes(args.dataset, split="test")
    records: list[dict[str, object]] = []
    for scene in scenes:
        count = min(args.frame_count, len(scene.rgb_paths))
        # Use uniformly indexed RGB frames so every control boundary is an
        # exact source timestamp, avoiding interpolation in the audit probe.
        indices = np.linspace(0, count - 1, count, dtype=np.int64)
        rgb = load_scene_rgb(scene, indices)
        rgb = np.stack([cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA) for frame in rgb])
        timestamps = scene.timestamps[indices]
        edge_indices = np.linspace(0, count - 1, args.window_count + 1, dtype=np.int64)
        edges = timestamps[edge_indices]
        duration = float(edges[-1] - edges[0])
        window_duration = float(np.mean(np.diff(edges)))
        candidates = [replace(parameters, duration_s=window_duration) for parameters in trajectory_grid(duration).values()]
        limits = TrajectoryLimits(40.0, 500.0, min_duration_s=float(np.min(np.diff(edges))), max_duration_s=float(np.max(np.diff(edges))))
        config = SweepConfig(
            prism_angle_deg=1.0,
            contrast_threshold=0.15,
            event_window_seconds=window_duration,
            time_surface_decay_seconds=window_duration / 3.0,
            temporal_bins=5,
        )
        imu = None if args.no_imu else load_scene_imu(scene)
        report = rollout_adaptive_controller(
            rgb,
            timestamps,
            scene.intrinsics.resized(args.width, args.height),
            edges,
            candidates,
            limits,
            config=config,
            initial_index=args.initial_index,
            use_flow=not args.no_flow,
            imu_stream=imu,
            useful_speed_cap_rad_s=args.useful_speed_cap,
        )
        aggregate = report["aggregate"]
        records.append({
            "sequence_id": scene.sequence_id,
            "scene_id": scene.scene_id,
            "motion_regime": scene.motion_regime,
            "frame_count": count,
            "resolution": [args.width, args.height],
            "duration_s": duration,
            "imu_used": imu is not None,
            "flow_used": not args.no_flow,
            "imu_sample_count": 0 if imu is None else len(imu.t),
            "mean_event_density": float(aggregate["mean_event_density"]),
            "mean_edge_coverage": float(aggregate["mean_edge_coverage"]),
            "mean_event_efficiency": float(aggregate["mean_event_efficiency"]),
            "mean_temporal_diversity": float(aggregate["mean_temporal_diversity"]),
            "mean_reference_valid_fraction": float(aggregate["mean_reference_valid_fraction"]),
            "event_count": int(aggregate["event_count"]),
            "selected_indices": [int(item["selected_index"]) for item in report["windows"]],
            "selected_parameters": [item["selected_parameters"] for item in report["windows"]],
            "information_deficits": [float(item.get("information_deficit", 0.0)) for item in report["decisions"]],
            "states": report["states"],
        })
        print(json.dumps({"sequence_id": scene.sequence_id, "event_count": records[-1]["event_count"], "selected_indices": records[-1]["selected_indices"]}), flush=True)

    result = {
        "status": "synthetic_rendered_rgb_imu_adaptive_event_quality_probe_not_task_evidence",
        "provenance": "Blender-rendered RGB with trajectory-derived deterministic IMU",
        "dataset": str(args.dataset),
        "configuration": {"frame_count": args.frame_count, "window_count": args.window_count, "width": args.width, "height": args.height, "initial_index": args.initial_index, "no_flow": args.no_flow, "no_imu": args.no_imu, "useful_speed_cap_rad_s": args.useful_speed_cap},
        "scene_count": len(records),
        "records": records,
        "note": "Matching fields from rollout are not interpreted because this export has no dense correspondence/depth.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scene_count": len(records)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
