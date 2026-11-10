#!/usr/bin/env python3
"""Sweep prism trajectories on EveYES RGB data using event-quality metrics only."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

from ami_simulator.event_quality import event_quality_score
from ami_active_sensing.control.trajectory import TrajectoryLimits, make_trajectory
from ami_active_sensing.data import discover_eveyes_scenes, load_scene_imu, load_scene_rgb
from ami_active_sensing.evaluation.trajectory_sweep import SweepConfig, evaluate_trajectory
from run_scene_motion_sweep import trajectory_grid


def _event_utility(aggregate: dict[str, object], *, target_density: float) -> float:
    metrics = {
        "edge_f1": aggregate["mean_edge_f1"],
        "edge_coverage": aggregate["mean_edge_coverage"],
        "unique_pixel_ratio": aggregate["mean_unique_pixel_ratio"],
        "event_efficiency": aggregate["mean_event_efficiency"],
        "temporal_overlap": aggregate["mean_temporal_overlap"],
        "polarity_imbalance": aggregate.get("mean_polarity_imbalance", 0.0),
        "event_density": aggregate["mean_event_density"],
    }
    return event_quality_score(metrics, target_density=target_density)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("NEW/results/eveyes_trajectory_sweep.json"))
    parser.add_argument("--frame-count", type=int, default=31)
    parser.add_argument("--window-count", type=int, default=3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--bandwidth-limit", type=float, default=15.0)
    parser.add_argument("--target-density", type=float, default=10.0)
    parser.add_argument("--max-scenes", type=int, default=0, help="0 means all sequences in the selected split")
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.frame_count, args.window_count, args.width, args.height) < 2 or args.bandwidth_limit <= 0.0 or args.target_density <= 0.0:
        raise SystemExit("frame/window dimensions, bandwidth-limit, and target-density must be positive")
    torch.set_num_threads(args.torch_threads)
    scenes = discover_eveyes_scenes(args.dataset, split=args.split)
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]
    rows: list[dict[str, object]] = []
    conditions: list[dict[str, object]] = []
    for scene in scenes:
        count = min(args.frame_count, len(scene.rgb_paths))
        indices = np.linspace(0, count - 1, count, dtype=np.int64)
        frames = load_scene_rgb(scene, indices)
        frames = np.stack([cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA) for frame in frames])
        timestamps = scene.timestamps[indices]
        edge_indices = np.linspace(0, count - 1, args.window_count + 1, dtype=np.int64)
        edges = timestamps[edge_indices]
        window_duration = float(np.mean(np.diff(edges)))
        config = SweepConfig(prism_angle_deg=1.0, contrast_threshold=0.15, event_window_seconds=window_duration, time_surface_decay_seconds=window_duration / 3.0, temporal_bins=5)
        trajectories = {name: replace(parameters, duration_s=window_duration) for name, parameters in trajectory_grid(float(edges[-1] - edges[0])).items()}
        limits = TrajectoryLimits(40.0, 500.0, min_duration_s=window_duration, max_duration_s=window_duration)
        reports: list[dict[str, object]] = []
        for name, parameters in trajectories.items():
            report = evaluate_trajectory(name, frames, timestamps, scene.intrinsics.resized(args.width, args.height), make_trajectory(parameters, start_time=float(timestamps[0])), edges, config=config)
            aggregate = report["aggregate"]
            utility = _event_utility(aggregate, target_density=args.target_density)
            density = float(aggregate["mean_event_density"])
            row = {"sequence_id": scene.sequence_id, "split": scene.split, "motion_regime": scene.motion_regime, "trajectory": name, "event_quality_utility": utility, "within_bandwidth": density <= args.bandwidth_limit, **aggregate}
            rows.append(row)
            reports.append({"trajectory": name, "event_quality_utility": utility, "mean_event_density": density, "event_count": int(aggregate["event_count"])})
        feasible = [item for item in reports if bool(item["mean_event_density"] <= args.bandwidth_limit)]
        useful_winner = max(feasible or reports, key=lambda item: float(item["event_quality_utility"]))
        count_winner = max(reports, key=lambda item: float(item["event_count"]) if "event_count" in item else 0.0)
        conditions.append({"sequence_id": scene.sequence_id, "split": scene.split, "motion_regime": scene.motion_regime, "useful_winner": useful_winner["trajectory"], "count_winner": count_winner["trajectory"], "budget_feasible": bool(feasible)})
        print(json.dumps({"sequence_id": scene.sequence_id, "useful_winner": useful_winner["trajectory"], "count_winner": count_winner["trajectory"]}), flush=True)
    result = {
        "status": "synthetic_rendered_rgb_event_quality_sweep_not_task_evidence",
        "provenance": "Blender-rendered RGB; no real event-camera capture",
        "dataset": str(args.dataset),
        "split": args.split,
        "scene_count": len(scenes),
        "configuration": {"frame_count": args.frame_count, "window_count": args.window_count, "width": args.width, "height": args.height, "bandwidth_limit": args.bandwidth_limit, "target_density": args.target_density},
        "conditions": conditions,
        "aggregate": {
            "useful_winner_distribution": dict(sorted(Counter(str(item["useful_winner"]) for item in conditions).items())),
            "count_winner_distribution": dict(sorted(Counter(str(item["count_winner"]) for item in conditions).items())),
            "budget_infeasible_count": int(sum(not bool(item["budget_feasible"]) for item in conditions)),
        },
        "rows": rows,
        "note": "Task matching is excluded because the dataset export has no dense correspondence/depth.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output), "scene_count": len(scenes), "condition_count": len(conditions)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
