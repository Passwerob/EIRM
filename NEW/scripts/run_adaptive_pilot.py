#!/usr/bin/env python3
"""Evaluate the rule-based controller in autonomous rollout on the pilot grid."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ami_simulator.types import CameraIntrinsics
from ami_active_sensing.control.trajectory import TrajectoryLimits, TrajectoryParameters
from ami_active_sensing.evaluation.adaptive_rollout import rollout_adaptive_controller
from ami_active_sensing.evaluation.reports import matching_task_utility, useful_event_utility
from ami_active_sensing.evaluation.trajectory_sweep import SweepConfig
from run_scene_motion_sweep import MOTION_REGIMES, TEXTURES, make_texture, moving_sequence, trajectory_grid


def load_baseline_rows(path: Path) -> dict[tuple[str, str, str], dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        key = (row["texture"], row["motion_regime"], row["trajectory"])
        output[key] = row
    if not output:
        raise ValueError(f"No baseline rows found in {path}.")
    return output


def numeric_aggregate(row: dict[str, object]) -> dict[str, float | int]:
    aggregate: dict[str, float | int] = {}
    for key, value in row.items():
        if key.startswith("mean_"):
            aggregate[key] = float(value)
    aggregate["matching_failure_rate"] = float(row["matching_failure_rate"])
    aggregate["event_count"] = int(float(row["event_count"]))
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=Path("NEW/results/scene_motion_sweep.csv"))
    parser.add_argument("--output", type=Path, default=Path("NEW/results/adaptive_pilot.json"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--duration", type=float, default=0.4)
    parser.add_argument("--window", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--bandwidth-limit", type=float, default=15.0)
    parser.add_argument("--bandwidth-penalty", type=float, default=0.03)
    parser.add_argument("--initial-index", type=int, default=1, help="Initial candidate index before the first observation; run 0 and 1 as an exploration ablation.")
    parser.add_argument("--no-flow", action="store_true", help="Ablation: mask optical-flow features while retaining event and trajectory state.")
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    if args.size < 96 or min(args.fps, args.duration, args.window, args.bandwidth_limit) <= 0.0 or args.bandwidth_penalty < 0.0:
        raise SystemExit("Invalid dimensions, timing, or objective values.")
    window_count = int(round(args.duration / args.window))
    frame_count = int(round(args.duration * args.fps)) + 1
    timestamps = np.linspace(0.0, args.duration, frame_count, dtype=np.float64)
    edges = np.linspace(0.0, args.duration, window_count + 1, dtype=np.float64)
    if window_count < 3 or not np.isclose(window_count * args.window, args.duration) or not np.all(np.isin(np.round(edges, 12), np.round(timestamps, 12))):
        raise SystemExit("duration/window/fps must produce at least three aligned windows")
    torch.set_num_threads(args.torch_threads)
    intrinsics = CameraIntrinsics(args.size, args.size, float(args.size), float(args.size), (args.size - 1.0) / 2.0, (args.size - 1.0) / 2.0)
    limits = TrajectoryLimits(40.0, 500.0, min_duration_s=args.window, max_duration_s=args.window)
    config = SweepConfig(prism_angle_deg=1.0, contrast_threshold=0.15, event_window_seconds=args.window, time_surface_decay_seconds=args.window / 3.0, temporal_bins=5)
    baseline = load_baseline_rows(args.sweep)
    sweep_json_path = args.sweep.with_suffix(".json")
    if not sweep_json_path.exists():
        raise FileNotFoundError(f"Expected companion sweep manifest {sweep_json_path}.")
    sweep_manifest = json.loads(sweep_json_path.read_text(encoding="utf-8"))
    condition_manifest = {(item["texture"], item["motion_regime"]): item for item in sweep_manifest["conditions"]}
    candidates = [replace(parameters, duration_s=args.window) for parameters in trajectory_grid(args.duration).values()]
    records: list[dict[str, object]] = []
    for texture_index, texture in enumerate(TEXTURES):
        base = make_texture(texture, args.size, seed=args.seed + texture_index)
        for regime, velocity in MOTION_REGIMES.items():
            frames = moving_sequence(base, timestamps, velocity)
            homography = np.asarray([[1.0, 0.0, velocity[0] * args.window], [0.0, 1.0, velocity[1] * args.window], [0.0, 0.0, 1.0]], dtype=np.float64)
            homographies = np.repeat(homography[None], window_count - 1, axis=0)
            adaptive = rollout_adaptive_controller(frames, timestamps, intrinsics, edges, candidates, limits, config=config, homographies_reference_to_candidate=homographies, initial_index=args.initial_index, objective="useful_event_quality", use_flow=not args.no_flow)
            aggregate = adaptive["aggregate"]
            condition = condition_manifest[(texture, regime)]
            best_task_name = str(condition["winners"]["best_task"])
            best_useful_name = str(condition["winners"]["best_useful_under_budget"])
            best_task_aggregate = numeric_aggregate(baseline[(texture, regime, best_task_name)])
            best_useful_aggregate = numeric_aggregate(baseline[(texture, regime, best_useful_name)])
            adaptive_task = matching_task_utility(aggregate)
            adaptive_useful = useful_event_utility(aggregate, bandwidth_penalty=args.bandwidth_penalty)
            records.append({
                "texture": texture,
                "motion_regime": regime,
                "velocity_px_s": list(velocity),
                "adaptive_task_utility": adaptive_task,
                "adaptive_useful_utility": adaptive_useful,
                "adaptive_mean_event_density": float(aggregate["mean_event_density"]),
                "adaptive_event_count": int(aggregate["event_count"]),
                "adaptive_matching_failure_rate": float(aggregate["matching_failure_rate"]),
                "adaptive_selected_indices": [int(row["selected_index"]) for row in adaptive["windows"]],
                "adaptive_switch_count": int(sum(bool(item.get("switched", False)) for item in adaptive["decisions"])),
                "best_task_policy": best_task_name,
                "best_task_utility": matching_task_utility(best_task_aggregate) if best_task_aggregate else None,
                "best_useful_policy": best_useful_name,
                "best_useful_utility": useful_event_utility(best_useful_aggregate, bandwidth_penalty=args.bandwidth_penalty) if best_useful_aggregate else None,
            })
            print(json.dumps({"texture": texture, "regime": regime, "adaptive_task_utility": adaptive_task, "best_task_policy": best_task_name, "selected": records[-1]["adaptive_selected_indices"]}), flush=True)
    task_regrets = np.asarray([float(row["best_task_utility"] - row["adaptive_task_utility"]) for row in records])
    useful_regrets = np.asarray([float(row["best_useful_utility"] - row["adaptive_useful_utility"]) for row in records])
    policy_names = list(trajectory_grid(args.duration).keys())
    fixed_task_utilities: dict[str, float] = {}
    fixed_useful_utilities: dict[str, float] = {}
    for policy in policy_names:
        policy_rows = [numeric_aggregate(baseline[(row["texture"], row["motion_regime"], policy)]) for row in records]
        fixed_task_utilities[policy] = float(np.mean([matching_task_utility(item) for item in policy_rows]))
        fixed_useful_utilities[policy] = float(np.mean([useful_event_utility(item, bandwidth_penalty=args.bandwidth_penalty) for item in policy_rows]))
    adaptive_mean_task = float(np.mean([float(row["adaptive_task_utility"]) for row in records]))
    adaptive_mean_useful = float(np.mean([float(row["adaptive_useful_utility"]) for row in records]))
    global_fixed_task_delta = {policy: float(adaptive_mean_task - utility) for policy, utility in fixed_task_utilities.items()}
    global_fixed_useful_delta = {policy: float(adaptive_mean_useful - utility) for policy, utility in fixed_useful_utilities.items()}
    result = {
        "status": "autonomous_controller_pilot_not_paper_evidence",
        "controller": "rule_based_adaptive_trajectory",
        "state": "previous_event_quality_plus_previous_trajectory" if args.no_flow else "flow_plus_previous_event_quality_plus_previous_trajectory",
        "selection_is_autonomous": True,
        "baseline_sweep": str(args.sweep),
        "configuration": {"size": args.size, "fps": args.fps, "duration_s": args.duration, "window_s": args.window, "seed": args.seed, "bandwidth_limit": args.bandwidth_limit, "bandwidth_penalty": args.bandwidth_penalty, "initial_index": args.initial_index, "no_flow": args.no_flow},
        "aggregate": {
            "condition_count": len(records),
            "mean_best_task_minus_adaptive_task_regret": float(np.mean(task_regrets)),
            "median_best_task_minus_adaptive_task_regret": float(np.median(task_regrets)),
            "adaptive_beats_best_task_count": int(np.count_nonzero(task_regrets <= 0.0)),
            "mean_best_useful_minus_adaptive_useful_regret": float(np.mean(useful_regrets)),
            "median_best_useful_minus_adaptive_useful_regret": float(np.median(useful_regrets)),
            "adaptive_beats_best_useful_count": int(np.count_nonzero(useful_regrets <= 0.0)),
            "adaptive_mean_task_utility": adaptive_mean_task,
            "adaptive_mean_useful_utility": adaptive_mean_useful,
        },
        "global_fixed_baselines": {
            "task_utility_mean": fixed_task_utilities,
            "useful_utility_mean": fixed_useful_utilities,
            "adaptive_minus_fixed_task_utility": global_fixed_task_delta,
            "adaptive_minus_fixed_useful_utility": global_fixed_useful_delta,
        },
        "conditions": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"output": str(args.output), **result["aggregate"]}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
