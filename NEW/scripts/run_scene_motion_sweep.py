#!/usr/bin/env python3
"""Test whether texture and natural motion require different prism trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch

from ami_simulator.types import CameraIntrinsics
from ami_active_sensing.control.trajectory import ActuatorConstrainedTrajectory, TrajectoryLimits, TrajectoryParameters, make_trajectory
from ami_active_sensing.evaluation.reports import matching_task_utility, select_winners, useful_event_utility, winner_distribution
from ami_active_sensing.evaluation.trajectory_sweep import SweepConfig, evaluate_trajectory
from ami_active_sensing.motion.motion_features import extract_flow_features
from ami_active_sensing.motion.optical_flow import FarnebackFlowBackend


TEXTURES = ("checker", "blobs", "grid", "sparse")
MOTION_REGIMES = {
    "static": (0.0, 0.0),
    "slow": (8.0, 3.0),
    "medium": (25.0, 10.0),
    "fast": (60.0, 24.0),
}


def make_texture(name: str, size: int, *, seed: int) -> np.ndarray:
    """Create texture families with different edge density and ambiguity."""
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), 24, dtype=np.uint8)
    if name == "checker":
        cell = max(size // 14, 6)
        for y in range(0, size, cell):
            for x in range(0, size, cell):
                if (x // cell + y // cell) % 2 == 0:
                    image[y : y + cell, x : x + cell] = (210, 210, 210)
        for _ in range(20):
            x, y = rng.integers(6, size - 6, size=2)
            cv2.circle(image, (int(x), int(y)), 2, tuple(int(v) for v in rng.integers(60, 255, size=3)), -1)
    elif name == "blobs":
        for _ in range(75):
            x, y = rng.integers(5, size - 5, size=2)
            radius = int(rng.integers(2, 7))
            cv2.circle(image, (int(x), int(y)), radius, tuple(int(v) for v in rng.integers(45, 255, size=3)), -1)
    elif name == "grid":
        for offset in range(8, size, 13):
            cv2.line(image, (offset, 0), (offset, size - 1), (220, 80, 80), 2)
            cv2.line(image, (0, offset), (size - 1, offset), (80, 220, 180), 2)
        for _ in range(12):
            x, y = rng.integers(8, size - 14, size=2)
            cv2.rectangle(image, (int(x), int(y)), (int(x + 7), int(y + 5)), (250, 250, 70), -1)
    elif name == "sparse":
        for _ in range(22):
            x, y = rng.integers(10, size - 16, size=2)
            width, height = rng.integers(4, 13, size=2)
            cv2.rectangle(image, (int(x), int(y)), (int(x + width), int(y + height)), tuple(int(v) for v in rng.integers(100, 255, size=3)), -1)
    else:
        raise ValueError(f"Unknown texture {name!r}.")
    cv2.putText(image, name[0].upper(), (size // 2 - 10, size // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 30, 180), 2, cv2.LINE_AA)
    return image


def moving_sequence(base: np.ndarray, timestamps: np.ndarray, velocity_px_s: tuple[float, float]) -> np.ndarray:
    """Translate one texture with a known image-plane velocity."""
    height, width = base.shape[:2]
    frames = []
    for timestamp in timestamps:
        dx = velocity_px_s[0] * float(timestamp - timestamps[0])
        dy = velocity_px_s[1] * float(timestamp - timestamps[0])
        affine = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        frames.append(cv2.warpAffine(base, affine, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101))
    return np.stack(frames)


def trajectory_grid(duration_s: float) -> dict[str, TrajectoryParameters]:
    return {
        "constant_0": TrajectoryParameters("constant", angular_speed_rad_s=0.0, duration_s=duration_s),
        "constant_6": TrajectoryParameters("constant", angular_speed_rad_s=6.0, duration_s=duration_s),
        "constant_14": TrajectoryParameters("constant", angular_speed_rad_s=14.0, duration_s=duration_s),
        "constant_30": TrajectoryParameters("constant", angular_speed_rad_s=30.0, duration_s=duration_s),
        "constant_reverse_14": TrajectoryParameters("constant", angular_speed_rad_s=14.0, direction=-1, duration_s=duration_s),
        "sinusoid": TrajectoryParameters("sinusoid", amplitude_rad=0.50, frequency_hz=3.0, phase_rad=0.0, duration_s=duration_s),
        "triangle": TrajectoryParameters("triangle", amplitude_rad=0.70, frequency_hz=4.0, phase_rad=0.0, duration_s=duration_s),
        "chirp": TrajectoryParameters("chirp", amplitude_rad=0.45, frequency_hz=1.0, end_frequency_hz=5.0, phase_rad=0.0, duration_s=duration_s),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("NEW/results/scene_motion_sweep.json"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--duration", type=float, default=0.4)
    parser.add_argument("--window", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--bandwidth-limit", type=float, default=15.0, help="Mean events per pixel per window.")
    parser.add_argument("--bandwidth-penalty", type=float, default=0.03)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    if args.size < 96 or min(args.fps, args.duration, args.window, args.bandwidth_limit) <= 0.0 or args.bandwidth_penalty < 0.0:
        raise SystemExit("Invalid sweep dimensions, timing, or bandwidth objective.")
    window_count = int(round(args.duration / args.window))
    frame_count = int(round(args.duration * args.fps)) + 1
    timestamps = np.linspace(0.0, args.duration, frame_count, dtype=np.float64)
    edges = np.linspace(0.0, args.duration, window_count + 1, dtype=np.float64)
    if window_count < 3 or not np.isclose(window_count * args.window, args.duration) or not np.all(np.isin(np.round(edges, 12), np.round(timestamps, 12))):
        raise SystemExit("Use at least three aligned windows; duration must be an integer multiple of window.")
    torch.set_num_threads(args.torch_threads)
    intrinsics = CameraIntrinsics(args.size, args.size, float(args.size), float(args.size), (args.size - 1.0) / 2.0, (args.size - 1.0) / 2.0)
    limits = TrajectoryLimits(40.0, 500.0, min_duration_s=args.duration, max_duration_s=args.duration)
    sweep_config = SweepConfig(
        prism_angle_deg=1.0,
        contrast_threshold=0.15,
        event_window_seconds=args.window,
        time_surface_decay_seconds=args.window / 3.0,
        temporal_bins=5,
    )
    trajectories = trajectory_grid(args.duration)
    flow_backend = FarnebackFlowBackend()
    rows: list[dict[str, object]] = []
    conditions: list[dict[str, object]] = []
    total_conditions = len(TEXTURES) * len(MOTION_REGIMES)
    condition_index = 0
    for texture_index, texture in enumerate(TEXTURES):
        base = make_texture(texture, args.size, seed=args.seed + texture_index)
        for regime, velocity in MOTION_REGIMES.items():
            condition_index += 1
            frames = moving_sequence(base, timestamps, velocity)
            dx = velocity[0] * args.window
            dy = velocity[1] * args.window
            homography = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
            homographies = np.repeat(homography[None], window_count - 1, axis=0)
            flow_summaries = []
            for start, end in zip(edges[:-1], edges[1:]):
                start_index = int(np.argmin(np.abs(timestamps - start)))
                end_index = int(np.argmin(np.abs(timestamps - end)))
                flow = flow_backend.estimate(frames[start_index], frames[end_index], dt_seconds=float(end - start))
                flow_summaries.append(extract_flow_features(flow, low_motion_threshold_px_s=1.0, magnitude_histogram_max_px_s=100.0, reference_intensity=frames[start_index]))
            mean_flow = {name: float(np.mean([item[name] for item in flow_summaries])) for name in flow_summaries[0]}
            reports = []
            for trajectory_name, parameters in trajectories.items():
                desired = make_trajectory(parameters)
                constrained = ActuatorConstrainedTrajectory(desired, timestamps, limits)
                report = evaluate_trajectory(
                    trajectory_name,
                    frames,
                    timestamps,
                    intrinsics,
                    constrained,
                    edges,
                    config=sweep_config,
                    homographies_reference_to_candidate=homographies,
                )
                reports.append(report)
                aggregate = report["aggregate"]
                rows.append({
                    "texture": texture,
                    "motion_regime": regime,
                    "velocity_x_px_s": velocity[0],
                    "velocity_y_px_s": velocity[1],
                    "trajectory": trajectory_name,
                    "task_utility": matching_task_utility(aggregate),
                    "useful_utility": useful_event_utility(aggregate, bandwidth_penalty=args.bandwidth_penalty),
                    "within_bandwidth": float(aggregate["mean_event_density"]) <= args.bandwidth_limit,
                    **aggregate,
                })
            winners = select_winners(reports, bandwidth_limit=args.bandwidth_limit, bandwidth_penalty=args.bandwidth_penalty)
            conditions.append({
                "texture": texture,
                "motion_regime": regime,
                "velocity_px_s": list(velocity),
                "flow_features": mean_flow,
                "winners": winners,
            })
            print(json.dumps({"progress": f"{condition_index}/{total_conditions}", "texture": texture, "regime": regime, **winners}), flush=True)

    feasible_conditions = [item for item in conditions if bool(item["winners"]["budget_feasible"])]
    useful_distribution = winner_distribution([str(item["winners"]["best_useful_under_budget"]) for item in feasible_conditions])
    task_distribution = winner_distribution([str(item["winners"]["best_task"]) for item in conditions])
    count_distribution = winner_distribution([str(item["winners"]["max_event_count"]) for item in conditions])
    per_texture_variation = {
        texture: len({str(item["winners"]["best_useful_under_budget"]) for item in feasible_conditions if item["texture"] == texture})
        for texture in TEXTURES
    }
    count_regrets = np.asarray([float(item["winners"]["count_objective_task_regret"]) for item in conditions])
    useful_regrets = np.asarray([float(item["winners"]["useful_selection_task_regret"]) for item in feasible_conditions])
    conclusion = {
        "different_conditions_select_different_trajectories": not bool(useful_distribution["single_trajectory_wins_everywhere"]),
        "textures_with_motion_dependent_winner": int(sum(value > 1 for value in per_texture_variation.values())),
        "event_count_objective_has_positive_median_task_regret": float(np.median(count_regrets)) > 0.0,
        "gate_h2_pilot_pass": useful_distribution["unique_winner_count"] >= 2 and sum(value > 1 for value in per_texture_variation.values()) >= 2,
    }
    result = {
        "status": "controlled_synthetic_pilot_not_paper_evidence",
        "hypothesis": "different scene texture and natural motion conditions require different optical excitation",
        "seed": args.seed,
        "textures": list(TEXTURES),
        "motion_regimes_px_s": {name: list(value) for name, value in MOTION_REGIMES.items()},
        "trajectory_grid": {name: parameters.metadata() for name, parameters in trajectories.items()},
        "intrinsics": asdict(intrinsics),
        "limits": asdict(limits),
        "sweep_config": asdict(sweep_config),
        "selection_protocol": {
            "task_utility": "0.30 repeatability + 0.25 precision + 0.35 recall + 0.10 RANSAC inlier - 0.25 failure_rate",
            "bandwidth_limit_mean_events_per_pixel_per_window": args.bandwidth_limit,
            "bandwidth_penalty": args.bandwidth_penalty,
            "test_set_selection": False,
        },
        "conditions": conditions,
        "aggregate": {
            "useful_winner_distribution": useful_distribution,
            "budget_feasible_condition_count": len(feasible_conditions),
            "budget_infeasible_condition_count": len(conditions) - len(feasible_conditions),
            "task_winner_distribution": task_distribution,
            "count_winner_distribution": count_distribution,
            "unique_useful_winners_per_texture": per_texture_variation,
            "median_count_objective_task_regret": float(np.median(count_regrets)),
            "mean_count_objective_task_regret": float(np.mean(count_regrets)),
            "median_useful_selection_task_regret": float(np.median(useful_regrets)),
            "conclusion": conclusion,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output), "csv": str(csv_path), **result["aggregate"]}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
