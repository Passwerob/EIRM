#!/usr/bin/env python3
"""Run a deterministic five-policy active-excitation smoke experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
import torch

from ami_simulator.types import CameraIntrinsics
from ami_active_sensing.control.adaptive_controller import RuleBasedAdaptiveController
from ami_active_sensing.control.trajectory import (
    ActuatorConstrainedTrajectory,
    PiecewiseTrajectory,
    TrajectoryLimits,
    TrajectoryParameters,
    make_trajectory,
)
from ami_active_sensing.evaluation.trajectory_sweep import SweepConfig, evaluate_trajectory
from ami_active_sensing.motion.motion_features import build_state_features, extract_event_features, extract_flow_features
from ami_active_sensing.motion.optical_flow import FarnebackFlowBackend


def synthetic_static_scene(size: int, frame_count: int, *, seed: int) -> np.ndarray:
    """Create a repeatable textured scene with no natural temporal motion."""
    if size < 96 or frame_count < 2:
        raise ValueError("Synthetic smoke scene needs size >= 96 and at least two frames.")
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), 28, dtype=np.uint8)
    cell = max(size // 16, 6)
    for row in range(0, size, cell):
        for column in range(0, size, cell):
            if (row // cell + column // cell) % 2 == 0:
                image[row : row + cell, column : column + cell] = (190, 190, 190)
    for _ in range(45):
        x, y = rng.integers(8, size - 8, size=2)
        radius = int(rng.integers(2, 6))
        color = tuple(int(value) for value in rng.integers(45, 255, size=3))
        cv2.circle(image, (int(x), int(y)), radius, color, -1)
    cv2.putText(image, "AMI", (size // 8, size // 2), cv2.FONT_HERSHEY_SIMPLEX, size / 170.0, (255, 35, 120), 2, cv2.LINE_AA)
    return np.repeat(image[None], frame_count, axis=0)


def _constrained(desired: object, timestamps: np.ndarray, limits: TrajectoryLimits) -> ActuatorConstrainedTrajectory:
    return ActuatorConstrainedTrajectory(desired, timestamps, limits, initial_omega_rad_s=0.0)


def _piecewise(selected: list[TrajectoryParameters], edges: np.ndarray) -> PiecewiseTrajectory:
    segments = [(float(edges[index]), replace(parameters, duration_s=float(edges[index + 1] - edges[index]))) for index, parameters in enumerate(selected)]
    return PiecewiseTrajectory(segments)


def build_adaptive_schedule(
    frames: np.ndarray,
    timestamps: np.ndarray,
    intrinsics: CameraIntrinsics,
    edges: np.ndarray,
    candidates: list[TrajectoryParameters],
    limits: TrajectoryLimits,
    sweep_config: SweepConfig,
) -> tuple[ActuatorConstrainedTrajectory, list[dict[str, object]], list[dict[str, float]]]:
    """Choose each next segment from only the completed previous window."""
    controller = RuleBasedAdaptiveController(
        candidates,
        limits,
        initial_index=1,
        objective="useful_event_quality",
        minimum_dwell_s=float(edges[1] - edges[0]),
        switch_margin=0.01,
    )
    controller.reset(timestamp=float(edges[0]))
    flow_backend = FarnebackFlowBackend()
    selected = [candidates[controller.current_index]]
    decisions: list[dict[str, object]] = [{
        "timestamp": float(edges[0]),
        "selected_index": controller.current_index,
        "reason": "initial_safe_action_before_observation",
    }]
    states: list[dict[str, float]] = []
    for window_index in range(len(edges) - 1):
        end = float(edges[window_index + 1])
        end_frame_index = int(np.argmin(np.abs(timestamps - end)))
        if not np.isclose(timestamps[end_frame_index], end, rtol=0.0, atol=1e-12):
            raise ValueError("Adaptive window end does not align with a frame timestamp.")
        end_index = end_frame_index + 1
        partial_times = timestamps[:end_index]
        desired = _piecewise(selected, edges[: len(selected) + 1])
        trajectory = _constrained(desired, partial_times, limits)
        partial_report = evaluate_trajectory(
            "adaptive_partial",
            frames[:end_index],
            partial_times,
            intrinsics,
            trajectory,
            edges[: window_index + 2],
            config=sweep_config,
        )
        completed = partial_report["windows"][-1]
        event_features = extract_event_features(completed)
        start_index = int(np.argmin(np.abs(timestamps - edges[window_index])))
        flow = flow_backend.estimate(frames[start_index], frames[end_index - 1], dt_seconds=end - float(edges[window_index]))
        flow_features = extract_flow_features(
            flow,
            low_motion_threshold_px_s=1.0,
            magnitude_histogram_max_px_s=100.0,
            reference_intensity=frames[start_index],
        )
        state = build_state_features(flow_features, event_features, selected[-1], use_flow=True)
        states.append(state)
        if window_index + 1 < len(edges) - 1:
            next_parameters = controller.select(state, timestamp=end)
            selected.append(next_parameters)
            assert controller.last_decision is not None
            decisions.append({**asdict(controller.last_decision), "parameters": next_parameters.metadata()})
    desired = _piecewise(selected, edges)
    return _constrained(desired, timestamps, limits), decisions, states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("NEW/results/smoke_experiment.json"))
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--fps", type=float, default=200.0)
    parser.add_argument("--duration", type=float, default=0.6)
    parser.add_argument("--window", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    if args.fps <= 0.0 or args.duration <= 0.0 or args.window <= 0.0 or args.torch_threads < 1:
        raise SystemExit("fps, duration, window, and torch-threads must be positive")
    window_count = int(round(args.duration / args.window))
    if window_count < 2 or not np.isclose(window_count * args.window, args.duration):
        raise SystemExit("duration must be an integer multiple of window and contain at least two windows")
    frame_count = int(round(args.duration * args.fps)) + 1
    timestamps = np.linspace(0.0, args.duration, frame_count, dtype=np.float64)
    edges = np.linspace(0.0, args.duration, window_count + 1, dtype=np.float64)
    if not np.all(np.isin(np.round(edges, 12), np.round(timestamps, 12))):
        raise SystemExit("window edges must align with frame timestamps; choose compatible fps/window values")
    torch.set_num_threads(args.torch_threads)
    frames = synthetic_static_scene(args.size, frame_count, seed=args.seed)
    intrinsics = CameraIntrinsics(args.size, args.size, float(args.size), float(args.size), (args.size - 1.0) / 2.0, (args.size - 1.0) / 2.0)
    limits = TrajectoryLimits(max_speed_rad_s=40.0, max_acceleration_rad_s2=500.0, min_duration_s=args.window, max_duration_s=args.duration)
    segment_candidates = [
        TrajectoryParameters("constant", angular_speed_rad_s=0.0, duration_s=args.window),
        TrajectoryParameters("constant", angular_speed_rad_s=8.0, duration_s=args.window),
        TrajectoryParameters("constant", angular_speed_rad_s=35.0, duration_s=args.window),
        TrajectoryParameters("sinusoid", amplitude_rad=0.45, frequency_hz=2.0, phase_rad=0.0, duration_s=args.window),
        TrajectoryParameters("triangle", amplitude_rad=0.35, frequency_hz=3.0, phase_rad=0.0, duration_s=args.window),
    ]
    sweep_config = SweepConfig(
        prism_angle_deg=1.0,
        contrast_threshold=0.15,
        event_window_seconds=args.window,
        time_surface_decay_seconds=args.window / 3.0,
        temporal_bins=5,
    )

    full_duration = float(args.duration)
    desired_no_prism = make_trajectory(TrajectoryParameters("constant", angular_speed_rad_s=0.0, duration_s=full_duration))
    desired_slow = make_trajectory(TrajectoryParameters("constant", angular_speed_rad_s=8.0, duration_s=full_duration))
    desired_fast = make_trajectory(TrajectoryParameters("constant", angular_speed_rad_s=35.0, duration_s=full_duration))
    rng = np.random.default_rng(args.seed)
    random_selected = [segment_candidates[int(index)] for index in rng.integers(0, len(segment_candidates), window_count)]
    desired_random = _piecewise(random_selected, edges)
    adaptive, decisions, states = build_adaptive_schedule(frames, timestamps, intrinsics, edges, segment_candidates, limits, sweep_config)
    policies = {
        "no_prism": _constrained(desired_no_prism, timestamps, limits),
        "fixed_slow": _constrained(desired_slow, timestamps, limits),
        "fixed_fast": _constrained(desired_fast, timestamps, limits),
        "random_trajectory": _constrained(desired_random, timestamps, limits),
        "adaptive_trajectory": adaptive,
    }
    reports = [
        evaluate_trajectory(name, frames, timestamps, intrinsics, trajectory, edges, config=sweep_config)
        for name, trajectory in policies.items()
    ]
    result = {
        "status": "engineering_smoke_test_not_scientific_evidence",
        "scene": "synthetic_static_textured_low_natural_motion",
        "seed": args.seed,
        "timeline": {"fps": args.fps, "duration_s": args.duration, "window_s": args.window, "frame_count": frame_count},
        "intrinsics": asdict(intrinsics),
        "limits": asdict(limits),
        "sweep_config": asdict(sweep_config),
        "flow_backend": FarnebackFlowBackend().metadata(),
        "random_segments": [parameters.metadata() for parameters in random_selected],
        "adaptive_decisions": decisions,
        "adaptive_state_count": len(states),
        "policies": reports,
        "summary": {report["policy"]: report["aggregate"] for report in reports},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": result["summary"], "adaptive_decisions": decisions}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
