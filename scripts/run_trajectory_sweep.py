#!/usr/bin/env python3
"""Evaluate optical-excitation trajectory families on controlled scenes.

This is a standalone experiment in the main simulator.  It intentionally does
not import ``NEW``: the experimental redesign remains an independent reference
implementation while this script reuses the main repository's prism, warp,
event, and quality-metric contracts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import torch

from ami_simulator.event_quality import active_pixel_ids, compute_reference_edge_mask, event_quality_metrics
from ami_simulator.events import ThresholdEventSimulator
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.rotation import ScheduledRotation
from ami_simulator.types import CameraIntrinsics, EventStream
from ami_simulator.warp import PrismWarper, compensate_events


TRAJECTORIES = ("constant_0", "constant_180", "constant_360", "constant_720", "constant_1080", "sinusoidal_speed", "triangular_speed", "chirp_speed")


def _trajectory_rpm(name: str, times: np.ndarray, *, base_rpm: float = 360.0) -> np.ndarray:
    """Return a non-negative actuator schedule sampled at frame timestamps."""
    if name.startswith("constant_"):
        return np.full(len(times), float(name.split("_", 1)[1]), dtype=np.float64)
    duration = max(float(times[-1] - times[0]), 1e-12)
    phase = (times - float(times[0])) / duration
    if name == "sinusoidal_speed":
        # Speed modulation keeps the physical one-DoF direction fixed while
        # varying temporal excitation within the same mean-speed budget.
        values = base_rpm * (1.0 + 0.75 * np.sin(2.0 * np.pi * phase - 0.5 * np.pi))
        return np.clip(values, 0.0, None)
    if name == "triangular_speed":
        saw = 2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5)))
        return base_rpm * (0.25 + 1.5 * saw)
    if name == "chirp_speed":
        chirp_phase = 2.0 * np.pi * (0.5 * phase + 1.5 * phase * phase)
        values = base_rpm * (1.0 + 0.75 * np.sin(chirp_phase - 0.5 * np.pi))
        return np.clip(values, 0.0, None)
    raise ValueError(f"Unknown trajectory {name!r}.")


def _load_scene(scene: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, CameraIntrinsics]:
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s")
    intrinsics_data = json.loads((scene / "camera_intrinsics.json").read_text(encoding="utf-8"))
    intrinsics = CameraIntrinsics(
        frames.shape[2],
        frames.shape[1],
        float(intrinsics_data["fx"]),
        float(intrinsics_data["fy"]),
        float(intrinsics_data["cx"]),
        float(intrinsics_data["cy"]),
    )
    masks = np.stack([np.asarray(imageio.imread(path)) > 0 for path in sorted((scene / "object_masks").glob("*.png"))])
    if len(masks) != len(frames):
        raise ValueError(f"Scene {scene} requires one object mask per frame.")
    return rgb_to_linear_luminance(frames), timestamps, masks, intrinsics


def _useful_score(metrics: dict[str, float | int]) -> float:
    """Score support and diversity without rewarding raw event flooding."""
    return float(metrics["edge_f1"]) + 0.25 * float(metrics["edge_coverage"]) + 0.15 * float(metrics["unique_pixel_ratio"])

def _evaluate_trajectory(
    luminance: np.ndarray,
    timestamps: np.ndarray,
    roi_masks: np.ndarray,
    intrinsics: CameraIntrinsics,
    trajectory: str,
    *,
    prism_angle_deg: float,
    contrast_threshold: float,
    base_rpm: float,
) -> dict[str, float | int | str]:
    rpm = _trajectory_rpm(trajectory, timestamps, base_rpm=base_rpm)
    rotation = ScheduledRotation(timestamps, rpm)
    prism = RotatingWedgePrism(intrinsics, prism_angle_deg)
    warper = PrismWarper(prism, rotation, height=intrinsics.height, width=intrinsics.width, reference_time=float(timestamps[0]))
    with torch.inference_mode():
        sensor, valid = warper.warp(torch.from_numpy(luminance).unsqueeze(1), timestamps)
    events = ThresholdEventSimulator(contrast_threshold).simulate(sensor[:, 0].numpy(), timestamps)
    compensated = compensate_events(events, prism, rotation, theta0=0.0, reference_time=float(timestamps[0]))
    reference_events = EventStream(
        np.rint(compensated.extras["x_reference"]),
        np.rint(compensated.extras["y_reference"]),
        compensated.t,
        compensated.p,
    )
    records: list[dict[str, float]] = []
    previous_active: np.ndarray | None = None
    for index, (start, end) in enumerate(zip(timestamps[:-1], timestamps[1:])):
        selected = (reference_events.t >= start) & (reference_events.t < end)
        if index == len(timestamps) - 2:
            selected |= reference_events.t == end
        window_events = EventStream(
            reference_events.x[selected], reference_events.y[selected], reference_events.t[selected], reference_events.p[selected]
        )
        edge_mask = compute_reference_edge_mask(luminance[index], edge_quantile=0.75) & roi_masks[index]
        metrics = event_quality_metrics(window_events, reference_edge_mask=edge_mask, previous_active_pixels=previous_active, image_shape=luminance.shape[1:])
        metrics["event_density"] = float(len(window_events) / (intrinsics.width * intrinsics.height))
        records.append({key: float(value) for key, value in metrics.items() if isinstance(value, (float, int))})
        previous_active = active_pixel_ids(window_events, image_shape=luminance.shape[1:])
    useful = np.asarray([_useful_score(record) + 0.65 * record["event_efficiency"] - 0.5 * record["temporal_overlap"] - 0.05 * record["polarity_imbalance"] for record in records])
    counts = np.asarray([record["event_count"] for record in records])
    return {
        "trajectory": trajectory,
        "mean_event_count": float(np.mean(counts)),
        "mean_event_density": float(np.mean([record["event_density"] for record in records])),
        "mean_edge_f1": float(np.mean([record["edge_f1"] for record in records])),
        "mean_edge_coverage": float(np.mean([record["edge_coverage"] for record in records])),
        "mean_event_efficiency": float(np.mean([record["event_efficiency"] for record in records])),
        "mean_temporal_overlap": float(np.mean([record["temporal_overlap"] for record in records])),
        "mean_useful_score": float(np.mean(useful)),
        "mean_rpm": float(np.mean(rpm)),
        "peak_rpm": float(np.max(rpm)),
        "valid_fraction": float(valid.all(dim=0).float().mean().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/comotion_controlled"))
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("results/trajectory_sweep.json"))
    parser.add_argument("--prism-angle-deg", type=float, default=2.0)
    parser.add_argument("--contrast-threshold", type=float, default=0.2)
    parser.add_argument("--base-rpm", type=float, default=360.0)
    args = parser.parse_args()
    manifest = json.loads((args.dataset_root / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for record in manifest["records"]:
        if record["split"] != args.split:
            continue
        scene = args.dataset_root / record["scene"]
        luminance, timestamps, masks, intrinsics = _load_scene(scene)
        for trajectory in TRAJECTORIES:
            result = _evaluate_trajectory(luminance, timestamps, masks, intrinsics, trajectory, prism_angle_deg=args.prism_angle_deg, contrast_threshold=args.contrast_threshold, base_rpm=args.base_rpm)
            rows.append({"scene": record["scene"], "condition": record["condition"], **result})
        print(f"evaluated {record['scene']}")
    by_condition: dict[str, dict[str, object]] = {}
    for condition in manifest["conditions"]:
        selected = [row for row in rows if row["condition"] == condition]
        summary: dict[str, object] = {"scenes": len({row["scene"] for row in selected})}
        for metric in ("mean_event_count", "mean_event_density", "mean_edge_f1", "mean_edge_coverage", "mean_event_efficiency", "mean_temporal_overlap", "mean_useful_score", "mean_rpm", "peak_rpm", "valid_fraction"):
            summary[metric] = {trajectory: float(np.mean([float(row[metric]) for row in selected if row["trajectory"] == trajectory])) for trajectory in TRAJECTORIES}
        summary["event_count_winner"] = max(TRAJECTORIES, key=lambda trajectory: summary["mean_event_count"][trajectory])
        summary["useful_score_winner"] = max(TRAJECTORIES, key=lambda trajectory: summary["mean_useful_score"][trajectory])
        by_condition[condition] = summary
    report = {
        "evaluation": "controlled trajectory-family sweep",
        "dataset": manifest.get("dataset"),
        "split": args.split,
        "trajectory_families": list(TRAJECTORIES),
        "objective": "useful event support and temporal diversity; event count reported separately",
        "limitations": ["Synthetic procedural scenes; this is sensing-objective evidence, not downstream matching evidence.", "All schedules use the existing single-DoF rotating wedge prism model."],
        "by_condition": by_condition,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(by_condition, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
