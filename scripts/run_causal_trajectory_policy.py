#!/usr/bin/env python3
"""Evaluate a causal useful-event controller on controlled RGB scenes.

The controller observes only the completed RGB pair ending at the current
window and the previous event-quality feedback.  It then chooses the next
constant-speed segment of the single-DoF prism trajectory.
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
from ami_simulator.flow import estimate_flow_features
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.rotation import ConstantRotation
from ami_simulator.types import CameraIntrinsics, EventStream
from ami_simulator.warp import PrismWarper, compensate_events


SPEEDS = (0.0, 180.0, 360.0, 720.0)


def _load_scene(scene: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, CameraIntrinsics]:
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s")
    intrinsics_data = json.loads((scene / "camera_intrinsics.json").read_text(encoding="utf-8"))
    intrinsics = CameraIntrinsics(frames.shape[2], frames.shape[1], *(float(intrinsics_data[key]) for key in ("fx", "fy", "cx", "cy")))
    masks = np.stack([np.asarray(imageio.imread(path)) > 0 for path in sorted((scene / "object_masks").glob("*.png"))])
    return rgb_to_linear_luminance(frames), timestamps, masks, intrinsics


def _choose_speed(
    flow: float | None,
    previous: dict[str, float] | None,
    *,
    flow_threshold: float = 0.35,
    edge_threshold: float = 0.40,
    efficiency_threshold: float = 0.75,
    overlap_threshold: float = 0.35,
) -> float:
    """Small interpretable rule: quality feedback dominates flow prior."""
    if previous is None:
        return 180.0
    edge = previous["edge_f1"]
    efficiency = previous["event_efficiency"]
    overlap = previous["temporal_overlap"]
    # Keep the useful fixed operating point unless there is strong evidence
    # that natural motion already supplies redundant excitation.
    if flow is not None and flow > flow_threshold and edge > edge_threshold and efficiency > efficiency_threshold and overlap < overlap_threshold:
        return 0.0
    if edge < 0.22 and (flow is None or flow < 0.18):
        return 360.0
    return 180.0


def _run_scene(luminance: np.ndarray, timestamps: np.ndarray, masks: np.ndarray, intrinsics: CameraIntrinsics, *, policy: str = "causal", controller_config: dict[str, float] | None = None) -> dict[str, object]:
    prism = RotatingWedgePrism(intrinsics, 2.0)
    simulator = ThresholdEventSimulator(0.2)
    flow = estimate_flow_features(np.repeat(luminance[..., None], 3, axis=-1), timestamps)[:, 0]
    phase = 0.0
    previous_active: np.ndarray | None = None
    previous_metrics: dict[str, float] | None = None
    rows: list[dict[str, float | int]] = []
    selected_speeds: list[float] = []
    for index, (start, end) in enumerate(zip(timestamps[:-1], timestamps[1:])):
        speed = 180.0 if policy == "fixed_180" else _choose_speed(float(flow[index - 1]) if index else None, previous_metrics, **(controller_config or {}))
        rotation = ConstantRotation(speed)
        pair_times = np.asarray([start, end], dtype=np.float64)
        pair = torch.from_numpy(luminance[index : index + 2]).unsqueeze(1)
        warper = PrismWarper(prism, rotation, height=intrinsics.height, width=intrinsics.width, theta0=phase, reference_time=float(start))
        with torch.inference_mode():
            sensor, valid = warper.warp(pair, pair_times)
        events = simulator.simulate(sensor[:, 0].numpy(), pair_times)
        compensated = compensate_events(events, prism, rotation, theta0=phase, reference_time=float(start))
        reference_events = EventStream(np.rint(compensated.extras["x_reference"]), np.rint(compensated.extras["y_reference"]), compensated.t, compensated.p)
        edge_mask = compute_reference_edge_mask(luminance[index], edge_quantile=0.75) & masks[index]
        metrics = event_quality_metrics(reference_events, reference_edge_mask=edge_mask, previous_active_pixels=previous_active, image_shape=luminance.shape[1:])
        metrics["event_density"] = float(len(events) / (intrinsics.width * intrinsics.height))
        useful = float(metrics["edge_f1"] + 0.25 * metrics["edge_coverage"] + 0.15 * metrics["unique_pixel_ratio"] + 0.65 * metrics["event_efficiency"] - 0.5 * metrics["temporal_overlap"] - 0.05 * metrics["polarity_imbalance"])
        rows.append({**{key: float(value) for key, value in metrics.items()}, "useful_score": useful, "speed": speed, "valid_fraction": float(valid.all(dim=0).float().mean().item())})
        previous_metrics = {key: float(metrics[key]) for key in ("edge_f1", "event_efficiency", "temporal_overlap")}
        previous_active = active_pixel_ids(reference_events, image_shape=luminance.shape[1:])
        phase = float(np.remainder(phase + rotation.omega_rad_s * (end - start), 2.0 * np.pi))
        selected_speeds.append(speed)
    return {"mean_useful_score": float(np.mean([row["useful_score"] for row in rows])), "mean_event_density": float(np.mean([row["event_density"] for row in rows])), "mean_edge_f1": float(np.mean([row["edge_f1"] for row in rows])), "mean_event_efficiency": float(np.mean([row["event_efficiency"] for row in rows])), "mean_speed": float(np.mean(selected_speeds)), "speed_change_fraction": float(np.mean(np.diff(selected_speeds) != 0.0)), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/comotion_controlled_v2"))
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("results/causal_trajectory_policy_test.json"))
    parser.add_argument("--flow-threshold", type=float, default=0.35)
    parser.add_argument("--edge-threshold", type=float, default=0.40)
    parser.add_argument("--efficiency-threshold", type=float, default=0.75)
    parser.add_argument("--overlap-threshold", type=float, default=0.35)
    args = parser.parse_args()
    manifest = json.loads((args.dataset_root / "manifest.json").read_text(encoding="utf-8"))
    controller_config = {"flow_threshold": args.flow_threshold, "edge_threshold": args.edge_threshold, "efficiency_threshold": args.efficiency_threshold, "overlap_threshold": args.overlap_threshold}
    rows: list[dict[str, object]] = []
    for record in manifest["records"]:
        if record["split"] != args.split:
            continue
        loaded = _load_scene(args.dataset_root / record["scene"])
        for policy in ("causal", "fixed_180"):
            result = _run_scene(*loaded, policy=policy, controller_config=controller_config)
            rows.append({"scene": record["scene"], "condition": record["condition"], "policy": policy, **{key: value for key, value in result.items() if key != "rows"}, "windows": len(result["rows"])})
    aggregate = {policy: {key: float(np.mean([float(row[key]) for row in rows if row["policy"] == policy])) for key in ("mean_useful_score", "mean_event_density", "mean_edge_f1", "mean_event_efficiency", "mean_speed", "speed_change_fraction")} for policy in ("causal", "fixed_180")}
    report = {"evaluation": "causal trajectory-segment policy", "dataset": manifest.get("dataset"), "split": args.split, "controller_config": controller_config, "speeds_rpm": list(SPEEDS), "aggregate": aggregate, "delta_useful_vs_fixed_180": float(aggregate["causal"]["mean_useful_score"] - aggregate["fixed_180"]["mean_useful_score"]), "rows": rows, "limitations": ["Rule-based causal baseline; thresholds must be selected on train only for a final claim.", "Fixed 180 RPM is resimulated by the same per-window code path.", "Sensing-objective evidence only; no downstream matching task."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
