#!/usr/bin/env python3
"""Directly resimulate causal prism control with policy-specific continuous phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import torch

from ami_simulator.control import CausalGearSelector
from ami_simulator.event_quality import active_pixel_ids, compute_reference_edge_mask, event_quality_metrics
from ami_simulator.events import CudaThresholdEventSimulator, ThresholdEventSimulator
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.motion_net import CausalFlowFusionQPolicy, CausalGatedFlowQPolicy
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.rotation import ConstantRotation
from ami_simulator.types import CameraIntrinsics, EventStream
from ami_simulator.warp import PrismWarper, compensate_events


GEAR_RPM = np.asarray([1080.0, 720.0, 360.0, 180.0, 0.0], dtype=np.float64)


def _load_scene(scene: Path, oracle_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, CameraIntrinsics, np.lib.npyio.NpzFile]:
    with np.load(oracle_path) as probe:
        if "labels" in probe:
            windows = len(probe["labels"])
        elif "candidate_utility" in probe:
            windows = len(probe["candidate_utility"])
        else:
            raise ValueError(f"Oracle {oracle_path} lacks a window-count field.")
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=windows + 1)
    intrinsics_data = json.loads((scene / "camera_intrinsics.json").read_text(encoding="utf-8"))
    intrinsics = CameraIntrinsics(frames.shape[2], frames.shape[1], float(intrinsics_data["fx"]), float(intrinsics_data["fy"]), float(intrinsics_data["cx"]), float(intrinsics_data["cy"]))
    mask_paths = sorted((scene / "object_masks").glob("*.png"))[: len(frames)]
    if len(mask_paths) != len(frames):
        raise ValueError(f"Scene {scene} requires one object mask per frame.")
    masks = np.stack([np.asarray(imageio.imread(path)) > 0 for path in mask_paths])
    return rgb_to_linear_luminance(frames), timestamps, masks, intrinsics, np.load(oracle_path)


def _overlap(active: np.ndarray, previous: np.ndarray | None) -> float:
    if previous is None:
        return 0.0
    union = np.union1d(active, previous)
    return float(len(np.intersect1d(active, previous)) / len(union)) if len(union) else 0.0


def _simulate_action(
    luminance: np.ndarray,
    timestamps: np.ndarray,
    roi_masks: np.ndarray,
    prism: RotatingWedgePrism,
    *,
    index: int,
    rpm: float,
    phase: float,
    previous_active: np.ndarray | None,
    previous_rpm: float,
    target_density: float,
    objective: dict[str, float],
    contrast_threshold: float,
    max_phase_step_deg: float,
    sensor_reference_log: np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    start, end = float(timestamps[index]), float(timestamps[index + 1])
    rotation = ConstantRotation(rpm)
    phase_delta = abs(rotation.omega_rad_s * (end - start))
    subdivisions = max(1, int(np.ceil(phase_delta / np.deg2rad(max_phase_step_deg))))
    pair_times = np.linspace(start, end, subdivisions + 1, dtype=np.float64)
    fractions = (pair_times - start) / max(end - start, 1e-12)
    pair_values = luminance[index][None] * (1.0 - fractions[:, None, None]) + luminance[index + 1][None] * fractions[:, None, None]
    torch_device = torch.device(device)
    pair = torch.from_numpy(pair_values.astype(np.float32)).unsqueeze(1).to(torch_device)
    warper = PrismWarper(prism, rotation, height=prism.intrinsics.height, width=prism.intrinsics.width, theta0=phase, reference_time=start)
    with torch.inference_mode():
        sensor, valid = warper.warp(pair, pair_times)
    if torch_device.type == "cuda":
        events, sensor_reference_end = CudaThresholdEventSimulator(contrast_threshold).simulate_with_state(
            sensor[:, 0], pair_times, reference_log=sensor_reference_log
        )
        compensated = compensate_events(events, prism, rotation, theta0=phase, reference_time=start, device=str(torch_device))
    else:
        events, sensor_reference_end = ThresholdEventSimulator(contrast_threshold).simulate_with_state(
            sensor[:, 0].numpy(), pair_times, reference_log=sensor_reference_log
        )
        compensated = compensate_events(events, prism, rotation, theta0=phase, reference_time=start)
    reference_events = EventStream(np.rint(compensated.extras["x_reference"]), np.rint(compensated.extras["y_reference"]), compensated.t, compensated.p)
    edge_mask = compute_reference_edge_mask(luminance[index], edge_quantile=0.75) & roi_masks[index]
    metrics = event_quality_metrics(reference_events, reference_edge_mask=edge_mask, previous_active_pixels=previous_active, image_shape=luminance.shape[1:])
    density = len(events) / float(luminance.shape[1] * luminance.shape[2])
    metrics["event_density"] = density
    utility = float(metrics["edge_f1"]) + 0.25 * float(metrics["edge_coverage"])
    utility += 0.15 * float(metrics["unique_pixel_ratio"])
    utility += objective["event_efficiency_weight"] * float(metrics["event_efficiency"])
    utility += objective.get("coverage_efficiency_weight", 0.0) * float(metrics["coverage_efficiency"])
    utility -= objective["overlap_weight"] * float(metrics["temporal_overlap"])
    utility -= 0.05 * float(metrics["polarity_imbalance"])
    utility -= objective["density_weight"] * float((np.log1p(density) - np.log1p(target_density)) ** 2)
    utility -= objective["invalid_fraction_weight"] * (1.0 - float(valid.all(dim=0).float().mean()))
    utility -= objective["rpm_effort_weight"] * (rpm / GEAR_RPM[0]) ** 2
    utility -= objective["rpm_change_weight"] * ((rpm - previous_rpm) / GEAR_RPM[0]) ** 2
    return {
        "utility": utility,
        "metrics": metrics,
        "active": active_pixel_ids(reference_events, image_shape=luminance.shape[1:]),
        "reference_events": reference_events,
        "sensor_events": events,
        "sensor_reference_log": sensor_reference_end,
        "phase_end": float(np.remainder(phase + rotation.omega_rad_s * (end - start), 2.0 * np.pi)),
    }


def _summary(records: list[dict[str, float]], rpm: list[float], changes: list[bool], motion: np.ndarray) -> dict[str, object]:
    summary: dict[str, object] = {
        "windows": len(records),
        "mean_objective": float(np.mean([record["utility"] for record in records])),
        "mean_event_density": float(np.mean([record["event_density"] for record in records])),
        "mean_event_efficiency": float(np.mean([record["event_efficiency"] for record in records])),
        "mean_edge_event_efficiency": float(np.mean([record.get("edge_event_efficiency", 0.0) for record in records])),
        "mean_edge_precision": float(np.mean([record["edge_precision"] for record in records])),
        "mean_edge_recall": float(np.mean([record["edge_coverage"] for record in records])),
        "mean_edge_f1": float(np.mean([record["edge_f1"] for record in records])),
        "mean_coverage_efficiency": float(np.mean([record["coverage_efficiency"] for record in records])),
        "mean_temporal_overlap": float(np.mean([record["temporal_overlap"] for record in records])),
        "mean_rpm": float(np.mean(rpm)),
        "gear_change_fraction": float(np.mean(changes)) if changes else 0.0,
    }
    # Motion fractions are normalized from training-only IMU bounds in the
    # oracle. Fixed bins make speed robustness comparable across policies.
    bins = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.000001))
    motion_metrics: dict[str, dict[str, float | int]] = {}
    values = np.asarray(motion, dtype=np.float64)
    for lower, upper in bins:
        mask = (values >= lower) & (values < upper)
        if not np.any(mask):
            continue
        selected = [records[index] for index in np.flatnonzero(mask)]
        selected_rpm = [rpm[index] for index in np.flatnonzero(mask)]
        motion_metrics[f"{lower:.2f}-{min(upper, 1.0):.2f}"] = {
            "windows": int(len(selected)),
            "mean_objective": float(np.mean([record["utility"] for record in selected])),
            "mean_edge_f1": float(np.mean([record["edge_f1"] for record in selected])),
            "mean_edge_recall": float(np.mean([record["edge_coverage"] for record in selected])),
            "mean_coverage_efficiency": float(np.mean([record["coverage_efficiency"] for record in selected])),
            "mean_event_density": float(np.mean([record["event_density"] for record in selected])),
            "mean_rpm": float(np.mean(selected_rpm)),
        }
    summary["motion_bins"] = motion_metrics
    if motion_metrics:
        bin_objectives = [float(item["mean_objective"]) for item in motion_metrics.values()]
        summary["motion_bin_objective_range"] = float(max(bin_objectives) - min(bin_objectives))
        summary["motion_bin_worst_objective"] = float(min(bin_objectives))
    # Detect abrupt natural-motion changes and quantify short-horizon retention.
    # This is descriptive: it does not assume that the post-switch optimum is
    # equal to the pre-switch optimum.
    transition_indices = [int(index + 1) for index, delta in enumerate(np.diff(values)) if abs(float(delta)) >= 0.20]
    transitions: list[dict[str, float | int]] = []
    for index in transition_indices:
        pre = records[max(0, index - 3) : index]
        post = records[index : min(len(records), index + 8)]
        if not pre or not post:
            continue
        reference = float(np.mean([record["utility"] for record in pre]))
        # A zero-event baseline has no meaningful pre-transition reference;
        # exclude it from retention statistics instead of dividing by epsilon.
        if reference <= 1e-6:
            continue
        post_values = np.asarray([record["utility"] for record in post], dtype=np.float64)
        threshold = 0.9 * reference
        recovered = np.flatnonzero(post_values >= threshold)
        transitions.append({
            "window_index": index,
            "motion_before": float(values[index - 1]),
            "motion_after": float(values[index]),
            "pre_transition_objective": reference,
            "post_transition_objective": float(np.mean(post_values[: min(3, len(post_values))])),
            # Immediate post-switch retention isolates the controller's
            # response before later windows can hide a transient failure.
            "retention_ratio": float(post_values[0] / max(reference, 1e-8)),
            "recovery_latency_windows": int(recovered[0]) if len(recovered) else len(post),
        })
    summary["transitions"] = transitions
    if transitions:
        summary["mean_transition_retention"] = float(np.mean([item["retention_ratio"] for item in transitions]))
        summary["mean_recovery_latency_windows"] = float(np.mean([item["recovery_latency_windows"] for item in transitions]))
    return summary


def _rollout(
    luminance: np.ndarray,
    timestamps: np.ndarray,
    roi_masks: np.ndarray,
    intrinsics: CameraIntrinsics,
    oracle: np.lib.npyio.NpzFile,
    objective: dict[str, float],
    *,
    policy_name: str,
    model: CausalFlowFusionQPolicy | None,
    checkpoint: dict[str, object] | None,
    window: int,
    fixed_gear: int | None = None,
    fallback_gear: int | None = None,
    fallback_margin: float = 0.0,
) -> dict[str, float | int]:
    count = len(timestamps) - 1
    prism = RotatingWedgePrism(intrinsics, 2.0)
    feature_dim = int(checkpoint["feature_dim"]) if checkpoint is not None else 0
    features = np.zeros((count, feature_dim), dtype=np.float32) if model is not None else None
    selector = CausalGearSelector(GEAR_RPM, initial_rpm=720.0, initial_time=float(timestamps[0]), minimum_dwell_seconds=objective["minimum_dwell_seconds"], switch_margin=objective["switch_margin"])
    phase = 0.0
    previous_active = None
    previous_rpm = 720.0
    selected_rpm: list[float] = []
    changes: list[bool] = []
    outputs: list[dict[str, float]] = []
    motion_values = np.asarray(oracle["imu_motion"], dtype=np.float64)
    motion_delta = bool(checkpoint.get("motion_delta", False)) if checkpoint is not None else False
    flow_start = int(checkpoint.get("flow_start", 9)) if checkpoint is not None else 9
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32) if checkpoint is not None else None
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32) if checkpoint is not None else None
    phase_features = bool(checkpoint.get("phase_features", False)) if checkpoint is not None else False
    for index in range(count):
        if model is not None and phase_features:
            features[index, -2:] = (np.sin(phase), np.cos(phase))
        if fixed_gear is not None:
            gear = fixed_gear
        elif policy_name == "oracle_direct":
            candidates = [
                _simulate_action(luminance, timestamps, roi_masks, prism, index=index, rpm=float(rpm), phase=phase, previous_active=previous_active, previous_rpm=previous_rpm, target_density=float(oracle["target_density"][index]), objective=objective, contrast_threshold=0.2, max_phase_step_deg=30.0)
                for rpm in GEAR_RPM
            ]
            costs = -np.asarray([candidate["utility"] for candidate in candidates])
            gear, _ = selector.select(costs, timestamp=float(timestamps[index + 1]))
            selected = candidates[gear]
        else:
            if index < window - 1:
                prediction = np.square((GEAR_RPM - 720.0) / GEAR_RPM[0])
            else:
                normalized = np.clip((features[index - window + 1 : index + 1] - mean) / std, -float(checkpoint["feature_clip"]), float(checkpoint["feature_clip"]))
                with torch.inference_mode():
                    q = model(torch.from_numpy(normalized).unsqueeze(0))[0].numpy()
                prediction = -q
                if fallback_gear is not None:
                    if not 0 <= fallback_gear < len(GEAR_RPM):
                        raise ValueError("fallback_gear must index GEAR_RPM.")
                    best = int(np.argmax(q))
                    if float(q[best] - q[fallback_gear]) < fallback_margin:
                        prediction = np.ones(len(GEAR_RPM), dtype=np.float64)
                        prediction[fallback_gear] = 0.0
            gear, _ = selector.select(prediction, timestamp=float(timestamps[index + 1]))
        if policy_name != "oracle_direct":
            selected = _simulate_action(luminance, timestamps, roi_masks, prism, index=index, rpm=float(GEAR_RPM[gear]), phase=phase, previous_active=previous_active, previous_rpm=previous_rpm, target_density=float(oracle["target_density"][index]), objective=objective, contrast_threshold=0.2, max_phase_step_deg=30.0)
        rpm = float(GEAR_RPM[gear])
        metrics = {"utility": float(selected["utility"]), **{key: float(value) for key, value in selected["metrics"].items()}}
        outputs.append(metrics)
        selected_rpm.append(rpm)
        if index:
            changes.append(rpm != previous_rpm)
        overlap = float(selected["metrics"]["temporal_overlap"])
        if model is not None and index + 1 < count:
            features[index + 1, 0] = float(oracle["imu_motion"][index])
            if motion_delta:
                features[index + 1, 9] = float(oracle["imu_motion"][index] - (oracle["imu_motion"][index - 1] if index else oracle["imu_motion"][index]))
            features[index + 1, 1] = float(selected["metrics"]["event_density"]) / 10.0
            features[index + 1, 2] = float(selected["metrics"]["edge_coverage"])
            features[index + 1, 3] = float(selected["metrics"]["structural_ratio"])
            features[index + 1, 4] = float(selected["metrics"]["unique_pixel_ratio"])
            features[index + 1, 5] = overlap
            features[index + 1, 6] = float(selected["metrics"]["polarity_imbalance"])
            features[index + 1, 7] = float(selected["metrics"]["event_efficiency"])
            features[index + 1, 8] = rpm / GEAR_RPM[0]
            if bool(checkpoint["use_flow"]):
                features[index + 1, flow_start : flow_start + 8] = np.asarray(oracle["flow_features"][index], dtype=np.float32)
        previous_active = selected["active"]
        previous_rpm = rpm
        phase = float(selected["phase_end"])
    return _summary(outputs, selected_rpm, changes, motion_values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--baselines", action="store_true", help="Also directly resimulate fixed gears and a greedy oracle.")
    parser.add_argument("--fallback-rpm", type=float, default=None, help="Fallback RPM for low-confidence learned predictions.")
    parser.add_argument("--fallback-margin", type=float, default=0.0, help="Minimum predicted-Q advantage over fallback before switching.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fallback_margin < 0.0:
        parser.error("fallback-margin must be non-negative")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.policy, map_location="cpu", weights_only=True)
    if checkpoint.get("target_type") != "candidate_q":
        raise ValueError("Direct evaluator currently accepts candidate-Q checkpoints only.")
    checkpoint_flow_start = int(checkpoint.get("flow_start", 9))
    if checkpoint.get("architecture") == "gated_flow_lstm":
        model = CausalGatedFlowQPolicy(checkpoint["feature_dim"], checkpoint["hidden_dim"], checkpoint["num_gears"], base_feature_dim=checkpoint["base_feature_dim"], flow_start=checkpoint_flow_start)
    else:
        model = CausalFlowFusionQPolicy(checkpoint["feature_dim"], checkpoint["hidden_dim"], checkpoint["num_gears"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows: list[dict[str, object]] = []
    for record in manifest["oracles"]:
        if record["split"] != args.split:
            continue
        scene = args.dataset_root / record["scene"]
        luminance, timestamps, masks, intrinsics, oracle = _load_scene(scene, Path(record["output"]))
        try:
            policies: list[tuple[str, CausalFlowFusionQPolicy | None, dict[str, object] | None, int | None]] = [("learned_causal", model, checkpoint, None)]
            if args.baselines:
                policies.extend((f"fixed_{int(rpm)}rpm", None, None, gear) for gear, rpm in enumerate(GEAR_RPM))
                policies.append(("oracle_direct", None, None, None))
            for name, policy_model, policy_checkpoint, fixed_gear in policies:
                fallback_gear = None
                if args.fallback_rpm is not None and policy_model is not None:
                    fallback_gear = int(np.argmin(np.abs(GEAR_RPM - float(args.fallback_rpm))))
                result = _rollout(luminance, timestamps, masks, intrinsics, oracle, manifest["configuration"], policy_name=name, model=policy_model, checkpoint=policy_checkpoint, window=args.window, fixed_gear=fixed_gear, fallback_gear=fallback_gear, fallback_margin=args.fallback_margin)
                rows.append({"scene": record["scene"], "condition": record["condition"], "policy": name, **result})
        finally:
            oracle.close()
    aggregate: list[dict[str, object]] = []
    for policy in sorted({str(row["policy"]) for row in rows}):
        selected = [row for row in rows if row["policy"] == policy]
        total = sum(int(row["windows"]) for row in selected)
        item: dict[str, object] = {"policy": policy, "windows": total}
        for key in ("mean_objective", "mean_event_density", "mean_event_efficiency", "mean_edge_event_efficiency", "mean_coverage_efficiency", "mean_edge_precision", "mean_edge_recall", "mean_edge_f1", "mean_temporal_overlap", "mean_rpm", "gear_change_fraction"):
            item[key] = float(sum(float(row[key]) * int(row["windows"]) for row in selected) / total)
        bin_names = sorted({name for row in selected for name in row.get("motion_bins", {})})
        motion_bins: dict[str, dict[str, float | int]] = {}
        for name in bin_names:
            bin_rows = [(row, row["motion_bins"][name]) for row in selected if name in row.get("motion_bins", {})]
            bin_total = sum(int(values["windows"]) for _, values in bin_rows)
            motion_bins[name] = {"windows": bin_total}
            for key in ("mean_objective", "mean_edge_f1", "mean_edge_recall", "mean_coverage_efficiency", "mean_event_density", "mean_rpm"):
                motion_bins[name][key] = float(sum(float(values[key]) * int(values["windows"]) for _, values in bin_rows) / bin_total)
        if motion_bins:
            item["motion_bins"] = motion_bins
            objectives = [float(values["mean_objective"]) for values in motion_bins.values()]
            item["motion_bin_objective_range"] = float(max(objectives) - min(objectives))
            item["motion_bin_worst_objective"] = float(min(objectives))
        transitions = [
            transition
            for row in selected
            for transition in row.get("transitions", [])
            if float(transition.get("pre_transition_objective", 0.0)) > 1e-6
        ]
        if transitions:
            item["transitions"] = transitions
            item["mean_transition_retention"] = float(np.mean([transition["retention_ratio"] for transition in transitions]))
            item["mean_recovery_latency_windows"] = float(np.mean([transition["recovery_latency_windows"] for transition in transitions]))
        aggregate.append(item)
    report = {"evaluation": "direct policy-specific phase-continuous resimulation", "split": args.split, "policy_checkpoint": str(args.policy), "phase_state": "encoder phase integrated after every selected action", "fallback_rpm": args.fallback_rpm, "fallback_margin": args.fallback_margin, "aggregate": aggregate, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
