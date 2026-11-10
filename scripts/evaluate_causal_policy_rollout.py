#!/usr/bin/env python3
"""Evaluate autonomous causal RPM rollouts against fixed-gear baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ami_simulator.control import CausalGearSelector
from ami_simulator.motion_net import CausalFlowFusionGearPolicy, CausalFlowFusionQPolicy, CausalFlowFusionRPMPolicy, CausalGatedFlowQPolicy, CausalRPMPolicy


GEAR_RPM = np.asarray([1080.0, 720.0, 360.0, 180.0, 0.0], dtype=np.float64)


def _active_masks(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "active_masks_packed" not in data:
        raise ValueError("Oracle lacks active_masks_packed; regenerate it with --store-active-masks.")
    count = int(data["active_mask_pixel_count"])
    return np.unpackbits(data["active_masks_packed"], axis=-1)[..., :count].astype(bool)


def _overlap(current: np.ndarray, previous: np.ndarray | None) -> float:
    if previous is None:
        return 0.0
    union = np.count_nonzero(current | previous)
    return float(np.count_nonzero(current & previous) / union) if union else 0.0


def _evaluate(data: np.lib.npyio.NpzFile, labels: np.ndarray, masks: np.ndarray, objective: dict[str, float]) -> dict[str, float | int]:
    previous_mask = None
    previous_rpm = 720.0
    values: list[float] = []
    overlaps: list[float] = []
    for index, gear in enumerate(labels):
        overlap = _overlap(masks[index, gear], previous_mask)
        score = float(data["edge_f1"][index, gear]) + 0.25 * float(data["coverage"][index, gear])
        score += 0.15 * float(data["unique_ratio"][index, gear])
        score += objective["event_efficiency_weight"] * float(data["event_efficiency"][index, gear])
        score -= objective["overlap_weight"] * overlap + 0.05 * float(data["polarity_balance"][index, gear])
        density = float(data["densities"][index, gear])
        target = float(data["target_density"][index])
        score -= objective["density_weight"] * float((np.log1p(density) - np.log1p(target)) ** 2)
        rpm = float(GEAR_RPM[gear])
        score -= objective["invalid_fraction_weight"] * (1.0 - float(data["valid_fraction"][index, gear]))
        score -= objective["rpm_effort_weight"] * (rpm / GEAR_RPM[0]) ** 2
        score -= objective["rpm_change_weight"] * ((rpm - previous_rpm) / GEAR_RPM[0]) ** 2
        values.append(score)
        overlaps.append(overlap)
        previous_mask = masks[index, gear]
        previous_rpm = rpm
    indices = np.arange(len(labels))
    return {
        "windows": int(len(labels)),
        "mean_objective": float(np.mean(values)),
        "mean_event_density": float(np.mean(data["densities"][indices, labels])),
        "mean_event_efficiency": float(np.mean(data["event_efficiency"][indices, labels])),
        "mean_edge_precision": float(np.mean(data["edge_precision"][indices, labels])),
        "mean_edge_recall": float(np.mean(data["coverage"][indices, labels])),
        "mean_edge_f1": float(np.mean(data["edge_f1"][indices, labels])),
        "mean_temporal_overlap": float(np.mean(overlaps)),
        "mean_rpm": float(np.mean(GEAR_RPM[labels])),
        "gear_change_fraction": float(np.mean(np.diff(labels) != 0)) if len(labels) > 1 else 0.0,
    }


def _filtered_targets(targets: np.ndarray, timestamps: np.ndarray, *, dwell: float) -> np.ndarray:
    start = float(timestamps[0] - np.median(np.diff(timestamps))) if len(timestamps) > 1 else float(timestamps[0])
    selector = CausalGearSelector(GEAR_RPM, initial_rpm=720.0, initial_time=start, minimum_dwell_seconds=dwell, switch_margin=0.0)
    labels = np.empty(len(targets), dtype=np.int64)
    for index, (target, timestamp) in enumerate(zip(targets, timestamps)):
        costs = np.square((GEAR_RPM - float(target)) / GEAR_RPM[0])
        labels[index], _ = selector.select(costs, timestamp=float(timestamp))
    return labels


def _policy_rollout(data: np.lib.npyio.NpzFile, masks: np.ndarray, model: object, *, window: int, dwell: float, switch_margin: float, use_flow: bool, motion_input: str, target_type: str, feature_mean: np.ndarray, feature_std: np.ndarray, feature_clip: float) -> np.ndarray:
    count = len(data["labels"])
    features = np.zeros((count, model.feature_dim), dtype=np.float32)
    if use_flow and ("flow_features" not in data or "imu_motion" not in data):
        raise ValueError("Flow-fusion rollout requires flow_features and imu_motion in the oracle.")
    timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    start = float(timestamps[0] - np.median(np.diff(timestamps))) if len(timestamps) > 1 else float(timestamps[0])
    selector = CausalGearSelector(GEAR_RPM, initial_rpm=720.0, initial_time=start, minimum_dwell_seconds=dwell, switch_margin=switch_margin)
    labels = np.empty(count, dtype=np.int64)
    previous_mask = None
    for index in range(count):
        if index < window - 1:
            prediction = 720.0
        else:
            normalized = np.clip((features[index - window + 1 : index + 1] - feature_mean) / feature_std, -feature_clip, feature_clip)
            values = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
            with torch.inference_mode():
                output = model(values)
                if target_type == "gear":
                    prediction = int(torch.argmax(output, dim=-1).item())
                elif target_type == "candidate_q":
                    prediction = output[0].numpy()
                else:
                    prediction = float(output.item())
        if index < window - 1:
            costs = np.square((GEAR_RPM - float(prediction)) / GEAR_RPM[0])
        elif target_type == "gear":
            costs = np.ones(len(GEAR_RPM), dtype=np.float64)
            costs[int(prediction)] = 0.0
        elif target_type == "candidate_q":
            costs = -np.asarray(prediction, dtype=np.float64)
        else:
            costs = np.square((GEAR_RPM - float(prediction)) / GEAR_RPM[0])
        labels[index], _ = selector.select(costs, timestamp=float(timestamps[index]))
        gear = labels[index]
        current_mask = masks[index, gear]
        overlap = _overlap(current_mask, previous_mask)
        if index + 1 < count:
            motion_key = "imu_motion" if motion_input == "imu" else "natural_motion"
            features[index + 1, 0] = float(data[motion_key][index])
            features[index + 1, 1] = float(data["densities"][index, gear]) / 10.0
            features[index + 1, 2] = float(data["coverage"][index, gear])
            features[index + 1, 3] = float(data["structural_ratio"][index, gear])
            features[index + 1, 4] = float(data["unique_ratio"][index, gear])
            features[index + 1, 5] = overlap
            features[index + 1, 6] = float(data["polarity_balance"][index, gear])
            features[index + 1, 7] = float(data["event_efficiency"][index, gear])
            features[index + 1, 8] = float(GEAR_RPM[gear] / GEAR_RPM[0])
            if use_flow:
                features[index + 1, 9:] = np.asarray(data["flow_features"][index], dtype=np.float32)
        previous_mask = current_mask
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    policy_path = args.policy or Path(manifest["policy_report"]["output"])
    checkpoint = torch.load(policy_path, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture", "tcn")
    use_flow = bool(checkpoint.get("use_flow", False))
    motion_input = str(checkpoint.get("motion_input", "imu" if use_flow else "natural"))
    target_type = str(checkpoint.get("target_type", "continuous"))
    if target_type == "gear":
        model = CausalFlowFusionGearPolicy(checkpoint["feature_dim"], checkpoint["hidden_dim"], checkpoint["num_gears"])
    elif target_type == "candidate_q":
        if architecture == "gated_flow_lstm":
            model = CausalGatedFlowQPolicy(checkpoint["feature_dim"], checkpoint["hidden_dim"], checkpoint["num_gears"], base_feature_dim=checkpoint["base_feature_dim"])
        else:
            model = CausalFlowFusionQPolicy(checkpoint["feature_dim"], checkpoint["hidden_dim"], checkpoint["num_gears"])
    else:
        model_cls = CausalFlowFusionRPMPolicy if architecture == "lstm" else CausalRPMPolicy
        model = model_cls(checkpoint["feature_dim"], checkpoint["hidden_dim"], min_rpm=checkpoint["min_rpm"], max_rpm=checkpoint["max_rpm"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    feature_mean = np.asarray(checkpoint.get("feature_mean", np.zeros(checkpoint["feature_dim"])), dtype=np.float32)
    feature_std = np.asarray(checkpoint.get("feature_std", np.ones(checkpoint["feature_dim"])), dtype=np.float32)
    feature_clip = float(checkpoint.get("feature_clip", np.inf))
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    objective = manifest["configuration"]
    dwell = float(objective["minimum_dwell_seconds"])
    switch_margin = float(objective.get("switch_margin", 0.0))
    for record in manifest["oracles"]:
        if record["split"] != args.split:
            continue
        with np.load(record["output"]) as data:
            if "prism_phase_start" in data:
                raise ValueError(
                    "Phase-continuous oracle tables are valid only on their selected trajectory. "
                    "Autonomous policy and fixed baselines require direct phase-continuous resimulation."
                )
            masks = _active_masks(data)
            timestamps = np.asarray(data["timestamps"], dtype=np.float64)
            schedules = {
                "oracle_feasible": np.asarray(data["labels"], dtype=np.int64),
                "learned_causal": _policy_rollout(data, masks, model, window=args.window, dwell=dwell, switch_margin=switch_margin, use_flow=use_flow, motion_input=motion_input, target_type=target_type, feature_mean=feature_mean, feature_std=feature_std, feature_clip=feature_clip),
                "motion_prior": _filtered_targets(GEAR_RPM[0] * (1.0 - np.asarray(data["natural_motion"])), timestamps, dwell=dwell),
                "random_dwell": _filtered_targets(GEAR_RPM[rng.integers(0, len(GEAR_RPM), len(timestamps))], timestamps, dwell=dwell),
            }
            schedules.update({f"fixed_{int(rpm)}rpm": np.full(len(timestamps), gear, dtype=np.int64) for gear, rpm in enumerate(GEAR_RPM)})
            for policy, labels in schedules.items():
                rows.append({"scene": record["scene"], "time_scale": float(record["time_scale"]), "policy": policy, **_evaluate(data, labels, masks, objective)})
    aggregates: list[dict[str, object]] = []
    for policy in sorted({row["policy"] for row in rows}):
        selected = [row for row in rows if row["policy"] == policy]
        total = sum(int(row["windows"]) for row in selected)
        aggregate = {"policy": policy, "windows": total}
        for key in ("mean_objective", "mean_event_density", "mean_event_efficiency", "mean_edge_precision", "mean_edge_recall", "mean_edge_f1", "mean_temporal_overlap", "mean_rpm", "gear_change_fraction"):
            aggregate[key] = float(sum(float(row[key]) * int(row["windows"]) for row in selected) / total)
        aggregates.append(aggregate)
    report = {
        "manifest": str(args.manifest),
        "policy_checkpoint": str(policy_path),
        "split": args.split,
        "evaluation": "autonomous quantized five-gear rollout with schedule-specific temporal overlap",
        "policy_architecture": architecture,
        "target_type": target_type,
        "use_flow": use_flow,
        "motion_input": motion_input,
        "limitations": ["This evaluates the sensing objective, not a downstream perception task.", "Continuous policy RPM predictions are quantized to the five simulated gears."],
        "aggregate": aggregates,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(aggregates, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
