#!/usr/bin/env python3
"""Directly evaluate the anchor-residual policy with continuous prism phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.flow import estimate_dense_flow, summarize_motion_clusters
from ami_simulator.motion_net import CausalAnchorQualityResidualPolicy, CausalAnchorResidualGRUPolicy, CausalAnchorResidualPolicy
from scripts.evaluate_phase_continuous_control import GEAR_RPM, _load_scene, _simulate_action, _summary


FEATURE_DIM = 20
CLUSTER_FEATURE_DIM = 47
ANCHOR_RPM = 180.0
RESIDUAL_RPM = (0.0, 360.0)


def _state(
    oracle: np.lib.npyio.NpzFile,
    *,
    index: int,
    previous: dict[str, float] | None,
    phase: float,
    dwell_seconds: float,
    previous_rpm: float,
    cluster_features: np.ndarray | None = None,
) -> np.ndarray:
    values = np.zeros(CLUSTER_FEATURE_DIM if cluster_features is not None else FEATURE_DIM, dtype=np.float32)
    if previous is not None:
        values[0] = float(oracle["imu_motion"][index - 1])
        values[1] = float(previous["event_density"]) / 10.0
        values[2] = float(previous["edge_coverage"])
        values[3] = float(previous["structural_ratio"])
        values[4] = float(previous["unique_pixel_ratio"])
        values[5] = float(previous["temporal_overlap"])
        values[6] = float(previous["polarity_imbalance"])
        values[7] = float(previous["event_efficiency"])
        values[8] = float(previous_rpm / GEAR_RPM[0])
        values[9:17] = np.asarray(oracle["flow_features"][index - 1], dtype=np.float32)
        if cluster_features is not None:
            values[20:47] = np.asarray(cluster_features[index - 1], dtype=np.float32)
    values[17] = float(np.sin(phase))
    values[18] = float(np.cos(phase))
    values[19] = float(dwell_seconds)
    return values


def _run_scene(
    luminance: np.ndarray,
    timestamps: np.ndarray,
    masks: np.ndarray,
    intrinsics,
    oracle: np.lib.npyio.NpzFile,
    objective: dict[str, float],
    *,
    policy: str,
    model: CausalAnchorResidualPolicy | CausalAnchorQualityResidualPolicy | None,
    checkpoint: dict[str, object] | None,
    window: int,
    min_event_efficiency: float = 0.0,
    minimum_dwell_seconds: float = 0.0,
    predicted_efficiency_floor: float | None = None,
) -> dict[str, object]:
    prism = RotatingWedgePrism(intrinsics, 2.0)
    phase = 0.0
    previous_active = None
    previous_metrics: dict[str, float] | None = None
    previous_rpm = ANCHOR_RPM
    dwell_seconds = 0.0
    states: list[np.ndarray] = []
    records: list[dict[str, float]] = []
    selected_rpm: list[float] = []
    changes: list[bool] = []
    cluster_features = None
    if checkpoint is not None and int(checkpoint.get("feature_dim", FEATURE_DIM)) == CLUSTER_FEATURE_DIM:
        cluster_features = np.stack(
            [summarize_motion_clusters(estimate_dense_flow(luminance[i], luminance[i + 1]), clusters=3) for i in range(len(timestamps) - 1)]
        ).astype(np.float32)
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32) if checkpoint is not None else None
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32) if checkpoint is not None else None
    margin = float(checkpoint.get("switch_margin", 0.02)) if checkpoint is not None else 0.0
    confidence = float(checkpoint.get("switch_confidence", 0.5)) if checkpoint is not None else 0.0
    for index in range(len(timestamps) - 1):
        state = _state(oracle, index=index, previous=previous_metrics, phase=phase, dwell_seconds=dwell_seconds, previous_rpm=previous_rpm, cluster_features=cluster_features)
        states.append(state)
        if policy == "fixed_180" or model is None or index < window - 1:
            rpm = ANCHOR_RPM
        else:
            history = np.stack(states[-window:])
            normalized = np.clip((history - mean) / std, -float(checkpoint["feature_clip"]), float(checkpoint["feature_clip"]))
            with torch.inference_mode():
                outputs = model(torch.from_numpy(normalized[None].astype(np.float32)))
            if len(outputs) == 3:
                predicted_delta, switch_logit, predicted_efficiency_delta = outputs
                predicted_efficiency = predicted_efficiency_delta[0].numpy()
            else:
                predicted_delta, switch_logit = outputs
                predicted_efficiency = None
            delta = predicted_delta[0].numpy()
            probability = float(torch.sigmoid(switch_logit[0]).item())
            previous_efficiency = 1.0 if previous_metrics is None else float(previous_metrics["event_efficiency"])
            efficiency_ok = predicted_efficiency is None or predicted_efficiency_floor is None or float(predicted_efficiency[int(np.argmax(delta))]) >= predicted_efficiency_floor
            if (
                float(np.max(delta)) > margin
                and probability > confidence
                and previous_efficiency >= min_event_efficiency
                and dwell_seconds >= minimum_dwell_seconds
                and efficiency_ok
            ):
                rpm = RESIDUAL_RPM[int(np.argmax(delta))]
            else:
                rpm = ANCHOR_RPM
        selected = _simulate_action(
            luminance,
            timestamps,
            masks,
            prism,
            index=index,
            rpm=float(rpm),
            phase=phase,
            previous_active=previous_active,
            previous_rpm=previous_rpm,
            target_density=float(oracle["target_density"][index]),
            objective=objective,
            contrast_threshold=0.2,
            max_phase_step_deg=30.0,
        )
        metrics = {"utility": float(selected["utility"]), **{key: float(value) for key, value in selected["metrics"].items()}}
        records.append(metrics)
        selected_rpm.append(float(rpm))
        if index:
            changes.append(float(rpm) != previous_rpm)
        dt = float(timestamps[index + 1] - timestamps[index])
        if index and float(rpm) != previous_rpm:
            dwell_seconds = dt
        else:
            dwell_seconds += dt
        previous_metrics = {key: float(selected["metrics"][key]) for key in ("event_density", "edge_coverage", "structural_ratio", "unique_pixel_ratio", "temporal_overlap", "polarity_imbalance", "event_efficiency")}
        previous_active = selected["active"]
        previous_rpm = float(rpm)
        phase = float(selected["phase_end"])
    return {**_summary(records, selected_rpm, changes, np.asarray(oracle["imu_motion"], dtype=np.float64)), "policy": policy}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/comotion_v2_learned/manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/comotion_controlled_v2"))
    parser.add_argument("--policy", type=Path, default=Path("results/comotion_anchor_residual_policy/policy.pt"))
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--switch-margin", type=float, default=None, help="Override the checkpoint switch margin.")
    parser.add_argument("--switch-confidence", type=float, default=None, help="Override the checkpoint switch confidence.")
    parser.add_argument("--min-event-efficiency", type=float, default=0.0, help="Require the previous window to meet this efficiency before switching.")
    parser.add_argument("--minimum-dwell-seconds", type=float, default=0.0, help="Require this causal dwell time before a learned switch.")
    parser.add_argument("--predicted-efficiency-floor", type=float, default=None, help="For quality checkpoints, require predicted residual event-efficiency delta above this value.")
    parser.add_argument("--event-efficiency-weight", type=float, default=None, help="Override objective weight during direct rollout.")
    parser.add_argument("--output", type=Path, default=Path("results/comotion_anchor_residual_policy/test_rollout.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    objective = dict(manifest["configuration"])
    if args.event_efficiency_weight is not None:
        if args.event_efficiency_weight < 0.0:
            parser.error("event-efficiency-weight must be non-negative")
        objective["event_efficiency_weight"] = float(args.event_efficiency_weight)
    checkpoint = torch.load(args.policy, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") not in ("anchor_residual_mlp", "anchor_residual_gru", "anchor_quality_residual_mlp") or checkpoint.get("feature_dim") not in (FEATURE_DIM, CLUSTER_FEATURE_DIM):
        raise ValueError("The checkpoint is not an anchor-residual policy with the expected feature layout.")
    if checkpoint.get("architecture") == "anchor_quality_residual_mlp":
        model_class = CausalAnchorQualityResidualPolicy
    elif checkpoint.get("architecture") == "anchor_residual_gru":
        model_class = CausalAnchorResidualGRUPolicy
    else:
        model_class = CausalAnchorResidualPolicy
    model = model_class(feature_dim=int(checkpoint["feature_dim"]), hidden_dim=int(checkpoint["hidden_dim"]), residual_actions=int(checkpoint["residual_actions"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if args.switch_margin is not None:
        checkpoint["switch_margin"] = float(args.switch_margin)
    if args.switch_confidence is not None:
        checkpoint["switch_confidence"] = float(args.switch_confidence)
    if args.predicted_efficiency_floor is not None:
        checkpoint["predicted_efficiency_floor"] = float(args.predicted_efficiency_floor)
    if float(checkpoint["switch_margin"]) < 0.0 or not 0.0 <= float(checkpoint["switch_confidence"]) <= 1.0 or not 0.0 <= args.min_event_efficiency <= 1.0 or args.minimum_dwell_seconds < 0.0 or (args.predicted_efficiency_floor is not None and args.predicted_efficiency_floor < -1.0):
        raise ValueError("switch margin must be non-negative and confidence must lie in [0, 1]")
    rows: list[dict[str, object]] = []
    records = manifest.get("oracles", manifest.get("records", []))
    if not records:
        raise ValueError("Manifest must contain oracles or records.")
    for record in records:
        if record["split"] != args.split:
            continue
        luminance, timestamps, masks, intrinsics, oracle = _load_scene(args.dataset_root / str(record["scene"]), Path(str(record["output"])))
        try:
            for policy in ("fixed_180", "anchor_residual"):
                result = _run_scene(luminance, timestamps, masks, intrinsics, oracle, objective, policy=policy, model=model if policy == "anchor_residual" else None, checkpoint=checkpoint if policy == "anchor_residual" else None, window=int(checkpoint["window"]), min_event_efficiency=float(args.min_event_efficiency), minimum_dwell_seconds=float(args.minimum_dwell_seconds), predicted_efficiency_floor=float(checkpoint.get("predicted_efficiency_floor", 0.0)) if checkpoint.get("architecture") == "anchor_quality_residual_mlp" else None)
                rows.append({"scene": record["scene"], "condition": record["condition"], **result})
        finally:
            oracle.close()
    aggregate: dict[str, dict[str, float]] = {}
    for policy in ("fixed_180", "anchor_residual"):
        selected = [row for row in rows if row["policy"] == policy]
        aggregate[policy] = {key: float(np.mean([float(row[key]) for row in selected])) for key in ("mean_objective", "mean_event_density", "mean_event_efficiency", "mean_edge_f1", "mean_coverage_efficiency", "mean_temporal_overlap", "mean_rpm", "gear_change_fraction")}
    report = {"evaluation": "direct phase-continuous anchor residual rollout", "split": args.split, "policy": str(args.policy), "architecture": checkpoint["architecture"], "target_objective": checkpoint.get("target_objective", "utility residual"), "efficiency_bonus": checkpoint.get("efficiency_bonus"), "event_efficiency_weight": float(objective.get("event_efficiency_weight", 0.0)), "switch_margin": float(checkpoint["switch_margin"]), "switch_confidence": float(checkpoint["switch_confidence"]), "min_event_efficiency": float(args.min_event_efficiency), "minimum_dwell_seconds": float(args.minimum_dwell_seconds), "predicted_efficiency_floor": checkpoint.get("predicted_efficiency_floor"), "aggregate": aggregate, "delta_objective_vs_fixed180": float(aggregate["anchor_residual"]["mean_objective"] - aggregate["fixed_180"]["mean_objective"]), "rows": rows, "limitations": ["Residual model is trained on anchor-state rollouts; model-induced post-switch states are not yet in training.", "Event efficiency is reported as a spatial-novelty diagnostic; the primary objective weight can be set to zero when repeated temporal excitation is intentionally useful.", "Minimum dwell is an actuator-feasibility constraint applied to learned switches.", "Useful objective is a sensing surrogate; downstream matching is not included."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
