#!/usr/bin/env python3
"""Evaluate a cluster-flow residual student with persistent sensor history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import torch

from ami_simulator.flow import estimate_dense_flow, summarize_motion_clusters
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.motion_net import CausalAnchorResidualGRUPolicy, CausalAnchorResidualPolicy
from scripts.evaluate_anchor_residual_policy import CLUSTER_FEATURE_DIM, _state
from scripts.evaluate_phase_continuous_control import _load_scene
from scripts.run_control_headroom_stateful import ACTIONS, RolloutState, _transition


def _cluster_sequence(luminance: np.ndarray) -> np.ndarray:
    return np.stack([summarize_motion_clusters(estimate_dense_flow(luminance[i], luminance[i + 1]), clusters=3) for i in range(len(luminance) - 1)]).astype(np.float32)


def _run(scene, oracle, objective, *, policy: str, model=None, checkpoint=None, margin: float = 0.05, confidence: float = 0.70, dwell: float = 0.025) -> dict[str, float]:
    luminance, timestamps, masks, intrinsics = scene
    prism = RotatingWedgePrism(intrinsics, 2.0)
    cluster = _cluster_sequence(luminance)
    state = RolloutState(phase=0.0, previous_active=None, sensor_reference_log=None, previous_rpm=180.0, last_switch_time=float(timestamps[0]) - dwell)
    states: list[np.ndarray] = []
    metrics: list[dict[str, float]] = []
    rpms: list[float] = []
    changes: list[bool] = []
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32) if checkpoint is not None else None
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32) if checkpoint is not None else None
    clip = float(checkpoint.get("feature_clip", 5.0)) if checkpoint is not None else 5.0
    window = int(checkpoint.get("window", 8)) if checkpoint is not None else 8
    for index in range(len(timestamps) - 1):
        feature = _state(oracle, index=index, previous=state.previous_metrics, phase=state.phase, dwell_seconds=float(timestamps[index] - state.last_switch_time), previous_rpm=state.previous_rpm, cluster_features=cluster)
        states.append(feature)
        rpm = 180.0
        if policy != "fixed_180" and model is not None and index >= window - 1:
            history = np.stack(states[-window:])
            normalized = np.clip((history - mean) / std, -clip, clip)
            with torch.inference_mode():
                delta, switch_logit = model(torch.from_numpy(normalized[None].astype(np.float32)))
            advantage = delta[0].numpy()
            if float(np.max(advantage)) > margin and float(torch.sigmoid(switch_logit[0]).item()) > confidence and float(timestamps[index] - state.last_switch_time) >= dwell:
                rpm = 0.0 if int(np.argmax(advantage)) == 0 else 360.0
        result, next_state = _transition(scene, oracle, objective, prism, state, index, rpm, dwell=dwell)
        metrics.append({key: float(value) for key, value in result["metrics"].items()})
        if index:
            changes.append(rpm != state.previous_rpm)
        rpms.append(rpm)
        state = next_state
    return {
        "windows": float(len(metrics)),
        "mean_edge_f1": float(np.mean([item["edge_f1"] for item in metrics])),
        "mean_edge_recall": float(np.mean([item["edge_coverage"] for item in metrics])),
        "mean_event_density": float(np.mean([item["event_density"] for item in metrics])),
        "mean_event_efficiency": float(np.mean([item["event_efficiency"] for item in metrics])),
        "mean_temporal_overlap": float(np.mean([item["temporal_overlap"] for item in metrics])),
        "mean_rpm": float(np.mean(rpms)),
        "switch_transition_fraction": float(np.mean(changes)) if changes else 0.0,
        "anchor_deviation_fraction": float(np.mean(np.asarray(rpms) != 180.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/comotion_v2_learned/manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/comotion_controlled_v2"))
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--switch-margin", type=float, default=0.05)
    parser.add_argument("--switch-confidence", type=float, default=0.70)
    parser.add_argument("--minimum-dwell-seconds", type=float, default=0.025)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.student, map_location="cpu", weights_only=True)
    if int(checkpoint.get("feature_dim", 0)) != CLUSTER_FEATURE_DIM:
        raise ValueError("Stateful student evaluator expects a 47-dimensional cluster-flow checkpoint.")
    klass = CausalAnchorResidualGRUPolicy if checkpoint.get("architecture") == "anchor_residual_gru" else CausalAnchorResidualPolicy
    model = klass(feature_dim=CLUSTER_FEATURE_DIM, hidden_dim=int(checkpoint["hidden_dim"]), residual_actions=2)
    model.load_state_dict(checkpoint["model"]); model.eval()
    objective = {**manifest["configuration"], "event_efficiency_weight": 0.0}
    rows: list[dict[str, object]] = []
    for record in manifest["oracles"]:
        if record["split"] != args.split:
            continue
        loaded = _load_scene(args.dataset_root / str(record["scene"]), Path(str(record["output"])))
        luminance, timestamps, masks, intrinsics, oracle = loaded
        try:
            scene = (luminance, timestamps, masks, intrinsics)
            for policy in ("fixed_180", "student"):
                result = _run(scene, oracle, objective, policy=policy, model=model if policy == "student" else None, checkpoint=checkpoint if policy == "student" else None, margin=args.switch_margin, confidence=args.switch_confidence, dwell=args.minimum_dwell_seconds)
                rows.append({"scene": record["scene"], "seed": int(record["seed"]), "condition": record["condition"], "policy": policy, **result})
        finally:
            oracle.close()
    aggregate = {}
    for policy in ("fixed_180", "student"):
        selected = [row for row in rows if row["policy"] == policy]
        aggregate[policy] = {key: float(np.mean([float(row[key]) for row in selected])) for key in ("mean_edge_f1", "mean_edge_recall", "mean_event_density", "mean_event_efficiency", "mean_temporal_overlap", "mean_rpm", "switch_transition_fraction", "anchor_deviation_fraction")}
    report = {"evaluation": "stateful learned student rollout", "split": args.split, "student": str(args.student), "feature_dim": CLUSTER_FEATURE_DIM, "event_efficiency_weight": 0.0, "aggregate": aggregate, "rows": rows, "limitations": ["Student is evaluated with persistent threshold state and continuous phase.", "The current DAgger labels are one-step edge-F1 residuals; full horizon-2 return distillation remains pending.", "No motor ramp/readout noise model."]}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
