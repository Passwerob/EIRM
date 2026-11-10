#!/usr/bin/env python3
"""Stateful rollout of a Q student that sees privileged GT ROI flow.

This is the P0 observability upper bound: the student is causal in time and
prism phase, but object/background flow uses simulator masks. A passing result
does not justify a paper controller claim until a mask-free observer matches it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ami_simulator.flow import CAUSAL_BASE_FEATURE_DIM, PRIVILEGED_ROI_Q_FEATURE_DIM, privileged_roi_sequence
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.motion_net import CausalAnchorQGRUPolicy
from scripts.evaluate_anchor_residual_policy import _state
from scripts.evaluate_phase_continuous_control import _load_scene
from scripts.run_control_headroom_stateful import ACTIONS, RolloutState, _transition

CONDITIONS = ("camera_only", "object_only", "co_motion", "counter_motion")
METRIC_KEYS = (
    "mean_edge_f1",
    "mean_edge_recall",
    "mean_edge_precision",
    "mean_event_density",
    "mean_event_efficiency",
    "mean_temporal_overlap",
    "mean_rpm",
    "switch_transition_fraction",
    "anchor_deviation_fraction",
)


def _privileged_state(
    oracle: np.lib.npyio.NpzFile,
    roi_features: np.ndarray,
    *,
    index: int,
    previous: dict[str, float] | None,
    phase: float,
    dwell_seconds: float,
    previous_rpm: float,
) -> np.ndarray:
    base = _state(oracle, index=index, previous=previous, phase=phase, dwell_seconds=dwell_seconds, previous_rpm=previous_rpm)[:CAUSAL_BASE_FEATURE_DIM]
    return np.concatenate((base, np.asarray(roi_features[index], dtype=np.float32)))


def _run(
    scene: tuple[np.ndarray, np.ndarray, np.ndarray, object],
    oracle: np.lib.npyio.NpzFile,
    objective: dict[str, float],
    *,
    model: CausalAnchorQGRUPolicy | None = None,
    checkpoint: dict[str, object] | None = None,
    margin: float = 0.02,
    confidence: float = 0.50,
    dwell: float = 0.025,
) -> dict[str, float]:
    luminance, timestamps, masks, intrinsics = scene
    prism = RotatingWedgePrism(intrinsics, 2.0)
    roi = privileged_roi_sequence(luminance, masks)
    state = RolloutState(0.0, None, None, 180.0, float(timestamps[0]) - dwell)
    features: list[np.ndarray] = []
    metrics: list[dict[str, float]] = []
    rpms: list[float] = []
    changes: list[bool] = []
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32) if checkpoint is not None else None
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32) if checkpoint is not None else None
    clip = float(checkpoint.get("feature_clip", 5.0)) if checkpoint is not None else 5.0
    window = int(checkpoint.get("window", 8)) if checkpoint is not None else 8
    for index in range(len(timestamps) - 1):
        feature = _privileged_state(
            oracle,
            roi,
            index=index,
            previous=state.previous_metrics,
            phase=state.phase,
            dwell_seconds=float(timestamps[index] - state.last_switch_time),
            previous_rpm=state.previous_rpm,
        )
        features.append(feature)
        rpm = 180.0
        if model is not None and checkpoint is not None and index >= window - 1:
            history = np.clip((np.stack(features[-window:]) - mean) / std, -clip, clip)
            with torch.inference_mode():
                q_values = model(torch.from_numpy(history[None].astype(np.float32)))[0].numpy()
            best = int(np.argmax(q_values))
            probabilities = torch.softmax(torch.from_numpy(q_values), dim=0).numpy()
            advantage = float(q_values[best] - q_values[1])
            ready = float(timestamps[index] - state.last_switch_time) >= dwell
            if best != 1 and advantage > margin and float(probabilities[best]) > confidence and ready:
                rpm = ACTIONS[best]
        result, next_state = _transition(scene, oracle, objective, prism, state, index, rpm, dwell=dwell)
        metrics.append({key: float(value) for key, value in result["metrics"].items()})
        rpms.append(rpm)
        if index:
            changes.append(rpm != state.previous_rpm)
        state = next_state
    return {
        "windows": float(len(metrics)),
        "mean_edge_f1": float(np.mean([row["edge_f1"] for row in metrics])),
        "mean_edge_recall": float(np.mean([row["edge_coverage"] for row in metrics])),
        "mean_edge_precision": float(np.mean([row["edge_precision"] for row in metrics])),
        "mean_event_density": float(np.mean([row["event_density"] for row in metrics])),
        "mean_event_efficiency": float(np.mean([row["event_efficiency"] for row in metrics])),
        "mean_temporal_overlap": float(np.mean([row["temporal_overlap"] for row in metrics])),
        "mean_rpm": float(np.mean(rpms)),
        "switch_transition_fraction": float(np.mean(changes)) if changes else 0.0,
        "anchor_deviation_fraction": float(np.mean(np.asarray(rpms) != 180.0)),
    }


def _aggregate(rows: list[dict[str, object]], policy: str) -> dict[str, float]:
    selected = [row for row in rows if row["policy"] == policy]
    summary = {key: float(np.mean([float(row[key]) for row in selected])) for key in METRIC_KEYS}
    for condition in CONDITIONS:
        subset = [row for row in selected if row["condition"] == condition]
        if subset:
            summary[f"{condition}_edge_f1"] = float(np.mean([float(row["mean_edge_f1"]) for row in subset]))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/comotion_v2_learned/manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/comotion_controlled_v2"))
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--switch-margin", type=float, default=0.005)
    parser.add_argument("--switch-confidence", type=float, default=0.35)
    parser.add_argument("--minimum-dwell-seconds", type=float, default=0.025)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.student, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "anchor_q_gru" or int(checkpoint.get("feature_dim", 0)) != PRIVILEGED_ROI_Q_FEATURE_DIM:
        raise ValueError(f"Expected {PRIVILEGED_ROI_Q_FEATURE_DIM}-D privileged ROI anchor_q_gru checkpoint.")
    model = CausalAnchorQGRUPolicy(
        feature_dim=PRIVILEGED_ROI_Q_FEATURE_DIM,
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_actions=int(checkpoint["num_actions"]),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    objective = {**manifest["configuration"], "event_efficiency_weight": 0.0}
    rows: list[dict[str, object]] = []
    for record in manifest.get("oracles", manifest.get("records", [])):
        if record["split"] != args.split:
            continue
        luminance, timestamps, masks, intrinsics, oracle = _load_scene(args.dataset_root / str(record["scene"]), Path(str(record["output"])))
        try:
            scene = (luminance, timestamps, masks, intrinsics)
            for policy in ("fixed_180", "privileged_roi_q"):
                result = _run(
                    scene,
                    oracle,
                    objective,
                    model=model if policy == "privileged_roi_q" else None,
                    checkpoint=checkpoint if policy == "privileged_roi_q" else None,
                    margin=args.switch_margin,
                    confidence=args.switch_confidence,
                    dwell=args.minimum_dwell_seconds,
                )
                rows.append({"scene": record["scene"], "seed": int(record.get("seed", 0)), "condition": record["condition"], "policy": policy, **result})
        finally:
            oracle.close()
    aggregate = {policy: _aggregate(rows, policy) for policy in ("fixed_180", "privileged_roi_q")}
    student = aggregate["privileged_roi_q"]
    baseline = aggregate["fixed_180"]
    report = {
        "evaluation": "P0 privileged ROI-flow horizon-2 Q student rollout",
        "privileged": True,
        "split": args.split,
        "student": str(args.student),
        "feature_dim": PRIVILEGED_ROI_Q_FEATURE_DIM,
        "candidate_rpm": list(ACTIONS),
        "switch_margin": args.switch_margin,
        "switch_confidence": args.switch_confidence,
        "minimum_dwell_seconds": args.minimum_dwell_seconds,
        "event_efficiency_weight": 0.0,
        "gate_co_motion_edge_f1": 0.02,
        "aggregate": aggregate,
        "delta_edge_f1_vs_fixed180": float(student["mean_edge_f1"] - baseline["mean_edge_f1"]),
        "delta_co_motion_edge_f1_vs_fixed180": float(student.get("co_motion_edge_f1", np.nan) - baseline.get("co_motion_edge_f1", np.nan)),
        "passed_co_motion_gate": bool(student.get("co_motion_edge_f1", 0.0) - baseline.get("co_motion_edge_f1", 0.0) >= 0.02),
        "rows": rows,
        "limitations": [
            "Object/background flow uses simulator masks at train and test; this is an observability upper bound.",
            "A pass supports learning a mask-free observer next. It is not a deployable controller claim.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("split", "aggregate", "delta_edge_f1_vs_fixed180", "delta_co_motion_edge_f1_vs_fixed180", "passed_co_motion_gate")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
