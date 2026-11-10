#!/usr/bin/env python3
"""Collect exact horizon-2 edge-F1 returns on states visited by a student.

The teacher is privileged (it can evaluate the simulator target edge map) and
is used only for distillation.  Every candidate branch carries the same
persistent event-sensor threshold state, prism phase, previous RPM, and dwell
state, so labels are causal and do not reset the sensor between actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ami_simulator.flow import estimate_dense_flow, summarize_motion_clusters
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.motion_net import CausalAnchorResidualGRUPolicy, CausalAnchorResidualPolicy
from scripts.evaluate_anchor_residual_policy import CLUSTER_FEATURE_DIM, _state
from scripts.evaluate_phase_continuous_control import _load_scene
from scripts.run_control_headroom_stateful import ACTIONS, RolloutState, _transition


def _score(metrics: dict[str, float], *, precision_weight: float, coverage_weight: float, overlap_weight: float) -> float:
    """Score target support; event-count efficiency is deliberately excluded."""
    return float(
        metrics["edge_f1"]
        + precision_weight * metrics["edge_precision"]
        + coverage_weight * metrics["edge_coverage"]
        - overlap_weight * metrics["temporal_overlap"]
    )


def _teacher_q(scene, oracle, objective, prism, state: RolloutState, index: int, *, dwell: float, horizon: int = 2, precision_weight: float = 0.0, coverage_weight: float = 0.0, overlap_weight: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate action sequences and return weighted and raw horizon returns."""
    timestamps = scene[1]
    remaining = min(horizon, len(timestamps) - 1 - index)
    values = np.full(len(ACTIONS), -np.inf, dtype=np.float32)
    raw_values = np.full(len(ACTIONS), -np.inf, dtype=np.float32)
    for first_i, first in enumerate(ACTIONS):
        result1, state1 = _transition(scene, oracle, objective, prism, state, index, first, dwell=dwell)
        score1 = _score(result1["metrics"], precision_weight=precision_weight, coverage_weight=coverage_weight, overlap_weight=overlap_weight)
        raw1 = float(result1["metrics"]["edge_f1"])
        if remaining == 1:
            values[first_i] = score1
            raw_values[first_i] = raw1
            continue
        feasible2 = ACTIONS if (timestamps[index + 2] - state1.last_switch_time + 1e-12 >= dwell) else (state1.previous_rpm,)
        scores2 = []
        for second in feasible2:
            result2, _ = _transition(scene, oracle, objective, prism, state1, index + 1, second, dwell=dwell)
            scores2.append((score1 + _score(result2["metrics"], precision_weight=precision_weight, coverage_weight=coverage_weight, overlap_weight=overlap_weight)) / 2.0)
        raw_scores2 = []
        for second in feasible2:
            result2, _ = _transition(scene, oracle, objective, prism, state1, index + 1, second, dwell=dwell)
            raw_scores2.append((raw1 + float(result2["metrics"]["edge_f1"])) / 2.0)
        values[first_i] = max(scores2) if scores2 else score1
        raw_values[first_i] = max(raw_scores2) if raw_scores2 else raw1
    return values, raw_values


def _collect(record: dict[str, object], root: Path, objective: dict[str, float], model: torch.nn.Module, checkpoint: dict[str, object], *, margin: float, confidence: float, dwell: float, precision_weight: float, coverage_weight: float, overlap_weight: float) -> dict[str, np.ndarray]:
    luminance, timestamps, masks, intrinsics, oracle = _load_scene(root / str(record["scene"]), Path(str(record["output"])))
    try:
        scene = (luminance, timestamps, masks, intrinsics)
        prism = RotatingWedgePrism(intrinsics, 2.0)
        cluster = np.stack([summarize_motion_clusters(estimate_dense_flow(luminance[i], luminance[i + 1]), clusters=3) for i in range(len(timestamps) - 1)]).astype(np.float32)
        mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        clip = float(checkpoint.get("feature_clip", 5.0)); window = int(checkpoint.get("window", 8))
        state = RolloutState(0.0, None, None, 180.0, float(timestamps[0]) - dwell)
        states: list[np.ndarray] = []; q_values: list[np.ndarray] = []; raw_q_values: list[np.ndarray] = []; executed: list[float] = []
        objective = {**objective, "event_efficiency_weight": 0.0}
        for index in range(len(timestamps) - 1):
            feature = _state(oracle, index=index, previous=state.previous_metrics, phase=state.phase, dwell_seconds=float(timestamps[index] - state.last_switch_time), previous_rpm=state.previous_rpm, cluster_features=cluster)
            states.append(feature)
            q, raw_q = _teacher_q(scene, oracle, objective, prism, state, index, dwell=dwell, precision_weight=precision_weight, coverage_weight=coverage_weight, overlap_weight=overlap_weight)
            q_values.append(q)
            raw_q_values.append(raw_q)
            rpm = 180.0
            if index >= window - 1:
                history = np.clip((np.stack(states[-window:]) - mean) / std, -clip, clip)
                with torch.inference_mode():
                    delta, logit = model(torch.from_numpy(history[None].astype(np.float32)))
                pred = delta[0].numpy(); prob = float(torch.sigmoid(logit[0]).item())
                if float(np.max(pred)) > margin and prob > confidence and float(timestamps[index] - state.last_switch_time) >= dwell:
                    rpm = 0.0 if int(np.argmax(pred)) == 0 else 360.0
            _, state = _transition(scene, oracle, objective, prism, state, index, rpm, dwell=dwell)
            executed.append(rpm)
        q_array = np.asarray(q_values, dtype=np.float32)
        raw_array = np.asarray(raw_q_values, dtype=np.float32)
        return {"features": np.asarray(states, dtype=np.float32), "q_h2": q_array, "q_h2_edge_f1": raw_array, "teacher_action": np.argmax(q_array + np.asarray([0.0, 1e-7, 0.0], dtype=np.float32), axis=1).astype(np.int64), "teacher_action_edge_f1": np.argmax(raw_array + np.asarray([0.0, 1e-7, 0.0], dtype=np.float32), axis=1).astype(np.int64), "executed_rpm": np.asarray(executed, dtype=np.float32), "timestamps": np.asarray(timestamps[:-1], dtype=np.float32)}
    finally:
        oracle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/comotion_v2_learned/manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/comotion_controlled_v2"))
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test", "all"), default="train")
    parser.add_argument("--switch-margin", type=float, default=0.08)
    parser.add_argument("--switch-confidence", type=float, default=0.60)
    parser.add_argument("--minimum-dwell-seconds", type=float, default=0.025)
    parser.add_argument("--precision-weight", type=float, default=0.0)
    parser.add_argument("--coverage-weight", type=float, default=0.0)
    parser.add_argument("--overlap-weight", type=float, default=0.0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.student, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") not in ("anchor_residual_gru", "anchor_residual_mlp") or int(checkpoint.get("feature_dim", 0)) != CLUSTER_FEATURE_DIM:
        raise ValueError("Student must be a 47-dimensional cluster-flow residual checkpoint.")
    klass = CausalAnchorResidualGRUPolicy if checkpoint["architecture"] == "anchor_residual_gru" else CausalAnchorResidualPolicy
    model = klass(feature_dim=CLUSTER_FEATURE_DIM, hidden_dim=int(checkpoint["hidden_dim"]), residual_actions=2)
    model.load_state_dict(checkpoint["model"]); model.eval()
    records = []
    for record in manifest["oracles"]:
        if args.split != "all" and record["split"] != args.split:
            continue
        output = args.output_root / str(record["split"]) / f"{record['scene']}.npz"; output.parent.mkdir(parents=True, exist_ok=True)
        data = _collect(record, args.dataset_root, manifest["configuration"], model, checkpoint, margin=args.switch_margin, confidence=args.switch_confidence, dwell=args.minimum_dwell_seconds, precision_weight=args.precision_weight, coverage_weight=args.coverage_weight, overlap_weight=args.overlap_weight)
        np.savez_compressed(output, **data)
        records.append({"split": record["split"], "scene": record["scene"], "condition": record["condition"], "seed": record["seed"], "output": str(output), "windows": int(len(data["features"]))})
        print(f"wrote {output}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {"dataset": "stateful horizon-2 privileged optical-observability teacher", "source_manifest": str(args.manifest), "source_student": str(args.student), "feature_dim": CLUSTER_FEATURE_DIM, "candidate_rpm": list(ACTIONS), "horizon": 2, "return": "edge_f1 + precision_weight*edge_precision + coverage_weight*edge_coverage - overlap_weight*temporal_overlap", "weights": {"edge_f1": 1.0, "edge_precision": args.precision_weight, "edge_coverage": args.coverage_weight, "temporal_overlap": args.overlap_weight, "event_efficiency": 0.0}, "records": records}
    (args.output_root / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"records": len(records), "output_root": str(args.output_root)}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
