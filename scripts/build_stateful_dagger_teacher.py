#!/usr/bin/env python3
"""Collect privileged teacher labels on states visited by a causal student."""

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
from scripts.run_control_headroom_stateful import ACTIONS, RolloutState, _choose_action, _transition
from scripts.evaluate_phase_continuous_control import _load_scene


def _collect(record: dict[str, object], root: Path, objective: dict[str, float], model: torch.nn.Module | None, checkpoint: dict[str, object] | None, *, margin: float, confidence: float, dwell: float) -> dict[str, np.ndarray]:
    loaded = _load_scene(root / str(record["scene"]), Path(str(record["output"])))
    luminance, timestamps, masks, intrinsics, oracle = loaded
    try:
        scene = (luminance, timestamps, masks, intrinsics)
        prism = RotatingWedgePrism(intrinsics, 2.0)
        cluster = np.stack([summarize_motion_clusters(estimate_dense_flow(luminance[i], luminance[i + 1]), clusters=3) for i in range(len(timestamps) - 1)]).astype(np.float32)
        feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32) if checkpoint is not None else np.zeros(CLUSTER_FEATURE_DIM, dtype=np.float32)
        feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32) if checkpoint is not None else np.ones(CLUSTER_FEATURE_DIM, dtype=np.float32)
        clip = float(checkpoint.get("feature_clip", 5.0)) if checkpoint is not None else 5.0
        window = int(checkpoint.get("window", 8)) if checkpoint is not None else 8
        state = RolloutState(phase=0.0, previous_active=None, sensor_reference_log=None, previous_rpm=180.0, last_switch_time=float(timestamps[0]) - dwell)
        states: list[np.ndarray] = []
        features: list[np.ndarray] = []
        deltas: list[np.ndarray] = []
        anchors: list[float] = []
        executed: list[float] = []
        for index in range(len(timestamps) - 1):
            feature = _state(oracle, index=index, previous=state.previous_metrics, phase=state.phase, dwell_seconds=float(timestamps[index] - state.last_switch_time), previous_rpm=state.previous_rpm, cluster_features=cluster)
            states.append(feature)
            candidates = [_transition(scene, oracle, {**objective, "event_efficiency_weight": 0.0}, prism, state, index, rpm, dwell=dwell)[0] for rpm in ACTIONS]
            anchor_f1 = float(candidates[1]["metrics"]["edge_f1"])
            deltas.append(np.asarray([float(candidates[0]["metrics"]["edge_f1"]) - anchor_f1, float(candidates[2]["metrics"]["edge_f1"]) - anchor_f1], dtype=np.float32))
            anchors.append(anchor_f1)
            if model is None or index < window - 1:
                rpm = 180.0
            else:
                values = np.stack(states[-window:])
                normalized = np.clip((values - feature_mean) / feature_std, -clip, clip)
                with torch.inference_mode():
                    predicted_delta, switch_logit = model(torch.from_numpy(normalized[None].astype(np.float32)))
                delta = predicted_delta[0].numpy()
                if float(np.max(delta)) > margin and float(torch.sigmoid(switch_logit[0]).item()) > confidence and float(timestamps[index] - state.last_switch_time) >= dwell:
                    rpm = 0.0 if int(np.argmax(delta)) == 0 else 360.0
                else:
                    rpm = 180.0
            _, state = _transition(scene, oracle, {**objective, "event_efficiency_weight": 0.0}, prism, state, index, rpm, dwell=dwell)
            executed.append(float(rpm))
            features.append(feature)
        return {"features": np.asarray(features, dtype=np.float32), "delta_edge_f1": np.asarray(deltas, dtype=np.float32), "anchor_edge_f1": np.asarray(anchors, dtype=np.float32), "executed_rpm": np.asarray(executed, dtype=np.float32)}
    finally:
        oracle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/comotion_v2_learned/manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/comotion_controlled_v2"))
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test", "all"), default="train")
    parser.add_argument("--switch-margin", type=float, default=0.05)
    parser.add_argument("--switch-confidence", type=float, default=0.70)
    parser.add_argument("--minimum-dwell-seconds", type=float, default=0.025)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.student, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") not in ("anchor_residual_gru", "anchor_residual_mlp") or int(checkpoint.get("feature_dim", 0)) != CLUSTER_FEATURE_DIM:
        raise ValueError("Student must be a 47-dimensional cluster-flow residual checkpoint.")
    klass = CausalAnchorResidualGRUPolicy if checkpoint["architecture"] == "anchor_residual_gru" else CausalAnchorResidualPolicy
    model = klass(feature_dim=CLUSTER_FEATURE_DIM, hidden_dim=int(checkpoint["hidden_dim"]), residual_actions=2)
    model.load_state_dict(checkpoint["model"]); model.eval()
    records: list[dict[str, object]] = []
    for record in manifest["oracles"]:
        if args.split != "all" and record["split"] != args.split:
            continue
        output = args.output_root / str(record["split"]) / f"{record['scene']}.npz"; output.parent.mkdir(parents=True, exist_ok=True)
        values = _collect(record, args.dataset_root, manifest["configuration"], model, checkpoint, margin=args.switch_margin, confidence=args.switch_confidence, dwell=args.minimum_dwell_seconds)
        np.savez_compressed(output, **values)
        records.append({"split": record["split"], "scene": record["scene"], "condition": record["condition"], "seed": record["seed"], "output": str(output), "windows": int(len(values["features"]))})
        print(f"wrote {output}")
    result = {"dataset": "stateful DAgger privileged edge-F1 teacher", "source_manifest": str(args.manifest), "source_student": str(args.student), "feature_dim": CLUSTER_FEATURE_DIM, "candidate_rpm": list(ACTIONS), "records": records, "label": "delta_edge_f1_on_student_visited_state"}
    args.output_root.mkdir(parents=True, exist_ok=True); (args.output_root / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"records": len(records), "output_root": str(args.output_root)}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
