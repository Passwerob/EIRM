#!/usr/bin/env python3
"""Collect residual labels on states visited by an anchor-residual policy.

The policy is allowed to execute its own switch decisions. At each visited
state, all three bounded RPM candidates are still evaluated at the current
continuous prism phase, producing labels for a DAgger-style second training
round. The simulator and optical model are shared with the existing rollout
evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.motion_net import CausalAnchorResidualPolicy
from scripts.evaluate_anchor_residual_policy import FEATURE_DIM, RESIDUAL_RPM, _state
from scripts.evaluate_phase_continuous_control import GEAR_RPM, _load_scene, _simulate_action


ANCHOR_RPM = 180.0
CANDIDATE_RPM = (0.0, ANCHOR_RPM, 360.0)


def _collect_scene(
    scene: Path,
    oracle_path: Path,
    objective: dict[str, float],
    *,
    model: CausalAnchorResidualPolicy,
    checkpoint: dict[str, object],
    switch_margin: float,
    switch_confidence: float,
    window: int,
    max_phase_step_deg: float,
) -> dict[str, np.ndarray | str]:
    luminance, timestamps, masks, intrinsics, oracle = _load_scene(scene, oracle_path)
    try:
        count = len(timestamps) - 1
        prism = RotatingWedgePrism(intrinsics, 2.0)
        feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        feature_clip = float(checkpoint["feature_clip"])
        states: list[np.ndarray] = []
        candidate_utilities = np.zeros((count, len(CANDIDATE_RPM)), dtype=np.float32)
        candidate_efficiency = np.zeros((count, len(CANDIDATE_RPM)), dtype=np.float32)
        anchor_utilities = np.zeros(count, dtype=np.float32)
        executed_rpm = np.zeros(count, dtype=np.float32)
        phase = 0.0
        previous_active = None
        previous_metrics: dict[str, float] | None = None
        previous_rpm = ANCHOR_RPM
        dwell_seconds = 0.0
        for index in range(count):
            state = _state(
                oracle,
                index=index,
                previous=previous_metrics,
                phase=phase,
                dwell_seconds=dwell_seconds,
                previous_rpm=previous_rpm,
            )
            states.append(state)
            candidates = [
                _simulate_action(
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
                    max_phase_step_deg=max_phase_step_deg,
                )
                for rpm in CANDIDATE_RPM
            ]
            candidate_utilities[index] = np.asarray([item["utility"] for item in candidates], dtype=np.float32)
            candidate_efficiency[index] = np.asarray([item["metrics"]["event_efficiency"] for item in candidates], dtype=np.float32)
            anchor_utilities[index] = float(candidates[1]["utility"])

            selected_index = 1
            if index >= window - 1:
                history = np.stack(states[-window:])
                normalized = np.clip((history - feature_mean) / feature_std, -feature_clip, feature_clip)
                with torch.inference_mode():
                    predicted_delta, switch_logit = model(torch.from_numpy(normalized[None].astype(np.float32)))
                delta = predicted_delta[0].numpy()
                probability = float(torch.sigmoid(switch_logit[0]).item())
                if float(np.max(delta)) > switch_margin and probability > switch_confidence:
                    selected_index = 0 if int(np.argmax(delta)) == 0 else 2

            selected = candidates[selected_index]
            rpm = float(CANDIDATE_RPM[selected_index])
            executed_rpm[index] = rpm
            dt = float(timestamps[index + 1] - timestamps[index])
            previous_metrics = {
                key: float(selected["metrics"][key])
                for key in (
                    "event_density",
                    "edge_coverage",
                    "structural_ratio",
                    "unique_pixel_ratio",
                    "temporal_overlap",
                    "polarity_imbalance",
                    "event_efficiency",
                )
            }
            previous_active = selected["active"]
            if index and rpm != previous_rpm:
                dwell_seconds = dt
            else:
                dwell_seconds += dt
            previous_rpm = rpm
            phase = float(selected["phase_end"])
        return {
            "features": np.asarray(states, dtype=np.float32),
            "candidate_rpm": np.asarray(CANDIDATE_RPM, dtype=np.float64),
            "candidate_utility": candidate_utilities,
            "candidate_event_efficiency": candidate_efficiency,
            "anchor_utility": anchor_utilities,
            "anchor_event_efficiency": candidate_efficiency[:, 1],
            "delta_utility": candidate_utilities[:, (0, 2)] - anchor_utilities[:, None],
            "delta_event_efficiency": candidate_efficiency[:, (0, 2)] - candidate_efficiency[:, 1, None],
            "executed_rpm": executed_rpm,
            "imu_motion": np.asarray(oracle["imu_motion"], dtype=np.float32),
            "timestamps": np.asarray(timestamps[1:], dtype=np.float64),
            "scene": scene.name,
        }
    finally:
        oracle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/comotion_v2_learned/manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/comotion_controlled_v2"))
    parser.add_argument("--policy", type=Path, default=Path("results/comotion_anchor_residual_policy/policy.pt"))
    parser.add_argument("--output-root", type=Path, default=Path("results/comotion_anchor_residual_dagger_dataset"))
    parser.add_argument("--split", choices=("train", "valid", "test", "all"), default="train")
    parser.add_argument("--switch-margin", type=float, default=0.02)
    parser.add_argument("--switch-confidence", type=float, default=0.70)
    parser.add_argument("--max-scenes-per-split", type=int, default=None)
    parser.add_argument("--max-phase-step-deg", type=float, default=30.0)
    args = parser.parse_args()
    if args.switch_margin < 0.0 or not 0.0 <= args.switch_confidence <= 1.0 or args.max_phase_step_deg <= 0.0:
        parser.error("switch margin must be non-negative, confidence must lie in [0, 1], and phase step must be positive")
    if args.max_scenes_per_split is not None and args.max_scenes_per_split < 1:
        parser.error("max-scenes-per-split must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.policy, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "anchor_residual_mlp" or checkpoint.get("feature_dim") != FEATURE_DIM:
        raise ValueError("The checkpoint is not the expected anchor-residual policy.")
    model = CausalAnchorResidualPolicy(feature_dim=FEATURE_DIM, hidden_dim=int(checkpoint["hidden_dim"]), residual_actions=int(checkpoint["residual_actions"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    records: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}
    for record in manifest["oracles"]:
        split = str(record["split"])
        if args.split != "all" and split != args.split:
            continue
        if args.max_scenes_per_split is not None and split_counts.get(split, 0) >= args.max_scenes_per_split:
            continue
        split_counts[split] = split_counts.get(split, 0) + 1
        output = args.output_root / split / f"{record['scene']}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        collected = _collect_scene(
            args.dataset_root / str(record["scene"]),
            Path(str(record["output"])),
            manifest["configuration"],
            model=model,
            checkpoint=checkpoint,
            switch_margin=float(args.switch_margin),
            switch_confidence=float(args.switch_confidence),
            window=int(checkpoint["window"]),
            max_phase_step_deg=float(args.max_phase_step_deg),
        )
        np.savez_compressed(output, **{key: value for key, value in collected.items() if isinstance(value, np.ndarray)})
        records.append({"split": split, "scene": record["scene"], "condition": record["condition"], "seed": record["seed"], "output": str(output), "windows": int(record["windows"])})
        print(f"wrote {output}")
    output_manifest = {
        "dataset": "AMI anchor residual DAgger deployment-state dataset",
        "source_manifest": str(args.manifest),
        "source_policy": str(args.policy),
        "switch_margin": float(args.switch_margin),
        "switch_confidence": float(args.switch_confidence),
        "anchor_rpm": ANCHOR_RPM,
        "candidate_rpm": list(CANDIDATE_RPM),
        "feature_dim": FEATURE_DIM,
        "records": records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(output_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"records": len(records), "output_root": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
