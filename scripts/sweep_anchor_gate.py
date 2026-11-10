#!/usr/bin/env python3
"""Sweep a small causal gate grid on validation scenes only.

The neural checkpoint is fixed. Only the interpretable acceptance rule is
changed, so this experiment measures objective/gate design rather than model
capacity. The selected configuration must be fixed before a test rollout.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from ami_simulator.motion_net import CausalAnchorResidualPolicy
from scripts.evaluate_anchor_residual_policy import FEATURE_DIM, _load_scene, _run_scene


def _condition_deltas(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["condition"]), str(row["policy"])), []).append(row)
    output: dict[str, dict[str, float]] = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        adaptive = grouped[(condition, "anchor_residual")]
        fixed = grouped[(condition, "fixed_180")]
        values: dict[str, float] = {}
        for key in ("mean_objective", "mean_edge_f1", "mean_event_efficiency", "mean_coverage_efficiency"):
            values[f"delta_{key[5:]}"] = float(np.mean([float(row[key]) for row in adaptive]) - np.mean([float(row[key]) for row in fixed]))
        output[condition] = values
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/comotion_v2_learned/manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/comotion_controlled_v2"))
    parser.add_argument("--policy", type=Path, default=Path("results/comotion_anchor_residual_policy/policy_full_anchor.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/comotion_anchor_residual_policy/validation_gate_sweep.json"))
    parser.add_argument("--margins", type=float, nargs="+", default=[0.01, 0.02])
    parser.add_argument("--confidences", type=float, nargs="+", default=[0.60, 0.70])
    parser.add_argument("--efficiency-floors", type=float, nargs="+", default=[0.60, 0.70])
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.policy, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "anchor_residual_mlp" or checkpoint.get("feature_dim") != FEATURE_DIM:
        raise ValueError("The gate sweep expects the utility-only anchor-residual checkpoint.")
    model = CausalAnchorResidualPolicy(feature_dim=FEATURE_DIM, hidden_dim=int(checkpoint["hidden_dim"]), residual_actions=int(checkpoint["residual_actions"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    records = [record for record in manifest["oracles"] if record["split"] == "valid"]
    if not records:
        raise ValueError("No validation records found.")
    rows: list[dict[str, object]] = []
    configurations: list[dict[str, float | bool]] = []
    for margin in args.margins:
        for confidence in args.confidences:
            for efficiency_floor in args.efficiency_floors:
                config_rows: list[dict[str, object]] = []
                config_checkpoint = copy.copy(checkpoint)
                config_checkpoint["switch_margin"] = float(margin)
                config_checkpoint["switch_confidence"] = float(confidence)
                for record in records:
                    luminance, timestamps, masks, intrinsics, oracle = _load_scene(args.dataset_root / str(record["scene"]), Path(str(record["output"])))
                    try:
                        for policy in ("fixed_180", "anchor_residual"):
                            result = _run_scene(
                                luminance,
                                timestamps,
                                masks,
                                intrinsics,
                                oracle,
                                manifest["configuration"],
                                policy=policy,
                                model=model if policy == "anchor_residual" else None,
                                checkpoint=config_checkpoint if policy == "anchor_residual" else None,
                                window=int(checkpoint["window"]),
                                min_event_efficiency=float(efficiency_floor),
                            )
                            config_rows.append({"scene": record["scene"], "condition": record["condition"], **result})
                    finally:
                        oracle.close()
                adaptive = [row for row in config_rows if row["policy"] == "anchor_residual"]
                fixed = [row for row in config_rows if row["policy"] == "fixed_180"]
                aggregate = {
                    key: float(np.mean([float(row[key]) for row in adaptive]) - np.mean([float(row[key]) for row in fixed]))
                    for key in ("mean_objective", "mean_edge_f1", "mean_event_efficiency", "mean_coverage_efficiency")
                }
                deltas = _condition_deltas(config_rows)
                condition_objective = [value["delta_objective"] for value in deltas.values()]
                condition_edge = [value["delta_edge_f1"] for value in deltas.values()]
                condition_efficiency = [value["delta_event_efficiency"] for value in deltas.values()]
                switch_fraction = float(np.mean([float(row["gear_change_fraction"]) for row in adaptive]))
                eligible = bool(
                    min(condition_objective) >= 0.0
                    and min(condition_edge) >= -0.01
                    and min(condition_efficiency) >= -0.01
                    and switch_fraction <= 0.40
                )
                configurations.append({
                    "margin": float(margin),
                    "confidence": float(confidence),
                    "efficiency_floor": float(efficiency_floor),
                    "eligible": eligible,
                    "switch_fraction": switch_fraction,
                    "aggregate": aggregate,
                    "condition_deltas": deltas,
                })
                rows.extend({"margin": margin, "confidence": confidence, "efficiency_floor": efficiency_floor, **row} for row in config_rows)
                print(f"evaluated margin={margin:g} confidence={confidence:g} efficiency_floor={efficiency_floor:g}")
    eligible = [item for item in configurations if item["eligible"]]
    def violation(item: dict[str, object]) -> tuple[float, float, float, float]:
        values = list(item["condition_deltas"].values())
        objective = min(float(value["delta_objective"]) for value in values)
        edge = min(float(value["delta_edge_f1"]) for value in values)
        efficiency = min(float(value["delta_event_efficiency"]) for value in values)
        return (max(0.0, -objective), max(0.0, -0.01 - edge), max(0.0, -0.01 - efficiency), max(0.0, float(item["switch_fraction"]) - 0.40))
    selected = max(eligible, key=lambda item: (min(value["delta_objective"] for value in item["condition_deltas"].values()), item["aggregate"]["mean_objective"], item["aggregate"]["mean_event_efficiency"], -item["switch_fraction"])) if eligible else None
    best_relaxed = min(configurations, key=lambda item: (violation(item), -min(value["delta_objective"] for value in item["condition_deltas"].values()), -item["aggregate"]["mean_objective"])) if configurations else None
    report = {
        "evaluation": "validation-only anchor gate sweep",
        "policy": str(args.policy),
        "constraints": {"min_condition_objective_delta": 0.0, "min_condition_edge_f1_delta": -0.01, "min_condition_event_efficiency_delta": -0.01, "max_switch_fraction": 0.40},
        "selected": selected,
        "best_relaxed": best_relaxed,
        "selection_status": "feasible" if selected is not None else "no_configuration_meets_declared_constraints",
        "configurations": configurations,
        "rows": rows,
        "limitations": ["The selected gate must be frozen before test evaluation.", "This sweep changes only a causal acceptance rule; the neural checkpoint is fixed.", "Metrics are sensing surrogates and do not replace downstream feature matching."]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(selected, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
