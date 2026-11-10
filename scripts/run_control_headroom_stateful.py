#!/usr/bin/env python3
"""Measure Stage B control headroom with persistent sensor threshold state.

This pilot compares fixed baselines, a causal event-activity feedback rule, a
one-step feasible hindsight reference, and a receding-horizon exact enumeration
over {0, 180, 360} RPM. Hindsight methods may use target edge F1 for selection;
they are diagnostic references, not deployable controllers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ami_simulator.geometry import RotatingWedgePrism
from scripts.evaluate_phase_continuous_control import _load_scene, _simulate_action


ACTIONS = (0.0, 180.0, 360.0)


@dataclass
class RolloutState:
    phase: float
    previous_active: np.ndarray | None
    sensor_reference_log: np.ndarray | None
    previous_rpm: float
    last_switch_time: float
    previous_metrics: dict[str, float] | None = None


def _feasible_actions(state: RolloutState, end_time: float, dwell: float) -> tuple[float, ...]:
    if end_time - state.last_switch_time + 1e-12 < dwell:
        return (state.previous_rpm,)
    return ACTIONS


def _transition(scene, oracle, objective, prism, state: RolloutState, index: int, rpm: float, *, dwell: float) -> tuple[dict[str, object], RolloutState]:
    luminance, timestamps, masks, _ = scene
    result = _simulate_action(
        luminance,
        timestamps,
        masks,
        prism,
        index=index,
        rpm=rpm,
        phase=state.phase,
        previous_active=state.previous_active,
        previous_rpm=state.previous_rpm,
        target_density=float(oracle["target_density"][index]),
        objective=objective,
        contrast_threshold=0.2,
        max_phase_step_deg=30.0,
        sensor_reference_log=state.sensor_reference_log,
    )
    switched = rpm != state.previous_rpm
    metrics = {key: float(value) for key, value in result["metrics"].items()}
    next_state = RolloutState(
        phase=float(result["phase_end"]),
        previous_active=np.asarray(result["active"]).copy(),
        sensor_reference_log=np.asarray(result["sensor_reference_log"]).copy(),
        previous_rpm=float(rpm),
        last_switch_time=float(timestamps[index + 1]) if switched else state.last_switch_time,
        previous_metrics=metrics,
    )
    return result, next_state


def _edge_score(result: dict[str, object]) -> float:
    return float(result["metrics"]["edge_f1"])


def _choose_action(policy: str, scene, oracle, objective, prism, state: RolloutState, index: int, *, dwell: float, horizon: int) -> float:
    timestamps = scene[1]
    feasible = _feasible_actions(state, float(timestamps[index + 1]), dwell)
    if policy == "fixed_180":
        return 180.0
    if policy == "fixed_360":
        return 360.0
    if policy == "stopped_prism":
        return 0.0
    if policy == "causal_feedback":
        metrics = state.previous_metrics
        if metrics is None:
            desired = 180.0
        elif metrics["event_density"] < 0.05:
            desired = 360.0
        elif metrics["event_density"] > 0.75 or metrics["temporal_overlap"] > 0.65:
            desired = 180.0
        else:
            desired = state.previous_rpm
        return desired if desired in feasible else state.previous_rpm
    if policy == "greedy_hindsight":
        candidates = [(_transition(scene, oracle, objective, prism, state, index, rpm, dwell=dwell)[0], rpm) for rpm in feasible]
        return max(candidates, key=lambda item: (_edge_score(item[0]), -item[1]))[1]
    if policy != "exact_horizon":
        raise ValueError(f"Unknown policy {policy!r}")

    remaining = min(horizon, len(timestamps) - 1 - index)
    best_score = -np.inf
    best_first = state.previous_rpm

    def search(branch_state: RolloutState, offset: int, total: float, first: float | None) -> None:
        nonlocal best_score, best_first
        if offset == remaining:
            mean = total / remaining
            candidate_first = state.previous_rpm if first is None else first
            if mean > best_score + 1e-12 or (abs(mean - best_score) <= 1e-12 and candidate_first < best_first):
                best_score, best_first = mean, candidate_first
            return
        step = index + offset
        actions = _feasible_actions(branch_state, float(timestamps[step + 1]), dwell)
        for action in actions:
            result, next_state = _transition(scene, oracle, objective, prism, branch_state, step, action, dwell=dwell)
            search(next_state, offset + 1, total + _edge_score(result), action if first is None else first)

    search(state, 0, 0.0, None)
    return float(best_first)


def _run_policy(scene, oracle, objective, *, policy: str, prism_angle_deg: float, dwell: float, horizon: int) -> dict[str, object]:
    luminance, timestamps, _, intrinsics = scene
    prism = RotatingWedgePrism(intrinsics, prism_angle_deg)
    state = RolloutState(phase=0.0, previous_active=None, sensor_reference_log=None, previous_rpm=180.0, last_switch_time=float(timestamps[0]) - dwell)
    records: list[dict[str, float]] = []
    rpms: list[float] = []
    changes: list[bool] = []
    deviations: list[bool] = []
    for index in range(len(timestamps) - 1):
        rpm = _choose_action(policy, scene, oracle, objective, prism, state, index, dwell=dwell, horizon=horizon)
        result, next_state = _transition(scene, oracle, objective, prism, state, index, rpm, dwell=dwell)
        metrics = {key: float(value) for key, value in result["metrics"].items()}
        records.append(metrics)
        rpms.append(rpm)
        deviations.append(rpm != 180.0)
        if index:
            changes.append(rpm != state.previous_rpm)
        state = next_state
    return {
        "windows": len(records),
        "mean_edge_f1": float(np.mean([row["edge_f1"] for row in records])),
        "mean_edge_recall": float(np.mean([row["edge_coverage"] for row in records])),
        "mean_edge_precision": float(np.mean([row["edge_precision"] for row in records])),
        "mean_event_density": float(np.mean([row["event_density"] for row in records])),
        "mean_event_efficiency": float(np.mean([row["event_efficiency"] for row in records])),
        "mean_edge_event_efficiency": float(np.mean([row["edge_event_efficiency"] for row in records])),
        "mean_temporal_overlap": float(np.mean([row["temporal_overlap"] for row in records])),
        "mean_rpm": float(np.mean(rpms)),
        "switch_transition_fraction": float(np.mean(changes)) if changes else 0.0,
        "anchor_deviation_fraction": float(np.mean(deviations)),
        "total_events_per_pixel": float(np.sum([row["event_density"] for row in records])),
    }


def _bootstrap_delta(rows: list[dict[str, object]], policy: str, metric: str, *, samples: int = 5000) -> dict[str, float]:
    by_seed: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {}).setdefault(str(row["policy"]), []).append(float(row[metric]))
    seeds = sorted(seed for seed, values in by_seed.items() if policy in values and "fixed_180" in values)
    delta = np.asarray([np.mean(by_seed[seed][policy]) - np.mean(by_seed[seed]["fixed_180"]) for seed in seeds], dtype=np.float64)
    rng = np.random.default_rng(0)
    boot = np.asarray([np.mean(delta[rng.integers(0, len(delta), len(delta))]) for _ in range(samples)]) if len(delta) else np.asarray([np.nan])
    return {"source_scenes": len(delta), "mean_delta": float(np.mean(delta)) if len(delta) else float("nan"), "ci95_low": float(np.quantile(boot, 0.025)), "ci95_high": float(np.quantile(boot, 0.975))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/comotion_v2_learned/manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/comotion_controlled_v2"))
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--max-scenes-per-condition", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--minimum-dwell-seconds", type=float, default=0.025)
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/control_headroom_stateful_smoke.json"))
    args = parser.parse_args()
    if args.max_scenes_per_condition < 1 or args.horizon < 1 or args.minimum_dwell_seconds < 0.0:
        parser.error("scene count and horizon must be positive, dwell must be non-negative")
    started = time.perf_counter()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    objective = dict(manifest["configuration"])
    objective["event_efficiency_weight"] = 0.0
    policies = ("no_prism", "stopped_prism", "fixed_180", "fixed_360", "causal_feedback", "greedy_hindsight", "exact_horizon")
    rows: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for record in manifest["oracles"]:
        condition = str(record["condition"])
        if record["split"] != args.split or counts.get(condition, 0) >= args.max_scenes_per_condition:
            continue
        counts[condition] = counts.get(condition, 0) + 1
        loaded = _load_scene(args.dataset_root / str(record["scene"]), Path(str(record["output"])))
        luminance, timestamps, masks, intrinsics, oracle = loaded
        scene = (luminance, timestamps, masks, intrinsics)
        try:
            for policy in policies:
                actual_policy = "stopped_prism" if policy == "no_prism" else policy
                prism_angle = 0.0 if policy == "no_prism" else 2.0
                result = _run_policy(scene, oracle, objective, policy=actual_policy, prism_angle_deg=prism_angle, dwell=args.minimum_dwell_seconds, horizon=args.horizon)
                rows.append({"scene_id": record["scene"], "seed": int(record["seed"]), "condition": condition, "split": args.split, "policy": policy, **result})
        finally:
            oracle.close()
    aggregate = []
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        item = {"policy": policy, "source_scenes": len({int(row["seed"]) for row in selected}), "scene_conditions": len(selected)}
        for metric in ("mean_edge_f1", "mean_edge_recall", "mean_edge_precision", "mean_event_density", "mean_event_efficiency", "mean_edge_event_efficiency", "mean_temporal_overlap", "mean_rpm", "switch_transition_fraction", "anchor_deviation_fraction", "total_events_per_pixel"):
            item[metric] = float(np.mean([float(row[metric]) for row in selected]))
        item["edge_f1_delta_vs_fixed180"] = _bootstrap_delta(rows, policy, "mean_edge_f1") if policy != "fixed_180" else {"source_scenes": item["source_scenes"], "mean_delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
        aggregate.append(item)
    config = {"split": args.split, "max_scenes_per_condition": args.max_scenes_per_condition, "horizon": args.horizon, "minimum_dwell_seconds": args.minimum_dwell_seconds, "actions_rpm": ACTIONS, "event_efficiency_weight": 0.0}
    report = {
        "run_id": f"ami_next_stateful_headroom_{args.split}_h{args.horizon}_n{args.max_scenes_per_condition}",
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "configuration": config,
        "method": "persistent_threshold_state_receding_horizon_pilot",
        "sensor_model": "deterministic_log_threshold_uncalibrated",
        "phase_source": "command_integrated_selected_action",
        "simulated_or_real": "simulated",
        "runtime_seconds": float(time.perf_counter() - started),
        "selection_metric": "target_edge_f1_hindsight_only; causal_feedback_uses_global_event_activity",
        "state_contract": "phase, previous active pixels, per-pixel threshold reference, previous RPM, last switch time are branched together",
        "aggregate": aggregate,
        "rows": rows,
        "limitations": ["Validation pilot only; source scene count is intentionally small.", "Horizon search is exact only for the declared finite action set and horizon.", "No motor acceleration/ramp or readout noise model.", "Hindsight methods use GT target edges and are not deployable."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    fields = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output), "runtime_seconds": report["runtime_seconds"], "aggregate": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
