#!/usr/bin/env python3
"""Run the controlled co-motion IMU-only versus IMU+flow LSTM experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from ami_simulator.dsec import read_imu_csv
from ami_simulator.event_quality import QUALITY_PROTOCOL_VERSION


def _run(command: list[str], *, environment: dict[str, str], root: Path) -> None:
    completed = subprocess.run(command, cwd=root, env=environment, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{completed.stderr}\n{completed.stdout}")


def _condition(scene: str) -> str:
    for name in ("camera_only", "object_only", "co_motion", "counter_motion"):
        if scene.endswith(name):
            return name
    raise ValueError(f"Cannot infer condition from {scene!r}.")


def _summarize(report: dict[str, object], policy: str = "learned_causal") -> dict[str, object]:
    learned = [row for row in report["rows"] if row["policy"] == policy]
    groups: dict[str, dict[str, float]] = {}
    for condition in ("camera_only", "object_only", "co_motion", "counter_motion"):
        rows = [row for row in learned if _condition(str(row["scene"])) == condition]
        weights = np.asarray([row["windows"] for row in rows], dtype=np.float64)
        groups[condition] = {
            key: float(np.average([row[key] for row in rows], weights=weights))
            for key in ("mean_objective", "mean_edge_f1", "mean_edge_recall", "mean_event_density", "mean_event_efficiency", "mean_coverage_efficiency", "mean_rpm")
        }
    objectives = np.asarray([value["mean_objective"] for value in groups.values()])
    f1 = np.asarray([value["mean_edge_f1"] for value in groups.values()])
    coverage_efficiency = np.asarray([value["mean_coverage_efficiency"] for value in groups.values()])
    aggregate = next((item for item in report.get("aggregate", []) if item.get("policy") == policy), {})
    speed_summary = {
        key: aggregate[key]
        for key in ("motion_bins", "motion_bin_objective_range", "motion_bin_worst_objective")
        if key in aggregate
    }
    return {
        "groups": groups,
        "balanced_objective_mean": float(objectives.mean()),
        "balanced_objective_std": float(objectives.std()),
        "worst_condition_objective": float(objectives.min()),
        "balanced_edge_f1_mean": float(f1.mean()),
        "worst_condition_edge_f1": float(f1.min()),
        "balanced_coverage_efficiency_mean": float(coverage_efficiency.mean()),
        "worst_condition_coverage_efficiency": float(coverage_efficiency.min()),
        "speed_robustness": speed_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/comotion_controlled"))
    parser.add_argument("--output", type=Path, default=Path("results/comotion_flow_lstm"))
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--motion-profile", choices=("smooth", "switching"), default="smooth", help="Motion profile passed to the controlled-scene generator.")
    parser.add_argument("--motion-delta", action="store_true", help="Train policies with a causal one-window IMU speed-difference feature.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--coverage-efficiency-weight", type=float, default=0.0, help="Weight target-support coverage by unique-event efficiency in the frozen controller objective.")
    parser.add_argument("--event-efficiency-weight", type=float, default=0.0, help="Optional redundancy penalty/reward weight. Keep at zero for the primary repeated-event sensing analysis.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    environment.setdefault("OMP_NUM_THREADS", "4")
    environment.setdefault("MKL_NUM_THREADS", "4")
    if args.coverage_efficiency_weight < 0.0 or args.event_efficiency_weight < 0.0:
        parser.error("efficiency weights must be non-negative")
    if args.overwrite or not (args.dataset_root / "manifest.json").exists():
        _run([sys.executable, "scripts/generate_comotion_scenes.py", "--output", str(args.dataset_root), "--frames", str(args.frames), "--fps", str(args.fps), "--size", str(args.size), "--motion-profile", args.motion_profile], environment=environment, root=root)
    dataset = json.loads((args.dataset_root / "manifest.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    train_motion: list[float] = []
    for record in dataset["records"]:
        if record["split"] == "train":
            imu = read_imu_csv(args.dataset_root / record["scene"] / "imu.csv", timestamp_unit="s")
            train_motion.extend(np.linalg.norm(imu.gyro, axis=1).tolist())
    imu_low, imu_high = np.quantile(np.asarray(train_motion), [0.05, 0.95])
    if imu_high <= imu_low:
        imu_high = imu_low + 1e-6
    records: list[dict[str, object]] = []
    for source in dataset["records"]:
        scene = str(source["scene"])
        destination = args.output / str(source["split"]) / f"{scene}.npz"
        report_path = destination.with_suffix(".json")
        if args.overwrite or not (destination.exists() and report_path.exists()):
            destination.parent.mkdir(parents=True, exist_ok=True)
            _run([
                sys.executable, "scripts/build_imu_gear_oracle.py",
                "--scene-root", str(args.dataset_root / scene),
                "--imu-csv", str(args.dataset_root / scene / "imu.csv"),
                "--timestamp-unit", "s", "--imu-timestamp-unit", "s",
                "--max-frames", str(args.frames), "--resize-width", str(args.size),
                "--motion-source", "imu", "--imu-motion-bounds", str(imu_low), str(imu_high),
                "--prism-angle-deg", "2.0", "--max-phase-step-deg", "30.0",
                "--target-density-low", "0.45", "--target-density-high", "1.35",
                "--event-efficiency-weight", str(args.event_efficiency_weight), "--density-weight", "0.2",
                "--coverage-efficiency-weight", str(args.coverage_efficiency_weight),
                "--rpm-effort-weight", "0.04", "--rpm-change-weight", "0.03",
                "--minimum-dwell-seconds", "0.025", "--switch-margin", "0.01",
                "--flow-features", "--store-active-masks", "--output", str(destination),
            ], environment=environment, root=root)
        oracle_report = json.loads(report_path.read_text(encoding="utf-8"))
        records.append({"scene": scene, "condition": source["condition"], "seed": source["seed"], "split": source["split"], "time_scale": 1.0, "output": str(destination), "windows": oracle_report["windows"]})
    phase_continuous_oracles = True
    for record in records:
        with np.load(record["output"]) as oracle:
            phase_continuous_oracles &= "prism_phase_start" in oracle.files
    configuration = {
        "base_rpm": 720.0,
        "event_efficiency_weight": float(args.event_efficiency_weight),
        "coverage_efficiency_weight": float(args.coverage_efficiency_weight),
        "density_weight": 0.2,
        "overlap_weight": 0.5,
        "invalid_fraction_weight": 0.25,
        "rpm_effort_weight": 0.04,
        "rpm_change_weight": 0.03,
        "minimum_dwell_seconds": 0.025,
        "switch_margin": 0.01,
    }
    manifest = {
        "dataset": dataset["dataset"],
        "motion_profile": dataset.get("motion_profile", args.motion_profile),
        "motion_delta_feature": bool(args.motion_delta),
        "quality_protocol_version": QUALITY_PROTOCOL_VERSION,
        "scientific_question": "Can flow disambiguate equal-IMU camera-only and camera-object co-motion and improve balanced active-event excitation?",
        "split_contract": "texture seeds are disjoint; all motion conditions for one seed stay in one split",
        "imu_motion_bounds": [float(imu_low), float(imu_high)],
        "configuration": configuration,
        "oracles": records,
        "phase_continuous_oracles": phase_continuous_oracles,
        "policy_report": {},
        "limitations": ["Procedural scenes isolate relative motion but are not photorealistic.", "The optical-flow input is computed from pre-prism RGB and therefore assumes an aligned frame stream.", "The composite score remains a controller surrogate, not downstream-task proof."],
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    train_paths = sorted(str(record["output"]) for record in records if record["split"] == "train")
    valid_paths = sorted(str(record["output"]) for record in records if record["split"] == "valid")
    policies = ({
        "imu_only_q": (False, "candidate_q"),
        "imu_flow_q_gated": (True, "candidate_q"),
    } if phase_continuous_oracles else {
        "imu_only_gear": (False, "gear"),
        "imu_flow_gear": (True, "gear"),
        "imu_only_q": (False, "candidate_q"),
        "imu_flow_q": (True, "candidate_q"),
    })
    policy_reports: dict[str, object] = {}
    rollout_reports: dict[str, object] = {}
    for name, (use_flow, target_type) in policies.items():
        policy_path = args.output / f"policy_{name}.pt"
        if target_type == "candidate_q":
            command = [sys.executable, "scripts/train_causal_q_policy.py", "--manifest", str(manifest_path), "--window", str(args.window), "--epochs", str(args.epochs), "--hidden-dim", "64", "--output", str(policy_path)]
            if phase_continuous_oracles:
                command.extend(("--candidate-dir", str(args.output / f"candidates_{name}")))
            if phase_continuous_oracles and use_flow:
                command.extend(("--base-policy", str(args.output / "policy_imu_only_q.pt"), "--freeze-base", "--group-robust-weight", "1.0"))
            if args.motion_delta:
                command.append("--motion-delta")
        else:
            command = [sys.executable, "scripts/train_causal_rpm_policy.py"]
            for path in train_paths:
                command.extend(("--train-oracle", path))
            for path in valid_paths:
                command.extend(("--valid-oracle", path))
            command.extend(("--architecture", "lstm", "--target-type", "gear", "--motion-input", "imu", "--window", str(args.window), "--epochs", str(args.epochs), "--hidden-dim", "64", "--output", str(policy_path)))
        if use_flow:
            command.append("--use-flow")
        _run(command, environment=environment, root=root)
        if phase_continuous_oracles:
            selection_path = args.output / f"selection_{name}.json"
            _run([
                sys.executable,
                "scripts/select_phase_continuous_checkpoint.py",
                str(manifest_path),
                "--dataset-root", str(args.dataset_root),
                "--candidate-manifest", str(args.output / f"candidates_{name}" / "manifest.json"),
                "--window", str(args.window),
                "--output", str(policy_path),
                "--report", str(selection_path),
            ], environment=environment, root=root)
        policy_reports[name] = json.loads(policy_path.with_suffix(".json").read_text(encoding="utf-8"))
        rollout_path = args.output / f"test_rollout_{name}.json"
        if phase_continuous_oracles:
            evaluation_command = [sys.executable, "scripts/evaluate_phase_continuous_control.py", str(manifest_path), "--dataset-root", str(args.dataset_root), "--policy", str(policy_path), "--split", "test", "--window", str(args.window), "--output", str(rollout_path)]
            if name == "imu_flow_q_gated":
                evaluation_command.append("--baselines")
        else:
            evaluation_command = [sys.executable, "scripts/evaluate_causal_policy_rollout.py", str(manifest_path), "--policy", str(policy_path), "--split", "test", "--window", str(args.window), "--output", str(rollout_path)]
        _run(evaluation_command, environment=environment, root=root)
        rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
        rollout_reports[name] = {"path": str(rollout_path), **_summarize(rollout)}
    flow_policy_name = "imu_flow_q_gated" if phase_continuous_oracles else "imu_flow_q"
    imu_only, imu_flow = rollout_reports["imu_only_q"], rollout_reports[flow_policy_name]
    reference_rollout = json.loads(Path(imu_flow["path"]).read_text(encoding="utf-8"))
    validation_rollout_path = args.output / "validation_rollout_imu_flow_q.json"
    if phase_continuous_oracles:
        validation_command = [sys.executable, "scripts/evaluate_phase_continuous_control.py", str(manifest_path), "--dataset-root", str(args.dataset_root), "--policy", str(args.output / f"policy_{flow_policy_name}.pt"), "--split", "valid", "--window", str(args.window), "--baselines", "--output", str(validation_rollout_path)]
    else:
        validation_command = [sys.executable, "scripts/evaluate_causal_policy_rollout.py", str(manifest_path), "--policy", str(args.output / "policy_imu_flow_q.pt"), "--split", "valid", "--window", str(args.window), "--output", str(validation_rollout_path)]
    _run(validation_command, environment=environment, root=root)
    validation_rollout = json.loads(validation_rollout_path.read_text(encoding="utf-8"))
    fixed_summaries = {f"fixed_{rpm}rpm": _summarize(reference_rollout, f"fixed_{rpm}rpm") for rpm in (0, 180, 360, 720, 1080)}
    fixed_validation_summaries = {f"fixed_{rpm}rpm": _summarize(validation_rollout, f"fixed_{rpm}rpm") for rpm in (0, 180, 360, 720, 1080)}
    best_fixed_name = max(fixed_validation_summaries, key=lambda key: fixed_validation_summaries[key]["balanced_objective_mean"])
    best_fixed = fixed_summaries[best_fixed_name]
    fusion_positive = imu_flow["worst_condition_objective"] > imu_only["worst_condition_objective"] and imu_flow["balanced_objective_mean"] >= imu_only["balanced_objective_mean"]
    beats_best_fixed = imu_flow["worst_condition_objective"] > best_fixed["worst_condition_objective"] and imu_flow["balanced_objective_mean"] > best_fixed["balanced_objective_mean"]
    comparison = {
        "status": ("paper_control_positive" if fusion_positive and beats_best_fixed else ("fusion_ablation_positive_but_not_best_fixed" if fusion_positive else "not_yet_positive")) if phase_continuous_oracles else "legacy_phase_diagnostic_not_paper_evidence",
        "algorithmic_status_ignoring_phase_limitation": "paper_control_positive" if fusion_positive and beats_best_fixed else ("fusion_ablation_positive_but_not_best_fixed" if fusion_positive else "not_yet_positive"),
        "paper_evidence_valid": phase_continuous_oracles,
        "phase_validity": "Direct policy-specific continuous-phase resimulation." if phase_continuous_oracles else "Legacy tables use gear-dependent counterfactual phase; new phase-continuous tables require direct resimulation after every selected action.",
        "fusion_success_rule": "IMU+flow must improve worst-condition objective without reducing the unweighted four-condition mean relative to IMU-only.",
        "paper_control_success_rule": "The learned controller must also exceed the validation-selected best global fixed RPM in balanced mean and worst-condition objective, or establish a declared Pareto advantage.",
        "policies": rollout_reports,
        "fixed_baselines": fixed_summaries,
        "fixed_validation_baselines": fixed_validation_summaries,
        "best_fixed_selected_on_validation": best_fixed_name,
        "beats_best_fixed": beats_best_fixed,
        "delta_imu_flow_minus_imu_only": {key: float(imu_flow[key] - imu_only[key]) for key in ("balanced_objective_mean", "balanced_objective_std", "worst_condition_objective", "balanced_edge_f1_mean", "worst_condition_edge_f1", "balanced_coverage_efficiency_mean", "worst_condition_coverage_efficiency")},
        "training_validation": {
            name: ({"rpm_mae": float(report["valid_mae_rpm"]), "gear_accuracy": report.get("valid_gear_accuracy")} if report["target_type"] == "gear" else report["valid"])
            for name, report in policy_reports.items()
        },
        "checkpoint_selection": {
            name: report.get("direct_rollout_selection") for name, report in policy_reports.items()
        },
        "interpretation_guardrail": "A negative result rejects this controller/dataset/score combination; it does not reject active optical modulation itself.",
    }
    (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    manifest["policy_report"] = policy_reports[flow_policy_name]
    manifest["policy_reports"] = policy_reports
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
