#!/usr/bin/env python3
"""Select a controller checkpoint using group-robust direct validation rollout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


CONDITIONS = ("camera_only", "object_only", "co_motion", "counter_motion")


def robust_rollout_metrics(report: dict[str, object]) -> dict[str, object]:
    rows = [row for row in report["rows"] if row["policy"] == "learned_causal"]
    groups: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        if not selected:
            raise ValueError(f"Direct rollout has no validation rows for {condition}.")
        weights = np.asarray([row["windows"] for row in selected], dtype=np.float64)
        groups[condition] = {
            key: float(np.average([row[key] for row in selected], weights=weights))
            for key in ("mean_objective", "mean_edge_f1", "mean_rpm")
        }
    objectives = np.asarray([groups[name]["mean_objective"] for name in CONDITIONS])
    edge_f1 = np.asarray([groups[name]["mean_edge_f1"] for name in CONDITIONS])
    balanced_mean = float(objectives.mean())
    worst = float(objectives.min())
    return {
        "groups": groups,
        "balanced_objective_mean": balanced_mean,
        "balanced_objective_std": float(objectives.std()),
        "worst_condition_objective": worst,
        "balanced_edge_f1_mean": float(edge_f1.mean()),
        "worst_condition_edge_f1": float(edge_f1.min()),
        "robust_selection_score": 0.5 * balanced_mean + 0.5 * worst,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    candidates = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))["candidates"]
    if not candidates:
        raise ValueError("Candidate manifest contains no checkpoints.")
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    evaluations: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="ami_rollout_selection_") as temporary:
        for candidate in candidates:
            rollout_path = Path(temporary) / f"epoch_{int(candidate['epoch']):04d}.json"
            command = [
                sys.executable,
                "scripts/evaluate_phase_continuous_control.py",
                str(args.manifest),
                "--dataset-root",
                str(args.dataset_root),
                "--policy",
                str(candidate["path"]),
                "--split",
                "valid",
                "--window",
                str(args.window),
                "--output",
                str(rollout_path),
            ]
            completed = subprocess.run(command, cwd=root, env=environment, text=True, capture_output=True)
            if completed.returncode:
                raise RuntimeError(f"Direct rollout failed for {candidate['path']}:\n{completed.stderr}\n{completed.stdout}")
            rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
            evaluations.append({"epoch": int(candidate["epoch"]), "path": str(candidate["path"]), **robust_rollout_metrics(rollout)})
    selected = max(
        evaluations,
        key=lambda item: (
            item["robust_selection_score"],
            item["worst_condition_edge_f1"],
            item["balanced_objective_mean"],
            -item["epoch"],
        ),
    )
    checkpoint = torch.load(selected["path"], map_location="cpu", weights_only=True)
    checkpoint["direct_rollout_selection"] = {
        "split": "valid",
        "criterion": "0.5 * unweighted four-condition mean objective + 0.5 * worst-condition objective",
        "selected_epoch": selected["epoch"],
        "score": selected["robust_selection_score"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    result = {
        "selection_split": "valid",
        "test_split_used_for_selection": False,
        "criterion": "0.5 * unweighted four-condition mean objective + 0.5 * worst-condition objective",
        "tie_break": "worst-condition edge F1, then balanced mean objective, then earlier epoch",
        "selected": selected,
        "candidates": evaluations,
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    training_report_path = args.output.with_suffix(".json")
    if training_report_path.exists():
        training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
        training_report["direct_rollout_selection"] = result
        training_report_path.write_text(json.dumps(training_report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
