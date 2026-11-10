#!/usr/bin/env python3
"""Run a fixed train/validation sensitivity grid for anti-collapse weights."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


CONFIGURATIONS = (
    ("efficiency_0p25", 0.25, 0.05),
    ("central", 0.50, 0.05),
    ("efficiency_0p75", 0.75, 0.05),
    ("effort_0p025", 0.50, 0.025),
    ("effort_0p1", 0.50, 0.10),
)


def _train_diagnostics(manifest: dict[str, object]) -> dict[str, object]:
    counts = np.zeros(5, dtype=np.int64)
    changes: list[float] = []
    mean_rpm: list[float] = []
    for record in manifest["oracles"]:
        if record["split"] != "train":
            continue
        with np.load(record["output"]) as data:
            labels = np.asarray(data["labels"], dtype=np.int64)
            counts += np.bincount(labels, minlength=5)
            changes.append(float(np.mean(np.diff(labels) != 0)) if len(labels) > 1 else 0.0)
            mean_rpm.append(float(np.mean(data["selected_rpm"])))
    return {
        "train_label_counts": counts.tolist(),
        "train_fastest_gear_fraction": float(counts[0] / counts.sum()),
        "train_mean_gear_change_fraction": float(np.mean(changes)),
        "train_mean_rpm": float(np.mean(mean_rpm)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practical-rife-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/dl3dv_objective_sensitivity"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/dl3dv_large5"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository)
    rows: list[dict[str, object]] = []
    for name, efficiency_weight, effort_weight in CONFIGURATIONS:
        destination = args.output / name
        command = [
            sys.executable,
            "scripts/run_dl3dv_rife_multiscale.py",
            "--dataset-root", str(args.dataset_root),
            "--practical-rife-root", str(args.practical_rife_root),
            "--output", str(destination),
            "--time-scales", "0.25", "0.5", "1", "2", "4",
            "--interpolation-fps", "120",
            "--max-phase-step-deg", "90",
            "--max-frames", "8",
            "--resize-width", "64",
            "--workers", str(args.workers),
            "--threads-per-worker", str(args.threads_per_worker),
            "--device", args.device,
            "--epochs", "200",
            "--window", "4",
            "--event-efficiency-weight", str(efficiency_weight),
            "--density-weight", "0.1",
            "--rpm-effort-weight", str(effort_weight),
            "--rpm-change-weight", "0.05",
            "--minimum-dwell-seconds", "0.05",
            "--switch-margin", "0.02",
        ]
        if args.overwrite:
            command.append("--overwrite")
        completed = subprocess.run(command, cwd=repository, env=environment, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(f"Sensitivity run {name} failed:\n{completed.stderr}\n{completed.stdout}")
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        policy = manifest["policy_report"]
        row = {
            "name": name,
            "event_efficiency_weight": efficiency_weight,
            "rpm_effort_weight": effort_weight,
            **_train_diagnostics(manifest),
            "validation_policy_mae_rpm": float(policy["valid_mae_rpm"]),
            "validation_mean_rpm_baseline_mae": float(policy["mean_rpm_baseline_mae"]),
            "validation_relative_mae_reduction": float(1.0 - policy["valid_mae_rpm"] / policy["mean_rpm_baseline_mae"]),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "selection_protocol": "Compare fixed configurations with training-label diagnostics and validation policy MAE only; test oracles must not select weights.",
        "central_configuration_is_predeclared": True,
        "rows": rows,
    }
    (args.output / "sensitivity.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with (args.output / "sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [key for key, value in rows[0].items() if not isinstance(value, list)]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key in fields} for row in rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
