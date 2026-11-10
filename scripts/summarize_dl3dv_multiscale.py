#!/usr/bin/env python3
"""Summarize useful-event and RPM-label trends from a multiscale manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON summary path.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional flat CSV path.")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for record in manifest["oracles"]:
        report_path = Path(record["output"]).with_suffix(".json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        data = np.load(record["output"], allow_pickle=True)
        labels = np.asarray(data["labels"], dtype=np.int64)
        unconstrained = np.asarray(data["unconstrained_labels"], dtype=np.int64) if "unconstrained_labels" in data else labels
        counts = np.asarray(data["counts"], dtype=np.float64)
        efficiency = np.asarray(data["event_efficiency"], dtype=np.float64) if "event_efficiency" in data else np.divide(data["unique_ratio"], data["densities"], out=np.zeros_like(data["unique_ratio"]), where=data["densities"] > 0.0)
        row_index = np.arange(len(labels))
        constraint_regret = np.asarray(data["cost"])[row_index, labels] - np.asarray(data["cost"])[row_index, unconstrained]
        rows.append(
            {
                "split": record["split"],
                "scene": record["scene"],
                "time_scale": float(record["time_scale"]),
                "windows": int(len(labels)),
                "mean_event_density_at_1080rpm": float(np.mean(data["densities"][:, 0])),
                "mean_event_density_at_720rpm": float(np.mean(data["densities"][:, 1])),
                "mean_unique_ratio_at_1080rpm": float(np.mean(data["unique_ratio"][:, 0])),
                "mean_event_efficiency_at_1080rpm": float(np.mean(efficiency[:, 0])),
                "mean_edge_precision_at_1080rpm": float(np.mean(data["edge_precision"][:, 0])) if "edge_precision" in data else float("nan"),
                "mean_edge_recall_at_1080rpm": float(np.mean(data["coverage"][:, 0])),
                "mean_edge_f1_at_1080rpm": float(np.mean(data["edge_f1"][:, 0])) if "edge_f1" in data else float("nan"),
                "mean_temporal_overlap_at_1080rpm": float(np.mean(data["temporal_overlap"][:, 0])),
                "mean_quality_at_1080rpm": float(np.mean(data["quality_score"][:, 0])),
                "mean_quality_at_0rpm": float(np.mean(data["quality_score"][:, -1])),
                "mean_event_count_at_1080rpm": float(np.mean(counts[:, 0])),
                "dominant_label_rpm": float(report["gear_scales"][int(np.bincount(labels, minlength=counts.shape[1]).argmax())] * report.get("base_rpm", manifest.get("configuration", {}).get("base_rpm", 720.0))),
                "dominant_label_fraction": float(np.max(np.bincount(labels, minlength=counts.shape[1])) / max(len(labels), 1)),
                "gear_change_fraction": float(np.mean(np.diff(labels) != 0)) if len(labels) > 1 else 0.0,
                "constraint_override_fraction": float(np.mean(labels != unconstrained)),
                "mean_constraint_regret": float(np.mean(constraint_regret)),
                "label_counts": np.bincount(labels, minlength=counts.shape[1]).tolist(),
            }
        )
    rows.sort(key=lambda row: (str(row["split"]), str(row["scene"]), float(row["time_scale"])))
    label_counts = np.sum([row["label_counts"] for row in rows], axis=0, dtype=np.int64) if rows else np.zeros(0, dtype=np.int64)
    summary = {
        "manifest": str(args.manifest),
        "rows": rows,
        "diagnostics": {
            "note": "Event density is reported alongside unique-pixel ratio and temporal overlap; density alone is not an information metric.",
            "dominant_label_fraction_mean": float(np.mean([row["dominant_label_fraction"] for row in rows])) if rows else 0.0,
            "aggregate_label_counts": label_counts.tolist(),
            "fastest_gear_fraction": float(label_counts[0] / label_counts.sum()) if label_counts.size and label_counts.sum() else 0.0,
            "mean_gear_change_fraction": float(np.mean([row["gear_change_fraction"] for row in rows])) if rows else 0.0,
            "mean_constraint_override_fraction": float(np.mean([row["constraint_override_fraction"] for row in rows])) if rows else 0.0,
            "mean_constraint_regret": float(np.mean([row["mean_constraint_regret"] for row in rows])) if rows else 0.0,
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [key for key, value in rows[0].items() if not isinstance(value, list)] if rows else []
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: value for key, value in row.items() if key in fields} for row in rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
