#!/usr/bin/env python3
"""Package the frozen CM-IWE geometric downstream evaluation for the paper."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/ami_next/cm_iwe_protocol_matched/cm_iwe_all_methods_final.json"
OUT = ROOT / "results/ami_next/downstream_geometric"
REGIMES = ("camera_only", "object_only", "co_motion", "counter_motion")
METHODS = ("dvs_voltmeter", "v2e", "ami_fixed_180", "ami_fixed_360", "adaptive_eirm")
LABELS = {
    "dvs_voltmeter": "DVS-Voltmeter",
    "v2e": "v2e",
    "ami_fixed_180": "Fixed180",
    "ami_fixed_360": "Fixed360",
    "adaptive_eirm": "EIRM",
}


def _rows(data: dict) -> list[dict]:
    return list(data["baseline_rows"]) + list(data["eirm_rows"])


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row and row[key] is not None]
    return float(np.mean(values)) if values else None


def main() -> None:
    data = json.loads(SOURCE.read_text())
    rows = _rows(data)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qualitative").mkdir(exist_ok=True)
    protocol = {
        "task": "deterministic event-to-edge geometric reconstruction (CM-IWE)",
        "source_result": str(SOURCE.relative_to(ROOT)),
        "methods": [LABELS[m] for m in METHODS],
        "event_representation": "polarity-separated CM-IWE",
        "windowing": "matched fixed physical-time windows; no event-count windowing",
        "velocity_grid": data["checks"]["4_identical_cm_grid"],
        "max_events_for_cm": data["checks"]["7_sample_counts"]["cm_input_cap_each"],
        "reference": "same RGB log-intensity edge map at each target timestamp",
        "compensation": data["checks"]["6_compensation_convention"],
        "normalization": "shared CM-IWE scoring; no method-specific intensity scaling or histogram matching",
        "primary_metrics": ["edge_f1", "edge_precision", "edge_coverage"],
        "aggregation": "arithmetic mean over matched evaluation windows; regime means are reported separately",
        "evaluated_regimes": list(REGIMES),
        "available_regimes": sorted({str(row["regime"]) for row in rows}),
        "fairness_checks": data["checks"],
        "limitations": [
            "counter_motion is absent from the frozen CM-IWE source protocol",
            "drive_stop_drive is retained only as a diagnostic and is not relabeled counter_motion",
            "CM-IWE is a deterministic geometric edge-reconstruction proxy, not a learned RGB reconstruction model",
        ],
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    aggregate = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method and row["regime"] in REGIMES]
        aggregate[LABELS[method]] = {
            "edge_f1": _mean(selected, "edge_f1"),
            "edge_precision": _mean(selected, "edge_precision"),
            "edge_coverage": _mean(selected, "edge_coverage"),
            "cm_contrast": _mean(selected, "cm_contrast"),
            "n_windows": len(selected),
        }
    (OUT / "aggregate_metrics.json").write_text(json.dumps({"status": "evaluated", "aggregate": aggregate}, indent=2) + "\n")

    per_regime = {}
    for regime in REGIMES:
        per_regime[regime] = {}
        for method in METHODS:
            selected = [row for row in rows if row["method"] == method and row["regime"] == regime]
            per_regime[regime][LABELS[method]] = {
                "edge_f1": _mean(selected, "edge_f1"),
                "edge_precision": _mean(selected, "edge_precision"),
                "edge_coverage": _mean(selected, "edge_coverage"),
                "cm_contrast": _mean(selected, "cm_contrast"),
                "n_windows": len(selected),
                "status": "evaluated" if selected else "absent_from_source_protocol",
            }
    (OUT / "per_regime_metrics.json").write_text(json.dumps(per_regime, indent=2) + "\n")

    fields = ["split", "scene", "regime", "hero", "method", "t0", "t1", "event_count", "edge_f1", "edge_precision", "edge_coverage", "cm_contrast", "cm_velocity_x", "cm_velocity_y", "cm_event_count"]
    with (OUT / "per_window_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "per_frame_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    stats = {LABELS[m]: {"n_windows": 0, "mean_event_count": None, "mean_cm_event_count": None} for m in METHODS}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        stats[LABELS[method]] = {
            "n_windows": len(selected),
            "mean_event_count": _mean(selected, "event_count"),
            "mean_cm_event_count": _mean(selected, "cm_event_count"),
        }
    (OUT / "event_statistics.json").write_text(json.dumps(stats, indent=2) + "\n")

    report = [
        "# Deterministic geometric downstream evaluation",
        "",
        "We evaluate a fixed CM-IWE event-to-edge reconstruction pipeline. The event stream is the only method-dependent input; all methods use matched physical-time windows, the same RGB reference edge map, the same 63-element translation grid, the same valid region, and the same event cap rule. Active methods use per-event timestamp-aware inverse prism compensation.",
        "",
        "## Aggregate",
        "",
        "| Method | Edge F1 | Precision | Coverage | Windows |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, metric in aggregate.items():
        report.append(f"| {method} | {metric['edge_f1']:.4f} | {metric['edge_precision']:.4f} | {metric['edge_coverage']:.4f} | {metric['n_windows']} |" if metric["edge_f1"] is not None else f"| {method} | n/a | n/a | n/a | 0 |")
    report += [
        "",
        "## Regimes",
        "",
        "| Regime | DVS-Voltmeter F1 | v2e F1 | Fixed180 F1 | Fixed360 F1 | EIRM F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for regime in REGIMES:
        vals = [per_regime[regime][LABELS[m]]["edge_f1"] for m in METHODS]
        fmt = ["n/a" if v is None else f"{v:.4f}" for v in vals]
        report.append(f"| {regime} | " + " | ".join(fmt) + " |")
    report += [
        "",
        "## Interpretation",
        "",
        "EIRM improves substantially over the passive DVS-Voltmeter and v2e baselines in the available controlled windows. It is close to Fixed180 and slightly below Fixed360 on aggregate edge F1. This supports transfer from acquisition-level observability to a deterministic geometric downstream representation without establishing that adaptive sensing dominates every fixed policy.",
        "",
        "The frozen source protocol contains camera-only, object-only, co-motion, and drive-stop-drive windows, but no counter-motion window. Drive-stop-drive is therefore reported only as a diagnostic and is not relabeled as counter-motion.",
        "",
        "Existing matched qualitative figures are under `results/ami_next/figures/qualitative_pdfs/blender/`; they are not used to select the quantitative windows.",
    ]
    (OUT / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
