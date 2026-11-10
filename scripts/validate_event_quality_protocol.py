#!/usr/bin/env python3
"""Run counterfactual contract tests for the event-quality benchmark metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ami_simulator.event_quality import QUALITY_PROTOCOL_VERSION, compute_reference_edge_mask, event_quality_metrics, event_quality_score, event_signal_fidelity_metrics
from ami_simulator.types import EventStream


def _stream(x: np.ndarray, y: np.ndarray, *, repeats: int = 1) -> EventStream:
    x = np.repeat(np.asarray(x, dtype=np.int32), repeats)
    y = np.repeat(np.asarray(y, dtype=np.int32), repeats)
    count = len(x)
    t = np.linspace(0.001, 0.099, count)
    p = np.where(np.arange(count) % 2, -1, 1).astype(np.int8)
    return EventStream(x, y, t, p)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/event_quality_protocol_validation.json"))
    args = parser.parse_args()
    height = width = 16
    image = np.zeros((height, width), dtype=np.float32)
    image[:, 4:12] = 0.5
    image[:, 12:] = 1.0
    frozen_edge_mask = compute_reference_edge_mask(image, edge_quantile=0.75)
    rows = np.arange(height, dtype=np.int32)
    clean = _stream(np.concatenate((np.full(height, 3), np.full(height, 11))), np.concatenate((rows, rows)))
    partial = _stream(np.full(height // 2, 3), rows[: height // 2])
    duplicated = _stream(np.concatenate((np.full(height, 3), np.full(height, 11))), np.concatenate((rows, rows)), repeats=4)
    equal_count_noise = _stream(np.concatenate((np.full(height, 0), np.full(height, 15))), np.concatenate((rows, rows)))
    hot_pixel = _stream(np.zeros(32, dtype=np.int32), np.zeros(32, dtype=np.int32))
    empty = EventStream.empty()
    mixed = EventStream(
        np.concatenate((clean.x, equal_count_noise.x)),
        np.concatenate((clean.y, equal_count_noise.y)),
        np.concatenate((clean.t, equal_count_noise.t)),
        np.concatenate((clean.p, equal_count_noise.p)),
    ).sorted_by_time()
    streams = {"clean": clean, "partial": partial, "duplicated": duplicated, "equal_count_noise": equal_count_noise, "hot_pixel": hot_pixel, "mixed_signal_noise": mixed, "empty_observation": empty}
    metrics: dict[str, dict[str, float | int]] = {}
    for name, events in streams.items():
        values = event_quality_metrics(events, reference_edge_mask=frozen_edge_mask, image_shape=image.shape)
        values["controller_surrogate"] = event_quality_score(values, target_density=0.5)
        fidelity = event_signal_fidelity_metrics(events, clean, image_shape=image.shape, time_range=(0.0, 0.1), temporal_bins=5, spatial_tolerance=1)
        metrics[name] = {**values, **fidelity}
    parameter_stability = []
    for temporal_bins in (3, 5, 10):
        for spatial_tolerance in (0, 1, 2):
            clean_fidelity = event_signal_fidelity_metrics(clean, clean, image_shape=image.shape, time_range=(0.0, 0.1), temporal_bins=temporal_bins, spatial_tolerance=spatial_tolerance)
            noise_fidelity = event_signal_fidelity_metrics(equal_count_noise, clean, image_shape=image.shape, time_range=(0.0, 0.1), temporal_bins=temporal_bins, spatial_tolerance=spatial_tolerance)
            parameter_stability.append({"temporal_bins": temporal_bins, "spatial_tolerance_px": spatial_tolerance, "clean_f1": clean_fidelity["spatiotemporal_f1"], "noise_f1": noise_fidelity["spatiotemporal_f1"], "ranking_correct": clean_fidelity["spatiotemporal_f1"] > noise_fidelity["spatiotemporal_f1"]})
    checks = {
        "equal_count_structure_beats_noise": metrics["clean"]["edge_f1"] > metrics["equal_count_noise"]["edge_f1"] and len(clean) == len(equal_count_noise),
        "new_edge_support_beats_partial": metrics["clean"]["edge_f1"] > metrics["partial"]["edge_f1"],
        "duplicates_do_not_increase_edge_f1": np.isclose(metrics["clean"]["edge_f1"], metrics["duplicated"]["edge_f1"]),
        "duplicates_reduce_event_efficiency": metrics["duplicated"]["event_efficiency"] < metrics["clean"]["event_efficiency"],
        "duplicates_reduce_controller_surrogate": metrics["duplicated"]["controller_surrogate"] < metrics["clean"]["controller_surrogate"],
        "hot_pixel_scores_below_clean": metrics["hot_pixel"]["controller_surrogate"] < metrics["clean"]["controller_surrogate"],
        "clean_fidelity_is_one": np.isclose(metrics["clean"]["spatiotemporal_f1"], 1.0),
        "mixed_noise_reduces_fidelity_precision": metrics["mixed_signal_noise"]["spatiotemporal_precision"] < metrics["clean"]["spatiotemporal_precision"],
        "mixed_noise_reduces_controller_surrogate": metrics["mixed_signal_noise"]["controller_surrogate"] < metrics["clean"]["controller_surrogate"],
        "noise_fidelity_below_clean": metrics["equal_count_noise"]["spatiotemporal_f1"] < metrics["clean"]["spatiotemporal_f1"],
        "empty_observation_has_zero_recall_when_signal_exists": metrics["empty_observation"]["spatiotemporal_recall"] == 0.0,
        "fidelity_ranking_stable_across_bin_tolerance_grid": all(row["ranking_correct"] for row in parameter_stability),
    }
    report = {
        "protocol_version": QUALITY_PROTOCOL_VERSION,
        "benchmark_policy": {
            "primary_synthetic_two_axis": {"exposure": ["edge_recall", "edge_f1", "downstream_task_utility"], "fidelity": ["spatiotemporal_precision", "spatiotemporal_recall", "spatiotemporal_f1", "reference_occupied_bins"]},
            "diagnostic": ["edge_precision", "edge_coverage", "edge_f1", "event_efficiency", "temporal_overlap", "event_density"],
            "primary_real": "downstream perception metric; no-reference composite score is not ground truth",
            "controller_surrogate_status": "provisional_until_rank_correlation_with_downstream_metric_is_established",
            "prohibited": "Never report fidelity without exposure/task utility; an empty ideal/observed stream can be perfectly faithful but uninformative.",
        },
        "all_checks_passed": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "fidelity_parameter_stability": parameter_stability,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
