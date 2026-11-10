#!/usr/bin/env python3
"""Score the shared CM-IWE downstream representation against RGB edges.

The event generators and event windows are fixed. Only the common
contrast-maximization representation is applied before comparing its edge map
with the RGB reference edge map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ami_simulator.contrast_maximization import contrast_maximization
from ami_simulator.event_quality import compute_reference_edge_mask
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import EventStream
from scripts.plot_paper_dvs_v2e_cm_coverage import (
    CLIP_CACHE,
    FULL_STEM,
    MOTION_REGIMES,
    MULTI_SCENE,
    READY,
    VELOCITY_GRID,
    _baseline_events,
    _resolve_ours,
    _scene_path,
    _timestamps,
)
from scripts.plot_eveyes_event_qualitative import _ami_window


def _edge_f1(edge: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    edge = np.asarray(edge, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    predicted = int(edge.sum())
    target = int(reference.sum())
    overlap = int(np.count_nonzero(edge & reference))
    precision = float(overlap / predicted) if predicted else 0.0
    recall = float(overlap / target) if target else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"edge_f1": f1, "edge_precision": precision, "edge_coverage": recall, "predicted_edge_pixels": predicted}


def _cm_edges(events: EventStream, shape: tuple[int, int], reference_time: float, max_events: int) -> tuple[np.ndarray, dict[str, float]]:
    if len(events) > max_events:
        indices = np.linspace(0, len(events) - 1, max_events).astype(np.int64)
        events = EventStream(events.x[indices], events.y[indices], events.t[indices], events.p[indices])
    result = contrast_maximization(events, shape, VELOCITY_GRID, reference_time=reference_time)
    image = np.asarray(result["image"], dtype=np.float64).sum(axis=0)
    log_image = np.log1p(image)
    gy, gx = np.gradient(log_image)
    magnitude = np.hypot(gx, gy)
    positive = magnitude[magnitude > 1e-12]
    threshold = float(np.quantile(positive, 0.75)) if len(positive) else float("inf")
    return magnitude >= threshold, {
        "cm_contrast": float(result["contrast"]),
        "cm_velocity_x": float(result["velocity"][0]),
        "cm_velocity_y": float(result["velocity"][1]),
        "cm_event_count": int(len(events)),
    }


def _event_for_spec(spec: dict, scene: Path, method: str, timestamps: np.ndarray, hero: int, device: str) -> EventStream:
    t0 = float(timestamps[hero])
    t1 = float(timestamps[min(hero + 1, len(timestamps) - 1)])
    if method in ("dvs_voltmeter", "v2e"):
        return _baseline_events(spec, scene, method, timestamps, hero, clip_warmup=48)[0]
    if method == "ami_fixed_180":
        return _ami_window(scene, hero, 180.0, device, plane="reference", n_windows=1)
    if method == "ami_fixed_360":
        return _ami_window(scene, hero, 360.0, device, plane="reference", n_windows=1)
    return _resolve_ours(scene, hero, device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=READY)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/cm_edge_reconstruction.json"))
    parser.add_argument("--max-events", type=int, default=80000)
    args = parser.parse_args()

    # Use one held-out Scene_005 block for every motion regime and the
    # train/valid/test object/co-motion blocks for cross-scene coverage.
    specs = list(MOTION_REGIMES) + [spec for spec in MULTI_SCENE if spec["split"] != "test" or spec["regime"] not in {"object_only", "co_motion"}]
    rows = []
    for spec in specs:
        scene = _scene_path(args.dataset_root, spec["split"], spec["scene"], spec["regime"])
        timestamps = _timestamps(scene)
        hero = int(spec["hero"])
        frames, ts = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=hero + 2)
        luminance = rgb_to_linear_luminance(frames)
        reference = compute_reference_edge_mask(luminance[hero], edge_quantile=0.75)
        shape = luminance.shape[1:]
        for method in ("dvs_voltmeter", "v2e", "ami_fixed_180", "ami_fixed_360"):
            events = _event_for_spec(spec, scene, method, timestamps, hero, args.device)
            predicted, cm = _cm_edges(events, shape, float(ts[min(hero + 1, len(ts) - 1)]), args.max_events)
            metrics = _edge_f1(predicted, reference)
            rows.append({"split": spec["split"], "scene": spec["scene"], "regime": spec["regime"], "hero": hero, "method": method, "event_count": int(len(events)), **metrics, **cm})
        print(f"evaluated {spec['split']}/{spec['scene']}/{spec['regime']}", flush=True)

    aggregate = {}
    for method in ("dvs_voltmeter", "v2e", "ami_fixed_180", "ami_fixed_360"):
        selected = [row for row in rows if row["method"] == method]
        aggregate[method] = {key: float(np.mean([float(row[key]) for row in selected])) for key in ("edge_f1", "edge_precision", "edge_coverage", "cm_contrast", "event_count", "cm_event_count")}
    per_regime = {}
    for regime in sorted({row["regime"] for row in rows}):
        per_regime[regime] = {}
        for method in ("dvs_voltmeter", "v2e", "ami_fixed_180", "ami_fixed_360"):
            selected = [row for row in rows if row["regime"] == regime and row["method"] == method]
            per_regime[regime][method] = {key: float(np.mean([float(row[key]) for row in selected])) for key in ("edge_f1", "edge_precision", "edge_coverage", "cm_contrast")}
    report = {
        "protocol": "ami_cm_iwe_edge_reconstruction_v1",
        "velocity_grid": {"count": len(VELOCITY_GRID), "vx": sorted({float(v[0]) for v in VELOCITY_GRID}), "vy": sorted({float(v[1]) for v in VELOCITY_GRID})},
        "max_events_for_cm": args.max_events,
        "edge_reference": "RGB log-intensity gradient, 75th percentile",
        "aggregate": aggregate,
        "per_regime": per_regime,
        "rows": rows,
        "limitations": ["CM contrast is a diagnostic and is sensitive to event count and velocity-grid resolution.", "The edge reconstruction result uses one fixed grid and one-frame windows."]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "per_regime": per_regime}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
