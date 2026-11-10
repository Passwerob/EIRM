#!/usr/bin/env python3
"""Controlled four-regime CM-IWE evaluation for true adaptive EIRM.

Uses the same procedural controlled scenes and causal rollout protocol as the
existing DVS/v2e/F1 table. CM-IWE is applied per causal window with the shared
velocity grid, then edge scores are aggregated over all held-out windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ami_simulator.contrast_maximization import contrast_maximization
from ami_simulator.event_quality import compute_reference_edge_mask
from ami_simulator.types import EventStream
from scripts.evaluate_dvs_v2e_ours import _load_scene, _ours
from scripts.evaluate_phase_continuous_control import _simulate_action
from scripts.plot_paper_dvs_v2e_cm_coverage import VELOCITY_GRID
from scripts.run_control_headroom_stateful import RolloutState, _transition
from scripts.evaluate_anchor_residual_policy import CLUSTER_FEATURE_DIM
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.motion_net import CausalAnchorQGRUPolicy


METHODS = ("fixed_0", "fixed_180", "fixed_360", "adaptive_eirm")


def _cm_edge(events: EventStream, reference: np.ndarray, reference_time: float, max_events: int) -> dict[str, float]:
    if len(events) > max_events:
        index = np.linspace(0, len(events) - 1, max_events).astype(np.int64)
        events = EventStream(events.x[index], events.y[index], events.t[index], events.p[index])
    cm = contrast_maximization(events, reference.shape, VELOCITY_GRID, reference_time=reference_time)
    image = np.asarray(cm["image"], dtype=np.float64).sum(axis=0)
    gy, gx = np.gradient(np.log1p(image))
    magnitude = np.hypot(gx, gy)
    positive = magnitude[magnitude > 1e-12]
    threshold = float(np.quantile(positive, 0.75)) if len(positive) else float("inf")
    predicted = magnitude >= threshold
    target = compute_reference_edge_mask(reference, edge_quantile=0.75)
    overlap = int(np.count_nonzero(predicted & target))
    npred, ntarget = int(predicted.sum()), int(target.sum())
    precision = float(overlap / npred) if npred else 0.0
    coverage = float(overlap / ntarget) if ntarget else 0.0
    f1 = float(2.0 * precision * coverage / (precision + coverage)) if precision + coverage else 0.0
    return {"edge_f1": f1, "edge_precision": precision, "edge_coverage": coverage, "cm_contrast": float(cm["contrast"]), "cm_event_count": int(len(events)), "event_count": int(len(events)), "cm_velocity_x": float(cm["velocity"][0]), "cm_velocity_y": float(cm["velocity"][1])}


def _fixed_streams(scene, oracle, objective, rpm: float) -> list[EventStream]:
    luminance, timestamps, masks, intrinsics = scene
    prism = RotatingWedgePrism(intrinsics, 2.0)
    state = RolloutState(0.0, None, None, 180.0, float(timestamps[0]) - 0.025)
    streams = []
    for index in range(len(timestamps) - 1):
        result, state = _transition(scene, oracle, objective, prism, state, index, float(rpm), dwell=0.025)
        streams.append(result["reference_events"])
    return streams


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("results/comotion_v2_learned/manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/comotion_controlled_v2"))
    parser.add_argument("--student", type=Path, default=Path("results/comotion_horizon2_q_policy/policy.pt"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/controlled_adaptive_cm_iwe.json"))
    parser.add_argument("--max-events", type=int, default=80000)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.student, map_location="cpu", weights_only=True)
    model = CausalAnchorQGRUPolicy(CLUSTER_FEATURE_DIM, int(checkpoint["hidden_dim"]), int(checkpoint["num_actions"]))
    model.load_state_dict(checkpoint["model"]); model.eval()
    objective = {**manifest["configuration"], "event_efficiency_weight": 0.0}
    rows = []
    for record in manifest["oracles"]:
        if record["split"] != args.split:
            continue
        oracle_path = Path(str(record["output"]))
        scene_data = _load_scene(args.dataset_root / str(record["scene"]), oracle_path)
        luminance, timestamps, masks, intrinsics, oracle = scene_data
        scene = (luminance, timestamps, masks, intrinsics)
        try:
            streams = {
                "fixed_0": _fixed_streams(scene, oracle, objective, 0.0),
                "fixed_180": _fixed_streams(scene, oracle, objective, 180.0),
                "fixed_360": _fixed_streams(scene, oracle, objective, 360.0),
                "adaptive_eirm": _ours(scene, oracle, objective, model, checkpoint, 0.005, 0.35, 0.025),
            }
            for method, sequence in streams.items():
                for index, events in enumerate(sequence):
                    score = _cm_edge(events, luminance[index], float(timestamps[index + 1]), args.max_events)
                    rows.append({"scene": record["scene"], "condition": record["condition"], "seed": int(record["seed"]), "window": index, "method": method, **score})
        finally:
            oracle.close()
        print(f"evaluated {record['scene']}", flush=True)
    aggregate = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        aggregate[method] = {key: float(np.mean([float(row[key]) for row in selected])) for key in ("edge_f1", "edge_precision", "edge_coverage", "cm_contrast", "event_count", "cm_event_count")}
    per_regime = {}
    for condition in sorted({row["condition"] for row in rows}):
        per_regime[condition] = {}
        for method in METHODS:
            selected = [row for row in rows if row["condition"] == condition and row["method"] == method]
            per_regime[condition][method] = {key: float(np.mean([float(row[key]) for row in selected])) for key in ("edge_f1", "edge_precision", "edge_coverage", "cm_contrast", "event_count")}
    report = {"protocol": "ami_controlled_adaptive_cm_iwe_v1", "split": args.split, "student": str(args.student), "velocity_grid_count": len(VELOCITY_GRID), "max_events_for_cm": args.max_events, "edge_reference": "RGB log-intensity gradient, 75th percentile", "aggregate": aggregate, "per_regime": per_regime, "rows": rows, "limitations": ["Procedural 64x64 controlled benchmark; not photorealistic EVEYES.", "CM contrast is a diagnostic; edge F1/precision/coverage are the scored downstream representation metrics."]}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "per_regime": per_regime}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
