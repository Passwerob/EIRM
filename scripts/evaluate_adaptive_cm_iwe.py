#!/usr/bin/env python3
"""Evaluate true causal EIRM adaptive events through the common CM-IWE grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ami_simulator.contrast_maximization import contrast_maximization
from ami_simulator.event_quality import compute_reference_edge_mask
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from scripts.evaluate_dl3dv_downstream import _ami_windows, _ours_windows, _visual_motion
from scripts.evaluate_eveyes_v2_downstream import _imu_fraction, _intrinsics, _scene_path
from scripts.plot_paper_dvs_v2e_cm_coverage import VELOCITY_GRID, _baseline_events, _cm_gray, _run_cm
from scripts.train_inverse_motion_policy import FEATURE_DIM
from ami_simulator.motion_net import TinyCausalGearPolicy


def _edges(events, shape, reference_time, max_events=80000):
    if len(events) > max_events:
        index = np.linspace(0, len(events) - 1, max_events).astype(np.int64)
        from ami_simulator.types import EventStream
        events = EventStream(events.x[index], events.y[index], events.t[index], events.p[index])
    cm = contrast_maximization(events, shape, VELOCITY_GRID, reference_time=reference_time)
    image = np.asarray(cm["image"], dtype=float).sum(axis=0)
    gy, gx = np.gradient(np.log1p(image))
    magnitude = np.hypot(gx, gy)
    predicted = magnitude >= np.quantile(magnitude[magnitude > 1e-12], 0.75)
    return predicted, cm, len(events)


def _score(predicted, reference):
    overlap = int(np.count_nonzero(predicted & reference))
    npred, nref = int(predicted.sum()), int(reference.sum())
    precision = overlap / npred if npred else 0.0
    recall = overlap / nref if nref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"edge_f1": float(f1), "edge_precision": float(precision), "edge_coverage": float(recall)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/eveyes_comotion_v2_ready"))
    parser.add_argument("--student", type=Path, default=Path("results/eveyes_v2_regime_f1_argmax_m0/policy.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/adaptive_cm_iwe.json"))
    args = parser.parse_args()
    checkpoint = torch.load(args.student, map_location="cpu", weights_only=True)
    model = TinyCausalGearPolicy(feature_dim=FEATURE_DIM, hidden_dim=int(checkpoint["hidden_dim"]), num_gears=int(checkpoint["num_gears"]))
    model.load_state_dict(checkpoint["model"]); model.eval()
    specs = (("object_only", 120), ("co_motion", 150))
    rows = []
    for regime, hero in specs:
        scene = _scene_path(args.dataset_root, "test", "Scene_005", regime)
        frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=hero + 2)
        timestamps = timestamps[: len(frames)]
        luminance = rgb_to_linear_luminance(frames)
        intrinsics = _intrinsics(scene)
        valid = np.ones((len(frames), intrinsics.height, intrinsics.width), dtype=bool)
        visual = _visual_motion(luminance, timestamps)
        imu = _imu_fraction(scene / "imu.csv", timestamps, (float(checkpoint["imu_motion_bounds"][0]), float(checkpoint["imu_motion_bounds"][1])))
        from ami_simulator.geometry import RotatingWedgePrism
        prism = RotatingWedgePrism(intrinsics, 0.5)
        ours, rpms = _ours_windows(luminance, timestamps, valid, prism, visual=visual, imu=imu, model=model, checkpoint=checkpoint, confidence=0.5, require_imu=False, dwell=0.025, device=args.device)
        t0, t1 = float(timestamps[hero]), float(timestamps[hero + 1])
        events = ours[hero]
        reference = compute_reference_edge_mask(luminance[hero], edge_quantile=0.75)
        predicted, cm, ncm = _edges(events, luminance.shape[1:], t1)
        metrics = _score(predicted, reference)
        rows.append({"scene": f"test/Scene_005/{regime}", "regime": regime, "hero": hero, "rpm": float(rpms[hero]), "event_count": int(len(events)), "cm_event_count": int(ncm), "cm_contrast": float(cm["contrast"]), "cm_velocity": [float(cm["velocity"][0]), float(cm["velocity"][1])], **metrics})
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    report = {"protocol": "ami_adaptive_cm_iwe_v1", "student": str(args.student), "velocity_grid_count": len(VELOCITY_GRID), "rows": rows, "note": "These are true causal EIRM event streams; older qualitative Ours caches use fixed 180 and are not used here."}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
