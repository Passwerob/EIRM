#!/usr/bin/env python3
"""Render RGB, sensor events, inverse events, and accumulated BW IWE."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.event_visualization import eirm_from_counts
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from scripts.evaluate_eveyes_v2_downstream import _intrinsics
from scripts.render_eveyes_scene001_storm_vs_ours import SCENE, _step
from scripts.render_eveyes_camera_nfb_event_video import _events_to_map


def _accumulate(counts: np.ndarray, positive: np.ndarray, negative: np.ndarray, events) -> None:
    if not len(events.x):
        return
    x = np.rint(events.x).astype(np.int64)
    y = np.rint(events.y).astype(np.int64)
    valid = (x >= 0) & (x < counts.shape[1]) & (y >= 0) & (y < counts.shape[0])
    np.add.at(counts, (y[valid], x[valid]), 1.0)
    np.add.at(positive, (y[valid & (events.p > 0)], x[valid & (events.p > 0)]), 1.0)
    np.add.at(negative, (y[valid & (events.p <= 0)], x[valid & (events.p <= 0)]), 1.0)


def _polarity_map(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    return eirm_from_counts(positive, negative)


def _bw(counts: np.ndarray) -> np.ndarray:
    positive = counts[counts > 0]
    peak = max(float(np.percentile(positive, 99.5)) if positive.size else 1.0, 1.0)
    level = np.clip(np.log1p(counts) / np.log1p(peak), 0.0, 1.0)
    return np.repeat((255.0 - 235.0 * level)[..., None], 3, axis=-1).astype(np.uint8)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, default=SCENE)
    p.add_argument("--rpm", type=float, default=180.0)
    p.add_argument("--duration", type=float, default=0.2, help="Accumulation window in seconds")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, default=Path("results/eveyes_scene001/figures/scene001_four_panel"))
    args = p.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    rgb, timestamps = load_rgb_sequence(args.scene / "rgb", args.scene / "timestamps.csv", timestamp_unit="us", max_frames=480)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    luminance = rgb_to_linear_luminance(rgb)
    h, w = rgb.shape[1:3]
    prism = RotatingWedgePrism(_intrinsics(args.scene), 0.5)
    sensor_counts = np.zeros((h, w), np.float32)
    reference_counts = np.zeros((h, w), np.float32)
    sensor_pos = np.zeros((h, w), np.float32); sensor_neg = np.zeros((h, w), np.float32)
    reference_pos = np.zeros((h, w), np.float32); reference_neg = np.zeros((h, w), np.float32)
    state = {"phase": 0.0, "active": None, "ref": None, "rpm": float(args.rpm)}
    last = None
    end_index = int(np.searchsorted(timestamps, timestamps[0] + args.duration, side="right") - 1)
    end_index = max(1, min(end_index, len(luminance) - 1))
    for i in range(end_index):
        result = _step(luminance[i:i + 2], np.asarray(timestamps[i:i + 2]), prism, args.rpm,
                       state["phase"], state["active"], state["rpm"], state["ref"], args.device)
        _accumulate(sensor_counts, sensor_pos, sensor_neg, result["sensor_events"])
        _accumulate(reference_counts, reference_pos, reference_neg, result["reference_events"])
        state.update(phase=float(result["phase_end"]), active=result["active"],
                     ref=np.asarray(result["sensor_reference_log"]), rpm=float(args.rpm))
        last = result
        if i % 5 == 0:
            print(f"{i + 1}/{end_index} sensor={len(result['sensor_events'].x)} reference={len(result['reference_events'].x)}", flush=True)
    sensor = _polarity_map(sensor_pos, sensor_neg)
    reference = _polarity_map(reference_pos, reference_neg)
    panels = (rgb[-1], sensor, reference, _bw(reference_counts))
    labels = ("RGB image", "Sensor-plane events", "Inverse-warped events", "Accumulated BW reconstruction")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), dpi=120)
    for ax, image, label in zip(axes, panels, labels):
        ax.imshow(image)
        ax.set_title(label, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.suptitle(f"Scene_001 · camera static / object moving · {args.rpm:g} RPM · accumulated {timestamps[end_index] - timestamps[0]:.3f} s", fontsize=14)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=170)
    fig.savefig(args.output.with_suffix(".pdf"))
    np.savez_compressed(args.output.with_name(args.output.name + "_counts.npz"), sensor=sensor_counts, reference=reference_counts)
    (args.output.with_suffix(".json")).write_text(__import__("json").dumps({
        "scene": str(args.scene), "resolution": [w, h], "frames": len(rgb),
        "duration_s": float(timestamps[end_index] - timestamps[0]), "rpm": args.rpm,
        "prism_angle_deg": 0.5, "contrast_threshold": 0.2,
        "sensor_event_count": int(sensor_counts.sum()),
        "inverse_event_count": int(reference_counts.sum()),
        "inverse_warp_before_accumulation": True,
        "reconstruction": "log-count grayscale IWE; not a trained 3D reconstruction"
    }, indent=2) + "\n")
    print(args.output.with_suffix(".png"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
