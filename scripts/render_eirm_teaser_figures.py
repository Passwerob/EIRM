#!/usr/bin/env python3
"""Render EIRM teaser figures for the car-motion Blender scene."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ami_simulator.event_visualization import eirm_from_counts, eirm_from_events
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import rgb_to_linear_luminance
from scripts.evaluate_eveyes_v2_downstream import _intrinsics
from scripts.plot_eveyes_scene001_motion_regimes import _motion_blur_rgb, _motion_events
from scripts.plot_eveyes_scene001_dvs_vs_ours import CACHE as DVS_CACHE
from scripts.evaluate_dvs_v2e_ours import _read_npz, _slice
from scripts.render_eveyes_scene001_storm_vs_ours import SCENE, _hero_events, _load
from scripts.render_scene001_four_panel import _accumulate, _bw


def _counts(events, height, width):
    pos = np.zeros((height, width), dtype=np.float32)
    neg = np.zeros((height, width), dtype=np.float32)
    _accumulate(np.zeros_like(pos), pos, neg, events)
    return pos, neg


def _recon(events, height, width):
    pos, neg = _counts(events, height, width)
    return _bw(pos + neg)


def _save_four_panel(scene: Path, output: Path, duration: float) -> None:
    rgb, timestamps = _load(scene, 480)
    lum = rgb_to_linear_luminance(rgb)
    h, w = rgb.shape[1:3]
    prism = RotatingWedgePrism(_intrinsics(scene), 0.5)
    end = int(np.searchsorted(timestamps, timestamps[0] + duration, side="right") - 1)
    end = max(1, min(end, len(lum) - 1))
    sensor_parts = []
    ref_parts = []
    for i in range(end):
        from scripts.render_eveyes_scene001_storm_vs_ours import _step
        state = getattr(_save_four_panel, "_state", None)
        if state is None or i == 0:
            state = {"phase": 0.0, "active": None, "ref": None, "rpm": 180.0}
        result = _step(lum[i:i + 2], timestamps[i:i + 2], prism, 180.0, state["phase"], state["active"], state["rpm"], state["ref"], "cpu")
        sensor_parts.append(result["sensor_events"]); ref_parts.append(result["reference_events"])
        state.update(phase=float(result["phase_end"]), active=result["active"], ref=np.asarray(result["sensor_reference_log"]), rpm=180.0)
        _save_four_panel._state = state
    from scripts.render_eveyes_scene001_storm_vs_ours import _concat
    sensor = _concat(sensor_parts); ref = _concat(ref_parts)
    panels = [rgb[end], eirm_from_events(sensor, h, w), eirm_from_events(ref, h, w), _recon(ref, h, w)]
    labels = ["RGB image", "Sensor-plane events", "Inverse-warped events", "Accumulated event reconstruction"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), dpi=140)
    for ax, image, label in zip(axes, panels, labels):
        ax.imshow(image); ax.set_title(label, fontsize=12); ax.axis("off")
    fig.suptitle(f"Scene_001 · camera static / object moving · EIRM visualization · {duration:.2f}s", fontsize=14)
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight"); fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output.with_suffix(".png"))


def _save_motion_storm(scene: Path, output: Path) -> None:
    rgb, timestamps = _load(scene, 8)
    lum = rgb_to_linear_luminance(rgb)
    hero = 0; h, w = rgb.shape[1:3]
    # Natural-motion baseline: no prism excitation. Ours is the controlled
    # optical excitation stream rendered below at 180 RPM.
    normal = _slice(_read_npz(DVS_CACHE), float(timestamps[hero]), float(timestamps[hero + 1]))
    ours = _hero_events(lum, timestamps, RotatingWedgePrism(_intrinsics(scene), 0.5), hero, 180.0, "cpu", plane="reference", n_windows=1)
    storm_rgb = _motion_blur_rgb(rgb[hero]); storm = _motion_events(lum[hero])
    controlled_rgb = rgb[hero]
    panels = [rgb[hero], eirm_from_events(normal, h, w), _recon(normal, h, w), storm_rgb, eirm_from_events(storm, h, w), _recon(storm, h, w), controlled_rgb, eirm_from_events(ours, h, w), _recon(ours, h, w)]
    labels = ["Natural motion\nRGB", "Natural motion\nno-prism events", "Natural motion\nreconstruction", "Camera too fast\nRGB blur", "Event storm\nsensor-plane", "Event storm\nreconstruction", "Ours control\nRGB", "Ours control\ninverse-warped events", "Ours control\nreconstruction"]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.2), dpi=140)
    for ax, image, label in zip(axes.flat, panels, labels):
        ax.imshow(image); ax.set_title(label, fontsize=11); ax.axis("off")
    fig.suptitle("Controlled optical excitation · normal motion versus camera-speed event storm", fontsize=15)
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight"); fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output.with_suffix(".png"))


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--scene", type=Path, default=SCENE); p.add_argument("--duration", type=float, default=0.2); p.add_argument("--output-dir", type=Path, default=Path("results/eveyes_scene001/figures/eirm")); a = p.parse_args()
    _save_four_panel(a.scene, a.output_dir / "scene001_eirm_four_panel", a.duration)
    _save_motion_storm(a.scene, a.output_dir / "scene001_eirm_motion_storm_ours")


if __name__ == "__main__":
    main()
