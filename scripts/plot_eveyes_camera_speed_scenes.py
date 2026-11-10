#!/usr/bin/env python3
"""Obvious wall-clock scenes: same t, open 1.0× vs camera 0.5× keep-360.

RGB + inverse-warped events. Not a Table 1 figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from scripts.evaluate_eveyes_camera_speed_nfb import DT, OBJECTIVE, _sample
from scripts.evaluate_eveyes_v2_downstream import _intrinsics, _scene_path
from scripts.evaluate_phase_continuous_control import _simulate_action
from scripts.render_eveyes_camera_nfb_event_video import _events_to_map, _sample_rgb

WARMUP = 8
TIMES = (2.0, 4.0, 8.0)


def _row_at(rows: list[dict], t: float) -> dict:
    return min(rows, key=lambda row: abs(float(row["t0"]) - t))


def _events_near(
    luminance: np.ndarray,
    prism: RotatingWedgePrism,
    *,
    path_s: float,
    rate: float,
    rpm: float,
    device: str,
) -> tuple[np.ndarray, int]:
    rate = max(float(rate), 0.25)
    start = max(0.0, float(path_s) - WARMUP * rate)
    phase = 0.0
    previous_active = None
    sensor_reference = None
    previous_rpm = float(rpm)
    events = None
    cursor = start
    for step in range(WARMUP + 1):
        lum_a = _sample(luminance, cursor)
        lum_b = _sample(luminance, cursor + rate)
        pair = np.stack((lum_a, lum_b), axis=0)
        now = step * DT
        times = np.asarray((now, now + DT), dtype=np.float64)
        valid = np.ones((2, pair.shape[1], pair.shape[2]), dtype=bool)
        result = _simulate_action(
            pair,
            times,
            valid,
            prism,
            index=0,
            rpm=float(rpm),
            phase=phase,
            previous_active=previous_active,
            previous_rpm=previous_rpm,
            target_density=0.2,
            objective=OBJECTIVE,
            contrast_threshold=0.2,
            max_phase_step_deg=30.0,
            sensor_reference_log=sensor_reference,
            device=device,
        )
        events = result["reference_events"]
        phase = float(result["phase_end"])
        sensor_reference = np.asarray(result["sensor_reference_log"])
        previous_active = result["active"]
        previous_rpm = float(rpm)
        cursor = min(len(luminance) - 1, cursor + rate)
    assert events is not None
    height, width = luminance.shape[1:]
    return _events_to_map(events, height, width), int(len(events.x))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/eveyes_motion_nfb/camera_speed_nfb.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/eveyes_motion_nfb_ready"))
    parser.add_argument("--scene", default="test/Scene_005/co_motion")
    parser.add_argument("--left", default="open_loop_camera")
    parser.add_argument("--right", default="camera_nfb_keep360")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("results/eveyes_motion_nfb/figures/camera_nfb_keep360_scenes"))
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")
    report = json.loads(args.input.read_text(encoding="utf-8"))
    block = report["scenes"][args.scene]
    split, scene, regime = args.scene.split("/")
    scene_path = _scene_path(args.dataset_root, split, scene, regime)
    frames, _timestamps = load_rgb_sequence(scene_path / "images", scene_path / "timestamps.csv", timestamp_unit="s", max_frames=480)
    rgb = np.clip(frames, 0, 255).astype(np.uint8)
    luminance = rgb_to_linear_luminance(frames)
    prism = RotatingWedgePrism(_intrinsics(scene_path), 0.5)
    fig, axes = plt.subplots(4, len(TIMES), figsize=(12.6, 10.4), dpi=140)
    row_titles = (
        "open loop RGB  ·  1.0×  ·  360 RPM",
        "camera NFB RGB  ·  0.5×  ·  360 RPM",
        "open loop events",
        "camera NFB events",
    )
    for col, t in enumerate(TIMES):
        left = _row_at(block[args.left]["rows"], t)
        right = _row_at(block[args.right]["rows"], t)
        left_rgb = _sample_rgb(rgb, float(left["path_s"]))
        right_rgb = _sample_rgb(rgb, float(right["path_s"]))
        left_evt, left_n = _events_near(
            luminance, prism, path_s=float(left["path_s"]), rate=float(left["camera_rate"]), rpm=float(left["rpm"]), device=args.device
        )
        right_evt, right_n = _events_near(
            luminance, prism, path_s=float(right["path_s"]), rate=float(right["camera_rate"]), rpm=float(right["rpm"]), device=args.device
        )
        panels = (left_rgb, right_rgb, left_evt, right_evt)
        captions = (
            f"t={left['t0']:.1f}s  path={left['path_s']:.0f}",
            f"t={right['t0']:.1f}s  path={right['path_s']:.0f}  {right['camera_rate']:.1f}×",
            f"{left_n/1e6:.2f}M events  occ={left['unique_pixel_ratio']:.2f}",
            f"{right_n/1e6:.2f}M events  occ={right['unique_pixel_ratio']:.2f}",
        )
        for row, (image, caption) in enumerate(zip(panels, captions)):
            axis = axes[row, col]
            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlabel(caption, fontsize=8)
            if col == 0:
                axis.set_ylabel(row_titles[row], fontsize=8)
            for spine in axis.spines.values():
                spine.set_visible(False)
        axes[0, col].set_title(f"wall-clock {t:.0f}s", fontsize=11, pad=8)
    fig.suptitle("Camera-speed NFB: same time, different place on the path  (prism locked 360)", fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"))
    fig.savefig(args.output.with_suffix(".png"), dpi=170)
    plt.close(fig)
    print(args.output.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
