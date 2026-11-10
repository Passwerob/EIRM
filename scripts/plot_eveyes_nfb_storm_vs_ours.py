#!/usr/bin/env python3
"""Storm vs ours scenes: left sensor-plane blizzard, right inverse-warped 180.

Qualitative exhibit, not Table 1. Left uses 1080 RPM / 3 windows / sensor plane
(same recipe as eveyes_event_qualitative). Right is dual camera-speed NFB at 180.
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
from ami_simulator.types import EventStream
from scripts.evaluate_dl3dv_downstream import _ami_windows
from scripts.evaluate_eveyes_v2_downstream import _intrinsics, _scene_path
from scripts.render_eveyes_camera_nfb_event_video import _concat, _events_to_map, _sample_rgb

TIMES = (2.0, 4.0, 8.0)
STORM_RPM = 1080.0
STORM_WINDOWS = 3
OURS_RPM = 180.0
WARMUP = 8


def _row_at(rows: list[dict], t: float) -> dict:
    return min(rows, key=lambda row: abs(float(row["t0"]) - t))


def _hero_events(
    luminance: np.ndarray,
    timestamps: np.ndarray,
    prism: RotatingWedgePrism,
    hero: int,
    rpm: float,
    device: str,
    *,
    plane: str,
    n_windows: int,
) -> EventStream:
    hero = int(np.clip(hero, 0, len(luminance) - 2))
    start = max(0, hero - max(WARMUP, n_windows - 1))
    end = min(len(luminance), hero + 2)
    lum = luminance[start:end]
    times = timestamps[start:end]
    valid = np.ones((len(lum), lum.shape[1], lum.shape[2]), dtype=bool)
    reference, sensor, _ = _ami_windows(lum, times, valid, prism, rpm, device=device)
    streams = sensor if plane == "sensor" else reference
    last = min(hero - start, len(streams) - 1)
    first = max(0, last - (n_windows - 1))
    return _concat(list(streams[first : last + 1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/eveyes_motion_nfb/camera_speed_nfb.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/eveyes_motion_nfb_ready"))
    parser.add_argument("--scene", default="test/Scene_005/co_motion")
    parser.add_argument("--left", default="open_loop_camera")
    parser.add_argument("--right", default="camera_speed_nfb")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("results/eveyes_motion_nfb/figures/camera_nfb_storm_vs_ours_scenes"))
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")
    report = json.loads(args.input.read_text(encoding="utf-8"))
    block = report["scenes"][args.scene]
    split, scene, regime = args.scene.split("/")
    scene_path = _scene_path(args.dataset_root, split, scene, regime)
    frames, timestamps = load_rgb_sequence(scene_path / "images", scene_path / "timestamps.csv", timestamp_unit="s", max_frames=480)
    rgb = np.clip(frames, 0, 255).astype(np.uint8)
    luminance = rgb_to_linear_luminance(frames)
    prism = RotatingWedgePrism(_intrinsics(scene_path), 0.5)
    height, width = rgb[0].shape[:2]
    fig, axes = plt.subplots(4, len(TIMES), figsize=(12.6, 10.6), dpi=140)
    row_titles = (
        "open loop RGB",
        "ours RGB  (180 RPM)",
        "STORM events  (sensor, 1080 RPM)",
        "OURS events  (inverse-warped, 180 RPM)",
    )
    for col, t in enumerate(TIMES):
        left = _row_at(block[args.left]["rows"], t)
        right = _row_at(block[args.right]["rows"], t)
        storm = _hero_events(
            luminance,
            timestamps,
            prism,
            int(round(float(left["path_s"]))),
            STORM_RPM,
            args.device,
            plane="sensor",
            n_windows=STORM_WINDOWS,
        )
        ours = _hero_events(
            luminance,
            timestamps,
            prism,
            int(round(float(right["path_s"]))),
            OURS_RPM,
            args.device,
            plane="reference",
            n_windows=1,
        )
        panels = (
            _sample_rgb(rgb, float(left["path_s"])),
            _sample_rgb(rgb, float(right["path_s"])),
            _events_to_map(storm, height, width, storm=True),
            _events_to_map(ours, height, width, storm=False),
        )
        captions = (
            f"t={left['t0']:.1f}s  path={left['path_s']:.0f}  1.0x",
            f"t={right['t0']:.1f}s  path={right['path_s']:.0f}  {right['camera_rate']:.1f}x",
            f"{len(storm.x)/1e6:.2f}M events  sensor blizzard",
            f"{len(ours.x)/1e6:.2f}M events  readable structure",
        )
        print(json.dumps({"t": t, "storm": int(len(storm.x)), "ours": int(len(ours.x)), "ours_rpm": right["rpm"]}), flush=True)
        for row, (image, caption) in enumerate(zip(panels, captions)):
            axis = axes[row, col]
            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlabel(caption, fontsize=8)
            if col == 0:
                axis.set_ylabel(row_titles[row], fontsize=7.5)
            for spine in axis.spines.values():
                spine.set_visible(False)
        axes[0, col].set_title(f"wall-clock {t:.0f}s", fontsize=11, pad=8)
    fig.suptitle("Left: uncontrolled storm    Right: ours (inverse-warped 180)", fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"))
    fig.savefig(args.output.with_suffix(".png"), dpi=170)
    plt.close(fig)
    print(args.output.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
