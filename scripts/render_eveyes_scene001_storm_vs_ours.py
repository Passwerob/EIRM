#!/usr/bin/env python3
"""Scene_001 cam_static_obj_moving: camera frozen, only the car moves.

Storm (1080 RPM, sensor plane, 3 windows) vs ours (180 RPM, inverse-warped).
Uses the original Blender pack under data/eveyes_scene001_cmm/.../cam_static_obj_moving.
Not Table 1, not m0, not the Scene_005 co_motion NFB pack.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import EventStream
from scripts.evaluate_dl3dv_downstream import _ami_windows
from scripts.evaluate_eveyes_camera_speed_nfb import OBJECTIVE
from scripts.evaluate_eveyes_v2_downstream import _intrinsics
from scripts.evaluate_phase_continuous_control import _simulate_action
from scripts.render_eveyes_camera_nfb_event_video import _concat, _events_to_map, _hud

SCENE = Path("data/eveyes_scene001_cmm/eveyes_single_car_dataset_final/train/Scene_001/cam_static_obj_moving")
STORM_RPM = 1080.0
STORM_WINDOWS = 3
OURS_RPM = 180.0
WARMUP = 8
TIMES = (0.0, 2.0, 4.0)


def _load(scene: Path, max_frames: int) -> tuple[np.ndarray, np.ndarray]:
    rgb_dir = scene / "rgb" if (scene / "rgb").exists() else scene / "images"
    frames, timestamps = load_rgb_sequence(rgb_dir, scene / "timestamps.csv", timestamp_unit="us", max_frames=max_frames)
    return np.clip(frames, 0, 255).astype(np.uint8), timestamps


def _hero_index(timestamps: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(timestamps - float(t))))


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


def _plot_scenes(rgb: np.ndarray, timestamps: np.ndarray, luminance: np.ndarray, prism: RotatingWedgePrism, device: str, output: Path) -> None:
    height, width = rgb[0].shape[:2]
    fig, axes = plt.subplots(3, len(TIMES), figsize=(12.4, 8.4), dpi=140)
    row_titles = (
        "RGB  (camera frozen)",
        "STORM  (sensor, 1080 RPM)",
        "OURS  (inverse-warped, 180 RPM)",
    )
    for col, t in enumerate(TIMES):
        hero = _hero_index(timestamps, t)
        storm = _hero_events(luminance, timestamps, prism, hero, STORM_RPM, device, plane="sensor", n_windows=STORM_WINDOWS)
        ours = _hero_events(luminance, timestamps, prism, hero, OURS_RPM, device, plane="reference", n_windows=1)
        panels = (
            rgb[hero],
            _events_to_map(storm, height, width, storm=True),
            _events_to_map(ours, height, width, storm=False),
        )
        captions = (
            f"t={timestamps[hero]:.2f}s  frame {hero}",
            f"{len(storm.x)/1e6:.2f}M events",
            f"{len(ours.x)/1e6:.2f}M events",
        )
        print({"t": t, "hero": hero, "storm": int(len(storm.x)), "ours": int(len(ours.x))}, flush=True)
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
        axes[0, col].set_title(f"{t:.0f}s" if t >= 1 else f"{t:.1f}s", fontsize=11, pad=6)
    fig.suptitle("Scene_001  ·  camera static, object moving  ·  storm vs ours", fontsize=12)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=170)
    plt.close(fig)
    print(output.with_suffix(".png"), flush=True)


def _step(luminance, times, prism, rpm, phase, previous_active, previous_rpm, sensor_reference, device):
    pair = np.stack((luminance[0], luminance[1]), axis=0)
    valid = np.ones((2, pair.shape[1], pair.shape[2]), dtype=bool)
    return _simulate_action(
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


def _render_video(rgb: np.ndarray, timestamps: np.ndarray, luminance: np.ndarray, prism: RotatingWedgePrism, device: str, output: Path, fps: int) -> None:
    height, width = rgb[0].shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.with_name(output.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height * 2))
    if not writer.isOpened():
        raise SystemExit(f"failed to open {raw}")
    storm_buf: deque[EventStream] = deque(maxlen=STORM_WINDOWS)
    storm_state = {"phase": 0.0, "active": None, "ref": None, "rpm": STORM_RPM}
    ours_state = {"phase": 0.0, "active": None, "ref": None, "rpm": OURS_RPM}
    n = len(luminance) - 1
    for index in range(n):
        times = np.asarray((timestamps[index], timestamps[index + 1]), dtype=np.float64)
        pair = luminance[index : index + 2]
        storm = _step(pair, times, prism, STORM_RPM, storm_state["phase"], storm_state["active"], storm_state["rpm"], storm_state["ref"], device)
        ours = _step(pair, times, prism, OURS_RPM, ours_state["phase"], ours_state["active"], ours_state["rpm"], ours_state["ref"], device)
        storm_buf.append(storm["sensor_events"])
        storm_events = _concat(list(storm_buf))
        ours_events = ours["reference_events"]
        storm_state.update(phase=float(storm["phase_end"]), active=storm["active"], ref=np.asarray(storm["sensor_reference_log"]), rpm=STORM_RPM)
        ours_state.update(phase=float(ours["phase_end"]), active=ours["active"], ref=np.asarray(ours["sensor_reference_log"]), rpm=OURS_RPM)
        left_rgb = _hud(rgb[index], ["RGB  camera frozen", f"t={timestamps[index]:.2f}s  only the car moves"])
        right_rgb = _hud(rgb[index], ["RGB  camera frozen", f"t={timestamps[index]:.2f}s  only the car moves"])
        left_evt = _hud(_events_to_map(storm_events, height, width, storm=True), ["STORM  sensor  1080 RPM", f"{len(storm_events.x):.0f} events"])
        right_evt = _hud(_events_to_map(ours_events, height, width, storm=False), ["OURS  inverse-warped  180 RPM", f"{len(ours_events.x):.0f} events"])
        frame = np.concatenate((np.concatenate((left_rgb, right_rgb), axis=1), np.concatenate((left_evt, right_evt), axis=1)), axis=0)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if index % 40 == 0:
            print(f"  frame={index}/{n} storm={len(storm_events.x)} ours={len(ours_events.x)}", flush=True)
    writer.release()
    print(raw, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scenes-output", type=Path, default=Path("results/eveyes_scene001/figures/scene001_storm_vs_ours_scenes"))
    parser.add_argument("--video-output", type=Path, default=Path("results/eveyes_scene001/figures/scene001_storm_vs_ours.mp4"))
    parser.add_argument("--skip-scenes", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")
    rgb, timestamps = _load(args.scene, args.max_frames)
    luminance = rgb_to_linear_luminance(rgb)
    prism = RotatingWedgePrism(_intrinsics(args.scene), 0.5)
    print({"scene": str(args.scene), "frames": len(rgb), "dt0": float(timestamps[1] - timestamps[0]), "fx": prism.intrinsics.fx, "fy": prism.intrinsics.fy}, flush=True)
    if not args.skip_scenes:
        _plot_scenes(rgb, timestamps, luminance, prism, args.device, args.scenes_output)
    if not args.skip_video:
        _render_video(rgb, timestamps, luminance, prism, args.device, args.video_output, args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
