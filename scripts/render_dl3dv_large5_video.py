#!/usr/bin/env python3
"""Render a native-resolution Passive/Fixed-AMI/IMU-Gear comparison video."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio.v3 as imageio
import numpy as np
import torch

from ami_simulator.dsec import read_imu_csv
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.rotation import ConstantRotation, ImuGearRotation
from ami_simulator.types import CameraIntrinsics
from ami_simulator.warp import PrismWarper


def _counts(sensor: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    log_frames = np.log(np.clip(sensor.astype(np.float64), 1e-6, 1.0))
    reference = log_frames[0].copy()
    positive = []
    negative = []
    for before, after in zip(log_frames[:-1], log_frames[1:]):
        delta = after - reference
        pos = np.floor(np.maximum(delta, 0.0) / threshold + 1e-6).astype(np.int32)
        neg = np.floor(np.maximum(-delta, 0.0) / threshold + 1e-6).astype(np.int32)
        positive.append(pos)
        negative.append(neg)
        reference += (pos - neg) * threshold
    shape = (len(positive),) + sensor.shape[1:]
    return np.stack(positive) if positive else np.zeros(shape, np.int32), np.stack(negative) if negative else np.zeros(shape, np.int32)


def _render_event_map(pos: np.ndarray, neg: np.ndarray, *, max_count: float = 4.0) -> np.ndarray:
    scale = 255.0 / np.log1p(max_count)
    red = np.clip(np.log1p(pos) * scale, 0, 255).astype(np.uint8)
    blue = np.clip(np.log1p(neg) * scale, 0, 255).astype(np.uint8)
    frame = np.zeros((*pos.shape, 3), dtype=np.uint8)
    frame[..., 0] = blue
    frame[..., 2] = red
    both = (pos > 0) & (neg > 0)
    frame[both] = np.maximum(frame[both], np.array([180, 180, 180], dtype=np.uint8))
    return frame


def _label(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 58), (18, 18, 18), -1)
    cv2.putText(output, title, (22, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(output, subtitle, (22, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (205, 205, 205), 1, cv2.LINE_AA)
    return output


def _load_native_scene(scene: Path, base_rpm: float, threshold: float, torch_threads: int):
    transforms = json.loads((scene / "transforms.json").read_text(encoding="utf-8"))
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s")
    intrinsics = CameraIntrinsics(int(transforms["w"]), int(transforms["h"]), float(transforms["fl_x"]), float(transforms["fl_y"]), float(transforms["cx"]), float(transforms["cy"]))
    imu = read_imu_csv(scene / "imu.csv", timestamp_unit="us", gyro_unit="rad/s")
    timestamps = np.clip(timestamps, imu.t[0], imu.t[-1])
    rotations = {"passive": ConstantRotation(0.0), "fixed": ConstantRotation(base_rpm), "gear": ImuGearRotation(imu, base_rpm=base_rpm)}
    luminance = rgb_to_linear_luminance(frames)
    model = RotatingWedgePrism(intrinsics, 0.5)
    sensors = {}
    for name, rotation in rotations.items():
        warper = PrismWarper(model, rotation, height=intrinsics.height, width=intrinsics.width, reference_time=float(timestamps[0]), padding_mode="border")
        with torch.inference_mode():
            sensor, _ = warper.warp(torch.from_numpy(luminance).unsqueeze(1), timestamps)
        sensors[name] = sensor[:, 0].numpy()
    maps = {name: _counts(value, threshold) for name, value in sensors.items()}
    return frames, timestamps, imu, rotations, maps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/dl3dv_large5"))
    parser.add_argument("--output", type=Path, default=Path("results/dl3dv_large5_native_comparison.mp4"))
    parser.add_argument("--contact-sheet", type=Path, default=Path("results/dl3dv_large5_native_contact_sheet.png"))
    parser.add_argument("--base-rpm", type=float, default=720.0)
    parser.add_argument("--contrast-threshold", type=float, default=0.2)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    if args.fps <= 0 or args.torch_threads < 1:
        raise SystemExit("fps and torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    scenes = sorted(args.dataset_root.glob("screen-*"))
    if len(scenes) != 5:
        raise SystemExit(f"Expected five scenes, found {len(scenes)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 540
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width * 3}x{height}", "-r", f"{args.fps:.6f}", "-i", "-", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(args.output)], stdin=subprocess.PIPE)
    representatives = []
    try:
        for scene_index, scene in enumerate(scenes):
            frames, timestamps, imu, rotations, maps = _load_native_scene(scene, args.base_rpm, args.contrast_threshold, args.torch_threads)
            gear = rotations["gear"]
            for frame_index in range(len(frames)):
                if frame_index == 0:
                    maps_now = {name: (np.zeros((height, width), np.int32), np.zeros((height, width), np.int32)) for name in maps}
                else:
                    maps_now = {name: (maps[name][0][frame_index - 1], maps[name][1][frame_index - 1]) for name in maps}
                panels = []
                for name, title in (("passive", "Passive"), ("fixed", "Fixed AMI"), ("gear", "IMU Gear")):
                    pos, neg = maps_now[name]
                    count = int(pos.sum() + neg.sum())
                    if name == "gear":
                        rpm = float(gear.omega_values(np.asarray([timestamps[frame_index]]))[0] * 60.0 / (2.0 * np.pi))
                        subtitle = f"t={timestamps[frame_index]:.3f}s  window={count:,} events  RPM={rpm:.0f}"
                    elif name == "fixed":
                        subtitle = f"t={timestamps[frame_index]:.3f}s  window={count:,} events  RPM={args.base_rpm:.0f}"
                    else:
                        subtitle = f"t={timestamps[frame_index]:.3f}s  window={count:,} events  prism=off"
                    panels.append(_label(_render_event_map(pos, neg), title, subtitle))
                joined = np.concatenate(panels, axis=1)
                if frame_index == len(frames) // 2:
                    representatives.append(joined.copy())
                assert ffmpeg.stdin is not None
                ffmpeg.stdin.write(joined.tobytes())
            print(f"rendered {scene.name} ({scene_index + 1}/{len(scenes)})", flush=True)
    finally:
        if ffmpeg.stdin is not None:
            ffmpeg.stdin.close()
        return_code = ffmpeg.wait()
    if return_code != 0:
        raise SystemExit(f"ffmpeg failed with exit code {return_code}")
    sheet = np.concatenate(representatives, axis=0)
    imageio.imwrite(args.contact_sheet, sheet)
    print(json.dumps({"video": str(args.output), "contact_sheet": str(args.contact_sheet), "resolution": [width * 3, height], "fps": args.fps, "scene_count": len(scenes), "frames_per_scene": 49}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
