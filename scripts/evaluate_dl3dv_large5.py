#!/usr/bin/env python3
"""Evaluate Passive, Fixed AMI, and IMU Gear on the DL3DV-large5 package."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ami_simulator.dsec import read_imu_csv
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.rotation import ConstantRotation, ImuGearRotation
from ami_simulator.types import CameraIntrinsics
from ami_simulator.warp import PrismWarper


def _count_threshold_events(intensity: np.ndarray, threshold: float) -> tuple[int, int, float]:
    """Count the deterministic threshold events without materializing event arrays."""
    log_frames = np.log(np.clip(np.asarray(intensity, dtype=np.float64), 1e-6, 1.0))
    reference = log_frames[0].copy()
    total = 0
    positive = 0
    covered = np.zeros(reference.shape, dtype=bool)
    for before, after in zip(log_frames[:-1], log_frames[1:]):
        delta = after - reference
        pos = np.floor(np.maximum(delta, 0.0) / threshold + 1e-6).astype(np.int64)
        neg = np.floor(np.maximum(-delta, 0.0) / threshold + 1e-6).astype(np.int64)
        total += int(pos.sum() + neg.sum())
        positive += int(pos.sum())
        covered |= (pos > 0) | (neg > 0)
        reference += (pos - neg) * threshold
    return total, int(covered.sum()), float(positive / total) if total else 0.0


def _run(frames: np.ndarray, timestamps: np.ndarray, intrinsics: CameraIntrinsics, rotation: object, threshold: float) -> dict[str, float]:
    height, width = frames.shape[1:3]
    luminance = rgb_to_linear_luminance(frames)
    model = RotatingWedgePrism(intrinsics, 0.5)
    warper = PrismWarper(model, rotation, height=height, width=width, reference_time=float(timestamps[0]), padding_mode="border")
    with torch.inference_mode():
        sensor, valid = warper.warp(torch.from_numpy(luminance).unsqueeze(1), timestamps)
    event_count, covered_pixels, positive_fraction = _count_threshold_events(sensor[:, 0].numpy(), threshold)
    return {"events": event_count, "events_per_megapixel": float(event_count / (height * width) * 1e6), "coverage": float(covered_pixels / (height * width)), "positive_fraction": positive_fraction, "valid_fraction": float(valid.all(dim=0).float().mean())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/dl3dv_large5"))
    parser.add_argument("--output", type=Path, default=Path("results/dl3dv_large5_comparison.json"))
    parser.add_argument("--base-rpm", type=float, default=720.0)
    parser.add_argument("--fps", type=float, default=30.0, help="Must match the nominal timeline used to build the package.")
    parser.add_argument("--contrast-threshold", type=float, default=0.2)
    parser.add_argument("--eval-width", type=int, default=480, help="Width used for CPU comparison; source package remains at native 960x540.")
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    if args.torch_threads < 1 or args.fps <= 0 or args.eval_width < 32:
        raise SystemExit("torch-threads, fps, and eval-width must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    scenes = sorted(args.dataset_root.glob("screen-*"))
    if len(scenes) != 5:
        raise SystemExit(f"Expected exactly five scene packages under {args.dataset_root}, found {len(scenes)}")
    records = []
    for index, scene in enumerate(scenes, start=1):
        transforms = json.loads((scene / "transforms.json").read_text(encoding="utf-8"))
        frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s")
        intrinsics = CameraIntrinsics(int(transforms["w"]), int(transforms["h"]), float(transforms["fl_x"]), float(transforms["fl_y"]), float(transforms["cx"]), float(transforms["cy"]))
        if args.eval_width != intrinsics.width:
            eval_height = int(round(intrinsics.height * args.eval_width / intrinsics.width))
            frames = np.stack([cv2.resize(frame, (args.eval_width, eval_height), interpolation=cv2.INTER_AREA) for frame in frames])
            intrinsics = intrinsics.resized(args.eval_width, eval_height)
        imu = read_imu_csv(scene / "imu.csv", timestamp_unit="us", gyro_unit="rad/s")
        # Decimal nominal timestamps can exceed the final IMU sample by ~1e-16.
        timestamps = np.clip(timestamps, imu.t[0], imu.t[-1])
        fixed = _run(frames, timestamps, intrinsics, ConstantRotation(args.base_rpm), args.contrast_threshold)
        gear_rotation = ImuGearRotation(imu, base_rpm=args.base_rpm)
        gear = _run(frames, timestamps, intrinsics, gear_rotation, args.contrast_threshold)
        passive = _run(frames, timestamps, intrinsics, ConstantRotation(0.0), args.contrast_threshold)
        motion = np.linalg.norm(imu.gyro, axis=1)
        source_events = json.loads((scene / "scene.json").read_text(encoding="utf-8"))["official_reference_events"]["source_directory"]
        records.append({"scene_id": scene.name, "source_scene": json.loads((scene / "scene.json").read_text(encoding="utf-8"))["source_scene"], "frame_count": len(frames), "duration_s": float(timestamps[-1] - timestamps[0]), "source_resolution": [int(transforms["w"]), int(transforms["h"])], "evaluation_resolution": [intrinsics.width, intrinsics.height], "imu_samples": len(imu.t), "gyro_norm_rad_s": {"mean": float(motion.mean()), "p95": float(np.quantile(motion, 0.95)), "max": float(motion.max())}, "passive": passive, "fixed_ami": fixed, "imu_gear": {**gear, "gear_counts": gear_rotation.metadata()["gear_counts"], "mean_rpm": float(np.mean(gear_rotation.omega_values(timestamps) * 60.0 / (2.0 * np.pi)))}, "official_reference_events": {"source_directory": source_events, "not_used_in_ami_comparison": True}})
        print(json.dumps({"scene": scene.name, "passive": passive["events"], "fixed": fixed["events"], "imu_gear": gear["events"], "gear_counts": gear_rotation.metadata()["gear_counts"]}), flush=True)
        del frames, imu
        gc.collect()
    summary = {"dataset": "DL3DV-large5", "dataset_root": str(args.dataset_root), "scene_count": len(records), "fps_assumption": args.fps, "base_rpm": args.base_rpm, "contrast_threshold": args.contrast_threshold, "evaluation_width": args.eval_width, "event_policy": "official preprocessed DVS-Voltmeter tensors are external references; only AMI simulator outputs enter the comparison", "scenes": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scene_count": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
