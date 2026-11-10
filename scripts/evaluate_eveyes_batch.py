#!/usr/bin/env python3
"""Batch in-memory comparison of fixed AMI and IMU-conditioned prism motion.

The Eveyes archive contains RGB and IMU but no event stream.  This script
therefore evaluates event counts directly in memory and writes only one small
JSON summary, avoiding a large HDF5 file for every sequence.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ami_simulator.dsec import read_imu_csv
from ami_simulator.events import ThresholdEventSimulator
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.rotation import ConstantRotation, ImuGearRotation
from ami_simulator.types import CameraIntrinsics
from ami_simulator.warp import PrismWarper


def _count_events(
    frames: np.ndarray,
    timestamps: np.ndarray,
    intrinsics: CameraIntrinsics,
    rotation: object,
    contrast_threshold: float,
) -> tuple[int, float]:
    """Form events without writing the event stream to disk."""
    height, width = frames.shape[1:3]
    model = RotatingWedgePrism(intrinsics, 0.5)
    warper = PrismWarper(
        model,
        rotation,
        height=height,
        width=width,
        reference_time=float(timestamps[0]),
        padding_mode="border",
    )
    luminance = rgb_to_linear_luminance(frames)
    with torch.inference_mode():
        sensor, valid = warper.warp(torch.from_numpy(luminance).unsqueeze(1), timestamps)
    events = ThresholdEventSimulator(contrast_threshold).simulate(sensor[:, 0].numpy(), timestamps)
    return int(len(events)), float(valid.float().mean().item())


def _sequence_roots(dataset_root: Path) -> list[Path]:
    roots = []
    for imu_path in sorted(dataset_root.rglob("imu.csv")):
        root = imu_path.parent
        required = (root / "rgb", root / "timestamps.csv", root / "camera_intrinsics.json")
        if all(path.exists() for path in required):
            roots.append(root)
    return roots


def evaluate_sequence(root: Path, *, max_frames: int, base_rpm: float, contrast_threshold: float) -> dict[str, object]:
    frames, timestamps = load_rgb_sequence(root / "rgb", root / "timestamps.csv", timestamp_unit="us", max_frames=max_frames)
    intrinsics_data = json.loads((root / "camera_intrinsics.json").read_text(encoding="utf-8"))
    height, width = frames.shape[1:3]
    intrinsics = CameraIntrinsics(
        width,
        height,
        float(intrinsics_data["fx"]),
        float(intrinsics_data["fy"]),
        float(intrinsics_data["cx"]),
        float(intrinsics_data["cy"]),
    )
    imu = read_imu_csv(root / "imu.csv", timestamp_unit="s", gyro_unit="rad/s")
    fixed_rotation = ConstantRotation(base_rpm)
    gear_rotation = ImuGearRotation(imu, base_rpm=base_rpm)
    fixed_count, fixed_valid = _count_events(frames, timestamps, intrinsics, fixed_rotation, contrast_threshold)
    gear_count, gear_valid = _count_events(frames, timestamps, intrinsics, gear_rotation, contrast_threshold)
    result = {
        "split": root.parts[-3],
        "scene": root.parts[-2],
        "motion_regime": root.parts[-1],
        "frame_count": int(len(frames)),
        "timestamp_span_s": float(timestamps[-1] - timestamps[0]),
        "imu_sample_count": int(len(imu.t)),
        "gyro_norm_max_rad_s": float(np.linalg.norm(imu.gyro, axis=1).max()),
        "fixed_raw_events": fixed_count,
        "imu_gear_raw_events": gear_count,
        "event_delta_percent": float(100.0 * (gear_count - fixed_count) / max(fixed_count, 1)),
        "fixed_valid_fraction": fixed_valid,
        "imu_gear_valid_fraction": gear_valid,
        "gear_metadata": gear_rotation.metadata(),
    }
    del frames, imu
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/eveyes_full"))
    parser.add_argument("--output", type=Path, default=Path("results/eveyes_batch_summary.json"))
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--base-rpm", type=float, default=720.0)
    parser.add_argument("--contrast-threshold", type=float, default=0.2)
    args = parser.parse_args()
    if args.max_frames < 2:
        raise SystemExit("--max-frames must be at least 2")
    roots = _sequence_roots(args.dataset_root)
    if args.max_sequences is not None:
        roots = roots[: args.max_sequences]
    if not roots:
        raise SystemExit(f"No complete Eveyes sequences found under {args.dataset_root}")
    results = []
    for index, root in enumerate(roots, start=1):
        print(f"[{index}/{len(roots)}] {root}", flush=True)
        # Keep peak memory bounded when evaluating the full archive.
        result = evaluate_sequence(root, max_frames=args.max_frames, base_rpm=args.base_rpm, contrast_threshold=args.contrast_threshold)
        results.append(result)
        print(json.dumps({key: result[key] for key in ("motion_regime", "fixed_raw_events", "imu_gear_raw_events", "event_delta_percent")}, sort_keys=True), flush=True)
        gc.collect()
    summary = {
        "dataset": "Eveyes",
        "dataset_root": str(args.dataset_root),
        "sequence_count": len(results),
        "max_frames": args.max_frames,
        "base_rpm": args.base_rpm,
        "contrast_threshold": args.contrast_threshold,
        "sequences": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    del results
    gc.collect()
    print(json.dumps({"output": str(args.output), "sequence_count": len(roots)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
