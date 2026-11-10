"""Command-line entry point for AMI Prism Simulator."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import SimulationConfig, load_rgb_sequence, run_simulation
from .dsec import read_imu_csv
from .rotation import ImuRotation
from .types import CameraIntrinsics
from .vfi import DebugLinearVFIBackend, PracticalRIFEBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate a rotating-wedge AMI event camera from timestamped RGB frames.")
    parser.add_argument("--rgb-dir", required=True, type=Path)
    parser.add_argument("--timestamps", required=True, type=Path)
    parser.add_argument("--timestamp-unit", choices=["s", "ms", "us", "ns"], default="s")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit a sequence for a short smoke run.")
    parser.add_argument("--fx", type=float)
    parser.add_argument("--fy", type=float)
    parser.add_argument("--cx", type=float)
    parser.add_argument("--cy", type=float)
    parser.add_argument("--allow-default-intrinsics", action="store_true")
    parser.add_argument("--prism-angle-deg", type=float, default=0.5)
    parser.add_argument("--rpm", type=float, default=720.0)
    parser.add_argument("--imu-csv", type=Path, default=None, help="Optional IMU CSV driving prism angular velocity.")
    parser.add_argument("--imu-time-unit", choices=["s", "ms", "us", "ns"], default="s")
    parser.add_argument("--imu-gyro-unit", choices=["rad/s", "deg/s", "dps"], default="rad/s")
    parser.add_argument("--imu-gyro-axis", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--imu-gyro-sign", type=float, default=1.0)
    parser.add_argument("--theta0", type=float, default=0.0)
    parser.add_argument("--contrast-threshold", type=float, default=0.2)
    parser.add_argument("--event-backend", choices=["threshold_cpu", "threshold_cuda", "threshold_cuda_compiled"], default="threshold_cpu")
    parser.add_argument("--max-events-per-pixel", type=int, default=128)
    parser.add_argument("--interpolation-fps", type=float, default=None)
    parser.add_argument("--vfi", choices=("none", "practical_rife", "debug_linear"), default="none")
    parser.add_argument("--practical-rife-root", type=Path, default=None)
    parser.add_argument("--rife-checkpoint", type=Path, default=None)
    parser.add_argument("--rife-fp16", action="store_true")
    parser.add_argument("--rife-inference-batch-size", type=int, default=8)
    parser.add_argument("--padding-mode", choices=["zeros", "border", "reflection"], default="border")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compensation-chunk-size", type=int, default=250_000)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frames, timestamps = load_rgb_sequence(args.rgb_dir, args.timestamps, timestamp_unit=args.timestamp_unit, max_frames=args.max_frames)
    height, width = frames.shape[1:3]
    supplied = (args.fx, args.fy, args.cx, args.cy)
    if all(value is not None for value in supplied):
        intrinsics = CameraIntrinsics(width, height, args.fx, args.fy, args.cx, args.cy)
        intrinsic_source = "command_line"
    elif args.allow_default_intrinsics:
        intrinsics = CameraIntrinsics(width, height, float(width), float(height), (width - 1.0) / 2.0, (height - 1.0) / 2.0)
        intrinsic_source = "image_centered_default_for_debug_only"
    else:
        raise SystemExit("Provide all of --fx --fy --cx --cy, or explicitly pass --allow-default-intrinsics for a debug run.")
    config = SimulationConfig(
        prism_angle_deg=args.prism_angle_deg,
        rpm=args.rpm,
        theta0=args.theta0,
        contrast_threshold=args.contrast_threshold,
        event_backend=args.event_backend,
        max_events_per_pixel=args.max_events_per_pixel,
        interpolation_fps=args.interpolation_fps,
        padding_mode=args.padding_mode,
        device=args.device,
        compensation_chunk_size=args.compensation_chunk_size,
    )
    rotation_policy = None
    source_metadata = {"input_timestamp_unit": args.timestamp_unit, "intrinsics_source": intrinsic_source}
    if args.imu_csv is not None:
        imu = read_imu_csv(args.imu_csv, timestamp_unit=args.imu_time_unit, gyro_unit=args.imu_gyro_unit)
        rotation_policy = ImuRotation(imu, axis=args.imu_gyro_axis, sign=args.imu_gyro_sign)
        source_metadata["imu_csv"] = str(args.imu_csv)
    if (args.interpolation_fps is None) != (args.vfi == "none"):
        raise SystemExit("Set both --interpolation-fps and a non-none --vfi, or set neither.")
    if args.vfi == "practical_rife":
        if args.practical_rife_root is None:
            raise SystemExit("--practical-rife-root is required for Practical-RIFE.")
        vfi_backend = PracticalRIFEBackend(args.practical_rife_root, args.rife_checkpoint, device=args.device, fp16=args.rife_fp16, inference_batch_size=args.rife_inference_batch_size)
    elif args.vfi == "debug_linear":
        vfi_backend = DebugLinearVFIBackend()
    else:
        vfi_backend = None
    summary = run_simulation(frames, timestamps, intrinsics, args.output, config=config, rotation_policy=rotation_policy, vfi_backend=vfi_backend, source_metadata=source_metadata)
    print(f"Wrote {summary['raw_event_count']} raw events and {summary['compensated_event_count']} compensated events to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
