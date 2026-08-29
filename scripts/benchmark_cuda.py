#!/usr/bin/env python3
"""Benchmark EIRM CUDA-capable stages without writing experiment artifacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ami_sim.events import DVSVoltmeterCudaFastBackend, DVSVoltmeterTorchBackend
from ami_sim.prism import ConstantRotationPolicy, PaperSimplifiedPrismModel, PrismWarper
from ami_sim.types import CameraIntrinsics


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(function, device: torch.device, repeats: int) -> tuple[float, object]:
    # Warm up kernels before measuring steady-state execution.
    result = function()
    _sync(device)
    durations: list[float] = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        result = function()
        _sync(device)
        durations.append(time.perf_counter() - start)
    return float(np.median(durations)), result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark EIRM CUDA stages.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--rpm", type=float, default=720.0)
    parser.add_argument("--prism-angle-deg", type=float, default=0.5)
    parser.add_argument("--warp-batch-size", type=int, default=16)
    parser.add_argument("--transfer-chunk-frames", type=int, default=16)
    parser.add_argument(
        "--dvs-precision",
        choices=["float32", "float64"],
        default="float64",
        help="DVS arithmetic precision (float64 reference, float32 experimental alternative).",
    )
    parser.add_argument("--event-backend", choices=["torch", "cuda_fast"], default="torch")
    parser.add_argument("--max-events-per-pixel", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-dvs", action="store_true")
    parser.add_argument("--dvs-root", type=Path, default=Path(__file__).resolve().parents[1] / "DVS-Voltmeter")
    return parser


def main() -> int:
    args = _parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")
    if args.frames < 2 or args.height <= 0 or args.width <= 0 or args.repeats <= 0:
        raise SystemExit("frames must be >=2 and dimensions/repeats must be positive.")

    torch.manual_seed(0)
    timestamps = np.linspace(0.0, (args.frames - 1) / 480.0, args.frames, dtype=np.float64)
    intrinsics = CameraIntrinsics(args.width, args.height, float(args.width), float(args.height), (args.width - 1) / 2.0, (args.height - 1) / 2.0)
    model = PaperSimplifiedPrismModel(intrinsics, args.prism_angle_deg)
    policy = ConstantRotationPolicy(args.rpm)
    frames = torch.rand((args.frames, 1, args.height, args.width), device=device, dtype=torch.float32)
    warper = PrismWarper(
        model=model,
        policy=policy,
        height=args.height,
        width=args.width,
        reference_time=float(timestamps[0]),
        batch_size=args.warp_batch_size,
    )
    warp_seconds, warped = _timed(lambda: warper.warp(frames, timestamps), device, args.repeats)
    result: dict[str, object] = {
        "device": str(device),
        "frames": args.frames,
        "resolution": [args.height, args.width],
        "warp_seconds_median": warp_seconds,
        "warp_frames_per_second": args.frames / warp_seconds,
        "warp_batch_size": args.warp_batch_size,
    }

    if not args.skip_dvs:
        backend_cls = DVSVoltmeterCudaFastBackend if args.event_backend == "cuda_fast" else DVSVoltmeterTorchBackend
        backend_kwargs = {
            "dvs_root": args.dvs_root,
            "device": device,
            "transfer_chunk_frames": args.transfer_chunk_frames,
            "seed": 0,
            "precision": args.dvs_precision,
        }
        if args.event_backend == "cuda_fast":
            backend_kwargs["max_events_per_pixel"] = args.max_events_per_pixel
        backend = backend_cls(
            **backend_kwargs,
        )
        intensity = frames[:, 0]
        dvs_seconds, events = _timed(lambda: backend.simulate(intensity, timestamps), device, args.repeats)
        result.update({
            "dvs_seconds_median": dvs_seconds,
            "dvs_frames_per_second": args.frames / dvs_seconds,
            "dvs_event_count": len(events),
            "dvs_transfer_chunk_frames": args.transfer_chunk_frames,
            "dvs_precision": args.dvs_precision,
            "dvs_backend": args.event_backend,
            "dvs_max_events_per_pixel": args.max_events_per_pixel,
        })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
