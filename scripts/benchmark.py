"""Compute-only benchmark for the current AMI simulator implementation."""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch

# Allow direct execution before the editable package has been installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ami_simulator.events import CompiledCudaThresholdEventSimulator, CudaThresholdEventSimulator, ThresholdEventSimulator
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.rotation import ConstantRotation
from ami_simulator.types import CameraIntrinsics
from ami_simulator.warp import PrismWarper, compensate_events


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _moving_intensity(frames: int, height: int, width: int) -> np.ndarray:
    """Structured motion avoids a random-frame event avalanche in a DVS benchmark."""
    v, u = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    values = []
    for index in range(frames):
        phase = float(index) * 0.65
        frame = 0.5 + 0.19 * np.sin((u + phase) / 13.0) + 0.17 * np.cos((v - 0.7 * phase) / 17.0)
        frame += 0.1 * np.sin((u + v + 1.3 * phase) / 31.0)
        values.append(np.clip(frame, 0.02, 0.98))
    return np.stack(values).astype(np.float32)


def _time_cuda(callback, device: torch.device) -> tuple[object, float]:
    _synchronize(device)
    start = time.perf_counter()
    result = callback()
    _synchronize(device)
    return result, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark AMI simulator stages without image or HDF5 I/O.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--fps", type=float, default=480.0)
    parser.add_argument("--prism-angle-deg", type=float, default=0.5)
    parser.add_argument("--rpm", type=float, default=720.0)
    parser.add_argument("--contrast-threshold", type=float, default=0.2)
    parser.add_argument("--event-backend", choices=["threshold_cpu", "threshold_cuda", "threshold_cuda_compiled"], default="threshold_cpu")
    parser.add_argument("--max-events-per-pixel", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--compensation-chunk-size", type=int, default=250_000)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This benchmark requires an available CUDA device.")
    if args.frames < 2 or args.height <= 0 or args.width <= 0 or args.fps <= 0:
        raise SystemExit("frames >= 2 and positive dimensions/FPS are required.")

    timestamps = np.arange(args.frames, dtype=np.float64) / args.fps
    intrinsics = CameraIntrinsics(args.width, args.height, float(args.width), float(args.width), (args.width - 1.0) / 2.0, (args.height - 1.0) / 2.0)
    prism = RotatingWedgePrism(intrinsics, args.prism_angle_deg)
    rotation = ConstantRotation(args.rpm)
    warper = PrismWarper(prism, rotation, height=args.height, width=args.width, reference_time=float(timestamps[0]))
    input_tensor = torch.from_numpy(_moving_intensity(args.frames, args.height, args.width)).to(device).unsqueeze(1)

    for _ in range(args.warmup):
        warped, _ = warper.warp(input_tensor, timestamps)
    _synchronize(device)
    (warped, valid), warp_seconds = _time_cuda(lambda: warper.warp(input_tensor, timestamps), device)

    if args.event_backend == "threshold_cpu":
        start = time.perf_counter()
        sensor_intensity = warped[:, 0].cpu().numpy()
        transfer_seconds = time.perf_counter() - start
        start = time.perf_counter()
        raw_events = ThresholdEventSimulator(args.contrast_threshold).simulate(sensor_intensity, timestamps)
        event_seconds = time.perf_counter() - start
    elif args.event_backend == "threshold_cuda":
        transfer_seconds = 0.0
        backend = CudaThresholdEventSimulator(args.contrast_threshold)
        # Exclude CUDA context/module initialization and allocator setup.
        backend.simulate(warped[:, 0], timestamps)
        _synchronize(device)
        raw_events, event_seconds = _time_cuda(
            lambda: backend.simulate(warped[:, 0], timestamps),
            device,
        )
    else:
        transfer_seconds = 0.0
        backend = CompiledCudaThresholdEventSimulator(args.contrast_threshold, args.max_events_per_pixel)
        # Lazily building/loading the extension is a one-time startup cost, not
        # steady-state simulator throughput.
        backend.simulate(warped[:, 0], timestamps)
        _synchronize(device)
        raw_events, event_seconds = _time_cuda(
            lambda: backend.simulate(warped[:, 0], timestamps),
            device,
        )
    _, compensation_seconds = _time_cuda(
        lambda: compensate_events(raw_events, prism, rotation, theta0=0.0, reference_time=float(timestamps[0]), device=str(device), chunk_size=args.compensation_chunk_size),
        device,
    )
    total_seconds = warp_seconds + transfer_seconds + event_seconds + compensation_seconds
    result = {
        "device": torch.cuda.get_device_name(device),
        "resolution": [args.width, args.height],
        "frames": args.frames,
        "event_backend": args.event_backend,
        "event_count": len(raw_events),
        "valid_sensor_fraction": float(valid.float().mean().item()),
        "seconds": {
            "warp_cuda": warp_seconds,
            "sensor_transfer_to_cpu": transfer_seconds,
            "threshold_events": event_seconds,
            "compensation_cuda": compensation_seconds,
            "compute_total": total_seconds,
        },
        "throughput": {
            "warp_frames_per_second": args.frames / warp_seconds,
            "end_to_end_compute_frames_per_second": args.frames / total_seconds,
            "threshold_events_per_second": len(raw_events) / event_seconds if event_seconds else float("inf"),
            "compensation_events_per_second": len(raw_events) / compensation_seconds if compensation_seconds else float("inf"),
        },
        "notes": (
            "Compute-only. HDF5/image I/O and compiled-extension build/load startup are excluded. "
            + ("The selected backend uses a hand-written CUDA extension; output transfer and temporal sorting remain included."
               if args.event_backend == "threshold_cuda_compiled"
               else "Output transfer and temporal sorting remain included.")
        ),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
