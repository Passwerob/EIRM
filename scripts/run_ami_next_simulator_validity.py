#!/usr/bin/env python3
"""Run the CPU-only Stage A simulator validity protocol.

The protocol deliberately avoids RIFE, learned policies, and real hardware. It
checks deterministic threshold behavior, phase reparameterization, and the
effect of temporal sampling on a static finite-width edge under prism motion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ami_simulator.events import ThresholdEventSimulator
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.rotation import ConstantRotation
from ami_simulator.types import CameraIntrinsics
from ami_simulator.warp import PrismWarper


def _static_edge(height: int, width: int) -> np.ndarray:
    image = np.full((height, width), 0.2, dtype=np.float32)
    image[:, width // 2 :] = 0.8
    # Add finite-width structure so the test is not an ideal discontinuity.
    image[:, width // 2 - 2 : width // 2 + 2] = np.linspace(0.35, 0.65, 4, dtype=np.float32)[None, :]
    return image


def _prism_stream(
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    rpm: float,
    theta0: float,
    samples: int,
    threshold: float,
    prism_angle_deg: float,
    duration_seconds: float | None = None,
    padding_mode: str = "border",
) -> tuple[object, float]:
    duration = float(60.0 / rpm if duration_seconds is None else duration_seconds)
    timestamps = np.linspace(0.0, duration, samples, dtype=np.float64)
    rotation = ConstantRotation(rpm)
    prism = RotatingWedgePrism(intrinsics, prism_angle_deg)
    warper = PrismWarper(
        prism,
        rotation,
        height=image.shape[0],
        width=image.shape[1],
        theta0=theta0,
        reference_time=0.0,
        padding_mode=padding_mode,
    )
    frames = torch.from_numpy(np.repeat(image[None, None], samples, axis=0))
    with torch.inference_mode():
        sensor, valid = warper.warp(frames, timestamps)
    events = ThresholdEventSimulator(threshold).simulate(sensor[:, 0].numpy(), timestamps)
    return events, float(valid.float().mean().item())


def _stream_signature(events: object, rpm: float) -> dict[str, object]:
    if len(events) == 0:
        return {"count": 0, "positive": 0, "negative": 0, "coordinates": [], "phase": []}
    phase = np.remainder(2.0 * np.pi * rpm * events.t / 60.0, 2.0 * np.pi)
    coords = np.column_stack((events.x, events.y, events.p)).astype(np.int64)
    return {
        "count": int(len(events)),
        "positive": int(np.count_nonzero(events.p > 0)),
        "negative": int(np.count_nonzero(events.p < 0)),
        "coordinates": coords.tolist(),
        "phase": phase.tolist(),
    }


def _relative_error(value: float, reference: float) -> float:
    return float(abs(value - reference) / max(abs(reference), 1.0))


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/simulator_validity.json"))
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--prism-angle-deg", type=float, default=0.5)
    parser.add_argument("--rpms", type=float, nargs="+", default=[90.0, 180.0, 360.0, 720.0, 1080.0])
    parser.add_argument("--phases", type=float, nargs="+", default=[0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0])
    parser.add_argument("--samples", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    args = parser.parse_args()
    if args.width < 8 or args.height < 8 or args.threshold <= 0.0 or args.prism_angle_deg <= 0.0:
        parser.error("image dimensions, threshold, and prism angle must be positive")
    if any(rpm <= 0.0 for rpm in args.rpms) or any(samples < 2 for samples in args.samples):
        parser.error("RPM and sample counts must be positive")

    image = _static_edge(args.height, args.width)
    intrinsics = CameraIntrinsics(args.width, args.height, 40.0, 40.0, (args.width - 1) / 2.0, (args.height - 1) / 2.0)
    simulator = ThresholdEventSimulator(args.threshold)

    constant = simulator.simulate(np.repeat(np.full((1, args.height, args.width), 0.5, dtype=np.float32), 4, axis=0), np.linspace(0.0, 1.0, 4))
    constant_check = {"event_count": int(len(constant)), "passed": len(constant) == 0}

    phase = np.linspace(0.0, 2.0 * np.pi, 129, dtype=np.float64)
    scalar = np.exp(0.55 * np.sin(phase)).astype(np.float32)[:, None, None]
    scalar_events = simulator.simulate(np.broadcast_to(scalar, (len(phase), args.height, args.width)), phase)
    periodic_check = {
        "event_count": int(len(scalar_events)),
        "endpoint_equal": bool(np.isclose(float(scalar[0, 0, 0]), float(scalar[-1, 0, 0]))),
        "passed": bool(len(scalar_events) > 0 and np.isclose(float(scalar[0, 0, 0]), float(scalar[-1, 0, 0]))),
    }

    equal_cycle_rows: list[dict[str, object]] = []
    phase_invariance: list[dict[str, object]] = []
    for theta0 in args.phases:
        reference_signature = None
        for rpm in args.rpms:
            events, valid_fraction = _prism_stream(
                image,
                intrinsics,
                rpm=float(rpm),
                theta0=float(theta0),
                samples=256,
                threshold=args.threshold,
                prism_angle_deg=args.prism_angle_deg,
            )
            signature = _stream_signature(events, float(rpm))
            row = {"theta0_rad": float(theta0), "rpm": float(rpm), "samples": 256, "valid_fraction": valid_fraction, "event_count": signature["count"], "positive": signature["positive"], "negative": signature["negative"]}
            equal_cycle_rows.append(row)
            if reference_signature is None:
                reference_signature = signature
                matches = True
                coordinate_mismatch_fraction = 0.0
                phase_rmse = 0.0
            else:
                reference_coords = np.asarray(reference_signature["coordinates"], dtype=np.int64)
                candidate_coords = np.asarray(signature["coordinates"], dtype=np.int64)
                coordinate_mismatch_fraction = float(np.mean(np.any(reference_coords != candidate_coords, axis=1))) if len(candidate_coords) == len(reference_coords) and len(candidate_coords) else 0.0
                reference_phase = np.sort(np.asarray(reference_signature["phase"], dtype=np.float64))
                candidate_phase = np.sort(np.asarray(signature["phase"], dtype=np.float64))
                phase_rmse = float(np.sqrt(np.mean((reference_phase - candidate_phase) ** 2))) if len(candidate_phase) == len(reference_phase) and len(candidate_phase) else 0.0
                matches = bool(
                    signature["count"] == reference_signature["count"]
                    and signature["positive"] == reference_signature["positive"]
                    and signature["negative"] == reference_signature["negative"]
                    and phase_rmse <= 1e-9
                )
            phase_invariance.append({"theta0_rad": float(theta0), "rpm": float(rpm), "matches_first_rpm": matches, "coordinate_mismatch_fraction": coordinate_mismatch_fraction, "phase_rmse_rad": phase_rmse})

    convergence_rows: list[dict[str, object]] = []
    for rpm in args.rpms:
        reference_events, _ = _prism_stream(image, intrinsics, rpm=float(rpm), theta0=0.0, samples=max(args.samples), threshold=args.threshold, prism_angle_deg=args.prism_angle_deg)
        reference_count = len(reference_events)
        for samples in args.samples:
            events, valid_fraction = _prism_stream(image, intrinsics, rpm=float(rpm), theta0=0.0, samples=int(samples), threshold=args.threshold, prism_angle_deg=args.prism_angle_deg)
            convergence_rows.append({"rpm": float(rpm), "samples": int(samples), "event_count": int(len(events)), "reference_samples": int(max(args.samples)), "relative_count_error": _relative_error(len(events), reference_count), "valid_fraction": valid_fraction})

    equal_time_rows: list[dict[str, object]] = []
    equal_time_seconds = 0.1
    for theta0 in args.phases:
        for rpm in args.rpms:
            events, valid_fraction = _prism_stream(
                image,
                intrinsics,
                rpm=float(rpm),
                theta0=float(theta0),
                samples=256,
                threshold=args.threshold,
                prism_angle_deg=args.prism_angle_deg,
                duration_seconds=equal_time_seconds,
            )
            equal_time_rows.append({"theta0_rad": float(theta0), "rpm": float(rpm), "duration_seconds": equal_time_seconds, "phase_span_rad": float(2.0 * np.pi * rpm * equal_time_seconds / 60.0), "event_count": int(len(events)), "positive": int(np.count_nonzero(events.p > 0)), "negative": int(np.count_nonzero(events.p < 0)), "valid_fraction": valid_fraction})

    border_rows: list[dict[str, object]] = []
    for padding_mode in ("border", "zeros"):
        events, valid_fraction = _prism_stream(image, intrinsics, rpm=180.0, theta0=0.0, samples=256, threshold=args.threshold, prism_angle_deg=args.prism_angle_deg, padding_mode=padding_mode)
        border_rows.append({"padding_mode": padding_mode, "event_count": int(len(events)), "positive": int(np.count_nonzero(events.p > 0)), "negative": int(np.count_nonzero(events.p < 0)), "valid_fraction": valid_fraction})

    resolution_rows: list[dict[str, object]] = []
    for resolution in (16, 32, 64):
        resolution_image = _static_edge(resolution, resolution)
        scale = resolution / args.width
        resolution_intrinsics = CameraIntrinsics(resolution, resolution, 40.0 * scale, 40.0 * scale, (resolution - 1) / 2.0, (resolution - 1) / 2.0)
        events, valid_fraction = _prism_stream(resolution_image, resolution_intrinsics, rpm=180.0, theta0=0.0, samples=256, threshold=args.threshold, prism_angle_deg=args.prism_angle_deg)
        resolution_rows.append({"width": resolution, "height": resolution, "rpm": 180.0, "event_count": int(len(events)), "event_density": float(len(events) / (resolution * resolution)), "valid_fraction": valid_fraction})

    report = {
        "protocol": "ami_next_stage_a_cpu_validity_v1",
        "run_id": "ami_next_stage_a_cpu_default_20260905",
        "config_hash": hashlib.sha256(json.dumps({"width": args.width, "height": args.height, "threshold": args.threshold, "prism_angle_deg": args.prism_angle_deg, "rpms": args.rpms, "phases": args.phases, "samples": args.samples}, sort_keys=True).encode("utf-8")).hexdigest(),
        "method": "deterministic_cpu_prism_validity_harness",
        "split": "analytic_smoke",
        "seed": 0,
        "sensor_model": "deterministic_log_threshold_uncalibrated",
        "phase_source": "command_integrated_constant_rotation",
        "window_ms": 100.0,
        "runtime_seconds": float(time.perf_counter() - started),
        "constraint_status": "no_motor_constraints; static analytic diagnostic",
        "simulated_or_real": "simulated",
        "dependencies": {"uses_rife": False, "uses_learned_policy": False, "uses_cuda": False},
        "configuration": {"width": args.width, "height": args.height, "threshold": args.threshold, "prism_angle_deg": args.prism_angle_deg, "rpms": [float(v) for v in args.rpms], "initial_phases_rad": [float(v) for v in args.phases], "sample_counts": [int(v) for v in args.samples]},
        "constant_luminance": constant_check,
        "periodic_endpoint": periodic_check,
        "equal_cycle": {"rows": equal_cycle_rows, "phase_invariance": phase_invariance, "all_passed": bool(all(row["matches_first_rpm"] for row in phase_invariance))},
        "sampling_convergence": {"rows": convergence_rows, "reference": "largest sample count per RPM; relative event-count error is diagnostic, not a universal error bound"},
        "equal_time": {"duration_seconds": equal_time_seconds, "rows": equal_time_rows, "interpretation": "RPM changes the covered phase span in a fixed observation window; differences are expected and must be evaluated against target support, overlap, bandwidth, and downstream utility."},
        "border_sensitivity": {"rows": border_rows, "interpretation": "Padding mode changes invalid-border values; zeros is a stress diagnostic, while scientific results must use a common valid FOV or an explicit invalid mask."},
        "resolution_sensitivity": {"rows": resolution_rows, "interpretation": "A coarse resolution sweep checks event density and valid area, not physical resolution invariance."},
        "limitations": [
            "Static finite-width edge and deterministic threshold model only.",
            "Equal-cycle invariance compares event count, polarity, and integer sensor coordinates; subpixel crossing tolerance is not claimed.",
            "No photoreceptor, refractory, threshold mismatch, readout, encoder, or motor dynamics model.",
            "CUDA/reference parity is pending because CUDA is unavailable in this environment.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["rpm", "samples", "event_count", "reference_samples", "relative_count_error", "valid_fraction"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(convergence_rows)
    print(json.dumps({"output": str(args.output), "csv": str(csv_path), "constant_passed": constant_check["passed"], "periodic_passed": periodic_check["passed"], "equal_cycle_passed": report["equal_cycle"]["all_passed"]}, indent=2))
    return 0 if constant_check["passed"] and periodic_check["passed"] and report["equal_cycle"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
