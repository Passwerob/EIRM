"""End-to-end AMI pipeline orchestration."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
import yaml

from ami_sim.datasets import DSECRGBSequence
from ami_sim.events import EventSimulatorBackend, write_events_h5
from ami_sim.metrics import (
    edge_coverage,
    edge_distance,
    event_observability_stats,
    event_spatial_variance,
    validate_event_stream,
    newly_recovered_structure,
    relative_motion_diagnostics,
    temporal_flicker_stats,
)
from ami_sim.prism import PrismModel, PrismWarper, RotationPolicy, compensate_events
from ami_sim.types import CameraIntrinsics, EventStream
from ami_sim.vfi import VFIBackend
from ami_sim.visualization import (
    save_event_comparison_panel,
    save_event_image,
    save_frame_sequence,
    save_panel,
    save_temporal_difference_video,
)


@dataclass(frozen=True)
class SimulationOptions:
    interp_fps: float = 480.0
    output_frame_count: int | None = None
    prism_model: str = "paper_simplified"
    prism_angle_deg: float = 0.5
    rotation_rpm: float = 720.0
    theta0: float = 0.0
    padding_mode: str = "border"
    align_corners: bool = False
    event_window_ms: float = 33.0
    valid_orientation_samples: int = 360
    save_debug_video: bool = False
    allow_default_intrinsics: bool = False
    random_seed: int = 0
    device: str | None = None
    warp_batch_size: int = 8
    compensation_chunk_size: int = 1_000_000
    cuda_fast_math: bool = False
    software_version: str = "0.1.0+cu"


def rgb_to_luminance(rgb_frames: np.ndarray) -> np.ndarray:
    """Convert RGB to a single photometrically consistent [0,1] signal."""
    rgb = np.asarray(rgb_frames, dtype=np.float32)
    if rgb.ndim != 4 or rgb.shape[-1] < 3:
        raise ValueError("RGB frames must have shape [T,H,W,3].")
    rgb = rgb[..., :3]
    if rgb.max(initial=0.0) > 1.0:
        rgb = rgb / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return np.clip(
        0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2],
        0.0,
        1.0,
    ).astype(np.float32)


def rgb_to_luminance_torch(rgb_frames: torch.Tensor) -> torch.Tensor:
    """GPU-friendly equivalent of :func:`rgb_to_luminance` for [T,3,H,W]."""
    if rgb_frames.ndim != 4 or rgb_frames.shape[1] < 3:
        raise ValueError("RGB tensors must have shape [T,3,H,W].")
    rgb = rgb_frames[:, :3].to(dtype=torch.float32)
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))
    return (0.2126 * linear[:, 0] + 0.7152 * linear[:, 1] + 0.0722 * linear[:, 2]).clamp(0.0, 1.0)


def _to_device(array: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if device.type == "cuda":
        return tensor.pin_memory().to(device=device, non_blocking=True)
    return tensor.to(device=device)


def _simulate_events(
    backend: EventSimulatorBackend,
    frames: torch.Tensor,
    timestamps: np.ndarray,
    config: dict[str, Any] | None,
) -> EventStream:
    """Use tensor input only for backends that explicitly support it."""
    if getattr(backend, "supports_torch", False):
        return backend.simulate(frames, timestamps, config)  # type: ignore[arg-type]
    return backend.simulate(frames.detach().cpu().numpy(), timestamps, config)


def interpolate_sequence(sequence: DSECRGBSequence, backend: VFIBackend, target_fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subdivide each real interval at or above target FPS and query VFI directly."""
    if target_fps <= 0:
        raise ValueError("target_fps must be positive.")
    if len(sequence) == 0:
        raise ValueError("Cannot interpolate an empty sequence.")
    original = [np.asarray(frame) for frame in sequence.iter_frames()]

    def normalise_rgb(frame: np.ndarray) -> np.ndarray:
        value = np.asarray(frame, dtype=np.float32)[..., :3]
        return value / 255.0 if value.max(initial=0.0) > 1.0 else value

    output_frames: list[np.ndarray] = [normalise_rgb(original[0])]
    output_times: list[float] = [float(sequence.timestamps[0])]
    for index in range(len(original) - 1):
        t0 = float(sequence.timestamps[index])
        t1 = float(sequence.timestamps[index + 1])
        delta = t1 - t0
        subdivisions = max(1, int(math.ceil(delta * target_fps)))
        fractions = np.arange(1, subdivisions, dtype=np.float64) / subdivisions
        if len(fractions):
            middle = backend.interpolate(original[index], original[index + 1], fractions)
            if len(middle) != len(fractions):
                raise ValueError("VFI backend returned a different number of samples.")
            output_frames.extend([normalise_rgb(frame) for frame in middle])
            output_times.extend((t0 + fractions * delta).tolist())
        output_frames.append(normalise_rgb(original[index + 1]))
        output_times.append(t1)
    # Sorting is intentionally not used: preserving construction order catches
    # accidental timestamp assumptions instead of masking them.
    timestamps = np.asarray(output_times, dtype=np.float64)
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("Interpolated timestamps are not strictly increasing.")
    return np.stack(output_frames, axis=0), timestamps, np.stack(original, axis=0)


def interpolate_sequence_to_count(
    sequence: DSECRGBSequence,
    backend: VFIBackend,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample one timestamped sequence at exactly ``frame_count`` outputs."""
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2.")
    if len(sequence) < 2:
        raise ValueError("At least two source frames are required.")
    original = [np.asarray(frame) for frame in sequence.iter_frames()]

    def normalise_rgb(frame: np.ndarray) -> np.ndarray:
        value = np.asarray(frame, dtype=np.float32)[..., :3]
        return value / 255.0 if value.max(initial=0.0) > 1.0 else value

    source_times = np.asarray(sequence.timestamps, dtype=np.float64)
    target_times = np.linspace(source_times[0], source_times[-1], frame_count)
    outputs: list[np.ndarray | None] = [None] * frame_count
    outputs[0] = normalise_rgb(original[0])
    outputs[-1] = normalise_rgb(original[-1])
    grouped: dict[int, list[tuple[int, float]]] = {}
    for output_index in range(1, frame_count - 1):
        timestamp = float(target_times[output_index])
        exact_source = np.flatnonzero(np.isclose(source_times, timestamp, rtol=0.0, atol=1e-12))
        if len(exact_source):
            outputs[output_index] = normalise_rgb(original[int(exact_source[0])])
            continue
        interval = int(np.searchsorted(source_times, timestamp, side="right") - 1)
        interval = min(max(interval, 0), len(original) - 2)
        fraction = (timestamp - source_times[interval]) / (source_times[interval + 1] - source_times[interval])
        grouped.setdefault(interval, []).append((output_index, float(fraction)))
    for interval, requests in grouped.items():
        fractions = np.asarray([fraction for _, fraction in requests], dtype=np.float64)
        predictions = backend.interpolate(original[interval], original[interval + 1], fractions)
        if len(predictions) != len(requests):
            raise ValueError("VFI backend returned a different number of fixed-count samples.")
        for (output_index, _), prediction in zip(requests, predictions):
            outputs[output_index] = normalise_rgb(prediction)
    if any(output is None for output in outputs):
        raise RuntimeError("Fixed-count interpolation left an output timestamp unresolved.")
    return np.stack([output for output in outputs if output is not None]), target_times, np.stack(original, axis=0)


def _normalise_rgb_torch(frame: torch.Tensor, device: torch.device) -> torch.Tensor:
    value = frame[..., :3].to(dtype=torch.float32)
    if float(value.amax().item()) > 1.0:
        value = value / 255.0
    if device.type == "cuda":
        return value.pin_memory().to(device=device, non_blocking=True)
    return value.to(device=device)


def interpolate_sequence_device(
    sequence: DSECRGBSequence,
    backend: VFIBackend,
    target_fps: float,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    """Tensor-native FPS interpolation for VFI backends exposing interpolate_torch."""
    if not hasattr(backend, "interpolate_torch"):
        raise TypeError("The selected VFI backend does not expose interpolate_torch.")
    original = [np.asarray(frame) for frame in sequence.iter_frames()]
    original_tensors = [_normalise_rgb_torch(torch.from_numpy(frame), device) for frame in original]
    output_frames: list[torch.Tensor] = [original_tensors[0]]
    output_times: list[float] = [float(sequence.timestamps[0])]
    for index in range(len(original) - 1):
        t0 = float(sequence.timestamps[index])
        t1 = float(sequence.timestamps[index + 1])
        delta = t1 - t0
        subdivisions = max(1, int(math.ceil(delta * target_fps)))
        fractions = np.arange(1, subdivisions, dtype=np.float64) / subdivisions
        if len(fractions):
            middle = backend.interpolate_torch(original_tensors[index], original_tensors[index + 1], fractions)  # type: ignore[attr-defined]
            if len(middle) != len(fractions):
                raise ValueError("VFI backend returned a different number of samples.")
            output_frames.extend(middle.unbind(0))
            output_times.extend((t0 + fractions * delta).tolist())
        output_frames.append(original_tensors[index + 1])
        output_times.append(t1)
    timestamps = np.asarray(output_times, dtype=np.float64)
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("Interpolated timestamps are not strictly increasing.")
    return torch.stack(output_frames, dim=0), timestamps, torch.stack(original_tensors, dim=0)


def interpolate_sequence_to_count_device(
    sequence: DSECRGBSequence,
    backend: VFIBackend,
    frame_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    """Tensor-native exact-count interpolation, keeping VFI output on device."""
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2.")
    if len(sequence) < 2 or not hasattr(backend, "interpolate_torch"):
        raise TypeError("Tensor-native exact-count interpolation requires at least two frames and interpolate_torch.")
    original = [np.asarray(frame) for frame in sequence.iter_frames()]
    original_tensors = [_normalise_rgb_torch(torch.from_numpy(frame), device) for frame in original]
    source_times = np.asarray(sequence.timestamps, dtype=np.float64)
    target_times = np.linspace(source_times[0], source_times[-1], frame_count)
    outputs: list[torch.Tensor | None] = [None] * frame_count
    outputs[0] = original_tensors[0]
    outputs[-1] = original_tensors[-1]
    grouped: dict[int, list[tuple[int, float]]] = {}
    for output_index in range(1, frame_count - 1):
        timestamp = float(target_times[output_index])
        exact_source = np.flatnonzero(np.isclose(source_times, timestamp, rtol=0.0, atol=1e-12))
        if len(exact_source):
            outputs[output_index] = original_tensors[int(exact_source[0])]
            continue
        interval = int(np.searchsorted(source_times, timestamp, side="right") - 1)
        interval = min(max(interval, 0), len(original) - 2)
        fraction = (timestamp - source_times[interval]) / (source_times[interval + 1] - source_times[interval])
        grouped.setdefault(interval, []).append((output_index, float(fraction)))
    for interval, requests in grouped.items():
        fractions = np.asarray([fraction for _, fraction in requests], dtype=np.float64)
        predictions = backend.interpolate_torch(original_tensors[interval], original_tensors[interval + 1], fractions)  # type: ignore[attr-defined]
        if len(predictions) != len(requests):
            raise ValueError("VFI backend returned a different number of fixed-count samples.")
        for (output_index, _), prediction in zip(requests, predictions.unbind(0)):
            outputs[output_index] = prediction
    if any(output is None for output in outputs):
        raise RuntimeError("Fixed-count tensor interpolation left an output timestamp unresolved.")
    return torch.stack([output for output in outputs if output is not None], dim=0), target_times, torch.stack(original_tensors, dim=0)


def _default_intrinsics(width: int, height: int) -> CameraIntrinsics:
    return CameraIntrinsics(width, height, float(width), float(height), (width - 1) / 2.0, (height - 1) / 2.0)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _write_rotation_csv(path: Path, policy: RotationPolicy, timestamps: np.ndarray, theta0: float) -> None:
    states = policy.states_at(timestamps, theta0=theta0, reference_time=float(timestamps[0]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "theta_rad", "theta_deg", "omega_rad_s", "rpm", "policy_state"])
        for state in states:
            writer.writerow([
                f"{state.timestamp:.17g}", f"{state.theta:.17g}", f"{np.degrees(state.theta):.17g}",
                f"{state.omega:.17g}", f"{state.omega * 60.0 / (2.0 * np.pi):.17g}", state.policy_state,
            ])


def _write_timestamps_csv(path: Path, timestamps: np.ndarray) -> None:
    """Persist a timeline without relying on a nominal FPS."""
    values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "timestamp_seconds"])
        for index, timestamp in enumerate(values):
            writer.writerow([index, f"{timestamp:.17g}"])


def _window(events: EventStream, start: float, duration: float) -> EventStream:
    return events.masked((events.t >= start) & (events.t < start + duration))


def _event_counts(events: EventStream) -> dict[str, int]:
    polarity = np.asarray(events.p)
    return {
        "count": int(len(events)),
        "positive": int(np.count_nonzero(polarity > 0)),
        "negative": int(np.count_nonzero(polarity < 0)),
    }


def _save_intensity_frames(frames: np.ndarray, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        np.save(directory / f"{index:06d}.npy", frame.astype(np.float32))


def run_simulation(
    sequence: DSECRGBSequence,
    vfi: VFIBackend,
    event_backend: EventSimulatorBackend,
    model: PrismModel,
    policy: RotationPolicy,
    output_dir: str | Path,
    *,
    options: SimulationOptions | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = options or SimulationOptions()
    requested_device = options.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    if options.warp_batch_size <= 0:
        raise ValueError("warp_batch_size must be positive.")
    if device.type == "cuda" and options.cuda_fast_math:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    random.seed(options.random_seed)
    np.random.seed(options.random_seed)
    torch.manual_seed(options.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(options.random_seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    use_tensor_vfi = device.type == "cuda" and hasattr(vfi, "interpolate_torch")
    if use_tensor_vfi:
        if options.output_frame_count is None:
            tensor_rgb, timestamps, tensor_original_rgb = interpolate_sequence_device(sequence, vfi, options.interp_fps, device)
        else:
            tensor_rgb, timestamps, tensor_original_rgb = interpolate_sequence_to_count_device(sequence, vfi, options.output_frame_count, device)
        interpolated_rgb = tensor_rgb.detach().cpu().permute(0, 2, 3, 1).numpy() if tensor_rgb.ndim == 4 and tensor_rgb.shape[1] == 3 else tensor_rgb.detach().cpu().numpy()
        original_rgb = tensor_original_rgb.detach().cpu().permute(0, 2, 3, 1).numpy() if tensor_original_rgb.ndim == 4 and tensor_original_rgb.shape[1] == 3 else tensor_original_rgb.detach().cpu().numpy()
        tensor_rgb = tensor_rgb.permute(0, 3, 1, 2).contiguous()
        tensor_original_rgb = tensor_original_rgb.permute(0, 3, 1, 2).contiguous()
    else:
        if options.output_frame_count is None:
            interpolated_rgb, timestamps, original_rgb = interpolate_sequence(sequence, vfi, options.interp_fps)
        else:
            interpolated_rgb, timestamps, original_rgb = interpolate_sequence_to_count(sequence, vfi, options.output_frame_count)
        tensor_rgb = _to_device(interpolated_rgb.astype(np.float32), device).permute(0, 3, 1, 2)
        tensor_original_rgb = _to_device(original_rgb.astype(np.float32), device).permute(0, 3, 1, 2)
    height, width = interpolated_rgb.shape[1:3]
    if options.prism_model == "paper_simplified" and sequence.calibration is None and not options.allow_default_intrinsics:
        raise ValueError(
            "PaperSimplifiedPrismModel requires DSEC calibration. Pass --calibration or explicitly "
            "use --allow-default-intrinsics for a synthetic/debug run."
        )
    tensor_intensity = rgb_to_luminance_torch(tensor_rgb).unsqueeze(1)
    tensor_original_intensity = rgb_to_luminance_torch(tensor_original_rgb).unsqueeze(1)
    warper = PrismWarper(
        model=model,
        policy=policy,
        height=height,
        width=width,
        theta0=options.theta0,
        reference_time=float(timestamps[0]),
        align_corners=options.align_corners,
        padding_mode=options.padding_mode,
        batch_size=options.warp_batch_size,
    )
    warped_intensity_tensor, per_frame_valid = warper.warp(tensor_intensity, timestamps)
    warped_rgb_tensor, _ = warper.warp(tensor_rgb, timestamps)
    # RGB tensors are no longer needed after the two warps; release them before
    # the three event passes to keep CUDA peak memory bounded.
    warped_rgb = warped_rgb_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
    del warped_rgb_tensor
    del tensor_rgb, tensor_original_rgb
    valid_mask = warper.all_orientation_valid_mask(
        samples=options.valid_orientation_samples,
        start_time=float(timestamps[0]),
        device=device,
    ).cpu().numpy()
    # Keep a true direct-original baseline. The existing vanilla stream is
    # intentionally the same VFI timeline without prism motion, so it is not
    # interchangeable with feeding the sparse source frames directly to DVS.
    original_direct = _simulate_events(event_backend, tensor_original_intensity[:, 0], sequence.timestamps, config)
    vanilla = _simulate_events(event_backend, tensor_intensity[:, 0], timestamps, config)
    raw = _simulate_events(event_backend, warped_intensity_tensor[:, 0], timestamps, config)
    compensated = compensate_events(
        raw,
        model,
        policy,
        theta0=options.theta0,
        reference_time=float(timestamps[0]),
        chunk_size=options.compensation_chunk_size,
        device=device,
    )

    # Fail early on malformed event output.  Stochastic backends are allowed to
    # differ in count and exact samples, but every canonical stream must retain
    # finite, bounded, sorted coordinates/timestamps and valid polarity values.
    event_quality = {
        "original_direct": validate_event_stream(
            original_direct, (height, width), frame_timestamps=sequence.timestamps
        ),
        "vanilla_480": validate_event_stream(
            vanilla, (height, width), frame_timestamps=timestamps
        ),
        "ami_raw_480": validate_event_stream(
            raw, (height, width), frame_timestamps=timestamps,
            require_integer_microseconds=event_backend.name == "dvs_voltmeter_cuda_fast",
        ),
        "ami_compensated_480": validate_event_stream(
            compensated, (height, width), frame_timestamps=timestamps
        ),
    }
    invalid_streams = [name for name, report in event_quality.items() if not report["valid"]]
    if invalid_streams:
        raise RuntimeError(f"Event quality gate failed for: {', '.join(invalid_streams)}")

    # Materialize CPU arrays once for serialization and CPU-side diagnostics.
    intensity = tensor_intensity[:, 0].detach().cpu().numpy()
    original_intensity = tensor_original_intensity[:, 0].detach().cpu().numpy()
    del tensor_intensity, tensor_original_intensity
    del warped_intensity_tensor

    for name in (
        "rgb_original", "rgb_interpolated", "rgb_ami_warped", "interpolated_intensity",
        "valid_mask", "prism_state", "events_original_direct", "events_vanilla", "events_ami_raw", "events_ami_compensated",
        "diagnostics", "visualizations",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)
    save_frame_sequence(original_rgb, output / "rgb_original")
    save_frame_sequence(interpolated_rgb, output / "rgb_interpolated")
    save_frame_sequence(warped_rgb, output / "rgb_ami_warped")
    _save_intensity_frames(intensity, output / "interpolated_intensity")
    iio.imwrite(output / "valid_mask" / "valid_mask.png", (valid_mask.astype(np.uint8) * 255))
    _write_timestamps_csv(output / "timestamps.csv", timestamps)
    _write_timestamps_csv(output / "source_timestamps.csv", sequence.timestamps)
    _write_rotation_csv(output / "prism_state" / "rotation.csv", policy, timestamps, options.theta0)

    common_metadata = {
        "resolution": {"width": width, "height": height},
        "time_unit": "seconds",
        "source_sequence": sequence.sequence_name,
        "camera": sequence.camera,
        "vfi": vfi.metadata(),
        "vfi_backend": vfi.name,
        "event_backend": event_backend.metadata(),
        "event_backend_name": event_backend.name,
        "prism": model.metadata(),
        "prism_model": model.name,
        "prism_angle": getattr(model, "prism_angle_deg", "not_applicable"),
        "rotation": policy.metadata(),
        "rpm": getattr(policy, "rpm", "not_applicable"),
        "reference_theta": options.theta0,
        "interp_fps_target": options.interp_fps,
        "timestamp_count": len(timestamps),
        "source_timestamp_count": len(sequence.timestamps),
        "source_timestamp_unit": sequence.timestamp_unit,
        "physical_order": "intensity_warp_then_event_formation_then_per_event_inverse_compensation",
        "random_seed": options.random_seed,
        "software_version": options.software_version,
        "compute_device": str(device),
        "warp_batch_size": options.warp_batch_size,
        "compensation_chunk_size": options.compensation_chunk_size,
        "cuda_fast_math": options.cuda_fast_math,
        "software_commit": "unavailable-no-git-root",
    }
    write_events_h5(output / "events_vanilla" / "events.h5", vanilla, {**common_metadata, "stage": "vanilla"})
    write_events_h5(output / "events_ami_raw" / "events.h5", raw, {**common_metadata, "stage": "raw_ami"})
    write_events_h5(output / "events_ami_compensated" / "events.h5", compensated, {**common_metadata, "stage": "compensated_ami"})
    write_events_h5(
        output / "events_original_direct" / "events.h5",
        original_direct,
        {
            **common_metadata,
            "stage": "original_direct",
            "direct_original_frames": True,
            "timestamp_count": len(sequence.timestamps),
        },
    )

    window_duration = options.event_window_ms / 1000.0
    start = float(timestamps[0])
    vanilla_window = _window(vanilla, start, window_duration)
    raw_window = _window(raw, start, window_duration)
    compensated_window = _window(compensated, start, window_duration)
    original_direct_window = _window(original_direct, start, window_duration)
    prism_flow = None
    if len(timestamps) > 1:
        points_y, points_x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing="ij",
        )
        points = torch.stack((points_x, points_y), dim=-1)
        theta_values = warper.theta_values(timestamps[:2])
        forward0 = model.forward_points(points, float(theta_values[0]))
        forward1 = model.forward_points(points, float(theta_values[1]))
        prism_flow = ((forward1 - forward0) / max(float(timestamps[1] - timestamps[0]), 1e-12)).detach().cpu().numpy()

    requested_windows = sorted({10.0, 20.0, 33.0, 50.0, float(options.event_window_ms)})
    window_metrics: dict[str, dict[str, Any]] = {}
    for window_ms in requested_windows:
        duration = window_ms / 1000.0
        vanilla_slice = _window(vanilla, start, duration)
        raw_slice = _window(raw, start, duration)
        compensated_slice = _window(compensated, start, duration)
        window_metrics[str(window_ms).rstrip("0").rstrip(".")] = {
            "window_ms": window_ms,
            "vanilla": edge_coverage(intensity[0], vanilla_slice, valid_mask),
            "raw_ami": edge_coverage(intensity[0], raw_slice, valid_mask),
            "compensated_ami": edge_coverage(intensity[0], compensated_slice, valid_mask, compensated=True),
            "newly_recovered": newly_recovered_structure(intensity[0], vanilla_slice, compensated_slice, valid_mask, ami_compensated=True),
            "raw_spatial_variance": event_spatial_variance(raw_slice),
            "compensated_spatial_variance": event_spatial_variance(compensated_slice, compensated=True),
            "raw_edge_distance": edge_distance(intensity[0], raw_slice, valid_mask=valid_mask),
            "compensated_edge_distance": edge_distance(intensity[0], compensated_slice, compensated=True, valid_mask=valid_mask),
            "orientation_coverage_rad": float(abs(policy.omega(start)) * duration),
            "orientation_coverage_below_quarter_turn": bool(abs(policy.omega(start)) * duration < np.pi / 2.0),
        }
        if prism_flow is not None and len(intensity) > 1:
            window_metrics[str(window_ms).rstrip("0").rstrip(".")]["event_observability"] = event_observability_stats(
                intensity[0], raw_slice, prism_flow, valid_mask=valid_mask
            )
    diagnostics: dict[str, Any] = {
        "event_counts": {
            "original_direct": _event_counts(original_direct),
            "vanilla_480": _event_counts(vanilla),
            "ami_raw_480": _event_counts(raw),
            "ami_compensated_480": _event_counts(compensated),
        },
        "event_quality": event_quality,
        "vfi_flicker": temporal_flicker_stats(intensity, valid_mask),
        "vanilla_coverage": edge_coverage(intensity[0], vanilla_window, valid_mask),
        "original_direct_coverage": edge_coverage(original_intensity[0], original_direct_window, valid_mask),
        "raw_ami_coverage": edge_coverage(intensity[0], raw_window, valid_mask),
        "compensated_ami_coverage": edge_coverage(intensity[0], compensated_window, valid_mask, compensated=True),
        "newly_recovered": newly_recovered_structure(intensity[0], vanilla_window, compensated_window, valid_mask, ami_compensated=True),
        "raw_spatial_variance": event_spatial_variance(raw_window),
        "compensated_spatial_variance": event_spatial_variance(compensated_window, compensated=True),
        "raw_edge_distance": edge_distance(intensity[0], raw_window, valid_mask=valid_mask),
        "compensated_edge_distance": edge_distance(intensity[0], compensated_window, compensated=True, valid_mask=valid_mask),
        "orientation_coverage_rad": float(abs(policy.omega(start)) * window_duration),
        "orientation_coverage_below_quarter_turn": bool(abs(policy.omega(start)) * window_duration < np.pi / 2.0),
        "per_frame_valid_fraction": float(per_frame_valid.float().mean().item()),
        "window_metrics": window_metrics,
        "event_window_ms": options.event_window_ms,
    }
    if prism_flow is not None and len(intensity) > 1:
        diagnostics["relative_motion"] = relative_motion_diagnostics(intensity[0], intensity[1], prism_flow)
        diagnostics["event_observability"] = event_observability_stats(intensity[0], raw_window, prism_flow, valid_mask=valid_mask)
    _write_json(output / "diagnostics" / "diagnostics.json", diagnostics)
    _write_json(output / "diagnostics" / "temporal_flicker_stats.json", diagnostics["vfi_flicker"])
    save_temporal_difference_video(intensity, output / "diagnostics" / "temporal_difference_video.mp4", fps=options.interp_fps)
    save_event_image(vanilla_window, (height, width), output / "visualizations" / "vanilla_events.png")
    save_event_image(raw_window, (height, width), output / "visualizations" / "raw_ami_events.png")
    save_event_image(compensated_window, (height, width), output / "visualizations" / "compensated_ami_events.png", compensated=True)
    save_event_image(original_direct_window, (height, width), output / "visualizations" / "original_direct_events.png")
    save_panel(intensity[0], vanilla_window, raw_window, compensated_window, output / "visualizations" / "aligned_panel.png")
    save_event_comparison_panel(
        original_rgb[0], original_direct_window, vanilla_window, raw_window, compensated_window,
        output / "visualizations" / "direct_vs_480fps_panel.png",
    )
    if options.save_debug_video:
        save_frame_sequence(warped_rgb, output / "rgb_ami_warped", fps=options.interp_fps, video_path=output / "visualizations" / "ami_forward_warp.mp4")
    metadata = {**common_metadata, "options": asdict(options), "diagnostics": diagnostics, "output_dir": str(output)}
    _write_json(output / "metadata.json", metadata)
    with (output / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_jsonable({**asdict(options), **(config or {})}), handle, sort_keys=True)
    return metadata
