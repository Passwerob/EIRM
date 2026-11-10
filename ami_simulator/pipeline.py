"""Self-contained RGB-to-AMI-event simulation pipeline."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import torch

from .events import CompiledCudaThresholdEventSimulator, CudaThresholdEventSimulator, ThresholdEventSimulator, write_events_h5
from .geometry import RotatingWedgePrism
from .rotation import ConstantRotation
from .types import CameraIntrinsics, EventStream
from .warp import PrismWarper, compensate_events
from .vfi import resample_with_vfi

_TIME_SCALES = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class SimulationConfig:
    prism_angle_deg: float = 0.5
    rpm: float = 720.0
    theta0: float = 0.0
    contrast_threshold: float = 0.2
    event_backend: str = "threshold_cpu"
    max_events_per_pixel: int = 128
    interpolation_fps: float | None = None
    padding_mode: str = "border"
    device: str = "cpu"
    compensation_chunk_size: int = 250_000


def rgb_to_linear_luminance(frames: np.ndarray) -> np.ndarray:
    """Convert RGB sRGB frames to one consistent linear-light intensity signal."""
    rgb = np.asarray(frames, dtype=np.float32)
    if rgb.ndim != 4 or rgb.shape[-1] < 3:
        raise ValueError("Expected RGB frames shaped [T,H,W,3].")
    rgb = rgb[..., :3]
    if rgb.max(initial=0.0) > 1.0:
        rgb = rgb / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return np.clip(0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2], 0.0, 1.0).astype(np.float32)


def temporal_resample_linear(frames: np.ndarray, timestamps: np.ndarray, target_fps: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Resample actual intervals without assuming a nominal source frame rate."""
    values = np.asarray(frames, dtype=np.float32)
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if len(values) != len(times) or len(times) < 2 or not np.all(np.diff(times) > 0.0):
        raise ValueError("Frames need two or more strictly increasing timestamps.")
    if target_fps is None:
        return values, times
    if target_fps <= 0:
        raise ValueError("interpolation_fps must be positive.")
    output_frames = [values[0]]
    output_times = [float(times[0])]
    for index, (start, end) in enumerate(zip(times[:-1], times[1:])):
        subdivisions = max(1, math.ceil((end - start) * target_fps))
        for step in range(1, subdivisions + 1):
            alpha = step / subdivisions
            output_frames.append(values[index] * (1.0 - alpha) + values[index + 1] * alpha)
            output_times.append(float(start + alpha * (end - start)))
    result_times = np.asarray(output_times, dtype=np.float64)
    if not np.all(np.diff(result_times) > 0.0):
        raise RuntimeError("Temporal resampling produced a non-monotonic timeline.")
    return np.stack(output_frames), result_times


def run_simulation(
    rgb_frames: np.ndarray,
    timestamps: np.ndarray,
    intrinsics: CameraIntrinsics,
    output_dir: str | Path,
    *,
    config: SimulationConfig | None = None,
    rotation_policy: object | None = None,
    vfi_backend: object | None = None,
    source_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run the physically ordered AMI simulation and write portable artifacts."""
    options = config or SimulationConfig()
    if options.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if options.interpolation_fps is None:
        rgb = np.asarray(rgb_frames, dtype=np.float32)
        source_times = np.asarray(timestamps, dtype=np.float64)
    elif vfi_backend is None:
        raise ValueError("interpolation_fps requires an explicit VFI backend; use Practical-RIFE for research runs.")
    else:
        rgb, source_times = resample_with_vfi(rgb_frames, timestamps, target_fps=options.interpolation_fps, backend=vfi_backend)
    height, width = rgb.shape[1:3]
    if (intrinsics.width, intrinsics.height) != (width, height):
        raise ValueError("Intrinsics must describe the exact input image resolution; call intrinsics.resized() explicitly when needed.")
    device = torch.device(options.device)
    luminance = rgb_to_linear_luminance(rgb)
    model = RotatingWedgePrism(intrinsics, options.prism_angle_deg)
    rotation = rotation_policy or ConstantRotation(options.rpm)
    warper = PrismWarper(model, rotation, height=height, width=width, theta0=options.theta0, reference_time=float(source_times[0]), padding_mode=options.padding_mode)
    luminance_tensor = torch.from_numpy(luminance).to(device=device).unsqueeze(1)
    with torch.inference_mode():
        sensor_tensor, sensor_valid = warper.warp(luminance_tensor, source_times)
    if options.event_backend == "threshold_cpu":
        simulator = ThresholdEventSimulator(options.contrast_threshold)
        raw_events = simulator.simulate(sensor_tensor[:, 0].cpu().numpy(), source_times)
    elif options.event_backend == "threshold_cuda":
        simulator = CudaThresholdEventSimulator(options.contrast_threshold)
        raw_events = simulator.simulate(sensor_tensor[:, 0], source_times)
    elif options.event_backend == "threshold_cuda_compiled":
        simulator = CompiledCudaThresholdEventSimulator(options.contrast_threshold, options.max_events_per_pixel)
        raw_events = simulator.simulate(sensor_tensor[:, 0], source_times)
    else:
        raise ValueError("event_backend must be threshold_cpu, threshold_cuda, or threshold_cuda_compiled.")
    compensated_events = compensate_events(raw_events, model, rotation, theta0=options.theta0, reference_time=float(source_times[0]), device=str(device), chunk_size=options.compensation_chunk_size)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "physical_order": "RGB -> linear_luminance -> rotating_prism_warp -> event_formation -> per_event_inverse_compensation",
        "resolution": {"width": width, "height": height},
        "source_frame_count": int(len(timestamps)),
        "simulated_frame_count": int(len(source_times)),
        "model": model.metadata(),
        "rotation": rotation.metadata(),
        "event_backend": simulator.metadata(),
        "vfi": {"vfi_backend": "none"} if vfi_backend is None else vfi_backend.metadata(),
        "config": asdict(options),
        **(source_metadata or {}),
    }
    write_events_h5(output / "events_raw.h5", raw_events, {**metadata, "stage": "sensor_plane"})
    write_events_h5(output / "events_compensated.h5", compensated_events, {**metadata, "stage": "reference_plane", "coordinate_fields": {"x_reference": "float pixel u", "y_reference": "float pixel v"}})
    np.save(output / "sensor_valid_mask.npy", sensor_valid.all(dim=0).cpu().numpy())
    with (output / "timestamps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "timestamp_seconds"])
        writer.writerows((index, f"{timestamp:.17g}") for index, timestamp in enumerate(source_times))
    summary = {"raw_event_count": len(raw_events), "compensated_event_count": len(compensated_events), "output_dir": str(output), **metadata}
    (output / "run.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _read_timestamp_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    rows: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = []
        for field in re.split(r"[,\s]+", line.strip()):
            if field:
                try:
                    values.append(float(field))
                except ValueError:
                    pass
        if values:
            rows.append(values[-1])
    if not rows:
        raise ValueError(f"No numeric timestamps found in {path}.")
    return np.asarray(rows, dtype=np.float64)


def load_rgb_sequence(
    rgb_dir: str | Path,
    timestamps_path: str | Path,
    *,
    timestamp_unit: str,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load same-sized RGB images with explicit timestamp-unit conversion."""
    if timestamp_unit not in _TIME_SCALES:
        raise ValueError(f"Unsupported timestamp unit {timestamp_unit!r}.")
    directory = Path(rgb_dir)
    image_paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES)
    if max_frames is not None:
        if max_frames < 2:
            raise ValueError("max_frames must be at least two when supplied.")
        image_paths = image_paths[:max_frames]
    if not image_paths:
        raise FileNotFoundError(f"No RGB images found in {directory}.")
    frames = [np.asarray(imageio.imread(path))[..., :3] for path in image_paths]
    if any(frame.ndim != 3 or frame.shape[-1] != 3 for frame in frames):
        raise ValueError("All source images must be RGB or RGBA.")
    if len({frame.shape for frame in frames}) != 1:
        raise ValueError("All source images must have the same dimensions.")
    timestamps = _read_timestamp_file(Path(timestamps_path))
    if len(timestamps) < len(frames):
        raise ValueError("Timestamp count must cover every selected RGB image.")
    timestamps = timestamps[:len(frames)] * _TIME_SCALES[timestamp_unit]
    if len(timestamps) < 2 or not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("Timestamps must contain two or more strictly increasing values.")
    return np.stack(frames), timestamps
