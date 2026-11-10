"""Deterministic threshold event formation and portable HDF5 output."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .types import EventStream

_COMPILED_CUDA_EXTENSION = None
_COMPILED_CUDA_EXTENSION_LOCK = threading.Lock()


def _load_compiled_cuda_extension():
    """Compile the hand-written CUDA kernel only when its backend is selected."""
    global _COMPILED_CUDA_EXTENSION
    if _COMPILED_CUDA_EXTENSION is not None:
        return _COMPILED_CUDA_EXTENSION
    with _COMPILED_CUDA_EXTENSION_LOCK:
        if _COMPILED_CUDA_EXTENSION is not None:
            return _COMPILED_CUDA_EXTENSION
        import torch
        from torch.utils.cpp_extension import load

        if not torch.cuda.is_available():
            raise RuntimeError("CompiledCudaThresholdEventSimulator requires CUDA.")
        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        source_dir = Path(__file__).with_name("cuda")
        _COMPILED_CUDA_EXTENSION = load(
            name="ami_prism_threshold_cuda",
            sources=[str(source_dir / "threshold_events.cpp"), str(source_dir / "threshold_events_kernel.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            with_cuda=True,
            verbose=os.environ.get("AMI_SIM_CUDA_EXT_VERBOSE", "0") == "1",
        )
        return _COMPILED_CUDA_EXTENSION


class ThresholdEventSimulator:
    """Log-intensity contrast threshold simulator without sensor-noise claims."""

    name = "deterministic_log_threshold"

    def __init__(self, contrast_threshold: float = 0.2, epsilon: float = 1e-6) -> None:
        if contrast_threshold <= 0:
            raise ValueError("contrast_threshold must be positive.")
        self.contrast_threshold = float(contrast_threshold)
        self.epsilon = float(epsilon)

    def simulate(self, intensity: np.ndarray, timestamps: np.ndarray) -> EventStream:
        events, _ = self.simulate_with_state(intensity, timestamps)
        return events

    def simulate_with_state(
        self,
        intensity: np.ndarray,
        timestamps: np.ndarray,
        *,
        reference_log: np.ndarray | None = None,
    ) -> tuple[EventStream, np.ndarray]:
        """Simulate a window and return its final per-pixel threshold state.

        ``reference_log`` is the sensor reference carried from the preceding
        window. Passing a copy of the same state to several calls enables fair
        counterfactual action branching without resetting threshold residuals.
        """
        frames = np.asarray(intensity, dtype=np.float64)
        times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if frames.ndim != 3 or len(frames) != len(times):
            raise ValueError("Expected intensity [T,H,W] and one timestamp per frame.")
        if len(times) < 2 or not np.all(np.diff(times) > 0.0):
            raise ValueError("Event timestamps must be strictly increasing and contain two frames.")
        if (times[-1] - times[0]) * 1e6 > np.iinfo(np.int32).max:
            raise ValueError("Compiled CUDA event timestamps require a sequence shorter than int32 microseconds.")
        log_frames = np.log(np.clip(frames, self.epsilon, 1.0))
        if reference_log is None:
            reference = log_frames[0].copy()
        else:
            reference = np.asarray(reference_log, dtype=np.float64).copy()
            if reference.shape != frames.shape[1:] or not np.all(np.isfinite(reference)):
                raise ValueError("reference_log must be finite and match the sensor image shape.")
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        event_times: list[np.ndarray] = []
        polarities: list[np.ndarray] = []
        for index in range(1, len(times)):
            before, after = log_frames[index - 1], log_frames[index]
            delta = after - reference
            for sign, counts in (
                (1, np.floor(np.maximum(delta, 0.0) / self.contrast_threshold + self.epsilon).astype(np.int32)),
                (-1, np.floor(np.maximum(-delta, 0.0) / self.contrast_threshold + self.epsilon).astype(np.int32)),
            ):
                v, u = np.nonzero(counts > 0)
                if not len(u):
                    continue
                repeated = np.repeat(np.arange(len(u)), counts[v, u])
                event_u, event_v = u[repeated], v[repeated]
                active_counts = counts[v, u].astype(np.int64)
                offsets = np.cumsum(active_counts) - active_counts
                ranks = np.arange(len(repeated), dtype=np.int64) - offsets[repeated] + 1
                levels = reference[event_v, event_u] + sign * self.contrast_threshold * ranks
                slope = after[event_v, event_u] - before[event_v, event_u]
                alpha = np.divide(levels - before[event_v, event_u], slope, out=np.full_like(levels, 0.5), where=np.abs(slope) > self.epsilon)
                xs.append(event_u.astype(np.int32))
                ys.append(event_v.astype(np.int32))
                event_times.append(times[index - 1] + np.clip(alpha, 0.0, 1.0) * (times[index] - times[index - 1]))
                polarities.append(np.full(len(event_u), sign, dtype=np.int8))
            reference += (np.floor(np.maximum(delta, 0.0) / self.contrast_threshold + self.epsilon).astype(np.int32) - np.floor(np.maximum(-delta, 0.0) / self.contrast_threshold + self.epsilon).astype(np.int32)) * self.contrast_threshold
        if not xs:
            return EventStream.empty(), reference
        events = EventStream(np.concatenate(xs), np.concatenate(ys), np.concatenate(event_times), np.concatenate(polarities)).sorted_by_time()
        return events, reference

    def metadata(self) -> dict[str, object]:
        return {"backend": self.name, "contrast_threshold": self.contrast_threshold, "sensor_calibrated": False}


class CudaThresholdEventSimulator:
    """CUDA tensor implementation of the deterministic log-threshold rule.

    This backend preserves the same reference-level update as
    :class:`ThresholdEventSimulator`, but uses float32 device arithmetic. It is
    a PyTorch CUDA implementation, not a custom compiled CUDA kernel.
    """

    name = "cuda_log_threshold"

    def __init__(self, contrast_threshold: float = 0.2, epsilon: float = 1e-6) -> None:
        if contrast_threshold <= 0:
            raise ValueError("contrast_threshold must be positive.")
        self.contrast_threshold = float(contrast_threshold)
        self.epsilon = float(epsilon)

    def _events_for_sign(
        self,
        counts: "torch.Tensor",
        sign: int,
        reference: "torch.Tensor",
        before: "torch.Tensor",
        after: "torch.Tensor",
        start_time: float,
        end_time: float,
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"] | None:
        import torch

        flat_counts = counts.reshape(-1)
        active = torch.nonzero(flat_counts > 0, as_tuple=False).flatten()
        if not len(active):
            return None
        active_counts = flat_counts[active]
        pixel_index = torch.repeat_interleave(active, active_counts)
        offsets = torch.cumsum(active_counts, dim=0) - active_counts
        event_offsets = torch.repeat_interleave(offsets, active_counts)
        ranks = torch.arange(len(pixel_index), device=counts.device, dtype=reference.dtype) - event_offsets.to(reference.dtype) + 1.0
        reference_flat = reference.reshape(-1)
        before_flat = before.reshape(-1)
        after_flat = after.reshape(-1)
        levels = reference_flat[pixel_index] + float(sign) * self.contrast_threshold * ranks
        slope = after_flat[pixel_index] - before_flat[pixel_index]
        alpha = torch.where(torch.abs(slope) > self.epsilon, (levels - before_flat[pixel_index]) / slope, torch.full_like(slope, 0.5)).clamp(0.0, 1.0)
        width = counts.shape[1]
        u = (pixel_index % width).to(torch.int32)
        v = torch.div(pixel_index, width, rounding_mode="floor").to(torch.int32)
        times = torch.as_tensor(start_time, device=counts.device, dtype=torch.float64) + alpha.to(torch.float64) * (end_time - start_time)
        polarity = torch.full((len(pixel_index),), sign, device=counts.device, dtype=torch.int8)
        return u, v, times, polarity

    def simulate(self, intensity: "torch.Tensor", timestamps: np.ndarray) -> EventStream:
        events, _ = self.simulate_with_state(intensity, timestamps)
        return events

    def simulate_with_state(
        self,
        intensity: "torch.Tensor",
        timestamps: np.ndarray,
        *,
        reference_log: np.ndarray | None = None,
    ) -> tuple[EventStream, np.ndarray]:
        import torch

        if not intensity.is_cuda:
            raise ValueError("CudaThresholdEventSimulator requires a CUDA tensor.")
        frames = intensity.to(dtype=torch.float32)
        times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if frames.ndim != 3 or len(frames) != len(times):
            raise ValueError("Expected CUDA intensity [T,H,W] and one timestamp per frame.")
        if len(times) < 2 or not np.all(np.diff(times) > 0.0):
            raise ValueError("Event timestamps must be strictly increasing and contain two frames.")
        log_frames = torch.log(frames.clamp(self.epsilon, 1.0))
        if reference_log is None:
            reference = log_frames[0].clone()
        else:
            reference = torch.as_tensor(reference_log, device=frames.device, dtype=log_frames.dtype)
            if reference.shape != frames.shape[1:] or not torch.isfinite(reference).all():
                raise ValueError("reference_log must be finite and match the sensor image shape.")
        chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for index in range(1, len(times)):
            before, after = log_frames[index - 1], log_frames[index]
            delta = after - reference
            positive = torch.floor(torch.clamp_min(delta, 0.0) / self.contrast_threshold + self.epsilon).to(torch.int64)
            negative = torch.floor(torch.clamp_min(-delta, 0.0) / self.contrast_threshold + self.epsilon).to(torch.int64)
            for sign, counts in ((1, positive), (-1, negative)):
                values = self._events_for_sign(counts, sign, reference, before, after, float(times[index - 1]), float(times[index]))
                if values is not None:
                    chunks.append(values)
            reference = reference + (positive - negative).to(reference.dtype) * self.contrast_threshold
        reference_cpu = reference.detach().cpu().numpy()
        if not chunks:
            return EventStream.empty(), reference_cpu
        u = torch.cat([value[0] for value in chunks])
        v = torch.cat([value[1] for value in chunks])
        event_times = torch.cat([value[2] for value in chunks])
        polarity = torch.cat([value[3] for value in chunks])
        order = torch.argsort(event_times, stable=True)
        return (
            EventStream(
                u[order].cpu().numpy(),
                v[order].cpu().numpy(),
                event_times[order].cpu().numpy(),
                polarity[order].cpu().numpy(),
            ),
            reference_cpu,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "contrast_threshold": self.contrast_threshold,
            "arithmetic": "torch_cuda_float32",
            "sensor_calibrated": False,
        }


class CompiledCudaThresholdEventSimulator:
    """Hand-written CUDA threshold-event backend with a per-pixel recurrence.

    The first kernel assigns one thread per pixel and advances that pixel's
    reference level through the complete sequence. A device prefix scan then
    allocates an exact event buffer; a second CUDA kernel writes the crossings
    without atomics. Python only transfers the final sorted canonical stream.
    """

    name = "compiled_cuda_log_threshold"

    def __init__(self, contrast_threshold: float = 0.2, max_events_per_pixel: int = 128, epsilon: float = 1e-6) -> None:
        if contrast_threshold <= 0:
            raise ValueError("contrast_threshold must be positive.")
        if max_events_per_pixel <= 0:
            raise ValueError("max_events_per_pixel must be positive.")
        self.contrast_threshold = float(contrast_threshold)
        self.max_events_per_pixel = int(max_events_per_pixel)
        self.epsilon = float(epsilon)

    def simulate(self, intensity: "torch.Tensor", timestamps: np.ndarray) -> EventStream:
        import torch

        if not intensity.is_cuda:
            raise ValueError("CompiledCudaThresholdEventSimulator requires a CUDA tensor.")
        frames = intensity.to(dtype=torch.float32).contiguous()
        times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if frames.ndim != 3 or len(frames) != len(times):
            raise ValueError("Expected CUDA intensity [T,H,W] and one timestamp per frame.")
        if len(times) < 2 or not np.all(np.diff(times) > 0.0):
            raise ValueError("Event timestamps must be strictly increasing and contain two frames.")
        extension = _load_compiled_cuda_extension()
        log_frames = torch.log(frames.clamp(self.epsilon, 1.0)).contiguous()
        timestamp_tensor = torch.as_tensor(times, device=frames.device, dtype=torch.float64).contiguous()
        events, overflow = extension.generate_threshold_event_sequence(
            log_frames, timestamp_tensor, self.contrast_threshold, self.max_events_per_pixel, float(times[0])
        )
        if bool(overflow.item()):
            raise RuntimeError(
                "compiled CUDA threshold backend exceeded max_events_per_pixel="
                f"{self.max_events_per_pixel}; increase the configured limit."
            )
        if not len(events):
            return EventStream.empty()
        order = torch.argsort(events[:, 0], stable=True)
        events = events[order].cpu().numpy()
        event_times = events[:, 0].astype(np.float64) * 1e-6 + times[0]
        return EventStream(events[:, 1], events[:, 2], event_times, events[:, 3])

    def metadata(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "contrast_threshold": self.contrast_threshold,
            "max_events_per_pixel": self.max_events_per_pixel,
            "implementation": "handwritten_cuda_extension_two_pass_per_pixel_threshold_recurrence",
            "event_storage": "int32 [relative_timestamp_us, x, y, polarity] on CUDA; seconds only at API boundary",
            "sensor_calibrated": False,
        }


def write_events_h5(path: str | Path, events: EventStream, metadata: dict[str, Any] | None = None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as handle:
        group = handle.create_group("events")
        for name, value in {"x": events.x, "y": events.y, "t": events.t, "p": events.p, **events.extras}.items():
            group.create_dataset(name, data=value, compression="gzip")
        for key, value in {**events.metadata(), **(metadata or {})}.items():
            group.attrs[key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value


def read_events_h5(path: str | Path) -> EventStream:
    with h5py.File(path, "r") as handle:
        group = handle["events"] if "events" in handle else handle
        required = {"x", "y", "t", "p"}
        if not required.issubset(group.keys()):
            raise ValueError("HDF5 event file must contain x, y, t, and p datasets.")
        order = np.argsort(np.asarray(group["t"]), kind="stable")
        extras = {key: np.asarray(group[key])[order] for key in group.keys() if key not in required}
        return EventStream(np.asarray(group["x"])[order], np.asarray(group["y"])[order], np.asarray(group["t"])[order], np.asarray(group["p"])[order], extras)
