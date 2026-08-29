"""Experimental custom-CUDA DVS-Voltmeter backend.

The official PyTorch implementation remains the reference backend. This module
compiles a small CUDA extension lazily and moves the per-pixel stochastic event
recursion into one kernel. It is deliberately opt-in because its stateless RNG
stream is not bitwise identical to PyTorch's RNG stream.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import torch

from ami_sim.types import EventStream
from .dvs_voltmeter import DVSVoltmeterTorchBackend, _SENSOR_PARAMETERS


_EXTENSION = None
_EXTENSION_LOCK = threading.Lock()


def _load_extension():
    """Build/load the extension only when the fast backend is explicitly used."""
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if not torch.cuda.is_available():
        raise RuntimeError("dvs_voltmeter_cuda_fast requires a CUDA-enabled PyTorch runtime")
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        from torch.utils.cpp_extension import load

        source_dir = Path(__file__).with_name("cuda")
        verbose = os.environ.get("EIRM_CUDA_EXT_VERBOSE", "0") == "1"
        _EXTENSION = load(
            name="eirm_dvs_cuda_fast",
            sources=[str(source_dir / "dvs_fast.cpp"), str(source_dir / "dvs_fast_kernel.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
            with_cuda=True,
            verbose=verbose,
        )
    return _EXTENSION


class DVSVoltmeterCudaFastBackend(DVSVoltmeterTorchBackend):
    """Custom CUDA event generator with the same high-level sensor model."""

    name = "dvs_voltmeter_cuda_fast"
    supports_torch = True

    def __init__(
        self,
        dvs_root: str | Path,
        *,
        camera_type: str = "DVS346",
        device: str | torch.device | None = None,
        transfer_chunk_frames: int = 16,
        seed: int = 0,
        precision: str = "float64",
        max_events_per_pixel: int = 32,
    ) -> None:
        super().__init__(
            dvs_root,
            camera_type=camera_type,
            device=device,
            transfer_chunk_frames=transfer_chunk_frames,
            seed=seed,
            precision=precision,
        )
        if self.device.type != "cuda":
            raise ValueError("DVSVoltmeterCudaFastBackend requires a CUDA device")
        if max_events_per_pixel <= 0:
            raise ValueError("max_events_per_pixel must be positive")
        self.max_events_per_pixel = int(max_events_per_pixel)

    @staticmethod
    def _flush(pending: list[torch.Tensor], cpu_chunks: list[np.ndarray]) -> None:
        if not pending:
            return
        columns = DVSVoltmeterTorchBackend._events_to_cpu(pending)
        if columns:
            cpu_chunks.append(np.stack(columns, axis=1))
        pending.clear()

    def simulate(
        self,
        frames: np.ndarray | torch.Tensor,
        timestamps: np.ndarray,
        config: dict[str, object] | None = None,
    ) -> EventStream:
        del config
        extension = _load_extension()
        if isinstance(frames, torch.Tensor):
            tensor_frames = frames
        else:
            tensor_frames = torch.from_numpy(np.asarray(frames, dtype=np.float32))
        if tensor_frames.ndim != 3:
            raise ValueError("DVS-Voltmeter expects [T,H,W] intensity frames")
        values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if len(tensor_frames) != len(values):
            raise ValueError("Frame count does not match timestamp count")
        if len(values) < 2 or not np.all(np.diff(values) > 0):
            raise ValueError("DVS-Voltmeter timestamps must be strictly increasing")

        dtype = torch.float32 if self.precision == "float32" else torch.float64
        tensor_frames = tensor_frames.to(device=self.device, dtype=torch.float32, non_blocking=True)
        height, width = int(tensor_frames.shape[1]), int(tensor_frames.shape[2])
        k1, k2, k3, k4, k5, k6 = _SENSOR_PARAMETERS[self.camera_type]
        base_frame: torch.Tensor | None = None
        residual: torch.Tensor | None = None
        previous_t = float(values[0] * 1e6)
        pending: list[torch.Tensor] = []
        cpu_chunks: list[np.ndarray] = []

        with torch.inference_mode():
            for frame_index, (frame, timestamp) in enumerate(zip(tensor_frames, values)):
                light = torch.round(torch.clamp(frame, 0.0, 1.0) * 255.0).to(dtype)
                timestamp_us = float(np.rint(timestamp * 1e6))
                if base_frame is None:
                    base_frame = light
                    residual = torch.zeros_like(light)
                    previous_t = timestamp_us
                    continue

                assert residual is not None
                delta_light = light - base_frame
                avg_light = (light + base_frame) / 2.0
                denominator = 1.0 / (avg_light + k2)
                mu = (k1 * delta_light / (timestamp_us - previous_t)) * denominator
                mu = mu + k4 + k5 * avg_light
                sigma = (k3 * torch.sqrt(avg_light)) * denominator + k6
                event_tensor, count_tensor, overflow_tensor, residual_next = extension.generate_events(
                    mu.contiguous(),
                    sigma.contiguous(),
                    residual.contiguous(),
                    previous_t,
                    timestamp_us,
                    width,
                    height,
                    self.seed,
                    frame_index,
                    self.max_events_per_pixel,
                )
                if bool(overflow_tensor.item()):
                    raise RuntimeError(
                        "dvs_voltmeter_cuda_fast exceeded max_events_per_pixel="
                        f"{self.max_events_per_pixel}; increase the backend limit"
                    )
                count = int(count_tensor.item())
                if count:
                    pending.append(event_tensor[:count])
                if (frame_index + 1) % self.transfer_chunk_frames == 0:
                    self._flush(pending, cpu_chunks)
                residual = residual_next
                base_frame = light
                previous_t = timestamp_us
        self._flush(pending, cpu_chunks)

        if not cpu_chunks:
            return EventStream.empty()
        values_us = np.concatenate(cpu_chunks, axis=0)
        order = np.argsort(values_us[:, 0], kind="stable")
        values_us = values_us[order]
        return EventStream(
            x=values_us[:, 1].astype(np.int32, copy=False),
            y=values_us[:, 2].astype(np.int32, copy=False),
            t=values_us[:, 0].astype(np.float64) * 1e-6,
            p=np.where(values_us[:, 3] > 0, 1, -1).astype(np.int8),
        )

    def metadata(self) -> dict[str, object]:
        return {
            **super().metadata(),
            "event_backend": self.name,
            "cuda_extension": "eirm_dvs_cuda_fast",
            "max_events_per_pixel": self.max_events_per_pixel,
            "event_storage": "int32 [timestamp_us, x, y, polarity] on CUDA; seconds only at API boundary",
            "rng": "stateless splitmix64 per pixel/frame (not bitwise reference-compatible)",
        }
