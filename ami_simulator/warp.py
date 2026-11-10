"""Image formation and event-plane compensation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from .geometry import RotatingWedgePrism
from .rotation import ConstantRotation
from .types import EventStream


class PrismWarper:
    """Inverse-sampled RGB/luminance warp with fixed pixel-center semantics."""

    def __init__(
        self,
        model: RotatingWedgePrism,
        rotation: object,
        *,
        height: int,
        width: int,
        theta0: float = 0.0,
        reference_time: float,
        padding_mode: str = "border",
    ) -> None:
        if padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError("padding_mode must be zeros, border, or reflection.")
        self.model = model
        self.rotation = rotation
        self.height = int(height)
        self.width = int(width)
        self.theta0 = float(theta0)
        self.reference_time = float(reference_time)
        self.padding_mode = padding_mode
        v, u = torch.meshgrid(torch.arange(height, dtype=torch.float32), torch.arange(width, dtype=torch.float32), indexing="ij")
        self._sensor_uv = torch.stack((u, v), dim=-1)

    def _grid(self, points_uv: torch.Tensor) -> torch.Tensor:
        # align_corners=False requires pixel-center normalization.
        u = 2.0 * (points_uv[..., 0] + 0.5) / self.width - 1.0
        v = 2.0 * (points_uv[..., 1] + 0.5) / self.height - 1.0
        return torch.stack((u, v), dim=-1)

    def warp(self, frames: torch.Tensor, timestamps: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if frames.ndim != 4 or tuple(frames.shape[-2:]) != (self.height, self.width):
            raise ValueError("Expected frames with shape [T,C,H,W] at the configured resolution.")
        times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if len(times) != len(frames):
            raise ValueError("Every frame needs exactly one timestamp.")
        theta = self.rotation.theta_values(times, theta0=self.theta0, reference_time=self.reference_time)
        sensor_uv = self._sensor_uv.to(device=frames.device, dtype=frames.dtype)
        source_uv = self.model.inverse_points(sensor_uv.unsqueeze(0).expand(len(frames), -1, -1, -1), torch.as_tensor(theta, device=frames.device, dtype=frames.dtype)[:, None, None])
        valid = (source_uv[..., 0] >= 0.0) & (source_uv[..., 0] <= self.width - 1.0) & (source_uv[..., 1] >= 0.0) & (source_uv[..., 1] <= self.height - 1.0)
        return functional.grid_sample(frames, self._grid(source_uv), mode="bilinear", padding_mode=self.padding_mode, align_corners=False), valid


def compensate_events(
    events: EventStream,
    model: RotatingWedgePrism,
    rotation: object,
    *,
    theta0: float,
    reference_time: float,
    device: str = "cpu",
    chunk_size: int = 250_000,
) -> EventStream:
    """Attach reference-plane float coordinates calculated from every event time."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    output_u = np.empty(len(events), dtype=np.float64)
    output_v = np.empty(len(events), dtype=np.float64)
    for start in range(0, len(events), chunk_size):
        end = min(start + chunk_size, len(events))
        theta = rotation.theta_values(events.t[start:end], theta0=theta0, reference_time=reference_time)
        points = torch.from_numpy(np.column_stack((events.x[start:end], events.y[start:end]))).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            reference = model.inverse_points(points, torch.from_numpy(theta.astype(np.float32)).to(device=device))
        values = reference.cpu().numpy()
        output_u[start:end] = values[:, 0]
        output_v[start:end] = values[:, 1]
    return events.with_extras(x_reference=output_u, y_reference=output_v)
