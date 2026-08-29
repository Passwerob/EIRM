"""GPU-capable inverse-sampled prism warp and event compensation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ami_sim.types import EventStream
from .base import PrismModel
from .rotation_policy import RotationPolicy


@dataclass
class PrismWarper:
    model: PrismModel
    policy: RotationPolicy
    height: int
    width: int
    theta0: float = 0.0
    reference_time: float | None = None
    align_corners: bool = False
    padding_mode: str = "border"
    batch_size: int = 8

    def __post_init__(self) -> None:
        if self.padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError("padding_mode must be zeros, border, or reflection.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        y, x = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float32),
            torch.arange(self.width, dtype=torch.float32),
            indexing="ij",
        )
        self._sensor_xy = torch.stack((x, y), dim=-1)

    def _normalise_grid(self, pixels_xy: torch.Tensor) -> torch.Tensor:
        if self.align_corners:
            x = 2.0 * pixels_xy[..., 0] / max(self.width - 1, 1) - 1.0
            y = 2.0 * pixels_xy[..., 1] / max(self.height - 1, 1) - 1.0
        else:
            x = 2.0 * (pixels_xy[..., 0] + 0.5) / self.width - 1.0
            y = 2.0 * (pixels_xy[..., 1] + 0.5) / self.height - 1.0
        return torch.stack((x, y), dim=-1)

    def theta_values(self, timestamps: np.ndarray) -> np.ndarray:
        values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if len(values) == 0:
            return values
        reference = float(values[0] if self.reference_time is None else self.reference_time)
        return self.policy.theta_values(values, theta0=self.theta0, reference_time=reference)

    def warp(self, frames: torch.Tensor, timestamps: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Warp [T,C,H,W] frames and return [T] validity masks."""
        if frames.ndim != 4 or tuple(frames.shape[-2:]) != (self.height, self.width):
            raise ValueError("PrismWarper expects frames with shape [T,C,H,W] at its configured resolution.")
        theta_values = self.theta_values(timestamps)
        sensor_grid = self._sensor_xy.to(device=frames.device, dtype=frames.dtype)
        warped_batches: list[torch.Tensor] = []
        valid_batches: list[torch.Tensor] = []
        for begin in range(0, len(theta_values), self.batch_size):
            end = min(begin + self.batch_size, len(theta_values))
            theta = torch.as_tensor(theta_values[begin:end], dtype=frames.dtype, device=frames.device)
            source_grid = sensor_grid.unsqueeze(0).expand(end - begin, -1, -1, -1)
            source_grid = self.model.inverse_points(source_grid, theta[:, None, None])
            grid = self._normalise_grid(source_grid)
            warped_batches.append(F.grid_sample(
                frames[begin:end], grid, mode="bilinear", padding_mode=self.padding_mode, align_corners=self.align_corners,
            ))
            valid_batches.append(
                (source_grid[..., 0] >= 0.0)
                & (source_grid[..., 0] <= self.width - 1)
                & (source_grid[..., 1] >= 0.0)
                & (source_grid[..., 1] <= self.height - 1)
            )
        return torch.cat(warped_batches, dim=0), torch.cat(valid_batches, dim=0)

    def dewarp(self, frames: torch.Tensor, timestamps: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample sensor-plane ``frames`` back onto the reference image plane.

        ``warp`` implements ``S_t(s) = I_t(W_t^{-1}(s))``.  This inverse-sampled
        counterpart implements ``I_hat_t(r) = S_t(W_t(r))`` and is therefore the
        correct image-space operation after reconstructing an AMI sensor frame.
        Both directions deliberately share the same grid convention and padding
        behavior so that a round trip does not introduce a coordinate mismatch.
        """
        if frames.ndim != 4 or tuple(frames.shape[-2:]) != (self.height, self.width):
            raise ValueError("PrismWarper expects frames with shape [T,C,H,W] at its configured resolution.")
        theta_values = self.theta_values(timestamps)
        reference_xy = self._sensor_xy.to(device=frames.device, dtype=frames.dtype)
        dewarped_batches: list[torch.Tensor] = []
        valid_batches: list[torch.Tensor] = []
        for begin in range(0, len(theta_values), self.batch_size):
            end = min(begin + self.batch_size, len(theta_values))
            theta = torch.as_tensor(theta_values[begin:end], dtype=frames.dtype, device=frames.device)
            target_grid = reference_xy.unsqueeze(0).expand(end - begin, -1, -1, -1)
            source_grid = self.model.forward_points(target_grid, theta[:, None, None])
            grid = self._normalise_grid(source_grid)
            dewarped_batches.append(F.grid_sample(
                frames[begin:end], grid, mode="bilinear", padding_mode=self.padding_mode, align_corners=self.align_corners,
            ))
            valid_batches.append(
                (source_grid[..., 0] >= 0.0)
                & (source_grid[..., 0] <= self.width - 1)
                & (source_grid[..., 1] >= 0.0)
                & (source_grid[..., 1] <= self.height - 1)
            )
        return torch.cat(dewarped_batches, dim=0), torch.cat(valid_batches, dim=0)

    def all_orientation_valid_mask(
        self,
        *,
        samples: int = 360,
        start_time: float = 0.0,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if samples < 2:
            raise ValueError("At least two orientation samples are required for a valid mask.")
        period = 2.0 * np.pi / max(abs(self.policy.omega(start_time)), 1e-12)
        timestamps = np.linspace(start_time, start_time + period, samples, endpoint=False)
        target_device = torch.device(device) if device is not None else self._sensor_xy.device
        sensor_grid = self._sensor_xy.to(device=target_device)
        valid_all = torch.ones((self.height, self.width), dtype=torch.bool, device=target_device)
        for begin in range(0, len(timestamps), self.batch_size):
            end = min(begin + self.batch_size, len(timestamps))
            theta = torch.as_tensor(self.theta_values(timestamps[begin:end]), dtype=torch.float32, device=target_device)
            source_grid = sensor_grid.unsqueeze(0).expand(end - begin, -1, -1, -1)
            source_grid = self.model.inverse_points(source_grid, theta[:, None, None])
            valid = (
                (source_grid[..., 0] >= 0.0)
                & (source_grid[..., 0] <= self.width - 1)
                & (source_grid[..., 1] >= 0.0)
                & (source_grid[..., 1] <= self.height - 1)
            )
            valid_all &= valid.all(dim=0)
        return valid_all


def compensate_events(
    events: EventStream,
    model: PrismModel,
    policy: RotationPolicy,
    *,
    theta0: float = 0.0,
    reference_time: float | None = None,
    chunk_size: int = 100_000,
    device: torch.device | str | None = None,
) -> EventStream:
    """Apply W^-1 at each event's own timestamp, preserving timing and polarity."""
    if len(events) == 0:
        return events.with_extras(x_float=np.empty(0, dtype=np.float64), y_float=np.empty(0, dtype=np.float64))
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    values = events.t
    reference = float(values[0] if reference_time is None else reference_time)
    # DVS backends emit time-sorted streams. Avoid making an O(N) permutation and
    # an additional full-size index array in the common case; retain the fallback
    # for external/legacy backends that do not guarantee ordering.
    already_sorted = len(values) < 2 or bool(np.all(values[1:] >= values[:-1]))
    order = None if already_sorted else np.argsort(values, kind="stable")
    sorted_times = values if order is None else values[order]
    sorted_theta = np.asarray(
        policy.theta_values(sorted_times, theta0=theta0, reference_time=reference),
        dtype=np.float32,
    )
    output_x = np.empty(len(events), dtype=np.float64)
    output_y = np.empty(len(events), dtype=np.float64)
    target_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # A single inference context avoids autograd bookkeeping for every chunk.
    with torch.inference_mode():
        for begin in range(0, len(events), chunk_size):
            end = min(begin + chunk_size, len(events))
            indices = slice(begin, end) if order is None else order[begin:end]
            points_np = np.column_stack((events.x[indices], events.y[indices]))
            points = torch.from_numpy(points_np).to(device=target_device, dtype=torch.float32, non_blocking=True)
            theta = torch.from_numpy(sorted_theta[begin:end]).to(device=target_device)
            compensated = model.inverse_points(points, theta)
            values_cpu = compensated.detach().cpu().numpy()
            output_x[indices] = values_cpu[:, 0]
            output_y[indices] = values_cpu[:, 1]
    return events.with_extras(x_float=output_x, y_float=output_y)
