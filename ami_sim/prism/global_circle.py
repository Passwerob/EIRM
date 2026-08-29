"""Debug-only global circular displacement approximation."""

from __future__ import annotations

import torch

from .base import PrismModel


class GlobalCircularApproximation(PrismModel):
    name = "global_circle"

    def __init__(self, intrinsics, amplitude_px: float = 4.0) -> None:
        super().__init__(intrinsics)
        self.amplitude_px = float(amplitude_px)

    def _offset(self, theta: torch.Tensor | float, points: torch.Tensor) -> torch.Tensor:
        angle = torch.as_tensor(theta, dtype=points.dtype, device=points.device)
        return self.amplitude_px * torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)

    def forward_points(self, points_xy: torch.Tensor, theta: torch.Tensor | float) -> torch.Tensor:
        return points_xy + self._offset(theta, points_xy)

    def inverse_points(self, points_xy: torch.Tensor, theta: torch.Tensor | float) -> torch.Tensor:
        return points_xy - self._offset(theta, points_xy)

    def metadata(self) -> dict[str, object]:
        return {**super().metadata(), "amplitude_px": self.amplitude_px, "debug_only": True}
