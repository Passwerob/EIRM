"""Calibrated paraxial ray-deflection model used by the simulator baseline.

The mapping is implemented as a camera-ray rotation around the in-plane axis
perpendicular to the current prism deflection direction, followed by projection
through the supplied DSEC intrinsics. It is deliberately an image formation warp,
not a 2-D image rotation. The paper's released implementation should be compared
against this formula before claiming exact reproduction; provenance is preserved in
the run metadata because the external release is not part of this checkout.
"""

from __future__ import annotations

import math

import torch

from .base import PrismModel


def _rotate_rays(rays: torch.Tensor, theta: torch.Tensor, angle: float) -> torch.Tensor:
    """Rodrigues rotation with a theta-dependent axis in camera coordinates."""
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    axis = torch.stack((-sin_theta, cos_theta, torch.zeros_like(theta)), dim=-1)
    while axis.ndim < rays.ndim:
        axis = axis.unsqueeze(-2)
    angle_tensor = torch.as_tensor(angle, dtype=rays.dtype, device=rays.device)
    cross = torch.cross(axis.expand_as(rays), rays, dim=-1)
    dot = (axis * rays).sum(dim=-1, keepdim=True)
    return (
        rays * torch.cos(angle_tensor)
        + cross * torch.sin(angle_tensor)
        + axis * dot * (1.0 - torch.cos(angle_tensor))
    )


class PaperSimplifiedPrismModel(PrismModel):
    name = "paper_simplified"

    def __init__(self, intrinsics, prism_angle_deg: float = 0.5) -> None:
        super().__init__(intrinsics)
        self.prism_angle_deg = float(prism_angle_deg)
        self.deflection_angle_rad = torch.pi * self.prism_angle_deg / 180.0
        self._inverse_sin_angle = math.sin(-float(self.deflection_angle_rad))
        self._inverse_cos_angle = math.cos(-float(self.deflection_angle_rad))

    def _rays(self, points_xy: torch.Tensor) -> torch.Tensor:
        fx, fy, cx, cy = self.intrinsics.fx, self.intrinsics.fy, self.intrinsics.cx, self.intrinsics.cy
        x = (points_xy[..., 0] - cx) / fx
        y = (points_xy[..., 1] - cy) / fy
        return torch.stack((x, y, torch.ones_like(x)), dim=-1)

    def _pixels(self, rays: torch.Tensor) -> torch.Tensor:
        fx, fy, cx, cy = self.intrinsics.fx, self.intrinsics.fy, self.intrinsics.cx, self.intrinsics.cy
        z = rays[..., 2].clamp_min(1e-8)
        return torch.stack((fx * rays[..., 0] / z + cx, fy * rays[..., 1] / z + cy), dim=-1)

    def forward_points(self, points_xy: torch.Tensor, theta: torch.Tensor | float) -> torch.Tensor:
        theta_tensor = torch.as_tensor(theta, dtype=points_xy.dtype, device=points_xy.device)
        rays = _rotate_rays(self._rays(points_xy), theta_tensor, float(self.deflection_angle_rad))
        return self._pixels(rays)

    def inverse_points(self, points_xy: torch.Tensor, theta: torch.Tensor | float) -> torch.Tensor:
        # Inline Rodrigues for the fixed prism angle. The generic cross-product
        # form materializes an axis and expanded ray tensor; this form is notably
        # cheaper for multi-million-event compensation while remaining algebraically
        # identical.
        theta_tensor = torch.as_tensor(theta, dtype=points_xy.dtype, device=points_xy.device)
        while theta_tensor.ndim < points_xy.ndim - 1:
            theta_tensor = theta_tensor.unsqueeze(-1)
        x, y, _ = self._rays(points_xy).unbind(dim=-1)
        sin_theta = torch.sin(theta_tensor)
        cos_theta = torch.cos(theta_tensor)
        sin_angle = self._inverse_sin_angle
        cos_angle = self._inverse_cos_angle
        dot = -sin_theta * x + cos_theta * y
        rotated_x = x * cos_angle + cos_theta * sin_angle - sin_theta * dot * (1.0 - cos_angle)
        rotated_y = y * cos_angle + sin_theta * sin_angle + cos_theta * dot * (1.0 - cos_angle)
        rotated_z = cos_angle + (-sin_theta * y - cos_theta * x) * sin_angle
        return self._pixels(torch.stack((rotated_x, rotated_y, rotated_z), dim=-1))

    def metadata(self) -> dict[str, object]:
        return {
            **super().metadata(),
            "prism_angle_deg": self.prism_angle_deg,
            "formula": "camera-ray Rodrigues deflection then pinhole projection",
            "provenance_status": "calibrated_simplification_requires_official-code-comparison",
        }
