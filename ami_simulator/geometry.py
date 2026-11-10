"""Pinhole and rotating-wedge geometry in an explicit OpenCV convention."""

from __future__ import annotations

import math

import torch

from .types import CameraIntrinsics


def _rotate_rays(rays: torch.Tensor, theta: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Rodrigues rotation about the in-plane axis perpendicular to deflection."""
    theta = torch.broadcast_to(theta, rays.shape[:-1])
    axis = torch.stack((-torch.sin(theta), torch.cos(theta), torch.zeros_like(theta)), dim=-1)
    angle = torch.as_tensor(angle_rad, dtype=rays.dtype, device=rays.device)
    cross = torch.cross(axis, rays, dim=-1)
    dot = (axis * rays).sum(dim=-1, keepdim=True)
    return rays * torch.cos(angle) + cross * torch.sin(angle) + axis * dot * (1.0 - torch.cos(angle))


class RotatingWedgePrism:
    """Paraxial wedge deflection represented as a camera-ray rotation.

    `forward_points` maps reference pixels to the prism sensor plane. The inverse
    maps sensor events back to the reference plane at the same prism angle.
    """

    name = "ray_deflection_wedge"

    def __init__(self, intrinsics: CameraIntrinsics, prism_angle_deg: float) -> None:
        self.intrinsics = intrinsics
        self.prism_angle_deg = float(prism_angle_deg)
        self.deflection_angle_rad = math.radians(self.prism_angle_deg)

    def _rays(self, points_uv: torch.Tensor) -> torch.Tensor:
        k = self.intrinsics
        x = (points_uv[..., 0] - k.cx) / k.fx
        y = (points_uv[..., 1] - k.cy) / k.fy
        return torch.stack((x, y, torch.ones_like(x)), dim=-1)

    def _pixels(self, rays: torch.Tensor) -> torch.Tensor:
        k = self.intrinsics
        z = rays[..., 2].clamp_min(1e-8)
        return torch.stack((k.fx * rays[..., 0] / z + k.cx, k.fy * rays[..., 1] / z + k.cy), dim=-1)

    def forward_points(self, points_uv: torch.Tensor, theta: torch.Tensor | float) -> torch.Tensor:
        angle = torch.as_tensor(theta, dtype=points_uv.dtype, device=points_uv.device)
        return self._pixels(_rotate_rays(self._rays(points_uv), angle, self.deflection_angle_rad))

    def inverse_points(self, points_uv: torch.Tensor, theta: torch.Tensor | float) -> torch.Tensor:
        angle = torch.as_tensor(theta, dtype=points_uv.dtype, device=points_uv.device)
        return self._pixels(_rotate_rays(self._rays(points_uv), angle, -self.deflection_angle_rad))

    def metadata(self) -> dict[str, object]:
        return {
            "model": self.name,
            "prism_angle_deg": self.prism_angle_deg,
            "intrinsics": self.intrinsics.__dict__,
            "coordinate_convention": "OpenCV pixels (u,v), +Z forward",
        }

