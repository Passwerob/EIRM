"""Image-space prism model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ami_sim.types import CameraIntrinsics


class PrismModel(ABC):
    """Maps reference pixel coordinates to sensor coordinates for a prism angle."""

    name = "base"

    def __init__(self, intrinsics: CameraIntrinsics) -> None:
        self.intrinsics = intrinsics

    @abstractmethod
    def forward_points(self, points_xy: torch.Tensor, theta: torch.Tensor | float) -> torch.Tensor:
        """Map reference coordinates to the prism sensor plane."""

    @abstractmethod
    def inverse_points(self, points_xy: torch.Tensor, theta: torch.Tensor | float) -> torch.Tensor:
        """Map prism sensor coordinates back to reference coordinates."""

    def displacement(self, points_xy: torch.Tensor, theta: torch.Tensor | float) -> torch.Tensor:
        return self.forward_points(points_xy, theta) - points_xy

    def metadata(self) -> dict[str, object]:
        return {"prism_model": self.name, "intrinsics": self.intrinsics.__dict__}
