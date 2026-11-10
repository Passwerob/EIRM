"""Frozen optical-flow backend for estimating available natural excitation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FarnebackConfig:
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.pyr_scale < 1.0:
            raise ValueError("pyr_scale must lie in (0, 1).")
        if min(self.levels, self.winsize, self.iterations, self.poly_n) <= 0 or self.poly_sigma <= 0.0:
            raise ValueError("Farneback iteration, window, polynomial, and scale parameters must be positive.")


@dataclass(frozen=True)
class OpticalFlowResult:
    """Dense flow in pixel displacement with an explicit frame interval."""

    flow_xy_px: np.ndarray
    valid_mask: np.ndarray
    dt_seconds: float
    backend_metadata: dict[str, object]

    def __post_init__(self) -> None:
        flow = np.asarray(self.flow_xy_px)
        valid = np.asarray(self.valid_mask)
        if flow.ndim != 3 or flow.shape[-1] != 2:
            raise ValueError("flow_xy_px must have shape [H, W, 2].")
        if valid.shape != flow.shape[:2] or valid.dtype != np.bool_:
            raise ValueError("valid_mask must be a boolean array with shape [H, W].")
        if not np.isfinite(self.dt_seconds) or self.dt_seconds <= 0.0:
            raise ValueError("dt_seconds must be positive and finite.")


def _gray_u8(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[-1] not in (3, 4)):
        raise ValueError("Expected grayscale [H,W] or RGB/RGBA [H,W,C] input.")
    if not np.all(np.isfinite(image)):
        raise ValueError("RGB frames must contain only finite values.")
    values = image.astype(np.float32)
    if np.issubdtype(image.dtype, np.integer):
        scale = float(np.iinfo(image.dtype).max)
        values = values / max(scale, 1.0)
    elif values.min(initial=0.0) < 0.0 or values.max(initial=0.0) > 1.0:
        raise ValueError("Floating-point image input must lie in [0, 1].")
    if values.ndim == 3:
        values = cv2.cvtColor(values[..., :3], cv2.COLOR_RGB2GRAY)
    return np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)


class FarnebackFlowBackend:
    """OpenCV Farneback flow used only as natural-motion context."""

    name = "opencv_farneback"

    def __init__(self, config: FarnebackConfig | None = None) -> None:
        self.config = config or FarnebackConfig()

    def estimate(self, previous_rgb: np.ndarray, current_rgb: np.ndarray, *, dt_seconds: float) -> OpticalFlowResult:
        if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
            raise ValueError("dt_seconds must be positive and finite.")
        previous = _gray_u8(previous_rgb)
        current = _gray_u8(current_rgb)
        if previous.shape != current.shape:
            raise ValueError("Optical-flow frame shapes must match.")
        cfg = self.config
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            cfg.pyr_scale,
            cfg.levels,
            cfg.winsize,
            cfg.iterations,
            cfg.poly_n,
            cfg.poly_sigma,
            cfg.flags,
        ).astype(np.float32)
        valid = np.isfinite(flow).all(axis=-1)
        flow = np.where(valid[..., None], flow, 0.0).astype(np.float32)
        return OpticalFlowResult(flow, valid, float(dt_seconds), self.metadata())

    def metadata(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "opencv_version": cv2.__version__,
            "flow_units": "pixel_displacement_between_frames",
            "coordinate_order": "dx_dy",
            "config": asdict(self.config),
        }
