"""Video-frame interpolation interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VFIBackend(ABC):
    """Generate direct samples at normalized timesteps strictly inside an interval."""

    name: str

    @abstractmethod
    def interpolate(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        target_timestamps: np.ndarray,
    ) -> np.ndarray:
        """Return [N,H,W,C] samples for requested normalized times in (0, 1)."""

    def metadata(self) -> dict[str, object]:
        return {"vfi_backend": self.name, "recursive_interpolation": False}


def validate_targets(target_timestamps: np.ndarray) -> np.ndarray:
    targets = np.asarray(target_timestamps, dtype=np.float64).reshape(-1)
    if np.any((targets <= 0.0) | (targets >= 1.0)):
        raise ValueError("VFI target timestamps must be normalized values strictly in (0, 1).")
    if len(targets) > 1 and not np.all(np.diff(targets) > 0):
        raise ValueError("VFI target timestamps must be strictly increasing.")
    return targets
