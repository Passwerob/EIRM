"""Explicitly non-research VFI backend used only for deterministic smoke tests."""

from __future__ import annotations

import numpy as np

from .base import VFIBackend, validate_targets


class DebugLinearVFIBackend(VFIBackend):
    name = "debug_linear"

    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
        targets = validate_targets(target_timestamps)
        first = np.asarray(frame0, dtype=np.float32)
        second = np.asarray(frame1, dtype=np.float32)
        if first.max(initial=0.0) > 1.0:
            first = first / 255.0
        if second.max(initial=0.0) > 1.0:
            second = second / 255.0
        if first.shape != second.shape:
            raise ValueError("VFI endpoint frames must have identical shapes.")
        samples = [first * (1.0 - tau) + second * tau for tau in targets]
        return np.stack(samples, axis=0) if samples else np.empty((0, *first.shape), dtype=np.float32)

    def interpolate_torch(self, frame0, frame1, target_timestamps):
        import torch

        targets = validate_targets(target_timestamps)
        first = frame0.to(dtype=torch.float32)
        second = frame1.to(device=first.device, dtype=torch.float32)
        if first.ndim != 3 or second.shape != first.shape:
            raise ValueError("Torch VFI endpoint frames must have shape [H,W,3].")
        values = torch.as_tensor(targets, dtype=first.dtype, device=first.device)
        view = (len(values),) + (1,) * first.ndim
        return first.unsqueeze(0) * (1.0 - values.view(view)) + second.unsqueeze(0) * values.view(view)

    def metadata(self) -> dict[str, object]:
        return {**super().metadata(), "debug_only": True}
