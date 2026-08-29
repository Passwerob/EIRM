"""Photometrically consistent temporal flicker diagnostics."""

from __future__ import annotations

import numpy as np


def temporal_flicker_stats(intensity_frames: np.ndarray, valid_mask: np.ndarray | None = None) -> dict[str, float]:
    frames = np.asarray(intensity_frames, dtype=np.float64)
    if frames.ndim != 3 or len(frames) < 2:
        return {"frame_count": float(len(frames)), "mean_abs_log_difference": 0.0, "std_log_difference": 0.0, "oscillation_score": 0.0}
    log_frames = np.log(np.clip(frames, 1e-6, 1.0))
    difference = np.diff(log_frames, axis=0)
    if valid_mask is not None:
        difference = difference[:, valid_mask]
    values = difference.reshape(-1)
    if len(values) == 0:
        values = np.zeros(1)
    signs = np.sign(values)
    sign_changes = np.mean(signs[1:] != signs[:-1]) if len(signs) > 1 else 0.0
    return {
        "frame_count": float(len(frames)),
        "mean_abs_log_difference": float(np.mean(np.abs(values))),
        "std_log_difference": float(np.std(values)),
        "oscillation_score": float(sign_changes),
    }
