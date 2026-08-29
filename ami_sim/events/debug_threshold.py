"""Deterministic log-threshold event simulator for tests and offline smoke runs."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import EventSimulatorBackend
from ami_sim.types import EventStream


class DebugThresholdEventSimulator(EventSimulatorBackend):
    """A small contrast-threshold reference, never presented as DVS-Voltmeter."""

    name = "debug_threshold"

    def __init__(self, contrast_threshold: float = 0.2, eps: float = 1e-6) -> None:
        if contrast_threshold <= 0:
            raise ValueError("contrast_threshold must be positive.")
        self.contrast_threshold = float(contrast_threshold)
        self.eps = float(eps)

    def simulate(self, frames: np.ndarray, timestamps: np.ndarray, config: dict[str, Any] | None = None) -> EventStream:
        del config
        intensity = np.asarray(frames, dtype=np.float64)
        times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if intensity.ndim != 3 or intensity.shape[0] != len(times):
            raise ValueError("Event simulation expects frames [T,H,W] and one timestamp per frame.")
        if len(times) < 2 or not np.all(np.diff(times) > 0):
            raise ValueError("Event timestamps must be strictly increasing.")
        log_frames = np.log(np.clip(intensity, self.eps, 1.0))
        reference = log_frames[0].copy()
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        event_times: list[np.ndarray] = []
        polarities: list[np.ndarray] = []
        height, width = intensity.shape[1:]
        yy, xx = np.indices((height, width), dtype=np.int32)
        for index in range(1, len(times)):
            before = log_frames[index - 1]
            after = log_frames[index]
            delta = after - reference
            positive_count = np.floor(np.maximum(delta, 0.0) / self.contrast_threshold + self.eps).astype(np.int32)
            negative_count = np.floor(np.maximum(-delta, 0.0) / self.contrast_threshold + self.eps).astype(np.int32)
            for sign, counts in ((1, positive_count), (-1, negative_count)):
                active_y, active_x = np.nonzero(counts > 0)
                if len(active_x) == 0:
                    continue
                repeated = np.repeat(np.arange(len(active_x)), counts[active_y, active_x])
                x_values = active_x[repeated]
                y_values = active_y[repeated]
                levels = np.concatenate([
                    reference[y, x] + sign * self.contrast_threshold * np.arange(1, counts[y, x] + 1)
                    for x, y in zip(active_x, active_y)
                ])
                denominator = after[y_values, x_values] - before[y_values, x_values]
                alpha = np.divide(
                    levels - before[y_values, x_values], denominator,
                    out=np.full_like(levels, 0.5, dtype=np.float64), where=np.abs(denominator) > self.eps,
                )
                alpha = np.clip(alpha, 0.0, 1.0)
                xs.append(x_values.astype(np.int32))
                ys.append(y_values.astype(np.int32))
                event_times.append(times[index - 1] + alpha * (times[index] - times[index - 1]))
                polarities.append(np.full(len(x_values), sign, dtype=np.int8))
            reference += (positive_count - negative_count) * self.contrast_threshold
        if not xs:
            return EventStream.empty()
        events = EventStream(np.concatenate(xs), np.concatenate(ys), np.concatenate(event_times), np.concatenate(polarities))
        return events.sorted_by_time()

    def metadata(self) -> dict[str, object]:
        return {**super().metadata(), "contrast_threshold": self.contrast_threshold, "debug_only": True}
