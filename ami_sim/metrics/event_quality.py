"""Quality gates for generated event streams and backend comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np

from ami_sim.types import EventStream


def _relative_difference(value: int, reference: int) -> float:
    return float(abs(int(value) - int(reference)) / max(abs(int(reference)), 1))


def _normalised_histogram(values: np.ndarray, start: float, end: float, bins: int) -> np.ndarray:
    if end <= start:
        result = np.zeros(1, dtype=np.float64)
        result[0] = 1.0 if len(values) else 0.0
        return result
    histogram, _ = np.histogram(values, bins=np.linspace(start, end, bins + 1))
    total = int(histogram.sum())
    return histogram.astype(np.float64) / max(total, 1)


def _spatial_counts(events: EventStream, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    x = events.x.astype(np.int64, copy=False)
    y = events.y.astype(np.int64, copy=False)
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height)
    indices = y[valid] * width + x[valid]
    counts = np.bincount(indices, minlength=height * width).astype(np.float64)
    return counts, counts > 0


def validate_event_stream(
    events: EventStream,
    shape: tuple[int, int],
    *,
    frame_timestamps: np.ndarray | None = None,
    require_integer_microseconds: bool = False,
    integer_timestamp_tolerance_us: float = 1e-3,
) -> dict[str, Any]:
    """Check invariants that must hold for a serialized event stream."""
    height, width = map(int, shape)
    if height <= 0 or width <= 0:
        raise ValueError("Event image shape must be positive.")
    count = len(events)
    x = events.x.astype(np.float64, copy=False)
    y = events.y.astype(np.float64, copy=False)
    t = events.t.astype(np.float64, copy=False)
    p = events.p
    finite_t = bool(np.all(np.isfinite(t)))
    finite_xy = bool(np.all(np.isfinite(x)) and np.all(np.isfinite(y)))
    coordinates_in_bounds = bool(
        finite_xy and np.all((x >= 0.0) & (x < width) & (y >= 0.0) & (y < height))
    )
    polarity_valid = bool(np.all(np.isin(p, (-1, 1))))
    time_sorted = bool(count < 2 or np.all(np.diff(t) >= 0.0))
    has_x_float = "x_float" in events.extras
    has_y_float = "y_float" in events.extras
    compensated_fields_complete = has_x_float == has_y_float
    compensated_coordinates_finite = True
    if has_x_float and has_y_float:
        compensated_coordinates_finite = bool(
            np.all(np.isfinite(events.extras["x_float"]))
            and np.all(np.isfinite(events.extras["y_float"]))
        )
    if finite_t and count:
        timestamp_integer_us_fraction = float(
            np.mean(np.isclose(t * 1e6, np.rint(t * 1e6), rtol=0.0, atol=integer_timestamp_tolerance_us))
        )
        timestamp_range_s = [float(np.min(t)), float(np.max(t))]
    else:
        timestamp_integer_us_fraction = 0.0 if count else 1.0
        timestamp_range_s = [None, None]

    timestamp_in_frame_range = True
    frame_range_s: list[float | None] = [None, None]
    if frame_timestamps is not None:
        timeline = np.asarray(frame_timestamps, dtype=np.float64).reshape(-1)
        if len(timeline) < 2 or not np.all(np.diff(timeline) > 0.0):
            raise ValueError("frame_timestamps must contain a strictly increasing timeline.")
        frame_range_s = [float(timeline[0]), float(timeline[-1])]
        if count:
            timestamp_in_frame_range = bool(np.all((t >= timeline[0]) & (t <= timeline[-1])))

    checks = {
        "finite_timestamps": finite_t,
        "finite_coordinates": finite_xy,
        "coordinates_in_bounds": coordinates_in_bounds,
        "polarity_values_valid": polarity_valid,
        "timestamps_sorted": time_sorted,
        "timestamps_in_frame_range": timestamp_in_frame_range,
        "timestamps_integer_microseconds": timestamp_integer_us_fraction >= 1.0,
        "compensated_fields_complete": compensated_fields_complete,
        "compensated_coordinates_finite": compensated_coordinates_finite,
    }
    required = [
        checks["finite_timestamps"],
        checks["finite_coordinates"],
        checks["coordinates_in_bounds"],
        checks["polarity_values_valid"],
        checks["timestamps_sorted"],
        checks["timestamps_in_frame_range"],
        checks["compensated_fields_complete"],
        checks["compensated_coordinates_finite"],
    ]
    if require_integer_microseconds:
        required.append(checks["timestamps_integer_microseconds"])
    return {
        "valid": bool(all(required)),
        "shape": [height, width],
        "events": count,
        "positive": int(np.count_nonzero(p > 0)),
        "negative": int(np.count_nonzero(p < 0)),
        "positive_fraction": float(np.mean(p > 0)) if count else 0.0,
        "timestamp_range_s": timestamp_range_s,
        "frame_range_s": frame_range_s,
        "timestamp_integer_us_fraction": timestamp_integer_us_fraction,
        "checks": checks,
    }


def compare_event_streams(
    reference: EventStream,
    candidate: EventStream,
    shape: tuple[int, int],
    *,
    frame_timestamps: np.ndarray | None = None,
    event_count_tolerance: float = 0.01,
    polarity_fraction_tolerance: float = 0.01,
    temporal_histogram_l1_tolerance: float = 0.10,
    spatial_occupancy_iou_min: float = 0.80,
    spatial_distribution_tv_max: float = 0.30,
    candidate_requires_integer_microseconds: bool = False,
) -> dict[str, Any]:
    """Compare stochastic backends without requiring bitwise-identical events."""
    reference_quality = validate_event_stream(reference, shape, frame_timestamps=frame_timestamps)
    candidate_quality = validate_event_stream(
        candidate,
        shape,
        frame_timestamps=frame_timestamps,
        require_integer_microseconds=candidate_requires_integer_microseconds,
    )
    all_times = np.concatenate((reference.t, candidate.t)) if len(reference) or len(candidate) else np.empty(0)
    if len(all_times) and np.all(np.isfinite(all_times)):
        start, end = float(np.min(all_times)), float(np.max(all_times))
        ref_time_hist = _normalised_histogram(reference.t, start, end, 100)
        candidate_time_hist = _normalised_histogram(candidate.t, start, end, 100)
        temporal_histogram_l1 = float(0.5 * np.abs(ref_time_hist - candidate_time_hist).sum())
    elif len(all_times):
        # Invalid timestamps already fail the invariant gate; use the maximum
        # finite divergence so JSON reports remain serializable.
        temporal_histogram_l1 = 1.0
    else:
        temporal_histogram_l1 = 0.0

    reference_counts, reference_occupancy = _spatial_counts(reference, shape)
    candidate_counts, candidate_occupancy = _spatial_counts(candidate, shape)
    intersection = int(np.count_nonzero(reference_occupancy & candidate_occupancy))
    union = int(np.count_nonzero(reference_occupancy | candidate_occupancy))
    occupancy_iou = float(intersection / max(union, 1))
    ref_total = max(float(reference_counts.sum()), 1.0)
    candidate_total = max(float(candidate_counts.sum()), 1.0)
    spatial_distribution_tv = float(
        0.5 * np.abs(reference_counts / ref_total - candidate_counts / candidate_total).sum()
    )
    checks = {
        "reference_valid": bool(reference_quality["valid"]),
        "candidate_valid": bool(candidate_quality["valid"]),
        "event_count_within_tolerance": _relative_difference(candidate_quality["events"], reference_quality["events"])
        <= event_count_tolerance,
        "polarity_fraction_within_tolerance": abs(
            candidate_quality["positive_fraction"] - reference_quality["positive_fraction"]
        )
        <= polarity_fraction_tolerance,
        "temporal_histogram_within_tolerance": temporal_histogram_l1 <= temporal_histogram_l1_tolerance,
        "spatial_occupancy_iou_above_minimum": occupancy_iou >= spatial_occupancy_iou_min,
        "spatial_distribution_within_tolerance": spatial_distribution_tv <= spatial_distribution_tv_max,
    }
    return {
        "quality_pass": bool(all(checks.values())),
        "reference": reference_quality,
        "candidate": candidate_quality,
        "relative_event_count_difference": _relative_difference(
            candidate_quality["events"], reference_quality["events"]
        ),
        "positive_fraction_difference": abs(
            candidate_quality["positive_fraction"] - reference_quality["positive_fraction"]
        ),
        "temporal_histogram_l1": temporal_histogram_l1,
        "spatial_occupancy_iou": occupancy_iou,
        "spatial_distribution_tv": spatial_distribution_tv,
        "thresholds": {
            "event_count_tolerance": event_count_tolerance,
            "polarity_fraction_tolerance": polarity_fraction_tolerance,
            "temporal_histogram_l1_tolerance": temporal_histogram_l1_tolerance,
            "spatial_occupancy_iou_min": spatial_occupancy_iou_min,
            "spatial_distribution_tv_max": spatial_distribution_tv_max,
        },
        "checks": checks,
    }
