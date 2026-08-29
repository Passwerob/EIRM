from __future__ import annotations

import numpy as np

from ami_sim.metrics import compare_event_streams, validate_event_stream
from ami_sim.types import EventStream


def _events() -> EventStream:
    return EventStream(
        np.array([1, 2, 1, 2], dtype=np.int32),
        np.array([1, 1, 2, 2], dtype=np.int32),
        np.array([0.001, 0.002, 0.003, 0.004], dtype=np.float64),
        np.array([1, -1, 1, -1], dtype=np.int8),
    )


def test_event_quality_checks_core_invariants() -> None:
    report = validate_event_stream(
        _events(),
        (4, 4),
        frame_timestamps=np.array([0.0, 0.01]),
        require_integer_microseconds=True,
    )
    assert report["valid"]
    assert report["checks"]["coordinates_in_bounds"]
    assert report["checks"]["timestamps_sorted"]
    assert report["timestamp_integer_us_fraction"] == 1.0


def test_event_quality_rejects_unsorted_or_out_of_bounds_stream() -> None:
    events = EventStream(
        np.array([0, 4]),
        np.array([0, 1]),
        np.array([0.002, 0.001]),
        np.array([1, -1], dtype=np.int8),
    )
    report = validate_event_stream(events, (4, 4))
    assert not report["valid"]
    assert not report["checks"]["coordinates_in_bounds"]
    assert not report["checks"]["timestamps_sorted"]


def test_event_quality_reports_nonfinite_timestamps_without_crashing() -> None:
    events = EventStream(
        np.array([0, 1]),
        np.array([0, 1]),
        np.array([0.001, np.nan]),
        np.array([1, -1], dtype=np.int8),
    )
    report = compare_event_streams(events, events, (4, 4))
    assert not report["quality_pass"]
    assert report["temporal_histogram_l1"] == 1.0


def test_event_quality_reference_comparison_passes_for_same_stream() -> None:
    report = compare_event_streams(_events(), _events(), (4, 4))
    assert report["quality_pass"]
    assert report["spatial_occupancy_iou"] == 1.0
    assert report["temporal_histogram_l1"] == 0.0
