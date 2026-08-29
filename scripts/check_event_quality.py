#!/usr/bin/env python3
"""Check event-stream invariants and compare a stochastic backend to a reference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ami_sim.events import read_events_h5
from ami_sim.metrics import compare_event_streams


def _timeline(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    values: list[float] = []
    for row in rows:
        for field in reversed(row):
            try:
                values.append(float(field))
                break
            except ValueError:
                continue
    result = np.asarray(values, dtype=np.float64)
    if len(result) > 1 and np.all(np.diff(result) > 0.0):
        return result
    raise ValueError(f"Could not read a strictly increasing timestamp timeline from {path}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Reference events.h5")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate events.h5")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--timestamps", type=Path, default=None, help="Optional frame timestamp CSV")
    parser.add_argument("--candidate-requires-integer-us", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    reference = read_events_h5(args.reference)
    candidate = read_events_h5(args.candidate)
    timeline = _timeline(args.timestamps) if args.timestamps is not None else None
    report = compare_event_streams(
        reference,
        candidate,
        (args.height, args.width),
        frame_timestamps=timeline,
        candidate_requires_integer_microseconds=args.candidate_requires_integer_us,
    )
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["quality_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
