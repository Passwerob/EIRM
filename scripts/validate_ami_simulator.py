#!/usr/bin/env python3
"""Generate all downstream diagnostics required by the AMI validation document."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ami_sim.analysis import run_validation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a completed EIRM AMI simulator run.")
    parser.add_argument("--run", required=True, type=Path, help="Completed simulator output directory.")
    parser.add_argument("--output", type=Path, default=None, help="Validation output directory (default: <run>/ami_validation).")
    parser.add_argument("--event-window-ms", type=float, default=20.0)
    parser.add_argument("--edge-quantile", type=float, default=0.80)
    parser.add_argument("--support-radius", type=int, default=1)
    parser.add_argument("--fps-root", type=Path, default=None, help="Directory containing 240/480/960 simulator output subdirectories.")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--debug-synthetic-dvs", action="store_true")
    parser.add_argument("--reference-run", type=Path, default=None, help="Optional completed run used for stochastic event-quality comparison.")
    args = parser.parse_args(argv)
    result = run_validation(
        args.run,
        output=args.output,
        event_window_ms=args.event_window_ms,
        edge_quantile=args.edge_quantile,
        support_radius=args.support_radius,
        fps_root=args.fps_root,
        run_synthetic=not args.skip_synthetic,
        official_synthetic_dvs=not args.debug_synthetic_dvs,
        reference_root=args.reference_run,
    )
    summary = {"validation_dir": result["validation_dir"], "fps_rows": len(result["fps_rows"])}
    if result.get("event_quality_vs_reference") is not None:
        summary["event_quality_pass"] = result["event_quality_vs_reference"]["quality_pass"]
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("event_quality_pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
