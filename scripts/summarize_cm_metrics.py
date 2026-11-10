#!/usr/bin/env python3
"""Aggregate the already-generated CM-IWE metadata into paper metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _blocks(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "blocks" in payload:
        return payload["blocks"]
    return [{"name": item["name"], "methods": item["methods"]} for item in payload.get("regimes", [])]


def _method_name(name: str) -> str:
    lowered = name.lower()
    if "dvs" in lowered:
        return "dvs_voltmeter"
    if "v2e" in lowered:
        return "v2e"
    return "ours"


def _summarize(blocks: list[dict]) -> dict[str, dict[str, float]]:
    values: dict[str, list[dict[str, float]]] = defaultdict(list)
    for block in blocks:
        for item in block.get("methods", []):
            values[_method_name(str(item["method"]))].append(
                {
                    "cm_contrast": float(item["cm_contrast"]),
                    "event_count": float(item.get("n_events", item.get("event_count", 0))),
                    "cm_events": float(item.get("n_cm_events", item.get("n_events", 0))),
                    "vx": float(item["cm_velocity"][0]),
                    "vy": float(item["cm_velocity"][1]),
                }
            )
    output = {}
    for method, rows in sorted(values.items()):
        output[method] = {
            "blocks": len(rows),
            "mean_cm_contrast": float(np.mean([row["cm_contrast"] for row in rows])),
            "median_cm_contrast": float(np.median([row["cm_contrast"] for row in rows])),
            "mean_event_count": float(np.mean([row["event_count"] for row in rows])),
            "mean_cm_event_count": float(np.mean([row["cm_events"] for row in rows])),
            "mean_cm_velocity_x": float(np.mean([row["vx"] for row in rows])),
            "mean_cm_velocity_y": float(np.mean([row["vy"] for row in rows])),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=Path, nargs="+", default=[Path("results/ami_next/figures/paper_multiscene_events_meta.json"), Path("results/ami_next/figures/main_dvs_v2e_cm_meta.json")])
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/cm_metrics_summary.json"))
    args = parser.parse_args()
    reports = {}
    all_blocks = []
    for path in args.meta:
        blocks = _blocks(path)
        reports[str(path)] = {"blocks": len(blocks), "metrics": _summarize(blocks)}
        all_blocks.extend(blocks)
    report = {
        "protocol": "ami_paper_cm_iwe_summary_v1",
        "sources": [str(path) for path in args.meta],
        "aggregate": _summarize(all_blocks),
        "per_source": reports,
        "interpretation": "CM contrast and selected translation velocity are downstream representation diagnostics; they are not an event-generator quality metric by themselves.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
