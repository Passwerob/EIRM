#!/usr/bin/env python3
"""Validate a frozen controller surrogate against held-out downstream utility."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and values[order[end]] == values[order[begin]]:
            end += 1
        ranks[order[begin:end]] = 0.5 * (begin + end - 1) + 1.0
        begin = end
    return ranks


def spearman_rho(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) != len(second) or len(first) < 3:
        raise ValueError("Spearman correlation requires aligned arrays with at least three samples.")
    x, y = average_ranks(first), average_ranks(second)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def normalized_selection_regret(surrogate: np.ndarray, downstream: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    regrets: list[float] = []
    agreements: list[float] = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        chosen = indices[int(np.argmax(surrogate[indices]))]
        best = float(np.max(downstream[indices]))
        span = best - float(np.min(downstream[indices]))
        regrets.append((best - float(downstream[chosen])) / span if span > 0.0 else 0.0)
        agreements.append(float(downstream[chosen] == best))
    return float(np.median(regrets)), float(np.mean(agreements))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Held-out CSV with surrogate, downstream, and group columns.")
    parser.add_argument("--surrogate-column", default="controller_surrogate")
    parser.add_argument("--downstream-column", required=True)
    parser.add_argument("--group-column", default="scene")
    parser.add_argument("--downstream-direction", choices=("higher", "lower"), default="higher")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--minimum-rho", type=float, default=0.5)
    parser.add_argument("--maximum-median-regret", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enforce", action="store_true", help="Exit nonzero when the predeclared gate fails.")
    args = parser.parse_args()
    if args.bootstrap < 100 or not -1.0 <= args.minimum_rho <= 1.0 or not 0.0 <= args.maximum_median_regret <= 1.0:
        parser.error("Bootstrap count and validation thresholds are invalid.")
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {args.surrogate_column, args.downstream_column, args.group_column}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"CSV must contain {sorted(required)}.")
    surrogate = np.asarray([float(row[args.surrogate_column]) for row in rows], dtype=np.float64)
    downstream = np.asarray([float(row[args.downstream_column]) for row in rows], dtype=np.float64)
    groups = np.asarray([row[args.group_column] for row in rows])
    if args.downstream_direction == "lower":
        downstream = -downstream
    rho = spearman_rho(surrogate, downstream)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Scene-level bootstrap requires at least two held-out groups.")
    rng = np.random.default_rng(args.seed)
    bootstrapped: list[float] = []
    for _ in range(args.bootstrap):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        x_parts, y_parts = [], []
        for group in sampled:
            indices = np.flatnonzero(groups == group)
            x_parts.append(surrogate[indices])
            y_parts.append(downstream[indices])
        try:
            bootstrapped.append(spearman_rho(np.concatenate(x_parts), np.concatenate(y_parts)))
        except ValueError:
            continue
    lower, upper = np.quantile(bootstrapped, [0.025, 0.975])
    median_regret, top1_agreement = normalized_selection_regret(surrogate, downstream, groups)
    gates = {
        "spearman_at_least_minimum": rho >= args.minimum_rho,
        "bootstrap_95pct_lower_above_zero": lower > 0.0,
        "median_normalized_selection_regret_within_limit": median_regret <= args.maximum_median_regret,
    }
    report = {
        "input": str(args.input),
        "protocol": "frozen_score_on_held_out_scenes",
        "sample_count": len(rows),
        "group_count": len(unique_groups),
        "spearman_rho": rho,
        "bootstrap_95pct_interval": [float(lower), float(upper)],
        "median_normalized_selection_regret": median_regret,
        "top1_agreement": top1_agreement,
        "thresholds": {"minimum_rho": args.minimum_rho, "maximum_median_regret": args.maximum_median_regret},
        "gates": {key: bool(value) for key, value in gates.items()},
        "all_gates_passed": bool(all(gates.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.enforce and not report["all_gates_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
