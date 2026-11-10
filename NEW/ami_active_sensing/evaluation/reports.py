"""Frozen selection and aggregation rules for trajectory-sweep reports."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np


def matching_task_utility(metrics: Mapping[str, float | int]) -> float:
    """Combine frozen matching metrics without using event count or density."""
    return float(
        0.30 * float(metrics["mean_repeatability"])
        + 0.25 * float(metrics["mean_matching_precision"])
        + 0.35 * float(metrics["mean_matching_recall"])
        + 0.10 * float(metrics["mean_ransac_inlier_ratio"])
        - 0.25 * float(metrics["matching_failure_rate"])
    )


def useful_event_utility(
    metrics: Mapping[str, float | int],
    *,
    bandwidth_penalty: float,
) -> float:
    """Task utility with a predeclared soft sensor-bandwidth cost."""
    if not np.isfinite(bandwidth_penalty) or bandwidth_penalty < 0.0:
        raise ValueError("bandwidth_penalty must be finite and non-negative.")
    density = float(metrics["mean_event_density"])
    if not np.isfinite(density) or density < 0.0:
        raise ValueError("mean_event_density must be finite and non-negative.")
    return matching_task_utility(metrics) - bandwidth_penalty * float(np.log1p(density))


def select_winners(
    records: Sequence[Mapping[str, object]],
    *,
    bandwidth_limit: float,
    bandwidth_penalty: float,
) -> dict[str, object]:
    """Select task, budgeted-useful, and event-count winners for one condition."""
    if not records:
        raise ValueError("At least one trajectory record is required.")
    if not np.isfinite(bandwidth_limit) or bandwidth_limit <= 0.0:
        raise ValueError("bandwidth_limit must be positive and finite.")

    def aggregate(record: Mapping[str, object]) -> Mapping[str, float | int]:
        value = record.get("aggregate")
        if not isinstance(value, Mapping):
            raise ValueError("Every trajectory record needs an aggregate mapping.")
        return value

    task_values = np.asarray([matching_task_utility(aggregate(record)) for record in records])
    useful_values = np.asarray([useful_event_utility(aggregate(record), bandwidth_penalty=bandwidth_penalty) for record in records])
    densities = np.asarray([float(aggregate(record)["mean_event_density"]) for record in records])
    counts = np.asarray([int(aggregate(record)["event_count"]) for record in records])
    within_budget = densities <= bandwidth_limit
    if not np.any(within_budget):
        # The lowest-bandwidth candidate is the only defensible fallback; the
        # report explicitly records that the hard budget was infeasible.
        useful_index = int(np.argmin(densities))
        budget_feasible = False
    else:
        masked = np.where(within_budget, useful_values, -np.inf)
        useful_index = int(np.argmax(masked))
        budget_feasible = True
    task_index = int(np.argmax(task_values))
    count_index = int(np.argmax(counts))

    def name(index: int) -> str:
        return str(records[index]["policy"])

    return {
        "best_task": name(task_index),
        "best_useful_under_budget": name(useful_index),
        "max_event_count": name(count_index),
        "budget_feasible": budget_feasible,
        "bandwidth_limit": float(bandwidth_limit),
        "best_task_utility": float(task_values[task_index]),
        "best_useful_utility": float(useful_values[useful_index]),
        "count_winner_task_utility": float(task_values[count_index]),
        "count_objective_task_regret": float(task_values[task_index] - task_values[count_index]),
        "useful_selection_task_regret": float(task_values[task_index] - task_values[useful_index]),
    }


def winner_distribution(winners: Sequence[str]) -> dict[str, object]:
    """Summarize whether one trajectory dominates all evaluated conditions."""
    if not winners:
        raise ValueError("winners must be non-empty.")
    counts = Counter(winners)
    probabilities = np.asarray(list(counts.values()), dtype=np.float64) / len(winners)
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    normalized = entropy / np.log(len(counts)) if len(counts) > 1 else 0.0
    return {
        "condition_count": len(winners),
        "unique_winner_count": len(counts),
        "counts": dict(sorted(counts.items())),
        "normalized_winner_entropy": normalized,
        "single_trajectory_wins_everywhere": len(counts) == 1,
    }
