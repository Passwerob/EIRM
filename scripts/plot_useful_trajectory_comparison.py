#!/usr/bin/env python3
"""Plot useful-event trajectory comparison from a trajectory sweep report.

The adaptive curve is a scene-level oracle: it selects the highest-useful
trajectory after observing the complete scene. It is an upper bound for a
future causal controller, not evidence that a real-time policy is finished.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _summarize(rows: list[dict[str, object]], trajectories: list[str], *, bootstrap_samples: int = 10000) -> tuple[dict[str, tuple[float, float]], list[dict[str, object]], dict[str, float]]:
    scenes = sorted({str(row["scene"]) for row in rows})
    fixed: dict[str, tuple[float, float]] = {}
    for trajectory in trajectories:
        values = np.asarray([float(row["mean_useful_score"]) for row in rows if row["trajectory"] == trajectory], dtype=np.float64)
        fixed[trajectory] = (float(values.mean()), float(values.std()))
    best_fixed = max(trajectories, key=lambda key: fixed[key][0])
    adaptive_rows: list[dict[str, object]] = []
    for scene in scenes:
        candidates = [row for row in rows if str(row["scene"]) == scene]
        selected = max(candidates, key=lambda row: float(row["mean_useful_score"]))
        baseline = next(row for row in candidates if row["trajectory"] == best_fixed)
        adaptive_rows.append({"scene": scene, "trajectory": selected["trajectory"], "useful_score": float(selected["mean_useful_score"]), "best_fixed_score": float(baseline["mean_useful_score"])})
    values = np.asarray([float(row["useful_score"]) for row in adaptive_rows], dtype=np.float64)
    fixed["adaptive_oracle"] = (float(values.mean()), float(values.std()))
    gains = values - np.asarray([float(row["best_fixed_score"]) for row in adaptive_rows], dtype=np.float64)
    rng = np.random.default_rng(0)
    bootstrap = np.asarray([np.mean(rng.choice(gains, size=len(gains), replace=True)) for _ in range(bootstrap_samples)], dtype=np.float64)
    return fixed, adaptive_rows, {"mean": float(gains.mean()), "positive_scene_count": int(np.count_nonzero(gains > 1e-12)), "scene_count": int(len(gains)), "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)), "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", default=[Path("results/trajectory_sweep_test.json")])
    parser.add_argument("--output", type=Path, default=Path("results/useful_trajectory_comparison.png"))
    parser.add_argument("--summary", type=Path, default=Path("results/useful_trajectory_comparison.json"))
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    trajectories = list(reports[0]["trajectory_families"])
    if any(list(report["trajectory_families"]) != trajectories for report in reports[1:]):
        raise ValueError("All sweep reports must use the same trajectory families.")
    rows = [row for report in reports for row in report["rows"]]
    fixed, adaptive_rows, gain_summary = _summarize(rows, trajectories)
    labels = trajectories + ["adaptive\noracle"]
    keys = trajectories + ["adaptive_oracle"]
    means = np.asarray([fixed[key][0] for key in keys])
    stds = np.asarray([fixed[key][1] for key in keys])
    best_fixed = max(trajectories, key=lambda key: fixed[key][0])

    figure, (axis, gain_axis) = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": (1.15, 1.0)})
    positions = np.arange(len(labels))
    colors = ["#9aa5b1"] * len(trajectories) + ["#d1495b"]
    axis.bar(positions, means, yerr=stds, capsize=4, color=colors, edgecolor="#263238", linewidth=0.8)
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylabel("Mean useful-event score")
    axis.set_title("Overall useful excitation")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.annotate(
        "adaptive oracle",
        xy=(positions[-1], means[-1]),
        xytext=(positions[-1] - 1.0, means[-1] + 0.035),
        arrowprops={"arrowstyle": "->", "color": "#d1495b"},
        color="#9b2534",
    )

    gains = np.asarray([float(row["useful_score"]) - float(row["best_fixed_score"]) for row in adaptive_rows])
    def condition_for(scene: str) -> str:
        return next((name for name in ("camera_only", "object_only", "co_motion", "counter_motion") if scene.endswith(name)), "unknown")

    conditions = ("camera_only", "object_only", "co_motion", "counter_motion")
    condition_means = np.asarray([np.mean([gain for gain, row in zip(gains, adaptive_rows) if condition_for(str(row["scene"])) == condition]) for condition in conditions])
    condition_stds = np.asarray([np.std([gain for gain, row in zip(gains, adaptive_rows) if condition_for(str(row["scene"])) == condition]) for condition in conditions])
    gain_colors = ["#2a9d8f" if value >= 0 else "#e76f51" for value in condition_means]
    gain_axis.bar(np.arange(len(conditions)), condition_means, yerr=condition_stds, capsize=4, color=gain_colors, edgecolor="#263238", linewidth=0.8)
    gain_axis.axhline(0.0, color="#263238", linewidth=0.8)
    gain_axis.set_xticks(np.arange(len(conditions)), [name.replace("_", "\n") for name in conditions])
    gain_axis.set_ylabel(f"Adaptive oracle - {best_fixed} useful score")
    gain_axis.set_title("Condition-wise gain over best fixed")
    gain_axis.grid(axis="y", alpha=0.25)
    gain_axis.set_axisbelow(True)
    figure.suptitle("Useful-event objective: local specialization, global gain", fontsize=14)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "input": [str(path) for path in args.input],
        "metric": "mean_useful_score",
        "best_fixed": best_fixed,
        "fixed": {key: {"mean": mean, "std": std} for key, (mean, std) in fixed.items() if key != "adaptive_oracle"},
        "adaptive_oracle": {"mean": fixed["adaptive_oracle"][0], "std": fixed["adaptive_oracle"][1], "scene_selections": adaptive_rows},
        "paired_gain_over_best_fixed": gain_summary,
        "interpretation": "The adaptive oracle is a scene-level upper bound, not a causal controller evaluation.",
        "output": str(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
