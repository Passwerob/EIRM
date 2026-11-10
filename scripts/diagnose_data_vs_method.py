#!/usr/bin/env python3
"""Separate data limitations from policy-method limitations.

This is an offline diagnostic on the already-generated phase-continuous
oracles. It does not regenerate events and does not modify the simulator. The
targets are the useful-event utilities of candidate gears evaluated along the
causal oracle path. Results are diagnostic only; final controller claims still
require direct phase-continuous rollout.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.train_causal_q_policy import (
    BASE_FEATURE_DIM,
    FLOW_FEATURE_DIM,
    GEAR_SCALES,
    _counterfactual_sequence,
    _windows,
)


def _entropy(labels: np.ndarray, classes: int) -> float:
    counts = np.bincount(labels, minlength=classes).astype(np.float64)
    probabilities = counts / max(float(counts.sum()), 1.0)
    probabilities = probabilities[probabilities > 0.0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _seed(scene: str) -> int:
    match = re.search(r"seed(\d+)", scene)
    if match is None:
        raise ValueError(f"Cannot infer seed from scene {scene!r}.")
    return int(match.group(1))


def _load_split(
    records: list[dict[str, object]],
    *,
    split: str,
    objective: dict[str, float],
    window: int,
    use_flow: bool,
    seed_filter: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    scene_reports: list[dict[str, object]] = []
    rpm_values = 720.0 * GEAR_SCALES
    for record in records:
        if record["split"] != split:
            continue
        scene = str(record["scene"])
        if seed_filter is not None and _seed(scene) not in seed_filter:
            continue
        with np.load(Path(str(record["output"]))) as oracle:
            schedule = np.asarray(oracle["labels"], dtype=np.int64)
            features, utilities = _counterfactual_sequence(
                oracle,
                schedule,
                use_flow=use_flow,
                motion_delta=False,
                rpm_values=rpm_values,
                objective=objective,
            )
            values, targets = _windows(features, utilities, window)
            if len(targets) == 0:
                continue
            x_parts.append(values)
            y_parts.append(targets)
            groups.append(np.full(len(targets), str(record["condition"]), dtype=object))
            scene_reports.append({
                "scene": scene,
                "condition": str(record["condition"]),
                "seed": _seed(scene),
                "windows": int(len(targets)),
            })
    if not x_parts:
        raise ValueError(f"No oracle windows found for split {split!r}.")
    return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(groups), scene_reports


def _headroom(targets: np.ndarray, baseline_gear: int) -> dict[str, float]:
    best = np.max(targets, axis=1)
    baseline = targets[:, baseline_gear]
    gain = best - baseline
    return {
        "mean_best_utility": float(np.mean(best)),
        "mean_fixed180_utility": float(np.mean(baseline)),
        "mean_best_minus_fixed180": float(np.mean(gain)),
        "median_best_minus_fixed180": float(np.median(gain)),
        "positive_fraction": float(np.mean(gain > 1e-6)),
        "large_gain_fraction_gt_005": float(np.mean(gain > 0.05)),
        "p95_gain": float(np.quantile(gain, 0.95)),
        "best_action_accuracy_of_fixed180": float(np.mean(np.argmax(targets, axis=1) == baseline_gear)),
    }


def _standardize(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-4, 1.0, std)
    return np.clip((train_x - mean) / std, -5.0, 5.0), np.clip((other_x - mean) / std, -5.0, 5.0)


def _prediction_metrics(predicted: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    oracle = np.argmax(targets, axis=1)
    rows = np.arange(len(targets))
    selected = targets[rows, predicted]
    best = targets[rows, oracle]
    return {
        "top1_accuracy": float(np.mean(predicted == oracle)),
        "mean_utility_regret": float(np.mean(best - selected)),
        "p95_utility_regret": float(np.quantile(best - selected, 0.95)),
        "mean_selected_utility": float(np.mean(selected)),
    }


def _nearest_neighbor(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
    train_flat = train_x.reshape(len(train_x), -1).astype(np.float64)
    eval_flat = eval_x.reshape(len(eval_x), -1).astype(np.float64)
    predictions = np.empty(len(eval_x), dtype=np.int64)
    chunk = 256
    for start in range(0, len(eval_x), chunk):
        values = eval_flat[start : start + chunk]
        distances = np.sum(np.square(values[:, None, :] - train_flat[None, :, :]), axis=-1)
        nearest = np.argmin(distances, axis=1)
        predictions[start : start + len(values)] = train_y[nearest]
    return predictions


class _FlatMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def _fit_mlp(
    train_x: np.ndarray,
    train_targets: np.ndarray,
    eval_x: np.ndarray,
    *,
    epochs: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    labels = np.argmax(train_targets, axis=1).astype(np.int64)
    values = torch.from_numpy(train_x.reshape(len(train_x), -1).astype(np.float32))
    target = torch.from_numpy(labels)
    model = _FlatMLP(values.shape[-1], 64, train_targets.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(values), target)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(eval_x.reshape(len(eval_x), -1).astype(np.float32)))
    return torch.argmax(logits, dim=-1).numpy()


def _scene_learning_curve(
    records: list[dict[str, object]],
    objective: dict[str, float],
    *,
    window: int,
    use_flow: bool,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    train_seeds: list[int],
    epochs: int,
    seed: int,
) -> list[dict[str, object]]:
    fractions = (0.25, 0.50, 1.0)
    rows: list[dict[str, object]] = []
    for fraction in fractions:
        count = max(1, int(np.ceil(len(train_seeds) * fraction)))
        selected_seeds = set(train_seeds[:count])
        train_x, train_y, _, reports = _load_split(records, split="train", objective=objective, window=window, use_flow=use_flow, seed_filter=selected_seeds)
        train_xn, valid_xn = _standardize(train_x, valid_x)
        train_labels = np.argmax(train_y, axis=1).astype(np.int64)
        valid_labels = np.argmax(valid_y, axis=1).astype(np.int64)
        nn_train = _nearest_neighbor(train_xn, train_labels, train_xn)
        nn_valid = _nearest_neighbor(train_xn, train_labels, valid_xn)
        mlp_train = _fit_mlp(train_xn, train_y, train_xn, epochs=epochs, seed=seed + count)
        mlp_valid = _fit_mlp(train_xn, train_y, valid_xn, epochs=epochs, seed=seed + count)
        rows.append({
            "fraction": fraction,
            "train_seeds": sorted(selected_seeds),
            "train_scenes": len(reports),
            "train_windows": len(train_y),
            "nearest_neighbor": {"train": _prediction_metrics(nn_train, train_y), "valid": _prediction_metrics(nn_valid, valid_y)},
            "mlp": {"train": _prediction_metrics(mlp_train, train_y), "valid": _prediction_metrics(mlp_valid, valid_y)},
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/comotion_v2_learned/manifest.json"))
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results/data_vs_method_diagnostic.json"))
    args = parser.parse_args()
    if args.window < 2 or args.epochs < 1:
        parser.error("window must be >= 2 and epochs must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = list(manifest["oracles"])
    objective = manifest["configuration"]
    use_flow = True
    train_x, train_y, train_groups, train_reports = _load_split(records, split="train", objective=objective, window=args.window, use_flow=use_flow)
    valid_x, valid_y, valid_groups, valid_reports = _load_split(records, split="valid", objective=objective, window=args.window, use_flow=use_flow)
    test_x, test_y, test_groups, test_reports = _load_split(records, split="test", objective=objective, window=args.window, use_flow=use_flow)
    baseline_gear = int(np.argmin(np.abs(720.0 * GEAR_SCALES - 180.0)))

    split_data = {"train": (train_x, train_y, train_groups), "valid": (valid_x, valid_y, valid_groups), "test": (test_x, test_y, test_groups)}
    headroom = {split: _headroom(values[1], baseline_gear) for split, values in split_data.items()}
    condition_headroom: dict[str, dict[str, float]] = {}
    for condition in sorted(set(test_groups.tolist())):
        condition_headroom[condition] = _headroom(test_y[test_groups == condition], baseline_gear)
    label_stats: dict[str, object] = {}
    for split, (_, targets, groups) in split_data.items():
        labels = np.argmax(targets, axis=1)
        by_condition: dict[str, object] = {}
        for condition in sorted(set(groups.tolist())):
            selected = labels[groups == condition]
            counts = np.bincount(selected, minlength=len(GEAR_SCALES)).tolist()
            by_condition[condition] = {"counts": counts, "entropy_bits": _entropy(selected, len(GEAR_SCALES)), "samples": int(len(selected))}
        label_stats[split] = {"counts": np.bincount(labels, minlength=len(GEAR_SCALES)).tolist(), "entropy_bits": _entropy(labels, len(GEAR_SCALES)), "by_condition": by_condition}

    train_xn, valid_xn = _standardize(train_x, valid_x)
    _, test_xn = _standardize(train_x, test_x)
    train_labels = np.argmax(train_y, axis=1).astype(np.int64)
    nn_valid = _nearest_neighbor(train_xn, train_labels, valid_xn)
    nn_test = _nearest_neighbor(train_xn, train_labels, test_xn)
    mlp_valid = _fit_mlp(train_xn, train_y, valid_xn, epochs=args.epochs, seed=7)
    mlp_test = _fit_mlp(train_xn, train_y, test_xn, epochs=args.epochs, seed=7)
    train_seeds = sorted({_seed(str(report["scene"])) for report in train_reports})
    curve = _scene_learning_curve(records, objective, window=args.window, use_flow=use_flow, valid_x=valid_x, valid_y=valid_y, train_seeds=train_seeds, epochs=args.epochs, seed=7)

    report = {
        "diagnostic": "data_vs_method_offline_oracle_path",
        "manifest": str(args.manifest),
        "window": args.window,
        "candidate_rpm": (720.0 * GEAR_SCALES).tolist(),
        "fixed180_gear": baseline_gear,
        "data_geometry": {
            "splits": {
                name: {
                    "scenes": len(reports),
                    "windows": int(len(values[1])),
                    "seeds": sorted({_seed(str(item["scene"])) for item in reports}),
                    "conditions": dict(Counter(groups.tolist())),
                }
                for name, (values, groups, reports) in {
                    "train": ((train_x, train_y), train_groups, train_reports),
                    "valid": ((valid_x, valid_y), valid_groups, valid_reports),
                    "test": ((test_x, test_y), test_groups, test_reports),
                }.items()
            },
            "feature_shape": list(train_x.shape),
            "flattened_feature_dim": int(np.prod(train_x.shape[1:])),
        },
        "adaptive_headroom": headroom,
        "test_headroom_by_condition": condition_headroom,
        "oracle_label_statistics": label_stats,
        "same_feature_method_checks": {
            "nearest_neighbor": {"valid": _prediction_metrics(nn_valid, valid_y), "test": _prediction_metrics(nn_test, test_y)},
            "flat_mlp": {"valid": _prediction_metrics(mlp_valid, valid_y), "test": _prediction_metrics(mlp_test, test_y)},
        },
        "scene_count_learning_curve": curve,
        "interpretation_rules": {
            "data_limited": "validation performance rises materially as train scenes increase, while train performance is already high",
            "state_oracle_ambiguous": "nearest-neighbor train is high but valid remains low and headroom is positive",
            "method_limited": "a simple same-feature MLP/nearest-neighbor model beats the deployed LSTM/Q policy at matched offline targets",
            "no_adaptive_signal": "best-minus-fixed180 headroom is near zero across splits",
        },
        "limitations": [
            "Offline targets come from phase-continuous oracle tables along the selected oracle path.",
            "This diagnostic does not establish downstream feature matching or causal rollout performance.",
            "The synthetic split has only six independent training seeds in v2; windows within a scene are correlated.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
