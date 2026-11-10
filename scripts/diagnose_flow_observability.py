#!/usr/bin/env python3
"""Measure whether region-conditioned optical flow exposes relative motion.

This is an observability upper-bound diagnostic: object masks come from the
simulator and must not be presented as deployable controller inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as imageio
import numpy as np

from ami_simulator.flow import FLOW_FEATURE_NAMES, MOTION_CLUSTER_FEATURE_NAMES, TILED_FLOW_FEATURE_NAMES, estimate_dense_flow, summarize_flow, summarize_motion_clusters, summarize_residual_flow, summarize_tiled_flow


CONDITIONS = ("camera_only", "object_only", "co_motion", "counter_motion")


def _records(manifest: dict[str, object], split: str) -> list[dict[str, object]]:
    records = manifest.get("records", manifest.get("oracles", []))
    return [record for record in records if record["split"] == split]


def _scene_features(root: Path, record: dict[str, object]) -> dict[str, np.ndarray]:
    scene = root / str(record["scene"])
    images = sorted((scene / "images").glob("*.png"))
    masks = sorted((scene / "object_masks").glob("*.png"))
    if len(images) != len(masks) or len(images) < 2:
        raise ValueError(f"Scene {scene} needs matching image/mask sequences.")
    values: dict[str, list[np.ndarray]] = {"global": [], "tiled_2x2": [], "tiled_4x4": [], "motion_clusters": [], "object_roi": [], "background_roi": [], "roi_residual": [], "dominant_residual": []}
    for index in range(len(images) - 1):
        previous = imageio.imread(images[index])
        current = imageio.imread(images[index + 1])
        flow = estimate_dense_flow(previous, current)
        object_mask = np.asarray(imageio.imread(masks[index])) > 0
        background_mask = ~object_mask
        global_features = summarize_flow(flow, previous)
        object_features = summarize_flow(flow, previous, object_mask)
        background_features = summarize_flow(flow, previous, background_mask)
        values["global"].append(global_features)
        values["tiled_2x2"].append(summarize_tiled_flow(flow, previous, rows=2, cols=2))
        values["tiled_4x4"].append(summarize_tiled_flow(flow, previous, rows=4, cols=4))
        values["motion_clusters"].append(summarize_motion_clusters(flow, clusters=3))
        values["object_roi"].append(object_features)
        values["background_roi"].append(background_features)
        # A compact relative-motion descriptor: subtract region summaries and
        # retain only displacement/magnitude/coherence terms with finite values.
        residual = object_features - background_features
        values["roi_residual"].append(np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0))
        values["dominant_residual"].append(summarize_residual_flow(flow, previous))
    return {key: np.stack(items).astype(np.float32) for key, items in values.items()}


def _nearest_centroid(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = np.where(train.std(axis=0) < 1e-6, 1.0, train.std(axis=0))
    train_norm = (train - mean) / std
    test_norm = (test - mean) / std
    centroids = np.stack([train_norm[train_labels == index].mean(axis=0) for index in range(len(CONDITIONS))])
    distances = np.linalg.norm(test_norm[:, None, :] - centroids[None, :, :], axis=-1)
    return np.argmin(distances, axis=1), distances


def _binary_centroid_accuracy(train: np.ndarray, labels: np.ndarray, test: np.ndarray, test_labels: np.ndarray) -> float:
    mean = train.mean(axis=0)
    std = np.where(train.std(axis=0) < 1e-6, 1.0, train.std(axis=0))
    train_norm, test_norm = (train - mean) / std, (test - mean) / std
    centroids = np.stack([train_norm[labels == index].mean(axis=0) for index in (0, 1)])
    prediction = np.argmin(np.linalg.norm(test_norm[:, None, :] - centroids[None, :, :], axis=-1), axis=1)
    return float(np.mean(prediction == test_labels))


def _evaluate(train_rows: list[dict[str, object]], test_rows: list[dict[str, object]], train_features: list[dict[str, np.ndarray]], test_features: list[dict[str, np.ndarray]], representation: str) -> dict[str, object]:
    train_values = np.concatenate([item[representation] for item in train_features])
    test_values = np.concatenate([item[representation] for item in test_features])
    train_labels = np.concatenate([np.full(len(item[representation]), CONDITIONS.index(str(row["condition"])), dtype=np.int64) for row, item in zip(train_rows, train_features)])
    test_labels = np.concatenate([np.full(len(item[representation]), CONDITIONS.index(str(row["condition"])), dtype=np.int64) for row, item in zip(test_rows, test_features)])
    predicted, distances = _nearest_centroid(train_values, train_labels, test_values)
    confusion = np.zeros((len(CONDITIONS), len(CONDITIONS)), dtype=np.int64)
    for actual, guess in zip(test_labels, predicted):
        confusion[actual, guess] += 1
    per_condition = {condition: float(np.mean(predicted[test_labels == index] == index)) for index, condition in enumerate(CONDITIONS)}
    pairwise: dict[str, float] = {}
    for left_index in range(len(CONDITIONS)):
        for right_index in range(left_index + 1, len(CONDITIONS)):
            mask_train = np.isin(train_labels, (left_index, right_index))
            mask_test = np.isin(test_labels, (left_index, right_index))
            pair_train_labels = (train_labels[mask_train] == right_index).astype(np.int64)
            pair_test_labels = (test_labels[mask_test] == right_index).astype(np.int64)
            pairwise[f"{CONDITIONS[left_index]}_vs_{CONDITIONS[right_index]}"] = _binary_centroid_accuracy(train_values[mask_train], pair_train_labels, test_values[mask_test], pair_test_labels)
    return {
        "representation": representation,
        "feature_names": list(MOTION_CLUSTER_FEATURE_NAMES if representation == "motion_clusters" else TILED_FLOW_FEATURE_NAMES if representation == "tiled_2x2" else tuple(f"tile_{index:02d}_{name}" for index in range(16) for name in FLOW_FEATURE_NAMES) if representation == "tiled_4x4" else FLOW_FEATURE_NAMES),
        "feature_dim": int(train_values.shape[1]),
        "train_windows": int(len(train_values)),
        "test_windows": int(len(test_values)),
        "accuracy": float(np.mean(predicted == test_labels)),
        "per_condition_accuracy": per_condition,
        "pairwise_accuracy": pairwise,
        "confusion_matrix_rows_actual": confusion.tolist(),
        "mean_nearest_centroid_distance": float(np.mean(np.min(distances, axis=1))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_rows, test_rows = _records(manifest, "train"), _records(manifest, "test")
    train_features = [_scene_features(args.dataset_root, row) for row in train_rows]
    test_features = [_scene_features(args.dataset_root, row) for row in test_rows]
    representations = {}
    for representation in ("global", "tiled_2x2", "tiled_4x4", "motion_clusters", "object_roi", "background_roi", "roi_residual", "dominant_residual"):
        representations[representation] = _evaluate(train_rows, test_rows, train_features, test_features, representation)
    condition_means = {}
    for condition in CONDITIONS:
        condition_means[condition] = {}
        for representation in representations:
            values = np.concatenate([item[representation] for row, item in zip(test_rows, test_features) if row["condition"] == condition])
            condition_means[condition][representation] = np.mean(values, axis=0).astype(float).round(6).tolist()
    report = {
        "diagnostic": "region-conditioned optical-flow observability upper bound",
        "mask_source": "simulator object masks; diagnostic only, not deployable input",
        "dataset": str(args.manifest),
        "train_scenes": len(train_rows),
        "test_scenes": len(test_rows),
        "condition_feature_means_test": condition_means,
        "representations": representations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
