#!/usr/bin/env python3
"""Evaluate lightweight downstream feature matching after causal control.

This is deliberately an evaluation-only module. It uses a fixed OpenCV ORB
configuration on the clean RGB frame and the compensated event image from each
causal window, so controller architecture and simulator physics are unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.motion_net import CausalAnchorResidualPolicy
from scripts.evaluate_anchor_residual_policy import ANCHOR_RPM, FEATURE_DIM, RESIDUAL_RPM, _load_scene, _state
from scripts.evaluate_phase_continuous_control import _simulate_action


def _event_image(events: object, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    image = np.zeros((height, width), dtype=np.float32)
    x = np.asarray(events.x, dtype=np.int64)
    y = np.asarray(events.y, dtype=np.int64)
    p = np.asarray(events.p, dtype=np.float32)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    np.add.at(image, (y[valid], x[valid]), p[valid])
    # Signed log compression preserves polarity edges while avoiding count
    # magnitude becoming a proxy for feature quality.
    return np.clip(127.0 + 48.0 * np.tanh(image / 4.0), 0.0, 255.0).astype(np.uint8)


def _feature_metrics(rgb: np.ndarray, events: object) -> dict[str, float | int]:
    reference = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    event_image = _event_image(events, reference.shape)
    orb = cv2.ORB_create(nfeatures=300, fastThreshold=5, edgeThreshold=5)
    keypoints_rgb, descriptors_rgb = orb.detectAndCompute(reference, None)
    keypoints_event, descriptors_event = orb.detectAndCompute(event_image, None)
    rgb_count = len(keypoints_rgb)
    event_count = len(keypoints_event)
    rgb_points = np.float32([item.pt for item in keypoints_rgb]) if rgb_count else np.empty((0, 2), dtype=np.float32)
    event_points = np.float32([item.pt for item in keypoints_event]) if event_count else np.empty((0, 2), dtype=np.float32)
    # Greedy one-to-one matching in the known shared reference plane. This is
    # stricter than nearest-neighbour repeatability because each keypoint can
    # be claimed only once.
    spatial_matches = 0
    if rgb_count and event_count:
        distances = np.linalg.norm(event_points[:, None, :] - rgb_points[None, :, :], axis=-1)
        candidates = [(float(distances[event_index, rgb_index]), event_index, rgb_index) for event_index, rgb_index in zip(*np.where(distances <= 3.0))]
        used_event: set[int] = set()
        used_rgb: set[int] = set()
        for _, event_index, rgb_index in sorted(candidates):
            if event_index not in used_event and rgb_index not in used_rgb:
                used_event.add(event_index)
                used_rgb.add(rgb_index)
        spatial_matches = len(used_event)
    spatial_repeatability = float(spatial_matches / min(rgb_count, event_count)) if min(rgb_count, event_count) else 0.0
    spatial_precision = float(spatial_matches / event_count) if event_count else 0.0
    spatial_recall = float(spatial_matches / rgb_count) if rgb_count else 0.0
    spatial_f1 = float(2.0 * spatial_precision * spatial_recall / (spatial_precision + spatial_recall)) if spatial_precision + spatial_recall else 0.0
    if descriptors_rgb is None or descriptors_event is None or not rgb_count or not event_count:
        return {"rgb_keypoints": rgb_count, "event_keypoints": event_count, "raw_matches": 0, "good_matches": 0, "inlier_matches": 0, "repeatability": spatial_repeatability, "matching_precision": 0.0, "matching_recall": 0.0, "inlier_ratio": 0.0, "spatial_matches": spatial_matches, "spatial_repeatability": spatial_repeatability, "spatial_precision": spatial_precision, "spatial_recall": spatial_recall, "spatial_f1": spatial_f1, "descriptor_true_matches": 0, "descriptor_true_precision": 0.0, "descriptor_true_recall": 0.0}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = matcher.knnMatch(descriptors_event, descriptors_rgb, k=2)
    good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    # Geometric inliers are evaluated in the shared compensated reference
    # plane. Homography is only attempted when enough correspondences exist.
    # Descriptor matches are considered true only when their coordinates also
    # agree in the known reference plane. No homography is fitted, so object
    # motion cannot be hidden by a flexible geometric model.
    descriptor_true = [item for item in good if np.linalg.norm(event_points[item.queryIdx] - rgb_points[item.trainIdx]) <= 3.0]
    inlier_count = len(descriptor_true)
    raw_matches = len(pairs)
    return {
        "rgb_keypoints": rgb_count,
        "event_keypoints": event_count,
        "raw_matches": raw_matches,
        "good_matches": len(good),
        "inlier_matches": inlier_count,
        "repeatability": spatial_repeatability,
        "matching_precision": float(inlier_count / len(good)) if good else 0.0,
        "matching_recall": float(inlier_count / rgb_count) if rgb_count else 0.0,
        "inlier_ratio": float(inlier_count / len(good)) if good else 0.0,
        "spatial_matches": spatial_matches,
        "spatial_repeatability": spatial_repeatability,
        "spatial_precision": spatial_precision,
        "spatial_recall": spatial_recall,
        "spatial_f1": spatial_f1,
        "descriptor_true_matches": len(descriptor_true),
        "descriptor_true_precision": float(len(descriptor_true) / len(good)) if good else 0.0,
        "descriptor_true_recall": float(len(descriptor_true) / rgb_count) if rgb_count else 0.0,
    }


def _run_scene(
    luminance: np.ndarray,
    timestamps: np.ndarray,
    masks: np.ndarray,
    intrinsics,
    oracle: np.lib.npyio.NpzFile,
    objective: dict[str, float],
    *,
    model: CausalAnchorResidualPolicy | None,
    checkpoint: dict[str, object] | None,
    margin: float,
    confidence: float,
    min_event_efficiency: float,
) -> dict[str, float | int | str]:
    prism = RotatingWedgePrism(intrinsics, 2.0)
    phase = 0.0
    previous_active = None
    previous_metrics: dict[str, float] | None = None
    previous_rpm = ANCHOR_RPM
    dwell_seconds = 0.0
    states: list[np.ndarray] = []
    values: list[dict[str, float | int]] = []
    selected_rpm: list[float] = []
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32) if checkpoint is not None else None
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32) if checkpoint is not None else None
    window = int(checkpoint["window"]) if checkpoint is not None else 0
    for index in range(len(timestamps) - 1):
        state = _state(oracle, index=index, previous=previous_metrics, phase=phase, dwell_seconds=dwell_seconds, previous_rpm=previous_rpm)
        states.append(state)
        rpm = ANCHOR_RPM
        if model is not None and index >= window - 1:
            history = np.stack(states[-window:])
            normalized = np.clip((history - mean) / std, -float(checkpoint["feature_clip"]), float(checkpoint["feature_clip"]))
            with torch.inference_mode():
                predicted_delta, switch_logit = model(torch.from_numpy(normalized[None].astype(np.float32)))
            delta = predicted_delta[0].numpy()
            probability = float(torch.sigmoid(switch_logit[0]).item())
            previous_efficiency = 1.0 if previous_metrics is None else float(previous_metrics["event_efficiency"])
            if float(np.max(delta)) > margin and probability > confidence and previous_efficiency >= min_event_efficiency:
                rpm = RESIDUAL_RPM[int(np.argmax(delta))]
        selected = _simulate_action(
            luminance,
            timestamps,
            masks,
            prism,
            index=index,
            rpm=float(rpm),
            phase=phase,
            previous_active=previous_active,
            previous_rpm=previous_rpm,
            target_density=float(oracle["target_density"][index]),
            objective=objective,
            contrast_threshold=0.2,
            max_phase_step_deg=30.0,
        )
        values.append(_feature_metrics(luminance[index], selected["reference_events"]))
        selected_rpm.append(float(rpm))
        dt = float(timestamps[index + 1] - timestamps[index])
        previous_metrics = {key: float(selected["metrics"][key]) for key in ("event_density", "edge_coverage", "structural_ratio", "unique_pixel_ratio", "temporal_overlap", "polarity_imbalance", "event_efficiency")}
        previous_active = selected["active"]
        previous_rpm = float(rpm)
        phase = float(selected["phase_end"])
        dwell_seconds = dt if index and rpm != selected_rpm[-2] else dwell_seconds + dt
    keys = ("repeatability", "matching_precision", "matching_recall", "inlier_ratio", "spatial_matches", "spatial_repeatability", "spatial_precision", "spatial_recall", "spatial_f1", "descriptor_true_matches", "descriptor_true_precision", "descriptor_true_recall", "rgb_keypoints", "event_keypoints", "raw_matches", "good_matches", "inlier_matches")
    return {"policy": "anchor_residual" if model is not None else "fixed_180", "mean_rpm": float(np.mean(selected_rpm)), **{f"mean_{key}": float(np.mean([float(item[key]) for item in values])) for key in keys}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/comotion_v2_learned/manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/comotion_controlled_v2"))
    parser.add_argument("--policy", type=Path, default=Path("results/comotion_anchor_residual_policy/policy_full_anchor.pt"))
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--switch-margin", type=float, default=0.02)
    parser.add_argument("--switch-confidence", type=float, default=0.70)
    parser.add_argument("--min-event-efficiency", type=float, default=0.70)
    parser.add_argument("--output", type=Path, default=Path("results/comotion_anchor_residual_policy/feature_matching_test.json"))
    args = parser.parse_args()
    if args.switch_margin < 0.0 or not 0.0 <= args.switch_confidence <= 1.0 or not 0.0 <= args.min_event_efficiency <= 1.0:
        parser.error("invalid gate arguments")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.policy, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "anchor_residual_mlp" or checkpoint.get("feature_dim") != FEATURE_DIM:
        raise ValueError("The feature evaluator expects a utility-only anchor-residual checkpoint.")
    model = CausalAnchorResidualPolicy(feature_dim=FEATURE_DIM, hidden_dim=int(checkpoint["hidden_dim"]), residual_actions=int(checkpoint["residual_actions"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows: list[dict[str, object]] = []
    records = [record for record in manifest["oracles"] if record["split"] == args.split]
    for record in records:
        luminance, timestamps, masks, intrinsics, oracle = _load_scene(args.dataset_root / str(record["scene"]), Path(str(record["output"])))
        try:
            for policy in ("fixed_180", "anchor_residual"):
                result = _run_scene(luminance, timestamps, masks, intrinsics, oracle, manifest["configuration"], model=model if policy == "anchor_residual" else None, checkpoint=checkpoint if policy == "anchor_residual" else None, margin=float(args.switch_margin), confidence=float(args.switch_confidence), min_event_efficiency=float(args.min_event_efficiency))
                rows.append({"scene": record["scene"], "condition": record["condition"], **result})
        finally:
            oracle.close()
        print(f"evaluated {record['scene']}")
    aggregate: dict[str, dict[str, float]] = {}
    metrics = ("repeatability", "matching_precision", "matching_recall", "inlier_ratio", "spatial_matches", "spatial_repeatability", "spatial_precision", "spatial_recall", "spatial_f1", "descriptor_true_matches", "descriptor_true_precision", "descriptor_true_recall", "rgb_keypoints", "event_keypoints", "raw_matches", "good_matches", "inlier_matches", "mean_rpm")
    for policy in ("fixed_180", "anchor_residual"):
        selected = [row for row in rows if row["policy"] == policy]
        aggregate[policy] = {key: float(np.mean([float(row[f"mean_{key}"] if f"mean_{key}" in row else row[key]) for row in selected])) for key in metrics}
    report = {"evaluation": "ORB feature detection and matching on causal event windows", "split": args.split, "policy": str(args.policy), "switch_margin": args.switch_margin, "switch_confidence": args.switch_confidence, "min_event_efficiency": args.min_event_efficiency, "orb": {"nfeatures": 300, "fastThreshold": 5, "edgeThreshold": 5, "ratio_test": 0.75, "ransac_reprojection_threshold": 3.0}, "aggregate": aggregate, "delta_anchor_residual_vs_fixed180": {key: float(aggregate["anchor_residual"][key] - aggregate["fixed_180"][key]) for key in metrics}, "rows": rows, "limitations": ["ORB is a lightweight downstream proxy, not a trained event feature extractor.", "Reference events are evaluated after the existing per-event prism compensation in the shared reference plane.", "The test split is used only after gate settings are fixed on validation."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
