#!/usr/bin/env python3
"""Train a small causal gear policy to imitate the deterministic IMU selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ami_simulator.dsec import read_imu_csv
from ami_simulator.motion_net import TinyCausalGearPolicy
from ami_simulator.rotation import ImuGearRotation, motion_adaptive_hysteresis, motion_quantile_thresholds
from ami_simulator.types import ImuStream


def make_features(gyro: np.ndarray, accel: np.ndarray, *, gyro_scale: float = 0.1) -> np.ndarray:
    """Build normalized causal IMU features; no future samples are used."""
    if not np.isfinite(gyro_scale) or gyro_scale <= 0.0:
        raise ValueError("gyro_scale must be finite and positive.")
    gyro = np.asarray(gyro, dtype=np.float32)
    accel = np.asarray(accel, dtype=np.float32)
    speed = np.linalg.norm(gyro, axis=1, keepdims=True)
    accel_norm = np.linalg.norm(accel, axis=1, keepdims=True)
    return np.concatenate((gyro / gyro_scale, accel / 10.0, speed / gyro_scale, accel_norm / 10.0), axis=1).astype(np.float32)


def windows(features: np.ndarray, labels: np.ndarray, *, length: int, stride: int, split: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x, train_y, valid_x, valid_y = [], [], [], []
    for end in range(length - 1, len(features), stride):
        start = end - length + 1
        item = features[start : end + 1]
        target = int(labels[end])
        if end < split:
            train_x.append(item)
            train_y.append(target)
        else:
            valid_x.append(item)
            valid_y.append(target)
    if not train_x:
        raise ValueError("The IMU sequence is too short for the requested window and split.")
    valid_array = np.stack(valid_x) if valid_x else np.empty((0, length, features.shape[-1]), dtype=features.dtype)
    return np.stack(train_x), np.asarray(train_y, dtype=np.int64), valid_array, np.asarray(valid_y, dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imu-csv", type=Path, action="append", required=True, help="IMU CSV; repeat for multiple scenes.")
    parser.add_argument("--oracle", type=Path, action="append", default=[], help="Optional event-aware oracle .npz; repeat in the same order as --imu-csv.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-mode", choices=["temporal", "random"], default="temporal")
    parser.add_argument("--adaptive-thresholds", action="store_true", help="Derive four gear thresholds from scene IMU quantiles.")
    args = parser.parse_args()
    if args.window < 4 or args.stride < 1 or args.epochs < 1:
        parser.error("window must be >= 4, stride and epochs must be positive")
    if args.oracle and len(args.oracle) != len(args.imu_csv):
        parser.error("Provide exactly one --oracle per --imu-csv when oracle labels are enabled.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    scene_windows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    scene_reports: list[dict[str, object]] = []
    for scene_index, imu_path in enumerate(args.imu_csv):
        imu = read_imu_csv(imu_path, timestamp_unit="us", gyro_unit="rad/s")
        split = int(len(imu.t) * 0.8)
        reference = imu
        if args.adaptive_thresholds and args.split_mode == "temporal":
            reference = ImuStream(imu.t[:split], imu.accel[:split], imu.gyro[:split])
        thresholds = motion_quantile_thresholds(reference) if args.adaptive_thresholds else (0.025, 0.032, 0.040, 0.048)
        hysteresis = motion_adaptive_hysteresis(reference) if args.adaptive_thresholds else 0.01
        selector = ImuGearRotation(imu, speed_thresholds_rad_s=thresholds, hysteresis_rad_s=hysteresis)
        labels = selector.gear_indices
        label_source = "fsm"
        if args.oracle:
            oracle = np.load(args.oracle[scene_index])
            oracle_times = np.asarray(oracle["timestamps"], dtype=np.float64)
            oracle_labels = np.asarray(oracle["labels"], dtype=np.int64)
            if len(oracle_times) != len(oracle_labels) or len(oracle_times) < 1 or not np.all(np.diff(oracle_times) > 0.0):
                raise ValueError(f"Invalid oracle timestamps/labels in {args.oracle[scene_index]}")
            labels = oracle_labels[np.clip(np.searchsorted(oracle_times, imu.t, side="left"), 0, len(oracle_labels) - 1)]
            label_source = "event_oracle"
        motion_scale = float(np.quantile(np.linalg.norm(reference.gyro, axis=1), 0.8)) if args.adaptive_thresholds else 0.1
        features = make_features(imu.gyro, imu.accel, gyro_scale=max(motion_scale, 1e-3))
        if args.split_mode == "temporal":
            split_data = windows(features, labels, length=args.window, stride=args.stride, split=split)
        else:
            all_x, all_y, _, _ = windows(features, labels, length=args.window, stride=args.stride, split=len(features))
            split_data = (all_x, all_y, np.empty((0, args.window, features.shape[-1]), dtype=features.dtype), np.empty((0,), dtype=np.int64))
        scene_windows.append(split_data)
        scene_name = imu_path.parent.name if imu_path.name == "imu.csv" else imu_path.stem
        scene_reports.append({"scene": scene_name, "samples": len(imu.t), "label_source": label_source, "gyro_scale_rad_s": motion_scale, "thresholds_rad_s": list(thresholds), "hysteresis_rad_s": hysteresis, "gear_counts": np.bincount(labels, minlength=5).tolist()})

    train_x = np.concatenate([item[0] for item in scene_windows], axis=0)
    train_y = np.concatenate([item[1] for item in scene_windows], axis=0)
    temporal_valid_x = [item[2] for item in scene_windows if len(item[2])]
    temporal_valid_y = [item[3] for item in scene_windows if len(item[3])]
    if args.split_mode == "temporal":
        valid_x = np.concatenate(temporal_valid_x, axis=0) if temporal_valid_x else np.empty((0, args.window, train_x.shape[-1]), dtype=train_x.dtype)
        valid_y = np.concatenate(temporal_valid_y, axis=0) if temporal_valid_y else np.empty((0,), dtype=np.int64)
    else:
        order = np.random.default_rng(args.seed).permutation(len(train_y))
        cut = max(1, int(len(order) * 0.8))
        valid_x, valid_y = train_x[order[cut:]], train_y[order[cut:]]
        train_x, train_y = train_x[order[:cut]], train_y[order[:cut]]
    if not len(valid_y):
        raise ValueError("The requested split produced no validation windows.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCausalGearPolicy(feature_dim=train_x.shape[-1], hidden_dim=16, num_gears=5).to(device)
    class_counts = np.bincount(train_y, minlength=model.num_gears).astype(np.float32)
    weights = np.zeros_like(class_counts)
    active = class_counts > 0
    weights[active] = class_counts[active].sum() / (active.sum() * class_counts[active])
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_values = torch.from_numpy(train_x).to(device)
    train_targets = torch.from_numpy(train_y).to(device)
    valid_values = torch.from_numpy(valid_x).to(device)
    valid_targets = torch.from_numpy(valid_y).to(device)
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_values), train_targets)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        train_pred = model(train_values).argmax(dim=-1)
        valid_pred = model(valid_values).argmax(dim=-1)
        train_accuracy = float((train_pred == train_targets).float().mean().item())
        valid_accuracy = float((valid_pred == valid_targets).float().mean().item())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "feature_dim": model.feature_dim, "hidden_dim": model.hidden_dim, "num_gears": model.num_gears}, args.output)
    majority_class = int(np.bincount(train_y, minlength=model.num_gears).argmax())
    valid_majority = float(np.mean(valid_y == majority_class))
    report = {"device": str(device), "split_mode": args.split_mode, "scenes": scene_reports, "samples": {"train": len(train_y), "valid": len(valid_y)}, "train_gear_counts": np.bincount(train_y, minlength=model.num_gears).tolist(), "valid_gear_counts": np.bincount(valid_y, minlength=model.num_gears).tolist(), "train_accuracy": train_accuracy, "valid_accuracy": valid_accuracy, "valid_majority_baseline": valid_majority, "output": str(args.output)}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
