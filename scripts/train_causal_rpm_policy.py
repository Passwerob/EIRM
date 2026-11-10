#!/usr/bin/env python3
"""Train a causal RPM policy from redundancy-aware optical-excitation oracles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ami_simulator.motion_net import CausalFlowFusionGearPolicy, CausalFlowFusionRPMPolicy, CausalRPMPolicy


BASE_FEATURE_DIM = 9
FLOW_FEATURE_DIM = 8


def _oracle_features(oracle: np.lib.npyio.NpzFile, *, base_rpm: float, gear_scales: np.ndarray, use_flow: bool, motion_input: str = "auto") -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    labels = np.asarray(oracle["labels"], dtype=np.int64).reshape(-1)
    if "selected_rpm" in oracle:
        target_rpm = np.asarray(oracle["selected_rpm"], dtype=np.float32).reshape(-1)
    else:
        target_rpm = (base_rpm * gear_scales[labels]).astype(np.float32)
    if len(labels) != len(target_rpm):
        raise ValueError("Oracle labels and selected_rpm must have the same length.")
    if motion_input == "imu" or (motion_input == "auto" and use_flow):
        motion_key = "imu_motion" if "imu_motion" in oracle else "motion"
    else:
        motion_key = "natural_motion" if "natural_motion" in oracle else "motion"
    motion = np.asarray(oracle[motion_key], dtype=np.float32).reshape(-1) if motion_key in oracle else np.zeros(len(labels), np.float32)
    if len(motion) != len(labels):
        raise ValueError("Oracle motion and labels must have the same length.")

    def field(name: str, fallback: np.ndarray) -> np.ndarray:
        values = np.asarray(oracle[name], dtype=np.float32) if name in oracle else fallback
        if values.shape != (len(labels), len(gear_scales)):
            raise ValueError(f"Oracle field {name!r} must have shape [windows, gears].")
        return values

    density = field("densities", np.zeros((len(labels), len(gear_scales)), np.float32))
    coverage = field("coverage", np.zeros_like(density))
    structural = field("structural_ratio", np.ones_like(density))
    unique = field("unique_ratio", coverage)
    event_efficiency = field(
        "event_efficiency",
        np.divide(unique, density, out=np.zeros_like(unique), where=density > 0.0),
    )
    overlap = field("temporal_overlap", np.zeros_like(density))
    polarity = field("polarity_balance", np.zeros_like(density))
    selected = np.clip(labels, 0, len(gear_scales) - 1)
    # Features at i contain only measurements available before command i:
    # shift selected previous-command statistics by one window.
    feature_dim = BASE_FEATURE_DIM + (FLOW_FEATURE_DIM if use_flow else 0)
    features = np.zeros((len(labels), feature_dim), dtype=np.float32)
    # Motion and flow for window i are available only after window i. Shift both
    # by one step so the action at i sees causal history rather than future data.
    features[1:, 0] = motion[:-1]
    features[1:, 1] = density[np.arange(len(labels) - 1), selected[:-1]] / 10.0
    features[1:, 2] = coverage[np.arange(len(labels) - 1), selected[:-1]]
    features[1:, 3] = structural[np.arange(len(labels) - 1), selected[:-1]]
    features[1:, 4] = unique[np.arange(len(labels) - 1), selected[:-1]]
    features[1:, 5] = overlap[np.arange(len(labels) - 1), selected[:-1]]
    features[1:, 6] = polarity[np.arange(len(labels) - 1), selected[:-1]]
    features[1:, 7] = event_efficiency[np.arange(len(labels) - 1), selected[:-1]]
    features[1:, 8] = target_rpm[:-1] / max(float(np.max(base_rpm * gear_scales)), 1.0)
    if use_flow:
        if "flow_features" not in oracle:
            raise ValueError("--use-flow requires flow_features in every oracle.")
        flow = np.asarray(oracle["flow_features"], dtype=np.float32)
        if flow.shape != (len(labels), FLOW_FEATURE_DIM):
            raise ValueError(f"flow_features must have shape {(len(labels), FLOW_FEATURE_DIM)}.")
        features[1:, BASE_FEATURE_DIM:] = flow[:-1]
    return features, target_rpm, np.asarray(oracle["timestamps"], dtype=np.float64), "selected_rpm" not in oracle


def _make_windows(features: np.ndarray, target: np.ndarray, *, length: int, stride: int, split: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x: list[np.ndarray] = []
    train_y: list[float] = []
    valid_x: list[np.ndarray] = []
    valid_y: list[float] = []
    for end in range(length - 1, len(features), stride):
        item = features[end - length + 1 : end + 1]
        if end < split:
            train_x.append(item)
            train_y.append(float(target[end]))
        else:
            valid_x.append(item)
            valid_y.append(float(target[end]))
    empty_x = np.empty((0, length, features.shape[-1]), dtype=np.float32)
    return (np.stack(train_x) if train_x else empty_x, np.asarray(train_y, np.float32), np.stack(valid_x) if valid_x else empty_x, np.asarray(valid_y, np.float32))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, action="append", default=[], help="Quality oracle NPZ; repeat for multiple scenes.")
    parser.add_argument("--train-oracle", type=Path, action="append", default=[], help="Scene-disjoint training oracle NPZ; repeat per scene.")
    parser.add_argument("--valid-oracle", type=Path, action="append", default=[], help="Scene-disjoint validation oracle NPZ; repeat per scene.")
    parser.add_argument("--manifest", type=Path, default=None, help="Load scene-disjoint train/valid oracle paths from a multiscale manifest.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-rpm", type=float, default=720.0)
    parser.add_argument("--gear-scales", type=float, nargs=5, default=(1.5, 1.0, 0.5, 0.25, 0.0))
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--use-flow", action="store_true", help="Use causal pre-prism RGB optical-flow summaries stored in each oracle.")
    parser.add_argument("--architecture", choices=("tcn", "lstm"), default="tcn")
    parser.add_argument("--motion-input", choices=("auto", "imu", "natural"), default="auto")
    parser.add_argument("--target-type", choices=("continuous", "gear"), default="continuous", help="Regress RPM or classify one of the simulated discrete gears.")
    args = parser.parse_args()
    if args.manifest is not None:
        if args.oracle or args.train_oracle or args.valid_oracle:
            parser.error("--manifest cannot be combined with explicit oracle arguments.")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        args.train_oracle = [Path(record["output"]) for record in manifest["oracles"] if record["split"] == "train"]
        args.valid_oracle = [Path(record["output"]) for record in manifest["oracles"] if record["split"] == "valid"]
    if args.window < 2 or args.stride < 1 or args.epochs < 1 or args.base_rpm <= 0.0 or args.lr <= 0.0:
        parser.error("window must be >= 2, stride/epochs/base-rpm/lr must be positive")
    if bool(args.train_oracle) != bool(args.valid_oracle):
        parser.error("--train-oracle and --valid-oracle must be provided together.")
    if not args.oracle and not args.train_oracle:
        parser.error("Provide --oracle or the scene-disjoint --train-oracle/--valid-oracle pair.")
    if args.train_oracle and args.oracle != []:
        parser.error("Use either --oracle (per-scene temporal split) or --train-oracle/--valid-oracle (scene-disjoint split), not both.")
    oracle_groups = [("train", args.train_oracle), ("valid", args.valid_oracle)] if args.train_oracle else [("temporal", args.oracle)]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gear_scales = np.asarray(args.gear_scales, dtype=np.float64)
    train_parts: list[tuple[np.ndarray, np.ndarray]] = []
    valid_parts: list[tuple[np.ndarray, np.ndarray]] = []
    reports: list[dict[str, object]] = []
    for group, paths in oracle_groups:
        for path in paths:
            with np.load(path) as oracle:
                features, target, _, legacy = _oracle_features(oracle, base_rpm=args.base_rpm, gear_scales=gear_scales, use_flow=args.use_flow, motion_input=args.motion_input)
            if group == "temporal":
                split = max(args.window, int(len(features) * 0.8))
                train_x, train_y, valid_x, valid_y = _make_windows(features, target, length=args.window, stride=args.stride, split=split)
                if len(train_y):
                    train_parts.append((train_x, train_y))
                if len(valid_y):
                    valid_parts.append((valid_x, valid_y))
            else:
                # Scene-disjoint mode uses every complete window from the
                # designated split and never mixes windows across scenes.
                split = len(features) + 1 if group == "train" else 0
                train_x, train_y, valid_x, valid_y = _make_windows(features, target, length=args.window, stride=args.stride, split=split)
                parts = train_parts if group == "train" else valid_parts
                values = (train_x, train_y) if group == "train" else (valid_x, valid_y)
                if len(values[1]):
                    parts.append(values)
            reports.append({"group": group, "oracle": str(path), "windows": len(features), "legacy_metrics": legacy, "target_rpm_range": [float(target.min()), float(target.max())]})
    if not train_parts or not valid_parts:
        raise ValueError("Oracles must provide enough windows for both temporal train and validation splits.")
    train_x = np.concatenate([part[0] for part in train_parts])
    train_y = np.concatenate([part[1] for part in train_parts])
    valid_x = np.concatenate([part[0] for part in valid_parts])
    valid_y = np.concatenate([part[1] for part in valid_parts])
    feature_mean = np.mean(train_x, axis=(0, 1), keepdims=True).astype(np.float32)
    feature_std = np.std(train_x, axis=(0, 1), keepdims=True).astype(np.float32)
    # Constant features carry no training information; centering them while
    # dividing by one avoids amplifying numerical noise at test time.
    feature_std = np.where(feature_std < 1e-4, 1.0, feature_std)
    train_x = np.clip((train_x - feature_mean) / feature_std, -5.0, 5.0)
    valid_x = np.clip((valid_x - feature_mean) / feature_std, -5.0, 5.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = BASE_FEATURE_DIM + (FLOW_FEATURE_DIM if args.use_flow else 0)
    rpm_values = args.base_rpm * gear_scales
    if args.target_type == "gear":
        if args.architecture != "lstm":
            parser.error("--target-type gear currently requires --architecture lstm.")
        model = CausalFlowFusionGearPolicy(feature_dim=feature_dim, hidden_dim=args.hidden_dim, num_gears=len(gear_scales)).to(device)
    else:
        model_cls = CausalFlowFusionRPMPolicy if args.architecture == "lstm" else CausalRPMPolicy
        model = model_cls(feature_dim=feature_dim, hidden_dim=args.hidden_dim, min_rpm=0.0, max_rpm=float(np.max(rpm_values))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_values, train_targets = torch.from_numpy(train_x).to(device), torch.from_numpy(train_y).to(device)
    valid_values, valid_targets = torch.from_numpy(valid_x).to(device), torch.from_numpy(valid_y).to(device)
    train_gears = torch.from_numpy(np.argmin(np.abs(train_y[:, None] - rpm_values[None, :]), axis=1)).long().to(device)
    valid_gears = torch.from_numpy(np.argmin(np.abs(valid_y[:, None] - rpm_values[None, :]), axis=1)).long().to(device)
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(train_values)
        loss = nn.functional.cross_entropy(output, train_gears) if args.target_type == "gear" else nn.functional.smooth_l1_loss(output, train_targets)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        train_output = model(train_values)
        valid_output = model(valid_values)
    if args.target_type == "gear":
        train_class = torch.argmax(train_output, dim=-1)
        valid_class = torch.argmax(valid_output, dim=-1)
        gear_tensor = torch.as_tensor(rpm_values, dtype=torch.float32, device=device)
        train_pred, valid_pred = gear_tensor[train_class], gear_tensor[valid_class]
        train_accuracy = float(torch.mean((train_class == train_gears).float()).item())
        valid_accuracy = float(torch.mean((valid_class == valid_gears).float()).item())
    else:
        train_pred, valid_pred = train_output, valid_output
        train_accuracy = valid_accuracy = None
    train_mae = float(torch.mean(torch.abs(train_pred - train_targets)).item())
    valid_mae = float(torch.mean(torch.abs(valid_pred - valid_targets)).item())
    baseline = float(np.mean(np.abs(valid_y - np.mean(train_y))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    resolved_motion_input = "imu" if args.motion_input == "auto" and args.use_flow else ("natural" if args.motion_input == "auto" else args.motion_input)
    checkpoint = {"model": model.state_dict(), "feature_dim": feature_dim, "hidden_dim": args.hidden_dim, "architecture": args.architecture, "target_type": args.target_type, "use_flow": args.use_flow, "motion_input": resolved_motion_input, "feature_mean": feature_mean.reshape(-1).tolist(), "feature_std": feature_std.reshape(-1).tolist(), "feature_clip": 5.0, "gear_rpm": rpm_values.tolist()}
    if args.target_type == "continuous":
        checkpoint.update({"min_rpm": model.min_rpm, "max_rpm": model.max_rpm})
    else:
        checkpoint["num_gears"] = model.num_gears
    torch.save(checkpoint, args.output)
    split_mode = "scene_disjoint" if args.train_oracle else "temporal_per_scene"
    report = {"device": str(device), "split_mode": split_mode, "architecture": args.architecture, "target_type": args.target_type, "use_flow": args.use_flow, "motion_input": resolved_motion_input, "feature_dim": feature_dim, "causal_alignment": "previous_window_imu_flow_event_state_predicts_next_window_gear", "train_samples": len(train_y), "valid_samples": len(valid_y), "train_mae_rpm": train_mae, "valid_mae_rpm": valid_mae, "train_gear_accuracy": train_accuracy, "valid_gear_accuracy": valid_accuracy, "mean_rpm_baseline_mae": baseline, "scenes": reports, "output": str(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
