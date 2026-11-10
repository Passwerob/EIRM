#!/usr/bin/env python3
"""Train a conservative residual policy around a fixed 180 RPM anchor."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ami_simulator.motion_net import CausalAnchorResidualGRUPolicy, CausalAnchorResidualPolicy


def _windows(features: np.ndarray, targets: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets):
        raise ValueError("features and targets must be aligned two-dimensional arrays")
    if len(features) < length:
        return np.empty((0, length, features.shape[-1]), np.float32), np.empty((0, targets.shape[-1]), np.float32)
    values = np.stack([features[end - length + 1 : end + 1] for end in range(length - 1, len(features))])
    labels = np.stack([targets[end] for end in range(length - 1, len(features))])
    return values.astype(np.float32), labels.astype(np.float32)


def _load_split(manifest: dict[str, object], split: str, length: int, target_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    reports: list[dict[str, object]] = []
    for record in manifest["records"]:
        if record["split"] != split:
            continue
        with np.load(Path(str(record["output"]))) as data:
            anchor_key = target_key.replace("delta_", "anchor_", 1)
            x, y = _windows(np.asarray(data["features"], dtype=np.float32), np.asarray(data[target_key], dtype=np.float32), length)
            anchor = np.asarray(data[anchor_key], dtype=np.float32)[length - 1 :]
            if len(y) != len(anchor):
                raise ValueError(f"Anchor/target length mismatch in {record['output']}")
            xs.append(x)
            ys.append(y)
            anchors.append(anchor)
            reports.append({"scene": record["scene"], "condition": record["condition"], "windows": int(len(y))})
    if not xs:
        raise ValueError(f"No residual records found for split {split!r}")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(anchors), reports


def _normalize(train_x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = train_x.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-4, 1.0, std)
    return np.clip((train_x - mean) / std, -5.0, 5.0), np.clip((values - mean) / std, -5.0, 5.0), mean, std


def _selection(pred_delta: np.ndarray, switch_probability: np.ndarray, target_delta: np.ndarray, anchor: np.ndarray, *, margin: float, confidence: float) -> tuple[np.ndarray, dict[str, float]]:
    best = np.argmax(pred_delta, axis=1)
    switch = (np.max(pred_delta, axis=1) > margin) & (switch_probability > confidence)
    rows = np.arange(len(target_delta))
    selected_delta = np.where(switch, target_delta[rows, best], 0.0)
    selected = anchor + selected_delta
    oracle = anchor + np.max(target_delta, axis=1)
    return selected, {
        "mean_selected_utility": float(np.mean(selected)),
        "mean_oracle_utility": float(np.mean(oracle)),
        "mean_utility_regret": float(np.mean(oracle - selected)),
        "positive_switch_fraction": float(np.mean(switch)),
        "gain_vs_anchor": float(np.mean(selected - anchor)),
    }


def _predict(model: nn.Module, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        delta, switch_logit = model(torch.from_numpy(values.astype(np.float32)))
    return delta.numpy(), torch.sigmoid(switch_logit).numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/comotion_anchor_residual_dataset/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/comotion_anchor_residual_policy/policy.pt"))
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--label-margin", type=float, default=0.02)
    parser.add_argument("--switch-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-key", choices=("delta_utility", "delta_edge_f1", "delta_edge_coverage"), default="delta_utility", help="Residual target. Use edge-F1 to train for target observability rather than the mixed utility surrogate.")
    parser.add_argument("--architecture", choices=("mlp", "gru"), default="mlp", help="Temporal encoder architecture.")
    args = parser.parse_args()
    if args.window < 2 or args.hidden_dim < 4 or args.epochs < 1 or args.lr <= 0.0 or args.label_margin < 0.0 or args.switch_loss_weight < 0.0:
        parser.error("invalid training arguments")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_x, train_y, train_anchor, train_reports = _load_split(manifest, "train", args.window, args.target_key)
    valid_x, valid_y, valid_anchor, valid_reports = _load_split(manifest, "valid", args.window, args.target_key)
    train_x, valid_x, feature_mean, feature_std = _normalize(train_x, valid_x)
    train_values = torch.from_numpy(train_x)
    valid_values = torch.from_numpy(valid_x)
    train_targets = torch.from_numpy(train_y)
    valid_targets = torch.from_numpy(valid_y)
    train_switch = (torch.max(train_targets, dim=1).values > args.label_margin).float()
    valid_switch = (torch.max(valid_targets, dim=1).values > args.label_margin).float()
    positive = float(train_switch.sum().item())
    negative = float(len(train_switch) - positive)
    pos_weight = torch.tensor([negative / max(positive, 1.0)], dtype=torch.float32)
    model_class = CausalAnchorResidualGRUPolicy if args.architecture == "gru" else CausalAnchorResidualPolicy
    model = model_class(feature_dim=train_x.shape[-1], hidden_dim=args.hidden_dim, residual_actions=train_y.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predicted_delta, predicted_switch = model(train_values)
        delta_loss = nn.functional.smooth_l1_loss(predicted_delta, train_targets)
        switch_loss = nn.functional.binary_cross_entropy_with_logits(predicted_switch, train_switch, pos_weight=pos_weight)
        loss = delta_loss + args.switch_loss_weight * switch_loss
        if epoch > 0:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            valid_delta, valid_switch_logits = model(valid_values)
            valid_delta_loss = nn.functional.smooth_l1_loss(valid_delta, valid_targets)
            valid_switch_loss = nn.functional.binary_cross_entropy_with_logits(valid_switch_logits, valid_switch, pos_weight=pos_weight)
            validation = float((valid_delta_loss + args.switch_loss_weight * valid_switch_loss).item())
        if validation < best_validation:
            best_validation = validation
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch in (0, 10, 25, 50, 100, args.epochs):
            history.append({"epoch": epoch, "train_loss": float(loss.item()), "valid_loss": validation})
    model.load_state_dict(best_state)
    model.eval()
    train_pred, train_prob = _predict(model, train_x)
    valid_pred, valid_prob = _predict(model, valid_x)
    candidates = []
    for margin in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10):
        for confidence in (0.50, 0.60, 0.70, 0.80):
            _, metrics = _selection(valid_pred, valid_prob, valid_y, valid_anchor, margin=margin, confidence=confidence)
            candidates.append({"margin": margin, "confidence": confidence, **metrics})
    selected = max(candidates, key=lambda item: (item["gain_vs_anchor"], -item["mean_utility_regret"], -item["positive_switch_fraction"]))
    report = {
        "architecture": f"anchor_residual_{args.architecture}",
        "target_key": args.target_key,
        "feature_dim": int(train_x.shape[-1]),
        "hidden_dim": args.hidden_dim,
        "window": args.window,
        "anchor_rpm": 180.0,
        "residual_rpm": [0.0, 360.0],
        "label_margin": args.label_margin,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "history": history,
        "train": {"samples": len(train_y), "positive_label_fraction": float(train_switch.mean().item()), **_selection(train_pred, train_prob, train_y, train_anchor, margin=float(selected["margin"]), confidence=float(selected["confidence"]))[1]},
        "valid": {"samples": len(valid_y), "positive_label_fraction": float(valid_switch.mean().item()), **_selection(valid_pred, valid_prob, valid_y, valid_anchor, margin=float(selected["margin"]), confidence=float(selected["confidence"]))[1]},
        "validation_selection": {"criterion": "maximum gain versus anchor, then regret, then switch fraction", "selected": selected, "candidates": candidates},
        "data": {"train_scenes": len(train_reports), "valid_scenes": len(valid_reports), "train_reports": train_reports, "valid_reports": valid_reports},
    }
    checkpoint = {"model": model.state_dict(), "architecture": f"anchor_residual_{args.architecture}", "target_key": args.target_key, "feature_dim": int(train_x.shape[-1]), "hidden_dim": args.hidden_dim, "residual_actions": 2, "anchor_rpm": 180.0, "residual_rpm": [0.0, 360.0], "window": args.window, "feature_mean": feature_mean.reshape(-1).tolist(), "feature_std": feature_std.reshape(-1).tolist(), "feature_clip": 5.0, "switch_margin": float(selected["margin"]), "switch_confidence": float(selected["confidence"]), "label_margin": args.label_margin, "report": str(args.output.with_suffix(".json"))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
