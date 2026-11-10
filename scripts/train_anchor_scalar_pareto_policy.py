#!/usr/bin/env python3
"""Train the unchanged residual MLP with a scalar Pareto target.

The model still predicts utility residuals and switch confidence. A scalar
efficiency bonus is applied only to the training target, so this isolates
objective weighting from architecture changes.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ami_simulator.motion_net import CausalAnchorResidualPolicy
from scripts.train_anchor_residual_policy import _normalize, _windows


def _load_split(manifest: dict[str, object], split: str, length: int, efficiency_bonus: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    xs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    utility: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    reports: list[dict[str, object]] = []
    for record in manifest["records"]:
        if record["split"] != split:
            continue
        with np.load(Path(str(record["output"]))) as data:
            features = np.asarray(data["features"], dtype=np.float32)
            utility_values = np.asarray(data["delta_utility"], dtype=np.float32)
            efficiency_values = np.asarray(data["delta_event_efficiency"], dtype=np.float32)
            x, utility_window = _windows(features, utility_values, length)
            _, efficiency_window = _windows(features, efficiency_values, length)
            anchor = np.asarray(data["anchor_utility"], dtype=np.float32)[length - 1 :]
            if len(utility_window) != len(efficiency_window) or len(utility_window) != len(anchor):
                raise ValueError(f"Target/anchor length mismatch in {record['output']}")
            xs.append(x)
            utility.append(utility_window)
            targets.append(utility_window + float(efficiency_bonus) * efficiency_window)
            anchors.append(anchor)
            reports.append({"scene": record["scene"], "condition": record["condition"], "windows": int(len(utility_window))})
    if not xs:
        raise ValueError(f"No scalar Pareto records found for split {split!r}")
    return np.concatenate(xs), np.concatenate(targets), np.concatenate(utility), np.concatenate(anchors), reports


def _predict(model: nn.Module, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        delta, switch_logit = model(torch.from_numpy(values.astype(np.float32)))
    return delta.numpy(), torch.sigmoid(switch_logit).numpy()


def _selection(predicted: np.ndarray, probability: np.ndarray, target: np.ndarray, anchor: np.ndarray, *, margin: float, confidence: float) -> dict[str, float]:
    best = np.argmax(predicted, axis=1)
    rows = np.arange(len(target))
    switch = (np.max(predicted, axis=1) > margin) & (probability > confidence)
    selected = anchor + np.where(switch, target[rows, best], 0.0)
    return {"mean_selected_target": float(np.mean(selected)), "mean_target_gain": float(np.mean(selected - anchor)), "switch_fraction": float(np.mean(switch))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/comotion_anchor_quality_dataset/manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--efficiency-bonus", type=float, required=True, help="Scalar weight on candidate event-efficiency delta.")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--label-margin", type=float, default=0.02)
    parser.add_argument("--switch-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.window < 2 or args.hidden_dim < 4 or args.epochs < 1 or args.lr <= 0.0 or args.label_margin < 0.0 or args.switch_loss_weight < 0.0:
        parser.error("invalid training arguments")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_x, train_target, train_utility, train_anchor, train_reports = _load_split(manifest, "train", args.window, args.efficiency_bonus)
    valid_x, valid_target, valid_utility, valid_anchor, valid_reports = _load_split(manifest, "valid", args.window, args.efficiency_bonus)
    train_x, valid_x, feature_mean, feature_std = _normalize(train_x, valid_x)
    train_values, valid_values = torch.from_numpy(train_x), torch.from_numpy(valid_x)
    train_targets, valid_targets = torch.from_numpy(train_target), torch.from_numpy(valid_target)
    train_switch = (torch.max(train_targets, dim=1).values > args.label_margin).float()
    valid_switch = (torch.max(valid_targets, dim=1).values > args.label_margin).float()
    positive = float(train_switch.sum().item())
    negative = float(len(train_switch) - positive)
    pos_weight = torch.tensor([negative / max(positive, 1.0)], dtype=torch.float32)
    model = CausalAnchorResidualPolicy(feature_dim=train_x.shape[-1], hidden_dim=args.hidden_dim, residual_actions=train_target.shape[-1])
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
            validation = float((nn.functional.smooth_l1_loss(valid_delta, valid_targets) + args.switch_loss_weight * nn.functional.binary_cross_entropy_with_logits(valid_switch_logits, valid_switch, pos_weight=pos_weight)).item())
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
    margins = (0.0, 0.01, 0.02, 0.03, 0.05)
    confidences = (0.50, 0.60, 0.70, 0.80)
    candidates = [{"margin": margin, "confidence": confidence, **_selection(valid_pred, valid_prob, valid_target, valid_anchor, margin=margin, confidence=confidence)} for margin in margins for confidence in confidences]
    selected = max(candidates, key=lambda item: (item["mean_target_gain"], -item["switch_fraction"]))
    report = {"architecture": "anchor_residual_mlp", "target": "utility_delta + efficiency_bonus * event_efficiency_delta", "efficiency_bonus": args.efficiency_bonus, "feature_dim": int(train_x.shape[-1]), "hidden_dim": args.hidden_dim, "window": args.window, "anchor_rpm": 180.0, "residual_rpm": [0.0, 360.0], "label_margin": args.label_margin, "best_epoch": best_epoch, "best_validation_loss": best_validation, "history": history, "train": {"samples": len(train_target), **_selection(train_pred, train_prob, train_target, train_anchor, margin=selected["margin"], confidence=selected["confidence"])}, "valid": {"samples": len(valid_target), **_selection(valid_pred, valid_prob, valid_target, valid_anchor, margin=selected["margin"], confidence=selected["confidence"])}, "validation_selection": {"selected": selected, "candidates": candidates}, "data": {"train_scenes": len(train_reports), "valid_scenes": len(valid_reports)}}
    checkpoint = {"model": model.state_dict(), "architecture": "anchor_residual_mlp", "feature_dim": int(train_x.shape[-1]), "hidden_dim": args.hidden_dim, "residual_actions": 2, "anchor_rpm": 180.0, "residual_rpm": [0.0, 360.0], "window": args.window, "feature_mean": feature_mean.reshape(-1).tolist(), "feature_std": feature_std.reshape(-1).tolist(), "feature_clip": 5.0, "switch_margin": float(selected["margin"]), "switch_confidence": float(selected["confidence"]), "label_margin": args.label_margin, "target_objective": "utility_delta + efficiency_bonus * event_efficiency_delta", "efficiency_bonus": float(args.efficiency_bonus), "report": str(args.output.with_suffix(".json"))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
