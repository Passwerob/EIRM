#!/usr/bin/env python3
"""Train TinyCausalGearPolicy to imitate the camera-rate teacher.

Default teacher is occupancy-deficit + F1 gate so 1.5× is used when events
are scarce. Deploy never sees window F1.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ami_simulator.camera_nfb import ARCHITECTURE, FEATURE_DIM, FEATURE_NAMES, WINDOW, camera_feature
from ami_simulator.inverse_motion import CAMERA_HOLD_RATE, CAMERA_RATES, CAMERA_STORM_OCCUPANCY, inverse_rpm_index
from ami_simulator.motion_net import TinyCausalGearPolicy


def _records(manifest: dict[str, object], split: str) -> list[dict[str, object]]:
    return [record for record in manifest["oracles"] if record["split"] == split]


def _scene_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        arrays = {key: np.asarray(data[key]) for key in ("visual_motion", "speed_mps", "occupancy", "label_rate", "executed_rate")}
    count = len(arrays["label_rate"])
    if any(len(arrays[key]) != count for key in arrays):
        raise ValueError(f"{path} camera-NFB oracle arrays are misaligned.")
    return arrays


def _label_scene(arrays: dict[str, np.ndarray], speed_bounds: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    low, high = speed_bounds
    scale = max(high - low, 1e-9)
    count = len(arrays["label_rate"])
    features = np.zeros((count, FEATURE_DIM), dtype=np.float32)
    labels = np.zeros(count, dtype=np.int64)
    for index in range(count):
        lagged_visual = float(arrays["visual_motion"][index - 1] if index else 0.0)
        lagged_speed = float(np.clip(((arrays["speed_mps"][index - 1] if index else arrays["speed_mps"][0]) - low) / scale, 0.0, 1.0))
        previous_occupancy = float(arrays["occupancy"][index - 1] if index else 0.0)
        previous_rate = float(arrays["executed_rate"][index - 1] if index else CAMERA_HOLD_RATE)
        features[index] = camera_feature(lagged_visual, lagged_speed, previous_occupancy, previous_rate)
        labels[index] = inverse_rpm_index(float(arrays["label_rate"][index]), CAMERA_RATES)
    return features, labels


def _windows(features: np.ndarray, labels: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    if len(features) < length:
        return np.empty((0, length, features.shape[-1]), np.float32), np.empty((0,), np.int64)
    values = np.stack([features[end - length + 1 : end + 1] for end in range(length - 1, len(features))])
    targets = np.stack([labels[end] for end in range(length - 1, len(labels))])
    return values.astype(np.float32), targets.astype(np.int64)


def _load_split(records: list[dict[str, object]], speed_bounds: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for record in records:
        features, labels = _label_scene(_scene_arrays(Path(str(record["output"]))), speed_bounds)
        windows, targets = _windows(features, labels, WINDOW)
        if len(targets):
            xs.append(windows)
            ys.append(targets)
    if not xs:
        raise ValueError("Split produced no camera-NFB windows.")
    return np.concatenate(xs), np.concatenate(ys)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/eveyes_motion_nfb/camera_nfb_oracles_deficit/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/eveyes_motion_nfb/camera_nfb_deficit_policy.pt"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    if args.epochs < 1 or args.hidden_dim < 4 or args.lr <= 0.0:
        parser.error("invalid training arguments")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_records = _records(manifest, "train")
    valid_records = _records(manifest, "valid")
    if not train_records or not valid_records:
        raise SystemExit("Need train and valid camera-NFB oracles.")
    speeds = np.concatenate([_scene_arrays(Path(str(record["output"])))["speed_mps"] for record in train_records])
    speed_bounds = (float(np.min(speeds)), float(np.max(speeds)))
    train_x, train_y = _load_split(train_records, speed_bounds)
    valid_x, valid_y = _load_split(valid_records, speed_bounds)
    mean = train_x.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = train_x.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-4, 1.0, std)
    train_n = np.clip((train_x - mean) / std, -5.0, 5.0)
    valid_n = np.clip((valid_x - mean) / std, -5.0, 5.0)
    model = TinyCausalGearPolicy(feature_dim=FEATURE_DIM, hidden_dim=args.hidden_dim, num_gears=len(CAMERA_RATES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tx, ty = torch.from_numpy(train_n), torch.from_numpy(train_y)
    vx, vy = torch.from_numpy(valid_n), torch.from_numpy(valid_y)
    counts = np.bincount(train_y, minlength=len(CAMERA_RATES)).astype(np.float64)
    present = counts > 0
    weights = np.zeros_like(counts)
    n_present = int(present.sum())
    if n_present == 0:
        raise SystemExit("Training labels are empty.")
    weights[present] = counts[present].sum() / (n_present * counts[present])
    class_weight = torch.tensor(weights, dtype=torch.float32)
    best = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    hold_index = inverse_rpm_index(CAMERA_HOLD_RATE, CAMERA_RATES)
    fast_index = inverse_rpm_index(CAMERA_RATES[0], CAMERA_RATES)
    for epoch in range(args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(tx)
        loss = nn.functional.cross_entropy(logits, ty, weight=class_weight)
        if epoch:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            valid_loss = float(nn.functional.cross_entropy(model(vx), vy, weight=class_weight).item())
            train_acc = float((torch.argmax(model(tx), dim=1) == ty).float().mean().item())
            valid_acc = float((torch.argmax(model(vx), dim=1) == vy).float().mean().item())
            pred = torch.argmax(model(vx), dim=1)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch in (0, 10, 25, 50, args.epochs):
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(loss.item()),
                    "valid_loss": valid_loss,
                    "train_acc": train_acc,
                    "valid_acc": valid_acc,
                    "valid_pred_1.5_frac": float((pred == fast_index).float().mean().item()),
                    "valid_pred_1.0_frac": float((pred == hold_index).float().mean().item()),
                }
            )
            print(json.dumps(history[-1], sort_keys=True), flush=True)
    model.load_state_dict(best)
    teacher = str(manifest.get("teacher", "deficit-camera-rate"))
    label_frac = {str(rate): float(np.mean(train_y == index)) for index, rate in enumerate(CAMERA_RATES)}
    report = {
        "architecture": ARCHITECTURE,
        "teacher": teacher,
        "objective": "low occupancy may select 1.5x if F1 vs 1.0x stays within slack; storm refuses 1.5; otherwise F1-argmax",
        "feature_dim": FEATURE_DIM,
        "feature_names": list(FEATURE_NAMES),
        "candidate_rates": list(CAMERA_RATES),
        "storm_occupancy": float(manifest.get("storm_occupancy", CAMERA_STORM_OCCUPANCY)),
        "deficit_occupancy": manifest.get("deficit_occupancy"),
        "upshift_slack": manifest.get("upshift_slack"),
        "window": WINDOW,
        "hidden_dim": args.hidden_dim,
        "class_weight": class_weight.tolist(),
        "label_frac": label_frac,
        "speed_bounds": list(speed_bounds),
        "eval_width": manifest.get("eval_width"),
        "prism_rpm": manifest.get("prism_rpm"),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "train_samples": int(len(train_y)),
        "valid_samples": int(len(valid_y)),
        "train_accuracy": history[-1]["train_acc"],
        "valid_accuracy": history[-1]["valid_acc"],
        "train_label_counts": counts.tolist(),
    }
    checkpoint = {
        "model": model.state_dict(),
        "architecture": ARCHITECTURE,
        "feature_dim": FEATURE_DIM,
        "hidden_dim": args.hidden_dim,
        "num_gears": len(CAMERA_RATES),
        "candidate_rates": list(CAMERA_RATES),
        "teacher": teacher,
        "window": WINDOW,
        "feature_mean": mean.reshape(-1).tolist(),
        "feature_std": std.reshape(-1).tolist(),
        "feature_clip": 5.0,
        "storm_occupancy": float(manifest.get("storm_occupancy", CAMERA_STORM_OCCUPANCY)),
        "deficit_occupancy": manifest.get("deficit_occupancy"),
        "upshift_slack": manifest.get("upshift_slack"),
        "speed_bounds": list(speed_bounds),
        "eval_width": manifest.get("eval_width"),
        "prism_rpm": manifest.get("prism_rpm"),
        "dataset": str(manifest.get("dataset", "")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "best_epoch": best_epoch, "valid_acc": history[-1]["valid_acc"], "label_frac": label_frac}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
