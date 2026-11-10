#!/usr/bin/env python3
"""Train a tiny causal prism-speed policy.

Teachers:
- inverse: slow visual motion → 360 RPM, fast → 0 RPM
- three-gear: inverse 0/180/360, but 360 only when teacher F1 does not regress
- quality-floor: same inverse schedule, but never below 180 RPM, and 360 only
  when teacher edge F1 does not regress
- f1-argmax: 360 RPM iff teacher edge F1(360) >= F1(180) + margin
- f1-argmax-3: argmax teacher F1 among {0, 180, 360}; ties take the lower RPM
- deficit: 360 RPM when IMU says the camera is moving but events are scarce
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ami_simulator.inverse_motion import QUALITY_INVERSE_RPM, THREE_GEAR_RPM, excitation_deficit_rpm, f1_argmax_rpm, f1_argmax_three_rpm, inverse_rpm, inverse_rpm_index, motion_thresholds, quality_inverse_rpm, three_gear_rpm
from ami_simulator.motion_net import TinyCausalGearPolicy
from scripts.evaluate_phase_continuous_control import GEAR_RPM

FEATURE_DIM = 4
WINDOW = 8
TWO_GEAR_TEACHERS = {"quality-floor", "f1-argmax", "deficit"}


def _gears_for(teacher: str) -> tuple[float, ...]:
    return QUALITY_INVERSE_RPM if teacher in TWO_GEAR_TEACHERS else THREE_GEAR_RPM


def _records(manifest: dict[str, object], split: str) -> list[dict[str, object]]:
    return [record for record in manifest["oracles"] if record["split"] == split]


def _scene_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        visual = np.asarray(data["visual_motion"], dtype=np.float64)
        imu = np.asarray(data["imu_motion"], dtype=np.float64)
        densities = np.asarray(data["densities"], dtype=np.float64)
        overlap = np.asarray(data["temporal_overlap"], dtype=np.float64)
        edge_f1 = np.asarray(data["edge_f1"], dtype=np.float64)
        object_edge_f1 = np.asarray(data["object_edge_f1"], dtype=np.float64) if "object_edge_f1" in data.files else np.zeros_like(edge_f1)
        object_area = np.asarray(data["object_area"], dtype=np.float64) if "object_area" in data.files else np.zeros(len(visual))
    if visual.ndim != 1 or imu.shape != visual.shape or densities.shape[0] != len(visual) or overlap.shape[0] != len(visual) or edge_f1.shape[0] != len(visual):
        raise ValueError(f"{path} is missing aligned visual/IMU/density/overlap/edge_f1 arrays.")
    return {"visual": visual, "imu": imu, "densities": densities, "overlap": overlap, "edge_f1": edge_f1, "object_edge_f1": object_edge_f1, "object_area": object_area}


def _gear_index(rpm: float) -> int:
    index = int(np.argmin(np.abs(GEAR_RPM - float(rpm))))
    if not np.isclose(GEAR_RPM[index], rpm):
        raise ValueError(f"rpm {rpm} is not in GEAR_RPM.")
    return index


def _label_scene(
    arrays: dict[str, np.ndarray],
    thresholds: tuple[float, float],
    *,
    teacher: str,
    storm_density: float,
    storm_overlap: float,
    upshift_margin: float,
    imu_threshold: float,
    density_threshold: float,
    min_object_area: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(arrays["visual"])
    gears = _gears_for(teacher)
    features = np.zeros((count, FEATURE_DIM), dtype=np.float32)
    labels = np.zeros(count, dtype=np.int64)
    previous_density = 0.0
    previous_overlap = 0.0
    fast_gear = _gear_index(360.0)
    anchor_gear = _gear_index(180.0)
    off_gear = _gear_index(0.0)
    for index in range(count):
        motion = float(arrays["visual"][index - 1] if index else 0.0)
        imu = float(arrays["imu"][index - 1] if index else 0.0)
        features[index] = (motion, imu, previous_density, previous_overlap)
        if teacher == "inverse":
            rpm = inverse_rpm(motion, thresholds, previous_density=previous_density, previous_overlap=previous_overlap, storm_density=storm_density, storm_overlap=storm_overlap)
        elif teacher == "three-gear":
            rpm = three_gear_rpm(
                motion,
                thresholds,
                previous_density=previous_density,
                previous_overlap=previous_overlap,
                f1_fast=float(arrays["edge_f1"][index, fast_gear]),
                f1_anchor=float(arrays["edge_f1"][index, anchor_gear]),
                upshift_margin=upshift_margin,
                storm_density=storm_density,
                storm_overlap=storm_overlap,
            )
        elif teacher == "quality-floor":
            rpm = quality_inverse_rpm(
                motion,
                thresholds,
                previous_density=previous_density,
                previous_overlap=previous_overlap,
                f1_fast=float(arrays["edge_f1"][index, fast_gear]),
                f1_anchor=float(arrays["edge_f1"][index, anchor_gear]),
                upshift_margin=upshift_margin,
                storm_density=storm_density,
                storm_overlap=storm_overlap,
            )
        elif teacher == "f1-argmax":
            rpm = f1_argmax_rpm(
                float(arrays["edge_f1"][index, fast_gear]),
                float(arrays["edge_f1"][index, anchor_gear]),
                previous_density=previous_density,
                previous_overlap=previous_overlap,
                upshift_margin=upshift_margin,
                storm_density=storm_density,
                storm_overlap=storm_overlap,
            )
        elif teacher == "f1-argmax-3":
            rpm = f1_argmax_three_rpm(
                float(arrays["edge_f1"][index, off_gear]),
                float(arrays["edge_f1"][index, anchor_gear]),
                float(arrays["edge_f1"][index, fast_gear]),
                previous_density=previous_density,
                previous_overlap=previous_overlap,
                upshift_margin=upshift_margin,
                storm_density=storm_density,
                storm_overlap=storm_overlap,
            )
        elif teacher == "target-f1-argmax-3":
            on_screen = float(arrays["object_area"][index]) >= min_object_area and float(arrays["object_edge_f1"][index].max()) > 0.0
            source = arrays["object_edge_f1"] if on_screen else arrays["edge_f1"]
            rpm = f1_argmax_three_rpm(
                float(source[index, off_gear]),
                float(source[index, anchor_gear]),
                float(source[index, fast_gear]),
                previous_density=previous_density,
                previous_overlap=previous_overlap,
                upshift_margin=upshift_margin,
                storm_density=storm_density,
                storm_overlap=storm_overlap,
            )
        elif teacher == "deficit":
            rpm = excitation_deficit_rpm(
                imu,
                previous_density,
                previous_overlap,
                imu_threshold=imu_threshold,
                density_threshold=density_threshold,
                storm_density=storm_density,
                storm_overlap=storm_overlap,
            )
        else:
            raise ValueError(f"Unknown teacher {teacher!r}")
        labels[index] = inverse_rpm_index(rpm, gears)
        gear = _gear_index(rpm)
        previous_density = float(arrays["densities"][index, gear])
        previous_overlap = float(arrays["overlap"][index, gear])
    return features, labels


def _windows(features: np.ndarray, labels: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    if len(features) < length:
        return np.empty((0, length, features.shape[-1]), np.float32), np.empty((0,), np.int64)
    values = np.stack([features[end - length + 1 : end + 1] for end in range(length - 1, len(features))])
    targets = np.stack([labels[end] for end in range(length - 1, len(labels))])
    return values.astype(np.float32), targets.astype(np.int64)


def _load_split(
    records: list[dict[str, object]],
    thresholds: tuple[float, float],
    *,
    teacher: str,
    storm_density: float,
    storm_overlap: float,
    upshift_margin: float,
    imu_threshold: float,
    density_threshold: float,
    min_object_area: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for record in records:
        features, labels = _label_scene(
            _scene_arrays(Path(str(record["output"]))),
            thresholds,
            teacher=teacher,
            storm_density=storm_density,
            storm_overlap=storm_overlap,
            upshift_margin=upshift_margin,
            imu_threshold=imu_threshold,
            density_threshold=density_threshold,
            min_object_area=min_object_area,
        )
        windows, targets = _windows(features, labels, WINDOW)
        if len(targets):
            xs.append(windows)
            ys.append(targets)
    if not xs:
        raise ValueError("Split produced no inverse-motion windows.")
    return np.concatenate(xs), np.concatenate(ys)


def _fit_deficit_thresholds(records: list[dict[str, object]]) -> tuple[float, float]:
    """Pick IMU/density cut-points on train oracles to maximize mean edge F1 vs 180 RPM."""
    fast_gear = _gear_index(360.0)
    anchor_gear = _gear_index(180.0)
    imu_parts: list[np.ndarray] = []
    dens_parts: list[np.ndarray] = []
    f180_parts: list[np.ndarray] = []
    f360_parts: list[np.ndarray] = []
    for record in records:
        arrays = _scene_arrays(Path(str(record["output"])))
        imu_parts.append(arrays["imu"])
        dens_parts.append(arrays["densities"][:, anchor_gear])
        f180_parts.append(arrays["edge_f1"][:, anchor_gear])
        f360_parts.append(arrays["edge_f1"][:, fast_gear])
    imu = np.concatenate(imu_parts)
    dens = np.concatenate(dens_parts)
    f180 = np.concatenate(f180_parts)
    f360 = np.concatenate(f360_parts)
    best = (float("-inf"), 0.3, float(np.median(dens)))
    for imu_threshold in np.linspace(0.0, max(float(np.max(imu)), 1e-6), 9):
        for density_threshold in np.linspace(float(np.quantile(dens, 0.1)), float(np.quantile(dens, 0.9)), 9):
            selected = np.where((imu >= imu_threshold) & (dens <= density_threshold), f360, f180)
            score = float(selected.mean())
            if score > best[0]:
                best = (score, float(imu_threshold), float(density_threshold))
    return best[1], best[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/comotion_v2_learned/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/comotion_inverse_motion_policy/policy.pt"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--storm-density", type=float, default=0.75)
    parser.add_argument("--storm-overlap", type=float, default=0.65)
    parser.add_argument("--teacher", choices=("inverse", "three-gear", "quality-floor", "f1-argmax", "f1-argmax-3", "target-f1-argmax-3", "deficit"), default="inverse")
    parser.add_argument("--min-object-area", type=float, default=0.05, help="Screen-area floor below which target-f1-argmax-3 falls back to image-wide edge F1.")
    parser.add_argument("--quality-floor", action="store_true", help="Deprecated alias for --teacher quality-floor.")
    parser.add_argument("--upshift-margin", type=float, default=0.0, help="Require teacher F1(360) >= F1(180) + margin before leaving the 180 RPM floor.")
    args = parser.parse_args()
    teacher = "quality-floor" if args.quality_floor else args.teacher
    if args.epochs < 1 or args.hidden_dim < 4 or args.lr <= 0.0 or args.storm_density < 0.0 or not 0.0 <= args.storm_overlap <= 1.0 or args.upshift_margin < 0.0:
        parser.error("invalid training arguments")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_records = _records(manifest, "train")
    valid_records = _records(manifest, "valid")
    train_visual = np.concatenate([_scene_arrays(Path(str(record["output"])))["visual"] for record in train_records])
    thresholds = motion_thresholds(train_visual)
    imu_threshold, density_threshold = _fit_deficit_thresholds(train_records)
    split_kwargs = dict(teacher=teacher, storm_density=args.storm_density, storm_overlap=args.storm_overlap, upshift_margin=args.upshift_margin, imu_threshold=imu_threshold, density_threshold=density_threshold, min_object_area=args.min_object_area)
    train_x, train_y = _load_split(train_records, thresholds, **split_kwargs)
    valid_x, valid_y = _load_split(valid_records, thresholds, **split_kwargs)
    overlap_values = np.concatenate([_scene_arrays(Path(str(record["output"])))["overlap"].reshape(-1) for record in train_records])
    use_temporal_overlap = bool(np.any(np.abs(overlap_values) > 1e-12))
    gears = _gears_for(teacher)
    train_gear_frac_by_scene = {}
    for record in train_records:
        _, labels = _label_scene(
            _scene_arrays(Path(str(record["output"]))),
            thresholds,
            **split_kwargs,
        )
        train_gear_frac_by_scene[str(record["scene"])] = {str(int(rpm)): float(np.mean(labels == index)) for index, rpm in enumerate(gears)}
    train_360_by_scene = {scene: fracs.get("360", float("nan")) for scene, fracs in train_gear_frac_by_scene.items()}
    mean = train_x.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = train_x.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-4, 1.0, std)
    train_n = np.clip((train_x - mean) / std, -5.0, 5.0)
    valid_n = np.clip((valid_x - mean) / std, -5.0, 5.0)
    model = TinyCausalGearPolicy(feature_dim=FEATURE_DIM, hidden_dim=args.hidden_dim, num_gears=len(gears))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tx, ty = torch.from_numpy(train_n), torch.from_numpy(train_y)
    vx, vy = torch.from_numpy(valid_n), torch.from_numpy(valid_y)
    counts = np.bincount(train_y, minlength=len(gears)).astype(np.float64)
    class_weight = torch.tensor(counts.sum() / (len(gears) * np.maximum(counts, 1.0)), dtype=torch.float32)
    best = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
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
            train_360 = float((ty == inverse_rpm_index(360.0, gears)).float().mean().item())
            pred_360 = float((pred == inverse_rpm_index(360.0, gears)).float().mean().item())
            train_0 = float((ty == inverse_rpm_index(0.0, gears)).float().mean().item()) if 0.0 in gears else float("nan")
            pred_0 = float((pred == inverse_rpm_index(0.0, gears)).float().mean().item()) if 0.0 in gears else float("nan")
        if valid_loss < best_loss:
            best_loss = valid_loss
            best = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch in (0, 25, 50, 100, args.epochs):
            history.append({"epoch": epoch, "train_loss": float(loss.item()), "valid_loss": valid_loss, "train_acc": train_acc, "valid_acc": valid_acc, "train_360_frac": train_360, "valid_pred_360_frac": pred_360, "train_0_frac": train_0, "valid_pred_0_frac": pred_0})
    model.load_state_dict(best)
    objectives = {
        "inverse": "slow visual motion -> 360 RPM; fast visual motion -> 0 RPM; storm brake downshifts only",
        "three-gear": "too little motion may use 360 RPM if teacher F1 does not regress; too much motion may use 0 RPM; 180 RPM is the mid gear",
        "quality-floor": "slow motion -> 360 RPM if teacher F1 does not regress; otherwise 180 RPM floor",
        "f1-argmax": "360 RPM iff teacher edge F1(360) >= F1(180) + margin; 180 RPM floor; storm brake refuses the upshift",
        "f1-argmax-3": "argmax teacher edge F1 among {0, 180, 360}; ties take the lower RPM; storm refuses 360",
        "target-f1-argmax-3": "argmax teacher object-edge F1 among {0, 180, 360} while a target is on screen, else argmax image-wide edge F1; ties take the lower RPM; storm refuses 360",
        "deficit": "360 RPM when IMU motion is present but previous event density is low; 180 RPM floor",
    }
    report = {
        "architecture": "tiny_inverse_motion_gear",
        "objective": objectives[teacher],
        "teacher": teacher,
        "quality_floor": teacher in TWO_GEAR_TEACHERS,
        "upshift_margin": args.upshift_margin,
        "min_object_area": args.min_object_area if teacher == "target-f1-argmax-3" else None,
        "feature_dim": FEATURE_DIM,
        "feature_names": ["lagged_visual_motion", "lagged_imu_motion", "previous_event_density", "previous_temporal_overlap"],
        "candidate_rpm": list(gears),
        "motion_thresholds": list(thresholds),
        "imu_threshold": imu_threshold,
        "density_threshold": density_threshold,
        "storm_density": args.storm_density,
        "storm_overlap": args.storm_overlap,
        "window": WINDOW,
        "hidden_dim": args.hidden_dim,
        "class_weight": class_weight.tolist(),
        "label_frac": {str(int(rpm)): float(np.mean(train_y == index)) for index, rpm in enumerate(gears)},
        "label_360_frac": float(np.mean(train_y == inverse_rpm_index(360.0, gears))),
        "label_0_frac": float(np.mean(train_y == inverse_rpm_index(0.0, gears))) if 0.0 in gears else float("nan"),
        "train_gear_frac_by_scene": train_gear_frac_by_scene,
        "train_360_frac_by_scene": train_360_by_scene,
        "use_temporal_overlap": use_temporal_overlap,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "train_samples": int(len(train_y)),
        "valid_samples": int(len(valid_y)),
        "train_accuracy": history[-1]["train_acc"],
        "valid_accuracy": history[-1]["valid_acc"],
    }
    checkpoint = {
        "model": model.state_dict(),
        "architecture": "tiny_inverse_motion_gear",
        "feature_dim": FEATURE_DIM,
        "hidden_dim": args.hidden_dim,
        "num_gears": len(gears),
        "candidate_rpm": list(gears),
        "teacher": teacher,
        "quality_floor": teacher in TWO_GEAR_TEACHERS,
        "upshift_margin": args.upshift_margin,
        "window": WINDOW,
        "feature_mean": mean.reshape(-1).tolist(),
        "feature_std": std.reshape(-1).tolist(),
        "feature_clip": 5.0,
        "motion_thresholds": list(thresholds),
        "imu_threshold": imu_threshold,
        "density_threshold": density_threshold,
        "storm_density": args.storm_density,
        "storm_overlap": args.storm_overlap,
        "use_temporal_overlap": use_temporal_overlap,
        "report": str(args.output.with_suffix(".json")),
    }
    if "imu_motion_bounds" in manifest:
        checkpoint["imu_motion_bounds"] = [float(value) for value in manifest["imu_motion_bounds"]]
        report["imu_motion_bounds"] = checkpoint["imu_motion_bounds"]
    if "storm_density_feature" in manifest:
        checkpoint["storm_density_feature"] = str(manifest["storm_density_feature"])
        report["storm_density_feature"] = checkpoint["storm_density_feature"]
    if "dataset" in manifest:
        checkpoint["dataset"] = str(manifest["dataset"])
        report["dataset"] = checkpoint["dataset"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
