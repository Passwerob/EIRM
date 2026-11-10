#!/usr/bin/env python3
"""Train a causal LSTM to rank all candidate prism-speed utilities."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ami_simulator.motion_net import CausalFlowFusionQPolicy, CausalGatedFlowQPolicy
from scripts.train_causal_rpm_policy import BASE_FEATURE_DIM, FLOW_FEATURE_DIM


GEAR_SCALES = np.asarray([1.5, 1.0, 0.5, 0.25, 0.0], dtype=np.float64)
CONDITIONS = ("camera_only", "object_only", "co_motion", "counter_motion")
MOTION_DELTA_FEATURE_DIM = 1


def checkpoint_epochs(total_epochs: int, requested: list[int] | None = None) -> list[int]:
    """Return a small, predeclared set of epochs for direct-rollout selection."""
    values = requested if requested is not None else [0, 10, 20, 50, 100, 200, total_epochs]
    return sorted({int(epoch) for epoch in values if 0 <= int(epoch) <= total_epochs} | {total_epochs})


def _checkpoint_payload(
    model: nn.Module,
    *,
    feature_dim: int,
    hidden_dim: int,
    architecture: str,
    use_flow: bool,
    phase_features: bool,
    motion_delta: bool,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    base_rpm: float,
    temperature: float,
    epoch: int,
) -> dict[str, object]:
    return {
        "model": model.state_dict(),
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "num_gears": len(GEAR_SCALES),
        "architecture": architecture,
        "target_type": "candidate_q",
        "use_flow": use_flow,
        "motion_input": "imu",
        "phase_features": phase_features,
        "motion_delta": motion_delta,
        "flow_start": BASE_FEATURE_DIM + (MOTION_DELTA_FEATURE_DIM if motion_delta else 0),
        "feature_mean": feature_mean.reshape(-1).tolist(),
        "feature_std": feature_std.reshape(-1).tolist(),
        "feature_clip": 5.0,
        "gear_rpm": (base_rpm * GEAR_SCALES).tolist(),
        "temperature": temperature,
        "base_feature_dim": int(model.base_feature_dim) if isinstance(model, CausalGatedFlowQPolicy) else None,
        "training_epoch": epoch,
    }


def _windows(features: np.ndarray, utilities: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    if len(features) != len(utilities) or utilities.ndim != 2:
        raise ValueError("Features and candidate utilities must have aligned windows.")
    if len(features) < length:
        return np.empty((0, length, features.shape[-1]), np.float32), np.empty((0, utilities.shape[-1]), np.float32)
    x = np.stack([features[end - length + 1 : end + 1] for end in range(length - 1, len(features))])
    y = np.stack([utilities[end] for end in range(length - 1, len(features))])
    return x.astype(np.float32), y.astype(np.float32)


def _active_masks(oracle: np.lib.npyio.NpzFile) -> np.ndarray:
    if "active_masks_packed" not in oracle:
        raise ValueError("Counterfactual Q training requires active masks stored by the oracle.")
    pixel_count = int(oracle["active_mask_pixel_count"])
    return np.unpackbits(oracle["active_masks_packed"], axis=-1)[..., :pixel_count].astype(bool)


def _overlap(candidates: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    if previous is None:
        return np.zeros(len(candidates), dtype=np.float32)
    intersection = np.count_nonzero(candidates & previous[None, :], axis=1)
    union = np.count_nonzero(candidates | previous[None, :], axis=1)
    return np.divide(intersection, union, out=np.zeros(len(candidates), dtype=np.float64), where=union > 0).astype(np.float32)


def _counterfactual_sequence(
    oracle: np.lib.npyio.NpzFile,
    schedule: np.ndarray,
    *,
    use_flow: bool,
    motion_delta: bool,
    rpm_values: np.ndarray,
    objective: dict[str, float],
    baseline_gear: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(schedule)
    phase_features = "prism_phase_start" in oracle
    base_dim = BASE_FEATURE_DIM + (MOTION_DELTA_FEATURE_DIM if motion_delta else 0)
    feature_dim = base_dim + (FLOW_FEATURE_DIM if use_flow else 0) + (2 if phase_features else 0)
    features = np.zeros((count, feature_dim), dtype=np.float32)
    utilities = np.zeros((count, len(rpm_values)), dtype=np.float32)
    masks = _active_masks(oracle)
    previous_mask = None
    previous_rpm = 720.0
    motion = np.asarray(oracle["imu_motion"], dtype=np.float32)
    if phase_features:
        phase = np.asarray(oracle["prism_phase_start"], dtype=np.float64)
        features[:, -2] = np.sin(phase)
        features[:, -1] = np.cos(phase)
    for index in range(count):
        overlaps = _overlap(masks[index], previous_mask)
        score = np.asarray(oracle["edge_f1"][index], dtype=np.float64)
        score += 0.25 * np.asarray(oracle["coverage"][index], dtype=np.float64)
        score += 0.15 * np.asarray(oracle["unique_ratio"][index], dtype=np.float64)
        score += objective["event_efficiency_weight"] * np.asarray(oracle["event_efficiency"][index], dtype=np.float64)
        score += objective.get("coverage_efficiency_weight", 0.0) * np.asarray(oracle["coverage_efficiency"][index], dtype=np.float64) if "coverage_efficiency" in oracle else 0.0
        score -= objective["overlap_weight"] * overlaps
        score -= 0.05 * np.asarray(oracle["polarity_balance"][index], dtype=np.float64)
        density = np.asarray(oracle["densities"][index], dtype=np.float64)
        target_density = float(oracle["target_density"][index])
        score -= objective["density_weight"] * np.square(np.log1p(density) - np.log1p(target_density))
        score -= objective["invalid_fraction_weight"] * (1.0 - np.asarray(oracle["valid_fraction"][index], dtype=np.float64))
        score -= objective["rpm_effort_weight"] * np.square(rpm_values / rpm_values[0])
        score -= objective["rpm_change_weight"] * np.square((rpm_values - previous_rpm) / rpm_values[0])
        if baseline_gear is not None:
            if not 0 <= baseline_gear < len(rpm_values):
                raise ValueError("baseline_gear must index rpm_values.")
            score -= score[baseline_gear]
        utilities[index] = score.astype(np.float32)
        gear = int(schedule[index])
        if index + 1 < count:
            features[index + 1, 0] = motion[index]
            if motion_delta:
                features[index + 1, BASE_FEATURE_DIM] = motion[index] - (motion[index - 1] if index else motion[index])
            features[index + 1, 1] = float(oracle["densities"][index, gear]) / 10.0
            features[index + 1, 2] = float(oracle["coverage"][index, gear])
            features[index + 1, 3] = float(oracle["structural_ratio"][index, gear])
            features[index + 1, 4] = float(oracle["unique_ratio"][index, gear])
            features[index + 1, 5] = float(overlaps[gear])
            features[index + 1, 6] = float(oracle["polarity_balance"][index, gear])
            features[index + 1, 7] = float(oracle["event_efficiency"][index, gear])
            features[index + 1, 8] = float(rpm_values[gear] / rpm_values[0])
            if use_flow:
                flow_start = BASE_FEATURE_DIM + (MOTION_DELTA_FEATURE_DIM if motion_delta else 0)
                features[index + 1, flow_start : flow_start + FLOW_FEATURE_DIM] = np.asarray(oracle["flow_features"][index], dtype=np.float32)
        previous_mask = masks[index, gear]
        previous_rpm = float(rpm_values[gear])
    return features, utilities


def _condition_index(path: Path) -> int:
    for index, condition in enumerate(CONDITIONS):
        if path.stem.endswith(condition):
            return index
    raise ValueError(f"Cannot infer controlled motion condition from {path}.")


def _load_group(paths: list[Path], *, use_flow: bool, motion_delta: bool, window: int, base_rpm: float, objective: dict[str, float], seed: int, baseline_gear: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    x_parts, y_parts, group_parts, reports = [], [], [], []
    rpm_values = base_rpm * GEAR_SCALES
    for path_index, path in enumerate(paths):
        with np.load(path) as oracle:
            count = len(oracle["labels"])
            phase_continuous = "prism_phase_start" in oracle
            if use_flow and "flow_features" not in oracle:
                raise ValueError(f"Oracle {path} lacks flow features.")
            rng = np.random.default_rng(seed + path_index * 1009)
            schedules = {"oracle": np.asarray(oracle["labels"], dtype=np.int64)}
            if not phase_continuous:
                # Legacy tables lack phase-conditioned counterfactual metrics.
                # This augmentation is retained only to reproduce diagnostics;
                # new physical oracles train on their valid on-path states.
                schedules.update({f"fixed_{gear}": np.full(count, gear, dtype=np.int64) for gear in range(len(GEAR_SCALES))})
                schedules["random"] = rng.integers(0, len(GEAR_SCALES), count, dtype=np.int64)
            scene_counts = np.zeros(len(GEAR_SCALES), dtype=np.int64)
            scene_samples = 0
            for schedule in schedules.values():
                features, utility = _counterfactual_sequence(oracle, schedule, use_flow=use_flow, motion_delta=motion_delta, rpm_values=rpm_values, objective=objective, baseline_gear=baseline_gear)
                values, targets = _windows(features, utility, window)
                if len(targets):
                    x_parts.append(values)
                    y_parts.append(targets)
                    group_parts.append(np.full(len(targets), _condition_index(path), dtype=np.int64))
                    scene_counts += np.bincount(np.argmax(targets, axis=1), minlength=len(GEAR_SCALES))
                    scene_samples += len(targets)
        reports.append({"oracle": str(path), "windows": count, "behavior_policies": len(schedules), "samples": scene_samples, "best_action_counts": scene_counts.tolist()})
    if not x_parts:
        raise ValueError("No complete causal windows were available.")
    return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(group_parts), reports


def _sample_losses(prediction: torch.Tensor, target: torch.Tensor, *, temperature: float, ranking_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_advantage = prediction - prediction.mean(dim=-1, keepdim=True)
    target_advantage = target - target.mean(dim=-1, keepdim=True)
    regression = nn.functional.smooth_l1_loss(pred_advantage, target_advantage, reduction="none").mean(dim=-1)
    target_distribution = torch.softmax(target_advantage / temperature, dim=-1)
    ranking = torch.sum(-target_distribution * torch.log_softmax(pred_advantage / temperature, dim=-1), dim=-1)
    return regression + ranking_weight * ranking, regression, ranking


def _loss(prediction: torch.Tensor, target: torch.Tensor, *, temperature: float, ranking_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total, regression, ranking = _sample_losses(prediction, target, temperature=temperature, ranking_weight=ranking_weight)
    return total.mean(), regression.mean(), ranking.mean()


def _robust_loss(sample_loss: torch.Tensor, groups: torch.Tensor, weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    # EVEYES co-motion pilot data intentionally contains one controlled
    # condition.  Ignore absent groups instead of turning their empty mean
    # into NaN; four-condition training still produces the same group vector.
    present = [sample_loss[groups == index].mean() for index in range(len(CONDITIONS)) if torch.any(groups == index)]
    robust_term = torch.max(torch.stack(present)) if present else sample_loss.new_zeros(())
    group_losses = sample_loss.new_full((len(CONDITIONS),), float("nan"))
    for index in range(len(CONDITIONS)):
        if torch.any(groups == index):
            group_losses[index] = sample_loss[groups == index].mean()
    return sample_loss.mean() + weight * robust_term, group_losses


def _metrics(prediction: torch.Tensor, target: torch.Tensor, rpm: torch.Tensor) -> dict[str, float]:
    selected = torch.argmax(prediction, dim=-1)
    oracle = torch.argmax(target, dim=-1)
    row = torch.arange(len(selected), device=selected.device)
    regret = target[row, oracle] - target[row, selected]
    return {
        "top1_accuracy": float(torch.mean((selected == oracle).float()).item()),
        "mean_utility_regret": float(torch.mean(regret).item()),
        "p95_utility_regret": float(torch.quantile(regret, 0.95).item()),
        "mean_selected_rpm": float(torch.mean(rpm[selected]).item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--use-flow", action="store_true")
    parser.add_argument("--motion-delta", action="store_true", help="Append a causal one-window IMU speed difference feature.")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--base-policy", type=Path, default=None, help="Optional IMU/event/phase Q checkpoint used to initialize a residual flow adapter.")
    parser.add_argument("--freeze-base", action="store_true", help="Train only the residual flow adapter and gate.")
    parser.add_argument("--group-robust-weight", type=float, default=0.0, help="Weight on the worst controlled-condition training loss.")
    parser.add_argument("--candidate-dir", type=Path, default=None, help="Optionally save predeclared epoch checkpoints for direct validation-rollout selection.")
    parser.add_argument("--candidate-epochs", type=int, nargs="*", default=None, help="Epochs to save; defaults to 0,10,20,50,100,200,last.")
    parser.add_argument("--advantage-rpm", type=float, default=None, help="Center candidate utilities on a fixed RPM baseline, e.g. 180.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.window < 2 or args.epochs < 1 or args.hidden_dim < 2 or args.lr <= 0.0 or args.temperature <= 0.0 or min(args.ranking_weight, args.group_robust_weight) < 0.0:
        parser.error("Window/epochs/hidden-dim/lr/temperature must be positive and ranking-weight non-negative.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    base_rpm = float(manifest["configuration"]["base_rpm"])
    baseline_gear = None if args.advantage_rpm is None else int(np.argmin(np.abs(base_rpm * GEAR_SCALES - float(args.advantage_rpm))))
    train_paths = [Path(record["output"]) for record in manifest["oracles"] if record["split"] == "train"]
    valid_paths = [Path(record["output"]) for record in manifest["oracles"] if record["split"] == "valid"]
    objective = manifest["configuration"]
    train_x, train_q, train_group, train_reports = _load_group(train_paths, use_flow=args.use_flow, motion_delta=args.motion_delta, window=args.window, base_rpm=base_rpm, objective=objective, seed=args.seed, baseline_gear=baseline_gear)
    valid_x, valid_q, valid_group, valid_reports = _load_group(valid_paths, use_flow=args.use_flow, motion_delta=args.motion_delta, window=args.window, base_rpm=base_rpm, objective=objective, seed=args.seed + 100_000, baseline_gear=baseline_gear)
    feature_mean = np.mean(train_x, axis=(0, 1), keepdims=True).astype(np.float32)
    feature_std = np.std(train_x, axis=(0, 1), keepdims=True).astype(np.float32)
    feature_std = np.where(feature_std < 1e-4, 1.0, feature_std)
    train_x = np.clip((train_x - feature_mean) / feature_std, -5.0, 5.0)
    valid_x = np.clip((valid_x - feature_mean) / feature_std, -5.0, 5.0)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = int(train_x.shape[-1])
    if valid_x.shape[-1] != feature_dim:
        raise ValueError("Train and validation oracles disagree on phase-feature availability.")
    expected_without_phase = BASE_FEATURE_DIM + (MOTION_DELTA_FEATURE_DIM if args.motion_delta else 0) + (FLOW_FEATURE_DIM if args.use_flow else 0)
    phase_features = feature_dim == expected_without_phase + 2
    if args.base_policy is not None:
        if not args.use_flow or not phase_features:
            parser.error("--base-policy requires flow and phase features.")
        base_checkpoint = torch.load(args.base_policy, map_location="cpu", weights_only=True)
        if base_checkpoint.get("target_type") != "candidate_q" or bool(base_checkpoint.get("use_flow")) or not bool(base_checkpoint.get("phase_features")):
            parser.error("--base-policy must be a phase-aware candidate-Q checkpoint without flow.")
        model = CausalGatedFlowQPolicy(feature_dim, args.hidden_dim, len(GEAR_SCALES), base_feature_dim=int(base_checkpoint["feature_dim"])).to(device)
        model.base_policy.load_state_dict(base_checkpoint["model"])
        if args.freeze_base:
            for parameter in model.base_policy.parameters():
                parameter.requires_grad_(False)
        architecture = "gated_flow_lstm"
    else:
        model = CausalFlowFusionQPolicy(feature_dim, args.hidden_dim, len(GEAR_SCALES)).to(device)
        architecture = "lstm"
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr, weight_decay=1e-4)
    train_values = torch.from_numpy(train_x).to(device)
    train_targets = torch.from_numpy(train_q).to(device)
    valid_values = torch.from_numpy(valid_x).to(device)
    valid_targets = torch.from_numpy(valid_q).to(device)
    train_groups = torch.from_numpy(train_group).to(device)
    valid_groups = torch.from_numpy(valid_group).to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    requested_candidates = checkpoint_epochs(args.epochs, args.candidate_epochs) if args.candidate_dir is not None else []
    candidate_records: list[dict[str, object]] = []

    def save_candidate(epoch: int) -> None:
        if args.candidate_dir is None or epoch not in requested_candidates:
            return
        args.candidate_dir.mkdir(parents=True, exist_ok=True)
        path = args.candidate_dir / f"epoch_{epoch:04d}.pt"
        torch.save(
            _checkpoint_payload(
                model,
                feature_dim=feature_dim,
                hidden_dim=args.hidden_dim,
                architecture=architecture,
                use_flow=args.use_flow,
                phase_features=phase_features,
                motion_delta=args.motion_delta,
                feature_mean=feature_mean,
                feature_std=feature_std,
                base_rpm=base_rpm,
                temperature=args.temperature,
                epoch=epoch,
            ),
            path,
        )
        candidate_records.append({"epoch": epoch, "path": str(path)})

    save_candidate(0)
    with torch.inference_mode():
        initial_sample_loss, _, _ = _sample_losses(model(valid_values), valid_targets, temperature=args.temperature, ranking_weight=args.ranking_weight)
        best_validation, _ = _robust_loss(initial_sample_loss, valid_groups, args.group_robust_weight)
        best_validation_value = float(best_validation.item())
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sample_loss, _, _ = _sample_losses(model(train_values), train_targets, temperature=args.temperature, ranking_weight=args.ranking_weight)
        total, _ = _robust_loss(sample_loss, train_groups, args.group_robust_weight)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        save_candidate(epoch)
        if epoch == args.epochs or epoch % 10 == 0:
            model.eval()
            with torch.inference_mode():
                valid_sample_loss, _, _ = _sample_losses(model(valid_values), valid_targets, temperature=args.temperature, ranking_weight=args.ranking_weight)
                validation_score, _ = _robust_loss(valid_sample_loss, valid_groups, args.group_robust_weight)
            if float(validation_score.item()) < best_validation_value:
                best_validation_value = float(validation_score.item())
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        train_prediction = model(train_values)
        valid_prediction = model(valid_values)
        train_total, train_regression, train_ranking = _loss(train_prediction, train_targets, temperature=args.temperature, ranking_weight=args.ranking_weight)
        valid_total, valid_regression, valid_ranking = _loss(valid_prediction, valid_targets, temperature=args.temperature, ranking_weight=args.ranking_weight)
        train_sample_loss, _, _ = _sample_losses(train_prediction, train_targets, temperature=args.temperature, ranking_weight=args.ranking_weight)
        valid_sample_loss, _, _ = _sample_losses(valid_prediction, valid_targets, temperature=args.temperature, ranking_weight=args.ranking_weight)
        _, train_group_loss = _robust_loss(train_sample_loss, train_groups, args.group_robust_weight)
        _, valid_group_loss = _robust_loss(valid_sample_loss, valid_groups, args.group_robust_weight)
    gear_rpm = torch.as_tensor(base_rpm * GEAR_SCALES, dtype=torch.float32, device=device)
    train_metrics = _metrics(train_prediction, train_targets, gear_rpm)
    valid_metrics = _metrics(valid_prediction, valid_targets, gear_rpm)
    gate_summary = None
    if isinstance(model, CausalGatedFlowQPolicy):
        with torch.inference_mode():
            train_gate = model.gate_values(train_values)
            valid_gate = model.gate_values(valid_values)
        gate_summary = {"train_mean": float(train_gate.mean().item()), "train_p95": float(torch.quantile(train_gate, 0.95).item()), "valid_mean": float(valid_gate.mean().item()), "valid_p95": float(torch.quantile(valid_gate, 0.95).item())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint_payload(
        model,
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        architecture=architecture,
        use_flow=args.use_flow,
        phase_features=phase_features,
        motion_delta=args.motion_delta,
        feature_mean=feature_mean,
        feature_std=feature_std,
        base_rpm=base_rpm,
        temperature=args.temperature,
        epoch=best_epoch,
    )
    torch.save(checkpoint, args.output)
    if args.candidate_dir is not None:
        candidate_manifest = {
            "selection_split": "valid",
            "selection_requires_direct_phase_continuous_rollout": True,
            "candidates": candidate_records,
        }
        (args.candidate_dir / "manifest.json").write_text(json.dumps(candidate_manifest, indent=2, sort_keys=True), encoding="utf-8")
    report = {
        "device": str(device),
        "target_type": "candidate_q",
        "architecture": architecture,
        "use_flow": args.use_flow,
        "feature_dim": feature_dim,
        "phase_features": phase_features,
        "motion_delta": args.motion_delta,
        "advantage_rpm": args.advantage_rpm,
        "advantage_baseline_gear": baseline_gear,
        "causal_alignment": "history_through_i_minus_1_predicts_all_candidate_utilities_at_i",
        "utility_source": "versioned metrics recomputed for oracle, fixed, and random counterfactual behavior histories",
        "behavior_augmentation": "oracle_path_only_for_phase_continuous_oracles; legacy diagnostics also use fixed_each_gear_and_random",
        "base_policy": str(args.base_policy) if args.base_policy is not None else None,
        "base_frozen": bool(args.freeze_base),
        "gate": gate_summary,
        "candidate_checkpoints": candidate_records,
        "loss": {"temperature": args.temperature, "ranking_weight": args.ranking_weight, "group_robust_weight": args.group_robust_weight, "best_epoch": best_epoch, "best_validation_robust_loss": best_validation_value, "train_total": float(train_total.item()), "train_regression": float(train_regression.item()), "train_ranking": float(train_ranking.item()), "valid_total": float(valid_total.item()), "valid_regression": float(valid_regression.item()), "valid_ranking": float(valid_ranking.item()), "train_by_condition": {name: float(train_group_loss[index].item()) for index, name in enumerate(CONDITIONS)}, "valid_by_condition": {name: float(valid_group_loss[index].item()) for index, name in enumerate(CONDITIONS)}},
        "train": {"samples": len(train_x), **train_metrics},
        "valid": {"samples": len(valid_x), **valid_metrics},
        "scenes": {"train": train_reports, "valid": valid_reports},
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
