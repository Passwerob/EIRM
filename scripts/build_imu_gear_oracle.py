#!/usr/bin/env python3
"""Build compact event-aware gear labels by evaluating prism-speed candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
import imageio.v3 as imageio

from ami_simulator.events import ThresholdEventSimulator
from ami_simulator.control import CausalGearSelector
from ami_simulator.event_quality import QUALITY_PROTOCOL_VERSION, active_pixel_ids, compute_reference_edge_mask, event_quality_metrics, event_quality_score
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.rotation import ConstantRotation
from ami_simulator.types import CameraIntrinsics, EventStream
from ami_simulator.warp import PrismWarper, compensate_events
from ami_simulator.dsec import read_imu_csv
from ami_simulator.types import ImuStream
from ami_simulator.vfi import DebugLinearVFIBackend, PracticalRIFEBackend, resample_with_vfi
from ami_simulator.flow import estimate_flow_features, FLOW_FEATURE_NAMES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output .npz cost table.")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--base-rpm", type=float, default=720.0)
    parser.add_argument("--gear-scales", type=float, nargs=5, default=(1.5, 1.0, 0.5, 0.25, 0.0))
    parser.add_argument("--contrast-threshold", type=float, default=0.2)
    parser.add_argument("--imu-csv", type=Path, required=True)
    parser.add_argument("--target-density-low", type=float, default=0.8, help="Target density at the highest measured motion.")
    parser.add_argument("--target-density-high", type=float, default=1.6, help="Target density at the lowest measured motion.")
    parser.add_argument("--resize-width", type=int, default=360, help="Optional RGB width for faster oracle generation.")
    parser.add_argument("--max-phase-step-deg", type=float, default=5.0, help="Maximum prism phase step between oracle samples.")
    parser.add_argument("--timestamp-unit", choices=("s", "ms", "us", "ns"), default="us")
    parser.add_argument("--prism-angle-deg", type=float, default=0.5)
    parser.add_argument("--motion-source", choices=("visual", "imu", "max"), default="visual", help="Natural-motion proxy for the target-density schedule.")
    parser.add_argument("--time-scale", type=float, default=1.0, help="Scale physical interval duration; <1 is faster and >1 is slower.")
    parser.add_argument("--imu-timestamp-unit", choices=("s", "ms", "us", "ns"), default="us")
    parser.add_argument("--interpolation-fps", type=float, default=None)
    parser.add_argument("--vfi", choices=("none", "practical_rife", "debug_linear"), default="none")
    parser.add_argument("--practical-rife-root", type=Path, default=None)
    parser.add_argument("--rife-checkpoint", type=Path, default=None)
    parser.add_argument("--rife-device", default=None)
    parser.add_argument("--rife-fp16", action="store_true")
    parser.add_argument("--rife-batch-size", type=int, default=8)
    parser.add_argument("--visual-motion-quantile", type=float, default=0.9, help="Spatial quantile used to retain localized object motion.")
    parser.add_argument("--edge-quantile", type=float, default=0.75, help="Clean/source-reference gradient quantile used for the frozen edge mask.")
    parser.add_argument("--visual-motion-bounds", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"), help="Shared training-set bounds for absolute visual speed normalization.")
    parser.add_argument("--imu-motion-bounds", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"), help="Shared training-set bounds for IMU speed normalization.")
    parser.add_argument("--event-efficiency-weight", type=float, default=0.5, help="Reward unique active pixels per emitted event.")
    parser.add_argument("--coverage-efficiency-weight", type=float, default=0.0, help="Reward target-edge coverage weighted by unique-event efficiency.")
    parser.add_argument("--density-weight", type=float, default=0.1, help="Penalty for deviation from the motion-conditioned event-density target.")
    parser.add_argument("--rpm-effort-weight", type=float, default=0.05, help="Penalty on squared normalized prism speed.")
    parser.add_argument("--rpm-change-weight", type=float, default=0.05, help="Penalty on squared normalized speed change from the previous command.")
    parser.add_argument("--minimum-dwell-seconds", type=float, default=0.05, help="Minimum time between discrete RPM gear changes.")
    parser.add_argument("--switch-margin", type=float, default=0.02, help="Minimum objective improvement required to change gear.")
    parser.add_argument("--store-active-masks", action="store_true", help="Store packed candidate active-pixel masks for counterfactual policy evaluation.")
    parser.add_argument("--flow-features", action="store_true", help="Store causal dense-RGB optical-flow summaries computed before prism warping.")
    parser.add_argument("--roi-mask-dir", type=Path, default=None, help="Optional per-frame binary masks. If omitted, scene_root/object_masks is used when present.")
    args = parser.parse_args()
    if args.max_frames < 2 or args.resize_width < 0 or args.target_density_low <= 0.0 or args.target_density_high < args.target_density_low or args.max_phase_step_deg <= 0.0 or args.time_scale <= 0.0:
        parser.error("max-frames must be >= 2, resize-width non-negative, target densities positive and ordered, and max-phase-step-deg positive")
    if (args.interpolation_fps is None) != (args.vfi == "none"):
        parser.error("Set both --interpolation-fps and a non-none --vfi, or set neither.")
    if args.vfi == "practical_rife" and args.practical_rife_root is None:
        parser.error("--practical-rife-root is required for Practical-RIFE.")
    if not 0.0 < args.visual_motion_quantile < 1.0 or not 0.0 < args.edge_quantile < 1.0:
        parser.error("Visual-motion and edge quantiles must lie inside (0, 1).")
    for bounds in (args.visual_motion_bounds, args.imu_motion_bounds):
        if bounds is not None and (not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]):
            parser.error("Motion bounds must be finite and strictly increasing.")
    if min(args.event_efficiency_weight, args.coverage_efficiency_weight, args.density_weight, args.rpm_effort_weight, args.rpm_change_weight, args.minimum_dwell_seconds, args.switch_margin) < 0.0:
        parser.error("Quality-objective weights must be non-negative.")

    scene = args.scene_root
    image_dir = scene / "images" if (scene / "images").is_dir() else scene / "rgb"
    timestamp_candidates = list(scene.glob("*_image_timestamps.txt")) or [scene / "timestamps.csv"]
    if not timestamp_candidates[0].exists():
        raise FileNotFoundError(f"No image timestamp file found in {scene}.")
    timestamp_path = timestamp_candidates[0]
    frames, timestamps = load_rgb_sequence(image_dir, timestamp_path, timestamp_unit=args.timestamp_unit, max_frames=args.max_frames)
    roi_dir = args.roi_mask_dir or (scene / "object_masks")
    roi_masks = None
    if roi_dir.is_dir():
        mask_paths = sorted(path for path in roi_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})[: len(frames)]
        if len(mask_paths) != len(frames):
            raise ValueError(f"ROI mask count ({len(mask_paths)}) must match loaded frame count ({len(frames)}).")
        loaded_masks = [np.asarray(imageio.imread(path)) for path in mask_paths]
        roi_masks = np.stack([mask[..., 0] if mask.ndim == 3 else mask for mask in loaded_masks]) > 0
        if roi_masks.shape[1:3] != frames.shape[1:3]:
            raise ValueError("ROI masks must have the same spatial resolution as source frames.")
    original_height, original_width = frames.shape[1:3]
    intrinsics_path = scene / "camera_intrinsics.json"
    transforms_path = scene / "transforms.json"
    if intrinsics_path.exists():
        intrinsics_data = json.loads(intrinsics_path.read_text(encoding="utf-8"))
        fx, fy = float(intrinsics_data["fx"]), float(intrinsics_data["fy"])
        cx, cy = float(intrinsics_data["cx"]), float(intrinsics_data["cy"])
    elif transforms_path.exists():
        intrinsics_data = json.loads(transforms_path.read_text(encoding="utf-8"))
        fx, fy = float(intrinsics_data["fl_x"]), float(intrinsics_data["fl_y"])
        cx, cy = float(intrinsics_data["cx"]), float(intrinsics_data["cy"])
    else:
        fx = fy = 1164.6238115833075
        cx, cy = 713.5791168212891, 570.9349365234375
    intrinsics = CameraIntrinsics(original_width, original_height, fx, fy, cx, cy)
    if args.resize_width and args.resize_width < original_width:
        resized_width = args.resize_width
        resized_height = int(round(original_height * resized_width / original_width))
        frame_tensor = torch.from_numpy(frames.astype(np.float32)).permute(0, 3, 1, 2)
        frames = functional.interpolate(frame_tensor, size=(resized_height, resized_width), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).numpy()
        if roi_masks is not None:
            roi_tensor = torch.from_numpy(roi_masks.astype(np.float32)).unsqueeze(1)
            roi_masks = functional.interpolate(roi_tensor, size=(resized_height, resized_width), mode="nearest")[:, 0].numpy() > 0.5
        intrinsics = intrinsics.resized(resized_width, resized_height)
    source_timestamps = timestamps.copy()
    timestamps = timestamps[0] + (timestamps - timestamps[0]) * args.time_scale
    vfi_metadata: dict[str, object] = {"vfi_backend": "none"}
    gear_scales = np.asarray(args.gear_scales, dtype=np.float64)
    required_phase_fps = float(args.base_rpm * gear_scales.max() * 6.0 / args.max_phase_step_deg)
    effective_interpolation_fps = args.interpolation_fps
    if args.vfi != "none":
        if args.vfi == "practical_rife":
            vfi = PracticalRIFEBackend(args.practical_rife_root, args.rife_checkpoint, device=args.rife_device, inference_batch_size=args.rife_batch_size, fp16=args.rife_fp16)
        else:
            vfi = DebugLinearVFIBackend()
        effective_interpolation_fps = max(float(args.interpolation_fps), required_phase_fps)
        frames, timestamps = resample_with_vfi(frames, timestamps, target_fps=effective_interpolation_fps, backend=vfi)
        if roi_masks is not None:
            source_scaled = source_timestamps[0] + (source_timestamps - source_timestamps[0]) * args.time_scale
            nearest = np.abs(timestamps[:, None] - source_scaled[None, :]).argmin(axis=1)
            roi_masks = roi_masks[nearest]
        vfi_metadata = vfi.metadata()
    luminance = rgb_to_linear_luminance(frames)
    flow_features = estimate_flow_features(frames, timestamps) if args.flow_features else None
    height, width = luminance.shape[1:]
    simulator = ThresholdEventSimulator(args.contrast_threshold)
    imu = read_imu_csv(args.imu_csv, timestamp_unit=args.imu_timestamp_unit, gyro_unit="rad/s")
    # Reparameterize the same physical trajectory on the augmented time axis.
    # Angular velocity scales as 1/s and acceleration as 1/s^2.
    image_origin, augmented_origin = float(source_timestamps[0]), float(timestamps[0])
    imu = ImuStream(
        augmented_origin + (imu.t - image_origin) * args.time_scale,
        imu.accel / (args.time_scale**2),
        imu.gyro / args.time_scale,
    )
    motion = np.linalg.norm(imu.gyro, axis=1)
    window_motion = np.empty(len(timestamps) - 1, dtype=np.float64)
    for index, (start, end) in enumerate(zip(timestamps[:-1], timestamps[1:])):
        left = int(np.searchsorted(imu.t, start, side="left"))
        right = int(np.searchsorted(imu.t, end, side="right"))
        values = motion[left:right]
        window_motion[index] = float(values.mean() if len(values) else np.interp((start + end) * 0.5, imu.t, motion))
    visual_delta = np.quantile(
        np.abs(np.diff(np.log(np.clip(luminance, 1e-6, None)), axis=0)),
        args.visual_motion_quantile,
        axis=(1, 2),
    ).astype(np.float64)
    visual_motion = visual_delta / np.diff(timestamps)
    visual_low, visual_high = args.visual_motion_bounds if args.visual_motion_bounds is not None else np.quantile(visual_motion, [0.1, 0.9])
    visual_fraction = np.clip((visual_motion - visual_low) / max(visual_high - visual_low, 1e-9), 0.0, 1.0)
    imu_low, imu_high = args.imu_motion_bounds if args.imu_motion_bounds is not None else np.quantile(window_motion, [0.1, 0.9])
    imu_fraction = np.clip((window_motion - imu_low) / max(imu_high - imu_low, 1e-9), 0.0, 1.0)
    if args.motion_source == "imu":
        motion_fraction = imu_fraction
    elif args.motion_source == "max":
        motion_fraction = np.maximum(visual_fraction, imu_fraction)
    else:
        motion_fraction = visual_fraction
    target_density = args.target_density_high + motion_fraction * (args.target_density_low - args.target_density_high)
    model = RotatingWedgePrism(intrinsics, args.prism_angle_deg)
    counts = np.zeros((len(timestamps) - 1, len(gear_scales)), dtype=np.int64)
    densities = np.zeros_like(counts, dtype=np.float64)
    coverage = np.zeros_like(densities)
    edge_precision = np.zeros_like(densities)
    edge_f1 = np.zeros_like(densities)
    structural_ratio = np.zeros_like(densities)
    unique_ratio = np.zeros_like(densities)
    event_efficiency = np.zeros_like(densities)
    edge_event_efficiency = np.zeros_like(densities)
    coverage_efficiency = np.zeros_like(densities)
    repeat_event_fraction = np.zeros_like(densities)
    sensor_unique_ratio = np.zeros_like(densities)
    sensor_event_efficiency = np.zeros_like(densities)
    reference_valid_event_fraction = np.zeros_like(densities)
    temporal_overlap = np.zeros_like(densities)
    quality_score = np.zeros_like(densities)
    valid_fraction = np.zeros_like(densities)
    polarity_balance = np.zeros_like(densities)
    previous_active = None
    phase_step_rad = np.deg2rad(args.max_phase_step_deg)
    previous_rpm = float(args.base_rpm)
    current_phase = 0.0
    phase_starts = np.empty(len(timestamps) - 1, dtype=np.float64)
    maximum_rpm = max(float(args.base_rpm * gear_scales.max()), 1.0)
    rpm_values = args.base_rpm * gear_scales
    selector = CausalGearSelector(
        rpm_values,
        initial_rpm=previous_rpm,
        initial_time=float(timestamps[0]),
        minimum_dwell_seconds=args.minimum_dwell_seconds,
        switch_margin=args.switch_margin,
    )
    labels = np.empty(len(timestamps) - 1, dtype=np.int64)
    unconstrained_labels = np.empty_like(labels)
    cost = np.empty_like(quality_score)
    packed_active_masks = np.zeros((len(timestamps) - 1, len(gear_scales), (height * width + 7) // 8), dtype=np.uint8) if args.store_active_masks else None
    for index, (start, end) in enumerate(zip(timestamps[:-1], timestamps[1:])):
        phase_starts[index] = current_phase
        candidate_active = []
        for gear, scale in enumerate(gear_scales):
            rpm = args.base_rpm * float(scale)
            rotation = ConstantRotation(rpm)
            phase_delta = abs(rotation.omega_rad_s * float(end - start))
            subdivisions = max(1, int(np.ceil(phase_delta / phase_step_rad)))
            pair_times = np.linspace(float(start), float(end), subdivisions + 1, dtype=np.float64)
            fractions = (pair_times - float(start)) / max(float(end - start), 1e-12)
            pair_values = luminance[index][None] * (1.0 - fractions[:, None, None]) + luminance[index + 1][None] * fractions[:, None, None]
            pair = torch.from_numpy(pair_values.astype(np.float32)).unsqueeze(1)
            # Every candidate starts from the same physical encoder phase.
            # Only the selected action advances that shared phase state.
            warper = PrismWarper(model, rotation, height=height, width=width, theta0=current_phase, reference_time=float(start))
            with torch.inference_mode():
                sensor, valid = warper.warp(pair, pair_times)
            events = simulator.simulate(sensor[:, 0].numpy(), pair_times)
            count = len(events)
            counts[index, gear] = count
            densities[index, gear] = count / float(height * width)
            compensated = compensate_events(events, model, rotation, theta0=current_phase, reference_time=float(start))
            reference_events = EventStream(
                np.rint(compensated.extras["x_reference"]),
                np.rint(compensated.extras["y_reference"]),
                compensated.t,
                compensated.p,
            )
            sensor_metrics = event_quality_metrics(events, image_shape=(height, width))
            reference_edge_mask = None
            if roi_masks is not None:
                reference_edge_mask = compute_reference_edge_mask(luminance[index], edge_quantile=args.edge_quantile) & roi_masks[index]
            metrics = event_quality_metrics(
                reference_events,
                reference_intensity=None if reference_edge_mask is not None else luminance[index],
                reference_edge_mask=reference_edge_mask,
                previous_active_pixels=previous_active,
                edge_quantile=args.edge_quantile,
                image_shape=(height, width),
            )
            # The density target is a sensor bandwidth budget; structural and
            # redundancy metrics are measured in the compensated scene plane.
            metrics["event_density"] = densities[index, gear]
            sensor_unique_ratio[index, gear] = float(sensor_metrics["unique_pixel_ratio"])
            sensor_event_efficiency[index, gear] = float(sensor_metrics["event_efficiency"])
            reference_valid_event_fraction[index, gear] = float(metrics["event_count"] / count) if count else 1.0
            coverage[index, gear] = float(metrics["edge_coverage"])
            edge_precision[index, gear] = float(metrics["edge_precision"])
            edge_f1[index, gear] = float(metrics["edge_f1"])
            structural_ratio[index, gear] = float(metrics["structural_ratio"])
            unique_ratio[index, gear] = float(metrics["unique_pixel_ratio"])
            event_efficiency[index, gear] = float(metrics["event_efficiency"])
            edge_event_efficiency[index, gear] = float(metrics["edge_event_efficiency"])
            coverage_efficiency[index, gear] = float(metrics["coverage_efficiency"])
            repeat_event_fraction[index, gear] = float(metrics["repeat_event_fraction"])
            temporal_overlap[index, gear] = float(metrics["temporal_overlap"])
            polarity_balance[index, gear] = float(metrics["polarity_imbalance"])
            quality_score[index, gear] = event_quality_score(
                metrics,
                target_density=float(target_density[index]),
                density_weight=args.density_weight,
                efficiency_weight=args.event_efficiency_weight,
                coverage_efficiency_weight=args.coverage_efficiency_weight,
            )
            candidate_active.append(active_pixel_ids(reference_events, image_shape=(height, width)))
            if packed_active_masks is not None:
                active_mask = np.zeros(height * width, dtype=np.uint8)
                active_mask[candidate_active[-1]] = 1
                packed_active_masks[index, gear] = np.packbits(active_mask)
            valid_fraction[index, gear] = float(valid.all(dim=0).float().mean())
        # Advance the reference with the selected candidate only after scoring
        # all speeds. This makes overlap a causal quantity, not a future leak.
        cost[index] = -quality_score[index] + 0.25 * (1.0 - valid_fraction[index])
        cost[index] += args.rpm_effort_weight * np.square(rpm_values / maximum_rpm)
        cost[index] += args.rpm_change_weight * np.square((rpm_values - selector.rpm) / maximum_rpm)
        selected, unconstrained = selector.select(cost[index], timestamp=float(end))
        unconstrained_labels[index] = unconstrained
        labels[index] = selected
        previous_active = candidate_active[selected]
        current_phase = float(np.remainder(current_phase + rpm_values[selected] * (2.0 * np.pi / 60.0) * float(end - start), 2.0 * np.pi))
    selected_rpm = args.base_rpm * gear_scales[labels]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_fields = dict(counts=counts, densities=densities, coverage=coverage, coverage_efficiency=coverage_efficiency, edge_precision=edge_precision, edge_f1=edge_f1, structural_ratio=structural_ratio, unique_ratio=unique_ratio, event_efficiency=event_efficiency, edge_event_efficiency=edge_event_efficiency, repeat_event_fraction=repeat_event_fraction, sensor_unique_ratio=sensor_unique_ratio, sensor_event_efficiency=sensor_event_efficiency, reference_valid_event_fraction=reference_valid_event_fraction, temporal_overlap=temporal_overlap, quality_score=quality_score, valid_fraction=valid_fraction, polarity_balance=polarity_balance, cost=cost, labels=labels, unconstrained_labels=unconstrained_labels, selected_rpm=selected_rpm, prism_phase_start=phase_starts, timestamps=timestamps[1:], motion=window_motion, imu_motion=imu_fraction, visual_motion=visual_motion, natural_motion=motion_fraction, motion_source=args.motion_source, target_density=target_density)
    if flow_features is not None:
        output_fields["flow_features"] = flow_features
    if packed_active_masks is not None:
        output_fields["active_masks_packed"] = packed_active_masks
        output_fields["active_mask_pixel_count"] = np.asarray(height * width, dtype=np.int64)
    np.savez_compressed(args.output, **output_fields)
    summary = {
        "scene": scene.name,
        "windows": int(len(labels)),
        "resolution": {"width": width, "height": height},
        "base_rpm": float(args.base_rpm),
        "gear_scales": gear_scales.tolist(),
        "target_density_range": [args.target_density_low, args.target_density_high],
        "mean_target_density": float(target_density.mean()),
        "motion_range_rad_s": [float(imu_low), float(imu_high)],
        "visual_motion_range": [float(visual_motion.min()), float(visual_motion.max())],
        "visual_motion_quantile": args.visual_motion_quantile,
        "edge_reference": "source_rgb_proxy_with_object_roi" if roi_masks is not None else "source_rgb_proxy_not_clean_irradiance",
        "roi_evaluation": {"enabled": roi_masks is not None, "mask_directory": str(roi_dir) if roi_masks is not None else None, "contract": "ROI edge coverage is rewarded while all full-frame events remain in precision, efficiency, overlap, and bandwidth penalties."},
        "event_metric_coordinate_policy": {"sensor_plane": ["counts", "densities", "sensor_unique_ratio", "sensor_event_efficiency"], "reference_plane_after_per_event_inverse_prism_warp": ["coverage", "edge_precision", "edge_f1", "unique_ratio", "event_efficiency", "edge_event_efficiency", "coverage_efficiency", "temporal_overlap"]},
        "coverage_efficiency_weight": args.coverage_efficiency_weight,
        "edge_quantile": args.edge_quantile,
        "visual_motion_normalization_bounds": [float(visual_low), float(visual_high)],
        "imu_motion_normalization_bounds": [float(imu_low), float(imu_high)],
        "motion_bounds_source": "explicit_training_bounds" if args.visual_motion_bounds is not None or args.imu_motion_bounds is not None else "per_sequence_smoke_only",
        "flow_features": {"enabled": flow_features is not None, "names": list(FLOW_FEATURE_NAMES), "source": "pre_prism_rgb_pairs", "causal_alignment": "feature_at_i_uses_pair_i_minus_1"},
        "motion_source": args.motion_source,
        "quality_objective": {
            "protocol_version": QUALITY_PROTOCOL_VERSION,
            "role": "provisional_controller_surrogate_not_benchmark_ground_truth",
            "event_efficiency_weight": args.event_efficiency_weight,
            "coverage_efficiency_weight": args.coverage_efficiency_weight,
            "density_weight": args.density_weight,
            "rpm_effort_weight": args.rpm_effort_weight,
            "rpm_change_weight": args.rpm_change_weight,
            "invalid_fraction_weight": 0.25,
            "minimum_dwell_seconds": args.minimum_dwell_seconds,
            "switch_margin": args.switch_margin,
        },
        "stores_active_masks": args.store_active_masks,
        "candidate_phase_policy": "all_gears_share_oracle_path_encoder_phase_at_window_start",
        "phase_continuity": "selected_gear_advances_phase_to_next_window",
        "time_scale": args.time_scale,
        "source_frame_count": int(len(source_timestamps)),
        "simulated_frame_count": int(len(timestamps)),
        "requested_interpolation_fps": args.interpolation_fps,
        "effective_interpolation_fps": effective_interpolation_fps,
        "required_phase_fps": required_phase_fps,
        "vfi": vfi_metadata,
        "label_counts": np.bincount(labels, minlength=len(gear_scales)).tolist(),
        "unconstrained_label_counts": np.bincount(unconstrained_labels, minlength=len(gear_scales)).tolist(),
        "gear_change_fraction": float(np.mean(np.diff(labels) != 0)) if len(labels) > 1 else 0.0,
        "mean_density_by_gear": densities.mean(axis=0).tolist(),
        "mean_edge_precision_by_gear": edge_precision.mean(axis=0).tolist(),
        "mean_edge_recall_by_gear": coverage.mean(axis=0).tolist(),
        "mean_edge_f1_by_gear": edge_f1.mean(axis=0).tolist(),
        "mean_cost_by_gear": cost.mean(axis=0).tolist(),
        "mean_quality_by_gear": quality_score.mean(axis=0).tolist(),
        "mean_event_efficiency_by_gear": event_efficiency.mean(axis=0).tolist(),
        "mean_sensor_event_efficiency_by_gear": sensor_event_efficiency.mean(axis=0).tolist(),
        "mean_reference_valid_event_fraction_by_gear": reference_valid_event_fraction.mean(axis=0).tolist(),
        "mean_repeat_event_fraction_by_gear": repeat_event_fraction.mean(axis=0).tolist(),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
