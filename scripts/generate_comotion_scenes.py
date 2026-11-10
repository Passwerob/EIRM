#!/usr/bin/env python3
"""Generate controlled camera/object relative-motion scenes for RPM control."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import imageio.v3 as imageio
import numpy as np


CONDITIONS = ("camera_only", "object_only", "co_motion", "counter_motion")


def _texture(seed: int, size: int, *, object_texture: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse = rng.uniform(0.0, 1.0, (size // 4 + 2, size // 4 + 2, 3)).astype(np.float32)
    values = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[:size, :size]
    if object_texture:
        pattern = ((xx // 3 + yy // 3) % 2).astype(np.float32)[..., None]
        values = 0.35 * values + 0.65 * pattern
    else:
        bands = (0.5 + 0.5 * np.sin(0.31 * xx + 0.17 * yy))[..., None]
        values = 0.7 * values + 0.3 * bands
    return np.clip(35.0 + 205.0 * values, 0.0, 255.0).astype(np.uint8)


def _warp(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _render_scene(
    root: Path,
    *,
    split: str,
    seed: int,
    condition: str,
    frames: int,
    fps: float,
    size: int,
    motion_profile: str,
    motion_scale: float = 1.0,
    object_alpha: float | None = None,
    scene_name: str | None = None,
) -> dict[str, object]:
    """Render one controlled scene.

    ``motion_scale`` multiplies the nominal image-plane trajectory (pixels).
    ``object_alpha`` if set uses co-motion mismatch: world v_object / v_camera
    = alpha, so the image-plane object warp is ``(1 - alpha) * camera_warp``.
    alpha=1 is relative lock. This extra kwargs path does not change the
    default pack CLI.
    """
    if scene_name is None:
        scene_name = f"{split}_seed{seed:02d}_{condition}"
    scene = root / scene_name
    image_dir, mask_dir = scene / "images", scene / "object_masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(10_000 + seed)
    times = np.arange(frames, dtype=np.float64) / fps
    duration = max(float(times[-1]), 1.0 / fps)
    frequency = 1.2 + 0.16 * (seed % 5)
    chirp = 0.9 + 0.1 * (seed % 3)
    phase = rng.uniform(-np.pi, np.pi)
    amplitude_x = (7.0 + 0.5 * (seed % 4)) * float(motion_scale)
    amplitude_y = (2.5 + 0.3 * ((seed + 1) % 3)) * float(motion_scale)
    if motion_profile == "smooth":
        angle = 2.0 * np.pi * (frequency * times + 0.5 * chirp * times**2 / duration) + phase
    elif motion_profile == "switching":
        # Piecewise angular rates create explicit speed transitions while
        # integrating phase continuously, which is the controller stress test.
        multipliers = np.asarray((0.45, 1.55, 0.65, 1.35), dtype=np.float64)
        segment = np.minimum((times / max(duration, 1e-12) * len(multipliers)).astype(np.int64), len(multipliers) - 1)
        angular_rate = 2.0 * np.pi * frequency * multipliers[segment]
        angle = phase + np.concatenate(([0.0], np.cumsum(0.5 * (angular_rate[1:] + angular_rate[:-1]) * np.diff(times))))
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(motion_profile)
    camera_x = amplitude_x * np.sin(angle)
    camera_y = amplitude_y * np.sin(0.73 * angle + 0.4)
    if object_alpha is not None:
        background_x, background_y = camera_x, camera_y
        physical_camera_x, physical_camera_y = camera_x, camera_y
        object_x = (1.0 - float(object_alpha)) * camera_x
        object_y = (1.0 - float(object_alpha)) * camera_y
        condition = "mismatch"
    elif condition == "object_only":
        background_x = np.zeros_like(camera_x)
        background_y = np.zeros_like(camera_y)
        object_x, object_y = camera_x, camera_y
        physical_camera_x = np.zeros_like(camera_x)
        physical_camera_y = np.zeros_like(camera_y)
    else:
        background_x, background_y = camera_x, camera_y
        physical_camera_x, physical_camera_y = camera_x, camera_y
        if condition == "camera_only":
            object_x, object_y = camera_x, camera_y
        elif condition == "co_motion":
            object_x, object_y = np.zeros_like(camera_x), np.zeros_like(camera_y)
        elif condition == "counter_motion":
            object_x, object_y = -0.8 * camera_x, -0.8 * camera_y
        else:  # pragma: no cover - guarded by caller
            raise ValueError(condition)

    background = _texture(seed * 17 + 3, size)
    patch_size = max(20, int(round(size * 0.43)))
    patch = _texture(seed * 29 + 11, patch_size, object_texture=True)
    alpha = np.zeros((patch_size, patch_size), dtype=np.uint8)
    cv2.rectangle(alpha, (2, 2), (patch_size - 3, patch_size - 3), 255, thickness=-1)
    cv2.circle(alpha, (patch_size // 2, patch_size // 2), patch_size // 5, 0, thickness=-1)
    base_x = (size - patch_size) // 2
    base_y = (size - patch_size) // 2
    masks: list[np.ndarray] = []
    for index in range(frames):
        frame = _warp(background, float(background_x[index]), float(background_y[index]))
        object_layer = np.zeros_like(frame)
        mask_layer = np.zeros((size, size), dtype=np.uint8)
        object_layer[base_y : base_y + patch_size, base_x : base_x + patch_size] = patch
        mask_layer[base_y : base_y + patch_size, base_x : base_x + patch_size] = alpha
        object_layer = _warp(object_layer, float(object_x[index]), float(object_y[index]))
        mask = _warp(mask_layer, float(object_x[index]), float(object_y[index]))
        blend = (mask.astype(np.float32) / 255.0)[..., None]
        frame = np.clip(frame * (1.0 - blend) + object_layer * blend, 0.0, 255.0).astype(np.uint8)
        masks.append(mask > 127)
        imageio.imwrite(image_dir / f"{index:06d}.png", frame)
        imageio.imwrite(mask_dir / f"{index:06d}.png", mask)

    fx = fy = 0.9 * size
    vx = np.gradient(physical_camera_x, times)
    vy = np.gradient(physical_camera_y, times)
    gyro = np.column_stack((vy / fy, -vx / fx, np.zeros_like(vx)))
    with (scene / "timestamps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("index", "timestamp_seconds"))
        writer.writerows((index, f"{timestamp:.9f}") for index, timestamp in enumerate(times))
    with (scene / "imu.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("timestamp", "ax", "ay", "az", "gx", "gy", "gz"))
        writer.writerows((f"{timestamp:.9f}", 0.0, 0.0, 9.80665, *values) for timestamp, values in zip(times, gyro))
    intrinsics = {"width": size, "height": size, "fx": fx, "fy": fy, "cx": (size - 1) / 2.0, "cy": (size - 1) / 2.0}
    (scene / "camera_intrinsics.json").write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")
    image_motion = {
        "camera_only": "same_as_background",
        "object_only": "moving_with_static_background",
        "co_motion": "stationary_while_background_moves",
        "counter_motion": "opposes_background",
        "mismatch": "object_image_warp=(1-alpha)*camera_warp",
    }
    metadata = {
        "scene": scene.name,
        "split": split,
        "seed": seed,
        "condition": condition,
        "frame_count": frames,
        "fps": fps,
        "motion_profile": motion_profile,
        "motion_scale": float(motion_scale),
        "object_alpha": None if object_alpha is None else float(object_alpha),
        "camera_motion_present": bool(condition != "object_only"),
        "object_image_motion": image_motion[condition],
        "object_roi_fraction_mean": float(np.mean(masks)),
        "scientific_role": "controlled camera/object relative-motion test, not photorealistic evidence",
    }
    np.savez_compressed(
        scene / "image_plane_trajectory.npz",
        t=times,
        background_x=np.asarray(background_x, dtype=np.float64),
        background_y=np.asarray(background_y, dtype=np.float64),
        object_x=np.asarray(object_x, dtype=np.float64),
        object_y=np.asarray(object_y, dtype=np.float64),
    )
    (scene / "scene.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/comotion_controlled"))
    parser.add_argument("--train-seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4, 5))
    parser.add_argument("--valid-seeds", type=int, nargs="+", default=(6, 7, 8))
    parser.add_argument("--test-seeds", type=int, nargs="+", default=(9, 10, 11))
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--motion-profile", choices=("smooth", "switching"), default="smooth", help="Smooth chirped motion or piecewise speed transitions for controller stress tests.")
    args = parser.parse_args()
    if args.frames < 16 or args.fps <= 0.0 or args.size < 48:
        parser.error("frames must be >=16, fps positive, and size >=48")
    seed_groups = {"train": args.train_seeds, "valid": args.valid_seeds, "test": args.test_seeds}
    flat = [seed for seeds in seed_groups.values() for seed in seeds]
    if len(flat) != len(set(flat)):
        parser.error("Train, validation, and test seeds must be disjoint.")
    records = [
        _render_scene(args.output, split=split, seed=seed, condition=condition, frames=args.frames, fps=args.fps, size=args.size, motion_profile=args.motion_profile)
        for split, seeds in seed_groups.items()
        for seed in seeds
        for condition in CONDITIONS
    ]
    manifest = {"dataset": "AMI controlled co-motion v1", "splits": seed_groups, "conditions": list(CONDITIONS), "motion_profile": args.motion_profile, "same_texture_seed_conditions_share_split": True, "records": records}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scenes": len(records), "splits": {key: len(value) * len(CONDITIONS) for key, value in seed_groups.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
