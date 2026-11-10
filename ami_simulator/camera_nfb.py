"""Learned camera-path negative feedback.

The actuator is discrete path rate {1.5, 1.0, 0.5}× along an already recorded
RGB trajectory. Inverse warp still undoes the prism only, not camera motion.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from ami_simulator.inverse_motion import CAMERA_HOLD_RATE, CAMERA_RATES, CAMERA_STORM_OCCUPANCY, inverse_rpm_index
from ami_simulator.motion_net import TinyCausalGearPolicy
from ami_simulator.types import CameraIntrinsics

FEATURE_DIM = 4
WINDOW = 8
FEATURE_NAMES = ("lagged_visual_motion", "lagged_pose_speed", "previous_occupancy", "previous_camera_rate")
ARCHITECTURE = "tiny_camera_nfb_gear"
VISUAL_MOTION_QUANTILE = 0.9


def sample_path(volume: np.ndarray, path_s: float) -> np.ndarray:
    """Linearly interpolate a T×H×W (or T×H×W×C) volume along the path parameter."""
    last = len(volume) - 1
    path_s = float(np.clip(path_s, 0.0, last))
    index = int(np.floor(path_s))
    next_index = min(index + 1, last)
    alpha = path_s - index
    return (1.0 - alpha) * volume[index] + alpha * volume[next_index]


def pair_visual_motion(first: np.ndarray, second: np.ndarray, dt_seconds: float) -> float:
    """Causal 0.9-quantile log-luminance change over the sampled pair, per second."""
    delta = np.quantile(
        np.abs(np.log(np.clip(second, 1e-6, None)) - np.log(np.clip(first, 1e-6, None))),
        VISUAL_MOTION_QUANTILE,
    )
    return float(delta / max(float(dt_seconds), 1e-9))


def load_pose_speed(scene: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader((scene / "pose_speed.csv").open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"{scene} pose_speed.csv is empty.")
    times = np.asarray([float(row["timestamp_seconds"]) for row in rows], dtype=np.float64)
    speeds = np.asarray([float(row["speed_mps"]) for row in rows], dtype=np.float64)
    return times, speeds


def sample_pose_speed(times: np.ndarray, speeds: np.ndarray, path_s: float, timestamps: np.ndarray) -> float:
    last = len(timestamps) - 1
    query = float(np.interp(np.clip(path_s, 0.0, last), np.arange(len(timestamps), dtype=np.float64), timestamps))
    return float(np.interp(query, times, speeds))


def normalize_speed(speed_mps: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return float(np.clip((float(speed_mps) - low) / max(high - low, 1e-9), 0.0, 1.0))


def resize_luminance(frames: np.ndarray, intrinsics: CameraIntrinsics, eval_width: int) -> tuple[np.ndarray, CameraIntrinsics]:
    if eval_width == int(intrinsics.width):
        return frames, intrinsics
    if eval_width < 32:
        raise ValueError("eval_width must be at least 32.")
    height = int(round(intrinsics.height * eval_width / intrinsics.width))
    resized = np.stack([cv2.resize(frame, (eval_width, height), interpolation=cv2.INTER_AREA) for frame in frames])
    return resized, intrinsics.resized(eval_width, height)


def camera_feature(
    lagged_visual: float,
    lagged_speed: float,
    previous_occupancy: float,
    previous_rate: float,
) -> np.ndarray:
    return np.asarray((lagged_visual, lagged_speed, previous_occupancy, previous_rate), dtype=np.float32)


def choose_learned_camera_rate(
    features: list[np.ndarray],
    model: TinyCausalGearPolicy,
    checkpoint: dict[str, object],
    *,
    previous_occupancy: float,
    confidence: float = 0.0,
) -> float:
    """Causal {1.5, 1.0, 0.5}× choice. Warmup holds 1.0; 1.5× only in occupancy deficit."""
    gears = tuple(float(value) for value in checkpoint["candidate_rates"])
    if tuple(gears) != tuple(CAMERA_RATES):
        raise ValueError(f"Expected candidate_rates {CAMERA_RATES}, got {gears}.")
    window = int(checkpoint.get("window", WINDOW))
    if len(features) < window:
        return CAMERA_HOLD_RATE
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    clip = float(checkpoint.get("feature_clip", 5.0))
    history = np.clip((np.stack(features[-window:]) - mean) / std, -clip, clip)
    with torch.inference_mode():
        logits = model(torch.from_numpy(history[None].astype(np.float32)))[0]
        probs = torch.softmax(logits, dim=0)
        gear = int(torch.argmax(logits).item())
        if float(probs[gear].item()) < float(confidence):
            gear = inverse_rpm_index(CAMERA_HOLD_RATE, gears)
    rate = float(gears[gear])
    storm = float(checkpoint.get("storm_occupancy", CAMERA_STORM_OCCUPANCY))
    deficit = checkpoint.get("deficit_occupancy")
    if previous_occupancy >= storm:
        rate = min(rate, CAMERA_HOLD_RATE)
    elif deficit is not None and previous_occupancy > float(deficit):
        rate = min(rate, CAMERA_HOLD_RATE)
    return rate
