#!/usr/bin/env python3
"""Build a full-copy EVEYES pack for pose-speed negative-feedback control.

Copies every file (no symlinks). RGB and timestamps are unchanged. IMU is
rewritten from camera_pose finite differences so object_only / freeze are
near-zero and driving carries linear speed. The control signal is pose
speed, not gyro — this pack's cameras mostly translate.

Do not overwrite data/eveyes_comotion_v2_ready or the frozen m0 snapshot.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

GRAVITY_WORLD = np.asarray([0.0, 0.0, 9.80665], dtype=np.float64)
SPECS = (("train", "Scene_002"), ("valid", "Scene_004"), ("test", "Scene_005"))
REGIMES = ("co_motion", "camera_only", "object_only", "drive_stop_drive")


def _assert_no_symlinks(root: Path) -> None:
    leftover: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in (*dirnames, *filenames):
            path = Path(dirpath) / name
            if path.is_symlink():
                leftover.append(str(path))
    if leftover:
        raise SystemExit(f"symlinks remain in {root}: {leftover[:8]}")


def _copy_clip(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["cp", "-aL", f"{source}/.", str(destination)])
    _assert_no_symlinks(destination)


def _load_pose(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    times = np.asarray([float(row["timestamp_s"]) for row in rows], dtype=np.float64)
    positions = np.asarray(
        [[float(row["position_x"]), float(row["position_y"]), float(row["position_z"])] for row in rows],
        dtype=np.float64,
    )
    quats = np.asarray(
        [
            [
                float(row["quaternion_w"]),
                float(row["quaternion_x"]),
                float(row["quaternion_y"]),
                float(row["quaternion_z"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.clip(norms, 1e-12, None)
    return times, positions, quats


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left.T
    rw, rx, ry, rz = right.T
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def _quat_conj(quat: np.ndarray) -> np.ndarray:
    out = np.asarray(quat, dtype=np.float64).copy()
    out[..., 1:] *= -1.0
    return out


def _quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    if quat[..., 0] < 0.0:
        quat = -quat
    w = np.clip(quat[0], -1.0, 1.0)
    xyz = quat[1:]
    nxyz = float(np.linalg.norm(xyz))
    angle = 2.0 * float(np.arctan2(nxyz, w))
    if nxyz < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return xyz / nxyz * angle


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _finite_rate(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    dt = np.diff(times)
    dt = np.where(np.abs(dt) < 1e-9, 1e-9, dt)
    rate = np.diff(values, axis=0) / dt[:, None]
    return np.vstack((rate[0], rate))


def _interp_rows(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    out = np.empty((len(query), values.shape[1]), dtype=np.float64)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(query, times, values[:, col])
    return out


def _pose_kinematics(times: np.ndarray, positions: np.ndarray, quats: np.ndarray) -> dict[str, np.ndarray]:
    velocity = _finite_rate(positions, times)
    accel_world = _finite_rate(velocity, times)
    gyro = np.zeros_like(positions)
    for index in range(len(times) - 1):
        rel = _quat_mul(_quat_conj(quats[index]), quats[index + 1])
        dt = max(float(times[index + 1] - times[index]), 1e-9)
        gyro[index] = _quat_to_rotvec(rel) / dt
    gyro[-1] = gyro[-2] if len(gyro) > 1 else gyro[-1]
    speed = np.linalg.norm(velocity, axis=1)
    accel_body = np.zeros_like(accel_world)
    for index, quat in enumerate(quats):
        rotation_world_from_body = _quat_to_matrix(quat)
        accel_body[index] = rotation_world_from_body.T @ (accel_world[index] + GRAVITY_WORLD)
    return {
        "velocity_world_m_s": velocity,
        "speed_mps": speed,
        "gyro_body_rad_s": gyro,
        "accel_body_m_s2": accel_body,
        "accel_world_m_s2": accel_world,
    }


def _rewrite_imu(clip: Path) -> dict[str, float]:
    pose_times, positions, quats = _load_pose(clip / "camera_pose.csv")
    kin = _pose_kinematics(pose_times, positions, quats)
    imu_path = clip / "imu.csv"
    with imu_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    imu_times = np.asarray([float(row["timestamp_s"]) for row in rows], dtype=np.float64)
    gyro = _interp_rows(pose_times, kin["gyro_body_rad_s"], imu_times)
    accel = _interp_rows(pose_times, kin["accel_body_m_s2"], imu_times)
    speed = np.interp(imu_times, pose_times, kin["speed_mps"])
    fieldnames = [
        "timestamp_s",
        "timestamp_us",
        "gyro_x_rad_s",
        "gyro_y_rad_s",
        "gyro_z_rad_s",
        "gyro_norm_rad_s",
        "accel_x_m_s2",
        "accel_y_m_s2",
        "accel_z_m_s2",
        "speed_mps",
    ]
    with imu_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, omega, acc, spd in zip(rows, gyro, accel, speed):
            writer.writerow(
                {
                    "timestamp_s": row["timestamp_s"],
                    "timestamp_us": row["timestamp_us"],
                    "gyro_x_rad_s": float(omega[0]),
                    "gyro_y_rad_s": float(omega[1]),
                    "gyro_z_rad_s": float(omega[2]),
                    "gyro_norm_rad_s": float(np.linalg.norm(omega)),
                    "accel_x_m_s2": float(acc[0]),
                    "accel_y_m_s2": float(acc[1]),
                    "accel_z_m_s2": float(acc[2]),
                    "speed_mps": float(spd),
                }
            )
    with (clip / "pose_speed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("index", "timestamp_seconds", "speed_mps", "gyro_norm_rad_s", "accel_norm_m_s2"))
        accel_norm = np.linalg.norm(kin["accel_world_m_s2"], axis=1)
        gyro_norm = np.linalg.norm(kin["gyro_body_rad_s"], axis=1)
        for index, (t_seconds, spd, omega, acc) in enumerate(zip(pose_times, kin["speed_mps"], gyro_norm, accel_norm)):
            writer.writerow((index, float(t_seconds), float(spd), float(omega), float(acc)))
    config_path = clip / "motion_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["imu_signal_model"] = "pose_finite_difference_plus_gravity"
        config["control_motion_signal"] = "pose_speed_mps"
        config["timestamps_unchanged"] = True
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {
        "mean_speed_mps": float(kin["speed_mps"].mean()),
        "max_speed_mps": float(kin["speed_mps"].max()),
        "mean_gyro_norm": float(np.linalg.norm(kin["gyro_body_rad_s"], axis=1).mean()),
        "mean_accel_norm": float(np.linalg.norm(kin["accel_world_m_s2"], axis=1).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/eveyes_comotion_v2_ready"))
    parser.add_argument("--output", type=Path, default=Path("data/eveyes_motion_nfb_ready"))
    args = parser.parse_args()
    records: list[dict[str, object]] = []
    for split, scene in SPECS:
        for regime in REGIMES:
            source = args.source_root / split / scene / regime
            if not (source / "images").exists() and not (source / "images").is_symlink():
                continue
            destination = args.output / split / scene / regime
            print(f"copy {source} -> {destination}", flush=True)
            _copy_clip(source, destination)
            stats = _rewrite_imu(destination)
            n_images = len(list((destination / "images").glob("*.png")))
            records.append(
                {
                    "scene": f"{split}/{scene}/{regime}",
                    "split": split,
                    "condition": regime,
                    "n_images": n_images,
                    **stats,
                }
            )
            print(json.dumps(records[-1], sort_keys=True), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "EVEYES-motion-nfb",
        "source": str(args.source_root),
        "copy": "full dereference, no symlinks",
        "unchanged": ["images", "timestamps.csv", "camera_pose.csv", "neighbor_pose.csv"],
        "rewritten": ["imu.csv from camera_pose", "pose_speed.csv"],
        "control_signal": "pose_speed_mps = |d camera_position / dt|",
        "not_the_paper_pack": True,
        "records": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "README.txt").write_text(
        "Motion negative-feedback pack\n"
        "Full copy of eveyes_comotion_v2_ready (no symlinks).\n"
        "RGB and timestamps are the original 60 Hz render.\n"
        "IMU gyro/accel are recomputed from camera_pose. Gyro is ~0 because\n"
        "these clips mostly translate. Use pose_speed.csv for inverse RPM:\n"
        "slow -> 360, fast -> 180, storm brake downshift only.\n"
        "Do not use this pack for Table 1 / m0.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "clips": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
