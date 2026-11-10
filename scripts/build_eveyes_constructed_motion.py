#!/usr/bin/env python3
"""Build a new EVEYES motion construction from an already rendered sequence.

The Blender scene cannot be re-rendered here, so a new motion is a re-timing of
a recorded camera/object path: an output clock t is mapped to a path parameter
s(t) in source-frame units by integrating a piecewise playback rate.

  rate < 1  slow motion   -> intermediate frames come from Practical-RIFE
  rate = 1  native
  rate > 1  fast motion   -> real rendered frames, no interpolation

Everything else is regenerated to stay consistent with the new clock rather
than copied:

  camera/ego/neighbor pose   interpolated along s(t) (linear position, slerp rotation)
  motion_state               path speed times the instantaneous rate
  imu.csv at 1000 Hz         gyro from the re-timed ego rotation, accel from the
                             second derivative of the re-timed ego position plus
                             gravity, plus the recorded road-vibration residual
                             resampled by road position

Reusing the recorded imu.csv verbatim would be wrong: it is tied to the source
clock, so a 0.25x or 4x traversal of the same road would carry the wrong rates.

Object masks are nearest-neighbour: they are binary labels, not images.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import imageio.v3 as imageio
import numpy as np

from ami_simulator.pipeline import _IMAGE_SUFFIXES
from ami_simulator.vfi import DebugLinearVFIBackend, PracticalRIFEBackend

READY = Path("data/eveyes_comotion_v2_ready")
RIFE_ROOT = Path("/data/fcr/code/event_vggt/Data_Part/Practical-RIFE")
GRAVITY = 9.81
IMU_RATE_HZ = 4000.0
SOURCE_FPS = 60.0
POSE_FILES = ("camera_pose.csv", "ego_pose.csv", "neighbor_pose.csv")


def _read_pose(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    position = np.asarray([[float(row[f"position_{a}"]) for a in "xyz"] for row in rows], dtype=np.float64)
    quaternion = np.asarray([[float(row[f"quaternion_{a}"]) for a in ("w", "x", "y", "z")] for row in rows], dtype=np.float64)
    return position, quaternion


def _read_imu(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    t = np.asarray([float(row["timestamp_s"]) for row in rows], dtype=np.float64)
    gyro = np.asarray([[float(row[f"gyro_{a}_rad_s"]) for a in "xyz"] for row in rows], dtype=np.float64)
    accel = np.asarray([[float(row[f"accel_{a}_m_s2"]) for a in "xyz"] for row in rows], dtype=np.float64)
    return t, gyro, accel


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1, 1)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    small = theta[:, 0] < 1e-8
    sin_theta = np.sin(theta)
    result = np.empty_like(q0)
    if np.any(~small):
        w0 = np.sin((1.0 - alpha[~small]) * theta[~small]) / sin_theta[~small]
        w1 = np.sin(alpha[~small] * theta[~small]) / sin_theta[~small]
        result[~small] = w0 * q0[~small] + w1 * q1[~small]
    if np.any(small):
        result[small] = (1.0 - alpha[small]) * q0[small] + alpha[small] * q1[small]
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _sample_pose(position: np.ndarray, quaternion: np.ndarray, path_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    last = len(position) - 1
    s = np.clip(np.asarray(path_s, dtype=np.float64), 0.0, last)
    lower = np.floor(s).astype(np.int64)
    upper = np.minimum(lower + 1, last)
    alpha = (s - lower)[:, None]
    pos = (1.0 - alpha) * position[lower] + alpha * position[upper]
    quat = _slerp(quaternion[lower], quaternion[upper], alpha[:, 0])
    return pos, quat


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=1,
    )


def _body_rates(position: np.ndarray, quaternion: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Gyro (rad/s) and specific force (m/s^2) in the body frame on a uniform clock."""
    # The constructed trajectory can end between source frames. Extrapolate
    # one position sample with the terminal velocity before differentiating so
    # finite differences do not turn the endpoint clamp into a fake hard stop.
    if len(position) >= 3:
        terminal_step = position[-2] - position[-3]
        terminal = position[-1] + terminal_step
        position_for_diff = np.concatenate([position, terminal[None]], axis=0)
    elif len(position) >= 2:
        terminal = position[-1] + (position[-1] - position[-2])
        position_for_diff = np.concatenate([position, terminal[None]], axis=0)
    else:
        position_for_diff = position
    velocity = np.gradient(position_for_diff, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)[: len(position)]
    if len(acceleration) >= 5:
        acceleration[-4:] = acceleration[-5]
    rotation = _quat_to_matrix(quaternion)
    gravity = np.zeros_like(acceleration)
    gravity[:, 2] = GRAVITY
    specific_force = np.einsum("nji,nj->ni", rotation, acceleration + gravity)

    conjugate = quaternion.copy()
    conjugate[:, 1:] *= -1.0
    forward = np.empty_like(quaternion)
    forward[:-1] = quaternion[1:]
    forward[-1] = quaternion[-1]
    w0, x0, y0, z0 = conjugate.T
    w1, x1, y1, z1 = forward.T
    delta = np.stack(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ],
        axis=1,
    )
    delta = np.where(delta[:, :1] < 0.0, -delta, delta)
    angle = 2.0 * np.arccos(np.clip(delta[:, 0], -1.0, 1.0))
    axis_norm = np.linalg.norm(delta[:, 1:], axis=1, keepdims=True)
    axis = np.divide(delta[:, 1:], np.maximum(axis_norm, 1e-12))
    gyro = axis * (angle / dt)[:, None]
    return gyro, specific_force


def _walk_profile(segments: list[dict], start_frame: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Step the output clock. Each segment sets a scene speed and a sensor clock.

    ``rate``  scene speed relative to the recorded traversal
    ``fps``   output frame rate for the segment (the sensor clock)

    Source frames advanced per output frame is ``rate * SOURCE_FPS / fps``, so a
    fast segment on a fast clock keeps the per-window displacement native while
    the prism, at a fixed RPM, sweeps ``omega / fps`` instead of ``omega / 60``.
    A segment ends after ``duration_s`` or when the path reaches ``until_frame``.
    """
    times, rates, fps_log, path = [0.0], [float(segments[0]["rate"])], [float(segments[0].get("fps", SOURCE_FPS))], [float(start_frame)]
    previous_rate = float(segments[0]["rate"])
    labels = np.zeros(1, dtype=np.int64)
    for order, segment in enumerate(segments):
        rate = float(segment["rate"])
        fps = float(segment.get("fps", SOURCE_FPS))
        dt = 1.0 / fps
        ramp = float(segment.get("ramp_s", 0.0))
        duration = segment.get("duration_s")
        until = segment.get("until_frame")
        if (duration is None) == (until is None):
            raise SystemExit("each segment needs exactly one of duration_s or until_frame")
        marks = []
        ramp_steps = int(round(ramp * fps))
        for step in range(ramp_steps):
            blend = 0.5 - 0.5 * np.cos(np.pi * (step + 1) / max(ramp_steps, 1))
            current = previous_rate + (rate - previous_rate) * blend
            path.append(path[-1] + current * SOURCE_FPS * dt)
            times.append(times[-1] + dt)
            rates.append(current)
            fps_log.append(fps)
            marks.append(order)
        if segment.get("snap_to_frame"):
            path[-1] = float(round(path[-1]))
        segment_start = times[-1]
        while True:
            if duration is not None and times[-1] - segment_start >= duration - 1e-9:
                break
            if until is not None and path[-1] >= float(until) - 1e-9:
                break
            path.append(path[-1] + rate * SOURCE_FPS * dt)
            times.append(times[-1] + dt)
            rates.append(rate)
            fps_log.append(fps)
            marks.append(order)
        labels = np.concatenate([labels, np.asarray(marks, dtype=np.int64)])
        previous_rate = rate
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(rates, dtype=np.float64),
        np.asarray(fps_log, dtype=np.float64),
        np.asarray(path, dtype=np.float64),
        labels,
    )


def _render_frames(scene: Path, path_s: np.ndarray, backend, out_dir: Path) -> dict[str, int]:
    directory = scene / "images"
    image_paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES)
    last = len(image_paths) - 1
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"copied": 0, "interpolated": 0}
    cache: dict[int, np.ndarray] = {}

    def source(index: int) -> np.ndarray:
        if index not in cache:
            if len(cache) > 8:
                cache.clear()
            cache[index] = np.asarray(imageio.imread(image_paths[index]))[..., :3]
        return cache[index]

    for out_index, s in enumerate(path_s):
        s = float(np.clip(s, 0.0, last))
        lower = int(np.floor(s))
        alpha = s - lower
        destination = out_dir / f"{out_index:06d}.png"
        if alpha < 1e-6 or lower >= last:
            imageio.imwrite(destination, source(min(lower, last)))
            counts["copied"] += 1
        else:
            middle = backend.interpolate(source(lower), source(lower + 1), np.asarray([alpha]))
            imageio.imwrite(destination, np.clip(middle[0] * 255.0, 0.0, 255.0).astype(np.uint8))
            counts["interpolated"] += 1
        if out_index % 30 == 0:
            print(f"    frame {out_index}/{len(path_s)}  s={s:.2f}  {counts}", flush=True)
    return counts


def _write_masks(scene: Path, path_s: np.ndarray, out_dir: Path) -> bool:
    mask_paths = sorted((scene / "object_masks").glob("*.png"))
    if not mask_paths:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    last = len(mask_paths) - 1
    for out_index, s in enumerate(path_s):
        nearest = int(np.clip(round(float(s)), 0, last))
        shutil.copyfile(mask_paths[nearest], out_dir / f"{out_index:06d}.png")
    return True


def _write_csv(path: Path, header: tuple[str, ...], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="test/Scene_005/drive_stop_drive")
    parser.add_argument("--regime", default="crawl_then_dash")
    parser.add_argument("--dataset-root", type=Path, default=READY)
    parser.add_argument("--profile", type=Path, required=True, help="JSON list of {rate, duration_s, ramp_s} segments")
    parser.add_argument("--start-frame", type=float, default=0.0)
    parser.add_argument("--vfi", choices=("practical_rife", "debug_linear"), default="practical_rife")
    parser.add_argument("--rife-root", type=Path, default=RIFE_ROOT)
    parser.add_argument("--rife-device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Report the path and IMU without writing frames.")
    parser.add_argument("--imu-bounds", nargs=2, type=float, default=(0.0002822150725105154, 0.0026775735903401166), help="Checkpoint gyro-norm bounds, for the IMU-saturation diagnostic.")
    args = parser.parse_args()

    split, scene_name, source_regime = args.source.split("/")
    scene = args.dataset_root / split / scene_name / source_regime
    if not (scene / "images").exists():
        raise SystemExit(f"missing source {scene}")
    destination = args.dataset_root / split / scene_name / args.regime
    if destination.exists() and not args.dry_run:
        raise SystemExit(f"{destination} already exists; remove it first")

    segments = json.loads(args.profile.read_text(encoding="utf-8"))
    times, rates, fps_log, path_s, labels = _walk_profile(segments, args.start_frame)
    n_source = len(sorted((scene / "images").glob("*.png")))
    if path_s.max() > n_source - 1:
        raise SystemExit(f"profile needs source frame {path_s.max():.1f} but only {n_source} exist")
    print(f"output {len(times)} frames / {times[-1]:.3f}s  path {path_s[0]:.1f}..{path_s[-1]:.1f} of {n_source}", flush=True)
    for order, segment in enumerate(segments):
        window = labels[:-1] == order
        if not window.any():
            continue
        fps = float(segment.get("fps", SOURCE_FPS))
        print(
            f"    [{order}] {segment.get('label', ''):<32} rate {segment['rate']:<6g} fps {fps:<6g} "
            f"windows {int(window.sum()):4d}  prism sweep at 180 RPM {180.0 / 60.0 * 360.0 / fps:6.2f} deg/window",
            flush=True,
        )

    imu_times = np.arange(0.0, times[-1] + 0.5 / IMU_RATE_HZ, 1.0 / IMU_RATE_HZ)
    imu_path_s = np.interp(imu_times, times, path_s)
    ego_position, ego_quaternion = _read_pose(scene / "ego_pose.csv")
    imu_p, imu_q = _sample_pose(ego_position, ego_quaternion, imu_path_s)
    gyro, accel = _body_rates(imu_p, imu_q, 1.0 / IMU_RATE_HZ)

    source_imu_t, source_gyro, source_accel = _read_imu(scene / "imu.csv")
    source_path_s = source_imu_t * SOURCE_FPS
    source_p, source_q = _sample_pose(ego_position, ego_quaternion, source_path_s)
    base_gyro, base_accel = _body_rates(source_p, source_q, float(np.median(np.diff(source_imu_t))))
    residual_gyro = source_gyro - base_gyro
    residual_accel = source_accel - base_accel
    vibration_gyro = np.stack([np.interp(imu_path_s, source_path_s, residual_gyro[:, axis]) for axis in range(3)], axis=1)
    vibration_accel = np.stack([np.interp(imu_path_s, source_path_s, residual_accel[:, axis]) for axis in range(3)], axis=1)
    gyro = gyro + vibration_gyro
    accel = accel + vibration_accel
    gyro_norm = np.linalg.norm(gyro, axis=1)

    low, high = args.imu_bounds
    window_fraction = []
    for start, end in zip(times[:-1], times[1:]):
        left = int(np.searchsorted(imu_times, start, side="left"))
        right = int(np.searchsorted(imu_times, end, side="right"))
        values = gyro_norm[left:right]
        mean = float(values.mean()) if len(values) else float(np.interp((start + end) * 0.5, imu_times, gyro_norm))
        window_fraction.append(float(np.clip((mean - low) / max(high - low, 1e-9), 0.0, 1.0)))
    window_fraction = np.asarray(window_fraction)
    saturated = window_fraction >= 1.0
    print(
        f"  IMU gyro_norm median {np.median(gyro_norm):.6f} max {gyro_norm.max():.6f}\n"
        f"  normalized fraction: mean {window_fraction.mean():.3f}  saturated (==1, lets Ours leave the 180 floor) "
        f"in {int(saturated.sum())}/{len(window_fraction)} windows",
        flush=True,
    )
    for order, segment in enumerate(segments):
        window = labels[:-1] == order
        if window.any():
            print(f"    [{order}] {segment.get('label', ''):<32} imu fraction mean {float(window_fraction[window].mean()):.3f}", flush=True)

    if args.dry_run:
        print(json.dumps({"dry_run": True, "frames": len(times), "duration_s": float(times[-1]), "path_range": [float(path_s[0]), float(path_s[-1])], "imu_saturated_fraction": float(saturated.mean())}, indent=2))
        return 0

    backend = (
        PracticalRIFEBackend(args.rife_root, device=args.rife_device, inference_batch_size=1)
        if args.vfi == "practical_rife"
        else DebugLinearVFIBackend()
    )
    destination.mkdir(parents=True, exist_ok=True)
    print("  rendering frames", flush=True)
    counts = _render_frames(scene, path_s, backend, destination / "images")
    has_masks = _write_masks(scene, path_s, destination / "object_masks")
    _write_csv(destination / "timestamps.csv", ("index", "timestamp_seconds"), ((index, f"{value:.9f}") for index, value in enumerate(times)))
    for name in POSE_FILES:
        source_path = scene / name
        if not source_path.exists():
            continue
        position, quaternion = _read_pose(source_path)
        sampled_p, sampled_q = _sample_pose(position, quaternion, path_s)
        _write_csv(
            destination / name,
            ("frame_id", "timestamp_s", "position_x", "position_y", "position_z", "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z"),
            (
                (index, f"{times[index]:.9f}", *(f"{value:.9f}" for value in sampled_p[index]), *(f"{value:.9f}" for value in sampled_q[index]))
                for index in range(len(times))
            ),
        )
    _write_csv(
        destination / "imu.csv",
        ("timestamp_s", "timestamp_us", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s", "gyro_norm_rad_s", "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2"),
        (
            (f"{imu_times[i]:.6f}", int(round(imu_times[i] * 1e6)), *(f"{v:.12f}" for v in gyro[i]), f"{gyro_norm[i]:.12f}", *(f"{v:.12f}" for v in accel[i]))
            for i in range(len(imu_times))
        ),
    )

    source_state = list(csv.DictReader((scene / "motion_state.csv").open(encoding="utf-8")))
    source_speed = np.asarray([float(row["speed_m_s"]) for row in source_state], dtype=np.float64)
    path_speed = np.interp(path_s, np.arange(len(source_speed), dtype=np.float64), source_speed) * rates
    longitudinal = np.gradient(path_speed, times)
    _write_csv(
        destination / "motion_state.csv",
        ("frame_id", "timestamp_s", "normalized_progress", "speed_m_s", "longitudinal_accel_m_s2", "playback_rate", "sensor_fps", "source_path_frame"),
        (
            (
                index,
                f"{times[index]:.9f}",
                f"{path_s[index] / max(len(source_speed) - 1, 1):.9f}",
                f"{path_speed[index]:.9f}",
                f"{longitudinal[index]:.9f}",
                f"{rates[index]:.6f}",
                f"{fps_log[index]:.3f}",
                f"{path_s[index]:.6f}",
            )
            for index in range(len(times))
        ),
    )

    for name in ("camera_intrinsics.json", "camera_extrinsics.json"):
        shutil.copyfile(scene / name, destination / name)
    source_motion = json.loads((scene / "motion_config.json").read_text(encoding="utf-8"))
    render_config = json.loads((scene / "render_config.json").read_text(encoding="utf-8"))
    (destination / "render_config.json").write_text(
        json.dumps(
            {
                **render_config,
                "rgb_samples": len(times),
                "duration_s": float(times[-1]),
                "last_rgb_timestamp_s": float(times[-1]),
                "rgb_rate_hz": sorted({float(value) for value in fps_log}),
                "uniform_frame_period": bool(len({round(float(value), 6) for value in fps_log}) == 1),
                "imu_rate_hz": IMU_RATE_HZ,
                "frames_copied_from_render": counts["copied"],
                "frames_from_vfi": counts["interpolated"],
                "vfi_backend": backend.metadata(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (destination / "motion_config.json").write_text(
        json.dumps(
            {
                "sequence_id": f"{scene_name}_{args.regime.upper()}_CONSTRUCTED",
                "base_scene": scene_name,
                "split": split,
                "motion_regime": args.regime.upper(),
                "construction": "retimed_playback_of_recorded_path",
                "source_sequence": args.source,
                "source_motion_regime": source_motion.get("motion_regime"),
                "not_re_rendered": "Blender scene unavailable; slow-motion frames come from Practical-RIFE",
                "rate_segments": segments,
                "start_frame": args.start_frame,
                "path_frame_range": [float(path_s[0]), float(path_s[-1])],
                "playback_rate_range": [float(rates.min()), float(rates.max())],
                "sensor_fps_values": sorted({float(value) for value in fps_log}),
                "knobs": {
                    "rate": "scene speed relative to the recorded traversal",
                    "fps": "sensor clock; at a fixed RPM the prism sweeps omega/fps per window",
                    "prism_sweep_deg_per_window_at_180rpm": {str(int(value)): 180.0 / 60.0 * 360.0 / float(value) for value in sorted({float(v) for v in fps_log})},
                },
                "imu_signal_model": "gyro from re-timed ego rotation, accel from re-timed ego position second derivative plus gravity, plus recorded road-vibration residual resampled by road position",
                "imu_is_not_copied": True,
                "object_masks": "nearest source frame" if has_masks else "absent",
                "inherits": {"scene_source_blend": source_motion.get("scene_source_blend"), "ego_camera_mount": source_motion.get("ego_camera_mount")},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "scene_id": f"{split}/{scene_name}/{args.regime}",
        "path": str(destination),
        "frames": len(times),
        "duration_s": float(times[-1]),
        "frames_copied": counts["copied"],
        "frames_interpolated": counts["interpolated"],
        "vfi": backend.metadata(),
        "imu_rows": len(imu_times),
        "gyro_norm_median": float(np.median(gyro_norm)),
        "gyro_norm_max": float(gyro_norm.max()),
        "speed_range_m_s": [float(path_speed.min()), float(path_speed.max())],
        "has_object_masks": has_masks,
    }
    (destination / "DONE.json").write_text(json.dumps({"status": "COMPLETE", **summary}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
