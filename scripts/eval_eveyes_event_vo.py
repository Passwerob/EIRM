#!/usr/bin/env python3
"""Event visual odometry on Scene_005 co_motion: 5-point VO, optional DEVO.

Compares Passive / Fixed 180 / Ours / Fixed 360 on the same E2VID clip
(t=1.00–2.48 s). Camera GT is from camera_pose.csv. This is ego-motion VO,
not object tracking: the moving car is optionally masked.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "data/eveyes_comotion_v2_ready/test/Scene_005/co_motion"
OUT = ROOT / "results/ami_next/devo_vo"
CACHE = OUT / "cache"

METHODS = (
    ("passive", ROOT / "results/ami_next/e2vid_eirm/passive_events.txt", "#4d4d4d"),
    ("fixed_180", ROOT / "results/ami_next/e2vid_eirm/eirm_180_events.txt", "#4c72b0"),
    ("ours", ROOT / "results/ami_next/ours_vs_fixed_track_match/ours_events.txt", "#c44e52"),
    ("fixed_360", ROOT / "results/ami_next/e2vid_eirm/eirm_360_events.txt", "#55a868"),
)


def _load_intrinsics(path: Path) -> tuple[np.ndarray, int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    k = np.array(
        [[float(data["fx"]), 0.0, float(data["cx"])], [0.0, float(data["fy"]), float(data["cy"])], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return k, int(data["height"]), int(data["width"])


def _load_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    t = np.array([float(r["timestamp_s"]) for r in rows], dtype=np.float64)
    xyz = np.array([[float(r["position_x"]), float(r["position_y"]), float(r["position_z"])] for r in rows], dtype=np.float64)
    quat = np.array(
        [[float(r["quaternion_x"]), float(r["quaternion_y"]), float(r["quaternion_z"]), float(r["quaternion_w"])] for r in rows],
        dtype=np.float64,
    )
    return t, np.concatenate([xyz, quat], axis=1)


def _load_timestamps(path: Path) -> np.ndarray:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return np.array([float(r["timestamp_seconds"]) for r in rows], dtype=np.float64)


def _load_masks(mask_dir: Path, n: int, height: int, width: int) -> np.ndarray:
    paths = sorted(mask_dir.glob("*.png"))[:n]
    if len(paths) != n:
        raise RuntimeError(f"expected {n} masks, got {len(paths)}")
    masks = np.stack([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0 for p in paths])
    if masks.shape[1:] != (height, width):
        raise RuntimeError(f"mask shape {masks.shape[1:]} != {(height, width)}")
    return masks


def _cache_events(src: Path, dst: Path) -> dict[str, np.ndarray]:
    if dst.is_file():
        with np.load(dst) as data:
            return {key: data[key] for key in ("t", "x", "y", "p")}
    print(f"caching {src} -> {dst}")
    import pandas as pd

    table = pd.read_csv(src, sep=r"\s+", skiprows=1, header=None, names=("t", "x", "y", "p"))
    payload = {
        "t": table["t"].to_numpy(np.float64),
        "x": table["x"].to_numpy(np.int16),
        "y": table["y"].to_numpy(np.int16),
        "p": table["p"].to_numpy(np.int8),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **payload)
    return payload


def _event_frames(events: dict[str, np.ndarray], stamps: np.ndarray, height: int, width: int) -> tuple[list[np.ndarray], np.ndarray]:
    t = events["t"]
    x = events["x"].astype(np.int32)
    y = events["y"].astype(np.int32)
    p = events["p"].astype(np.int8)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    t, x, y, p = t[valid], x[valid], y[valid], p[valid]
    frames: list[np.ndarray] = []
    used: list[float] = []
    left = 0
    for start, end in zip(stamps[:-1], stamps[1:]):
        while left < len(t) and t[left] <= start:
            left += 1
        right = left
        while right < len(t) and t[right] <= end:
            right += 1
        image = np.zeros((height, width), dtype=np.float32)
        if right > left:
            np.add.at(image, (y[left:right], x[left:right]), np.where(p[left:right] > 0, 1.0, -1.0))
        frames.append(image)
        used.append(0.5 * (start + end))
        left = right
    return frames, np.asarray(used, dtype=np.float64)


def _to_uint8(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    work = image.copy()
    if mask is not None:
        work[mask] = 0.0
    peak = float(np.percentile(np.abs(work), 99.5)) if work.size else 1.0
    peak = max(peak, 1.0)
    scaled = np.clip(127.5 + 127.5 * work / peak, 0, 255).astype(np.uint8)
    return scaled


def _relative_pose(img0: np.ndarray, img1: np.ndarray, k: np.ndarray) -> tuple[np.ndarray | None, int]:
    detector = cv2.ORB_create(nfeatures=1500)
    k0, d0 = detector.detectAndCompute(img0, None)
    k1, d1 = detector.detectAndCompute(img1, None)
    if d0 is None or d1 is None or len(k0) < 12 or len(k1) < 12:
        return None, 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d0, d1)
    if len(matches) < 12:
        return None, 0
    matches = sorted(matches, key=lambda m: m.distance)[:400]
    pts0 = np.float32([k0[m.queryIdx].pt for m in matches])
    pts1 = np.float32([k1[m.trainIdx].pt for m in matches])
    e, inliers = cv2.findEssentialMat(pts0, pts1, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if e is None or inliers is None:
        return None, 0
    n, r, t, mask = cv2.recoverPose(e, pts0, pts1, k, mask=inliers)
    if n < 8:
        return None, int(n)
    t = t.reshape(3)
    t = t / (np.linalg.norm(t) + 1e-12)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = r
    pose[:3, 3] = t
    return pose, int(n)


def _chain(relatives: list[np.ndarray | None]) -> np.ndarray:
    poses = [np.eye(4, dtype=np.float64)]
    current = np.eye(4, dtype=np.float64)
    for rel in relatives:
        if rel is None:
            poses.append(current.copy())
            continue
        current = current @ rel
        poses.append(current.copy())
    return np.stack(poses)


def _interpolate_gt(gt_t: np.ndarray, gt_pose: np.ndarray, query: np.ndarray) -> np.ndarray:
    xyz = np.empty((len(query), 3), dtype=np.float64)
    for i in range(3):
        xyz[:, i] = np.interp(query, gt_t, gt_pose[:, i])
    return xyz


def _sim3_ate(est: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    mu_e = est.mean(axis=0)
    mu_r = ref.mean(axis=0)
    xe = est - mu_e
    xr = ref - mu_r
    cov = xe.T @ xr / max(len(est), 1)
    u, s, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[2, 2] = -1
    rot = u @ d @ vt
    var = np.sum(xe**2) / max(len(est), 1)
    scale = float(np.trace(np.diag(s) @ d) / max(var, 1e-12))
    trans = mu_r - scale * rot @ mu_e
    aligned = (scale * (rot @ est.T)).T + trans
    err = np.linalg.norm(aligned - ref, axis=1)
    path = float(np.sum(np.linalg.norm(np.diff(ref, axis=0), axis=1)))
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(err**2))),
        "ate_rmse_cm": float(np.sqrt(np.mean(err**2)) * 100.0),
        "ate_mean_m": float(np.mean(err)),
        "path_m": path,
        "mpe_percent": float(100.0 * np.mean(err) / max(path, 1e-9)),
        "scale": scale,
        "n": float(len(est)),
    }


def _run_method(name: Path, events: dict[str, np.ndarray], stamps: np.ndarray, k: np.ndarray, height: int, width: int, masks: np.ndarray, gt_t: np.ndarray, gt_pose: np.ndarray, mask_car: bool) -> dict[str, object]:
    frames, mid_t = _event_frames(events, stamps, height, width)
    uints = []
    for index, frame in enumerate(frames):
        mask = masks[index] if mask_car and index < len(masks) else None
        uints.append(_to_uint8(frame, mask))
    relatives: list[np.ndarray | None] = []
    inliers: list[int] = []
    for a, b in zip(uints[:-1], uints[1:]):
        pose, n_in = _relative_pose(a, b, k)
        relatives.append(pose)
        inliers.append(n_in)
    chain = _chain(relatives)
    npose = min(len(chain), len(mid_t))
    pose_t = mid_t[:npose]
    est_xyz = chain[:npose, :3, 3]
    ref_xyz = _interpolate_gt(gt_t, gt_pose, pose_t)
    metrics = _sim3_ate(est_xyz, ref_xyz)
    metrics.update(
        {
            "method": name,
            "mean_inliers": float(np.mean(inliers)) if inliers else 0.0,
            "frac_failed": float(np.mean([rel is None for rel in relatives])) if relatives else 1.0,
            "n_windows": int(len(frames)),
            "n_events": int(len(events["t"])),
        }
    )
    return {"metrics": metrics, "est_xyz": est_xyz, "ref_xyz": ref_xyz, "pose_t": pose_t[: len(est_xyz)], "preview": uints[0]}


def _plot(results: dict[str, dict[str, object]], out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6))
    names = [key for key, *_ in METHODS]
    ate = [float(results[n]["metrics"]["ate_rmse_cm"]) for n in names]
    failed = [100.0 * float(results[n]["metrics"]["frac_failed"]) for n in names]
    colors = [c for *_, c in METHODS]
    axes[0].bar(names, ate, color=colors)
    axes[0].set_ylabel("ATE RMSE (cm)")
    axes[0].set_title("Sim(3)-aligned ATE")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(names, failed, color=colors)
    axes[1].set_ylabel("% failed relative poses")
    axes[1].set_title("Tracking dropouts")
    axes[1].tick_params(axis="x", rotation=20)
    ref = results[names[0]]["ref_xyz"]
    axes[2].plot(ref[:, 1], ref[:, 0], "k--", lw=1.5, label="GT")
    for name, color in zip(names, colors):
        est = results[name]["est_xyz"]
        ref_xyz = results[name]["ref_xyz"]
        stats = _sim3_ate(est, ref_xyz)
        mu_e = est.mean(0)
        mu_r = ref_xyz.mean(0)
        xe = est - mu_e
        xr = ref_xyz - mu_r
        cov = xe.T @ xr / max(len(est), 1)
        u, s, vt = np.linalg.svd(cov)
        d = np.eye(3)
        if np.linalg.det(u) * np.linalg.det(vt) < 0:
            d[2, 2] = -1
        rot = u @ d @ vt
        scale = float(np.trace(np.diag(s) @ d) / max(np.sum(xe**2) / max(len(est), 1), 1e-12))
        trans = mu_r - scale * rot @ mu_e
        aligned = (scale * (rot @ est.T)).T + trans
        axes[2].plot(aligned[:, 1], aligned[:, 0], color=color, lw=1.4, label=name)
    axes[2].set_title("XY trajectory (aligned)")
    axes[2].set_aspect("equal", adjustable="datalim")
    axes[2].legend(fontsize=8)
    fig.suptitle("Scene_005 co_motion event VO (ORB + 5-point, car masked)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _write_devo_scene(method: str, events: dict[str, np.ndarray], stamps: np.ndarray, k: np.ndarray, height: int, width: int, gt_t: np.ndarray, gt_pose: np.ndarray) -> Path:
    scene = OUT / "devo_scenes" / method
    scene.mkdir(parents=True, exist_ok=True)
    np.savetxt(scene / "calib_undist_left.txt", np.array([k[0, 0], k[1, 1], k[0, 2], k[1, 2]], dtype=np.float64))
    tss = stamps * 1e6
    np.savetxt(scene / "tss_imgs_us_left.txt", tss)
    evs = np.stack(
        [events["t"] * 1e6, events["x"].astype(np.float64), events["y"].astype(np.float64), events["p"].astype(np.float64)],
        axis=1,
    )
    np.save(scene / "evs_left.npy", evs)
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    rect = np.stack([xx, yy], axis=-1).astype(np.float32)
    import h5py

    with h5py.File(scene / "rectify_map_left.h5", "w") as handle:
        handle.create_dataset("rectify_map", data=rect)
    pose_t = stamps
    xyz = _interpolate_gt(gt_t, gt_pose, pose_t)
    quat = np.array([np.interp(pose_t, gt_t, gt_pose[:, i + 3]) for i in range(4)]).T
    quat /= np.clip(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12, None)
    gt = np.column_stack([pose_t * 1e6, xyz, quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]])
    header = "timestamp_us tx ty tz qx qy qz qw"
    np.savetxt(scene / "gt_stamped_left.txt", gt, header=header)
    (scene / "README.txt").write_text(
        "DEVO scene export for EVEYES Scene_005 co_motion.\n"
        "evs_left.npy is [t_us, x, y, p]. Convert to evs_left.txt if the official iterator is used.\n"
        "Intrinsics are pinhole, undistorted, identity rectify map.\n",
        encoding="utf-8",
    )
    return scene


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-car", action="store_true", default=True)
    parser.add_argument("--no-mask-car", action="store_false", dest="mask_car")
    parser.add_argument("--export-devo", action="store_true", default=True)
    args = parser.parse_args()

    k, height, width = _load_intrinsics(SCENE / "camera_intrinsics.json")
    gt_t, gt_pose = _load_poses(SCENE / "camera_pose.csv")
    stamps_all = _load_timestamps(SCENE / "timestamps.csv")
    masks_all = _load_masks(SCENE / "object_masks", len(stamps_all), height, width)

    # E2VID/event dumps cover t in (1.0, 2.483]
    t0, t1 = 1.0, 2.483333333
    keep = (stamps_all >= t0 - 1e-9) & (stamps_all <= t1 + 1e-6)
    idx = np.where(keep)[0]
    stamps = stamps_all[idx]
    masks = masks_all[idx[:-1]] if len(idx) > 1 else masks_all[idx]
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, object]] = {}
    for name, path, _color in METHODS:
        events = _cache_events(path, CACHE / f"{name}.npz")
        if args.export_devo:
            _write_devo_scene(name, events, stamps, k, height, width, gt_t, gt_pose)
        results[name] = _run_method(name, events, stamps, k, height, width, masks, gt_t, gt_pose, args.mask_car)
        print(json.dumps(results[name]["metrics"], indent=2, sort_keys=True))

    summary = {name: results[name]["metrics"] for name, *_ in METHODS}
    (OUT / "classical_vo_stats.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _plot(results, OUT / "fig_classical_event_vo.png")
    print(json.dumps({"output": str(OUT), "summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
