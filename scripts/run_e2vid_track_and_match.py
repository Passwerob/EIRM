#!/usr/bin/env python3
"""Scene_005 tracking + SIFT/ORB matching for passive vs EIRM 180/360.

Tracking: mask-free event CC detector, plus GT-init search-window centroid
tracker (masks init/score only). Matching: same-plane ORB (paper helper) and
SIFT with spatial inliers. E2VID reconstructions are used where they exist.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import EventStream
from scripts.evaluate_anchor_feature_matching import _event_image, _feature_metrics
from scripts.evaluate_dl3dv_downstream import _ami_windows, _dvs_windows
from scripts.evaluate_event_target_tracking import _box, _iou, _proposal
from scripts.evaluate_eveyes_v2_downstream import _intrinsics, _load_object_masks, _scene_path

SCENE = "Scene_005"
START = 60
N_FRAMES = 90
SPATIAL_PX = 5.0
OUT = ROOT / "results/ami_next/e2vid_track_match"
E2VID = ROOT / "results/ami_next/e2vid_eirm"
DATA = ROOT / "data/eveyes_comotion_v2_ready"
METHODS = {
    "passive": {"rpm": 0.0, "e2vid": "passive", "color": "#c44e52"},
    "eirm_180": {"rpm": 180.0, "e2vid": "eirm_180", "color": "#4c72b0"},
    "eirm_360": {"rpm": 360.0, "e2vid": "eirm_360", "color": "#55a868"},
}


def _xywh(box) -> np.ndarray | None:
    if box is None:
        return None
    return np.asarray(box, dtype=np.float64).reshape(4)


def _clip_xywh(box: np.ndarray, h: int, w: int) -> np.ndarray:
    x, y, bw, bh = [float(v) for v in box]
    x = min(max(x, 0.0), w - 1.0)
    y = min(max(y, 0.0), h - 1.0)
    bw = min(max(bw, 1.0), w - x)
    bh = min(max(bh, 1.0), h - y)
    return np.array([x, y, bw, bh], dtype=np.float64)


def _expand_xywh(box: np.ndarray, margin: float, h: int, w: int) -> np.ndarray:
    x, y, bw, bh = box
    return _clip_xywh(np.array([x - margin, y - margin, bw + 2 * margin, bh + 2 * margin]), h, w)


def _centroid_track(windows: list, init, shape: tuple[int, int], min_events: int = 40) -> list:
    h, w = shape
    box = _xywh(init)
    preds = []
    if box is None:
        return [None] * len(windows)
    bw0, bh0 = float(box[2]), float(box[3])
    for stream in windows:
        if box is None or len(stream) == 0:
            preds.append(None if box is None else box.copy())
            continue
        search = _expand_xywh(box, 48.0, h, w)
        x = np.asarray(stream.x, dtype=np.float64)
        y = np.asarray(stream.y, dtype=np.float64)
        keep = (x >= search[0]) & (x < search[0] + search[2]) & (y >= search[1]) & (y < search[1] + search[3])
        if int(keep.sum()) < min_events:
            preds.append(box.copy())
            continue
        cx, cy = float(x[keep].mean()), float(y[keep].mean())
        box = _clip_xywh(np.array([cx - 0.5 * bw0, cy - 0.5 * bh0, bw0, bh0]), h, w)
        preds.append(box.copy())
    return preds


def _target_coverage(stream, mask: np.ndarray) -> float:
    if mask.sum() == 0 or len(stream) == 0:
        return 0.0
    h, w = mask.shape
    hit = np.zeros((h, w), dtype=np.uint8)
    x = np.clip(np.asarray(stream.x, dtype=np.int64), 0, w - 1)
    y = np.clip(np.asarray(stream.y, dtype=np.int64), 0, h - 1)
    hit[y, x] = 1
    return float(hit[mask].mean())


def _score_track(windows: list, masks: np.ndarray, shape: tuple[int, int]) -> dict:
    gt = [_box(np.asarray(masks[min(i + 1, len(masks) - 1)], dtype=bool)) for i in range(len(windows))]
    det = [_proposal(stream, shape) for stream in windows]
    trk = _centroid_track(windows, _box(np.asarray(masks[0], dtype=bool)), shape)
    det_iou = np.array([_iou(p, g) for p, g in zip(det, gt)], dtype=np.float64)
    trk_iou = np.array([_iou(p, g) for p, g in zip(trk, gt)], dtype=np.float64)
    cover = np.array(
        [_target_coverage(stream, np.asarray(masks[min(i + 1, len(masks) - 1)], dtype=bool)) for i, stream in enumerate(windows)],
        dtype=np.float64,
    )
    n_evt = np.array([len(stream) for stream in windows], dtype=np.float64)
    return {
        "n_windows": int(len(windows)),
        "mean_events": float(n_evt.mean()) if n_evt.size else 0.0,
        "det_mean_iou": float(det_iou.mean()) if det_iou.size else 0.0,
        "det_recall_0.3": float((det_iou >= 0.3).mean()) if det_iou.size else 0.0,
        "det_recall_0.5": float((det_iou >= 0.5).mean()) if det_iou.size else 0.0,
        "trk_mean_iou": float(trk_iou.mean()) if trk_iou.size else 0.0,
        "trk_success_0.3": float((trk_iou >= 0.3).mean()) if trk_iou.size else 0.0,
        "trk_success_0.5": float((trk_iou >= 0.5).mean()) if trk_iou.size else 0.0,
        "target_event_coverage": float(cover.mean()) if cover.size else 0.0,
        "det_boxes": det,
        "trk_boxes": trk,
        "gt_boxes": gt,
    }


def _sift_match(src: np.ndarray, rgb: np.ndarray, mask: np.ndarray | None, px: float = SPATIAL_PX) -> dict:
    a = src if src.ndim == 2 else cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    b = rgb if rgb.ndim == 2 else cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    a = np.clip(a, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    sift = cv2.SIFT_create(nfeatures=1200)
    kpa, da = sift.detectAndCompute(a, None)
    kpb, db = sift.detectAndCompute(b, None)
    n_a = 0 if kpa is None else len(kpa)
    n_b = 0 if kpb is None else len(kpb)
    out = {
        "n_src": n_a,
        "n_rgb": n_b,
        "n_good": 0,
        "n_inlier_3px": 0,
        "n_inlier_5px": 0,
        "n_inlier_on_target": 0,
        "inlier_ratio_5px": 0.0,
        "pts_src": np.zeros((0, 2), np.float32),
        "pts_rgb": np.zeros((0, 2), np.float32),
        "inlier": np.zeros((0,), bool),
    }
    if da is None or db is None or n_a < 2 or n_b < 2:
        return out
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in (pair,) if m.distance < 0.75 * n.distance]
    if not good:
        return out
    pts_a = np.float32([kpa[m.queryIdx].pt for m in good])
    pts_b = np.float32([kpb[m.trainIdx].pt for m in good])
    dist = np.linalg.norm(pts_a - pts_b, axis=1)
    in3 = dist <= 3.0
    in5 = dist <= px
    on_target = np.zeros(len(good), dtype=bool)
    if mask is not None:
        h, w = mask.shape[:2]
        xb = np.clip(np.round(pts_b[:, 0]).astype(int), 0, w - 1)
        yb = np.clip(np.round(pts_b[:, 1]).astype(int), 0, h - 1)
        on_target = mask[yb, xb].astype(bool) & in5
    out.update(
        {
            "n_good": int(len(good)),
            "n_inlier_3px": int(in3.sum()),
            "n_inlier_5px": int(in5.sum()),
            "n_inlier_on_target": int(on_target.sum()),
            "inlier_ratio_5px": float(in5.mean()),
            "pts_src": pts_a,
            "pts_rgb": pts_b,
            "inlier": in5,
        }
    )
    return out


def _orb_row(metrics: dict) -> dict:
    return {
        "rgb_keypoints": int(metrics["rgb_keypoints"]),
        "event_keypoints": int(metrics["event_keypoints"]),
        "spatial_matches": int(metrics["spatial_matches"]),
        "descriptor_true_matches": int(metrics["descriptor_true_matches"]),
        "spatial_f1": float(metrics["spatial_f1"]),
        "inlier_ratio": float(metrics["inlier_ratio"]),
    }


def _load_e2vid(method: str) -> tuple[list[np.ndarray], np.ndarray]:
    recon = E2VID / METHODS[method]["e2vid"] / "reconstruction"
    ts = np.loadtxt(recon / "timestamps.txt")
    frames = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in sorted(recon.glob("frame_*.png"))]
    return frames, np.asarray(ts, dtype=np.float64)


def _simulate(luminance, timestamps, valid, prism, rpm: float, device: str) -> list:
    if rpm <= 0:
        windows, _ = _dvs_windows(luminance, timestamps)
        return windows
    windows, _, _ = _ami_windows(luminance, timestamps, valid, prism, rpm, device=device)
    return windows


def _draw_box(ax, box, color, lw=1.6, ls="-"):
    box = _xywh(box)
    if box is None:
        return
    ax.add_patch(Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor=color, linewidth=lw, linestyle=ls))


def _sift_summary(rows: list[dict], key: str) -> dict:
    if not rows:
        return {}
    return {
        "mean_inlier_3px": float(np.mean([r[key]["n_inlier_3px"] for r in rows])),
        "mean_inlier_5px": float(np.mean([r[key]["n_inlier_5px"] for r in rows])),
        "mean_inlier_on_target": float(np.mean([r[key]["n_inlier_on_target"] for r in rows])),
        "mean_src_kp": float(np.mean([r[key]["n_src"] for r in rows])),
        "mean_rgb_kp": float(np.mean([r[key]["n_rgb"] for r in rows])),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print("device", device, flush=True)
    report: dict = {
        "scene": SCENE,
        "start": START,
        "n_frames": N_FRAMES,
        "spatial_px": SPATIAL_PX,
        "protocol": {
            "detector": "event-count connected component; masks scoring-only",
            "tracker": "GT-init search-window event centroid; box size frozen",
            "matching": "ORB spatial<=3px (paper helper); SIFT Lowe 0.75 + spatial inliers",
        },
    }
    viz: dict = {}

    for regime in ("co_motion", "object_only"):
        print(f"\n=== {regime} ===", flush=True)
        scene = _scene_path(DATA, "test", SCENE, regime)
        end = START + N_FRAMES
        frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=end)
        frames = frames[START:]
        timestamps = timestamps[START : START + len(frames)]
        luminance = rgb_to_linear_luminance(frames)
        intrinsics = _intrinsics(scene)
        h, w = int(intrinsics.height), int(intrinsics.width)
        masks_all = _load_object_masks(scene, end, w, h)
        masks = masks_all[START : START + len(frames)]
        valid = np.ones((len(frames), h, w), dtype=bool)
        prism = RotatingWedgePrism(intrinsics, 0.5)
        rgb_u8 = np.clip(frames, 0, 255).astype(np.uint8) if frames.dtype != np.uint8 else frames
        vis_ids = [0, 30, min(len(frames) - 2, 88)]
        report[regime] = {"tracking": {}, "matching": {}}
        viz[regime] = {}

        e2vid_cache = {}
        if regime == "co_motion":
            for name in METHODS:
                e2vid_cache[name] = _load_e2vid(name)

        for name, spec in METHODS.items():
            print(f"  simulate {name} rpm={spec['rpm']}", flush=True)
            windows = _simulate(luminance, timestamps, valid, prism, spec["rpm"], device)
            print(f"    {sum(len(s) for s in windows)} events / {len(windows)} windows", flush=True)
            track = _score_track(windows, masks, (h, w))
            report[regime]["tracking"][name] = {k: track[k] for k in track if not k.endswith("boxes")}
            print(
                f"    det IoU={track['det_mean_iou']:.3f}  trk IoU={track['trk_mean_iou']:.3f}  "
                f"cover={track['target_event_coverage']:.3f}",
                flush=True,
            )

            event_sift_rows = []
            orb_rows = []
            e2vid_sift_rows = []
            vis_rows = []
            for i, stream in enumerate(windows):
                rgb = rgb_u8[min(i + 1, len(rgb_u8) - 1)]
                mask = np.asarray(masks[min(i + 1, len(masks) - 1)], dtype=bool)
                lum = luminance[min(i + 1, len(luminance) - 1)]
                ev_img = _event_image(stream, (h, w))
                sift_ev = _sift_match(ev_img, rgb, mask)
                orb = _feature_metrics(lum, stream)
                event_sift_rows.append({"event_sift": sift_ev})
                orb_rows.append(_orb_row(orb))
                if i in vis_ids:
                    vis_rows.append(
                        {
                            "i": i,
                            "t": float(timestamps[min(i + 1, len(timestamps) - 1)]),
                            "rgb": rgb,
                            "mask": mask,
                            "ev": ev_img,
                            "sift_ev": sift_ev,
                            "orb": orb_rows[-1],
                        }
                    )

            if name in e2vid_cache:
                recs, rec_t = e2vid_cache[name]
                for j, rec in enumerate(recs):
                    k = int(np.argmin(np.abs(timestamps - rec_t[j])))
                    rgb = rgb_u8[k]
                    mask = np.asarray(masks[k], dtype=bool)
                    sift_e2 = _sift_match(rec, rgb, mask)
                    e2vid_sift_rows.append({"e2vid_sift": sift_e2, "t": float(rec_t[j]), "frame": k})
                for row in vis_rows:
                    j = int(np.argmin(np.abs(rec_t - row["t"])))
                    row["e2vid"] = recs[j]
                    row["sift_e2"] = _sift_match(recs[j], row["rgb"], row["mask"])

            report[regime]["matching"][name] = {
                "event_sift": _sift_summary(event_sift_rows, "event_sift"),
                "event_orb": {
                    "mean_spatial_matches": float(np.mean([r["spatial_matches"] for r in orb_rows])),
                    "mean_descriptor_true_matches": float(np.mean([r["descriptor_true_matches"] for r in orb_rows])),
                    "mean_spatial_f1": float(np.mean([r["spatial_f1"] for r in orb_rows])),
                    "mean_event_keypoints": float(np.mean([r["event_keypoints"] for r in orb_rows])),
                },
            }
            if e2vid_sift_rows:
                report[regime]["matching"][name]["e2vid_sift"] = _sift_summary(e2vid_sift_rows, "e2vid_sift")
            es = report[regime]["matching"][name]["event_sift"]
            print(
                f"    event-SIFT inliers@5px={es['mean_inlier_5px']:.1f}  on-car={es['mean_inlier_on_target']:.1f}  "
                f"ORB spatial={report[regime]['matching'][name]['event_orb']['mean_spatial_matches']:.1f}",
                flush=True,
            )
            if "e2vid_sift" in report[regime]["matching"][name]:
                e2 = report[regime]["matching"][name]["e2vid_sift"]
                print(
                    f"    E2VID-SIFT inliers@5px={e2['mean_inlier_5px']:.1f}  on-car={e2['mean_inlier_on_target']:.1f}",
                    flush=True,
                )
            viz[regime][name] = {
                "track": track,
                "rgb": rgb_u8,
                "timestamps": timestamps,
                "vis": vis_rows,
                "h": h,
                "w": w,
            }

    for regime in ("co_motion", "object_only"):
        fig, axes = plt.subplots(3, 3, figsize=(11.2, 8.4))
        for r, name in enumerate(METHODS):
            pack = viz[regime][name]
            track = pack["track"]
            for c, vid in enumerate([0, 30, min(len(pack["rgb"]) - 2, 88)]):
                ax = axes[r, c]
                ax.imshow(pack["rgb"][min(vid + 1, len(pack["rgb"]) - 1)])
                _draw_box(ax, track["gt_boxes"][min(vid, len(track["gt_boxes"]) - 1)], "#f6c445", lw=2.0)
                _draw_box(ax, track["trk_boxes"][min(vid, len(track["trk_boxes"]) - 1)], METHODS[name]["color"], lw=1.8)
                _draw_box(ax, track["det_boxes"][min(vid, len(track["det_boxes"]) - 1)], "white", lw=1.0, ls="--")
                tq = float(pack["timestamps"][min(vid + 1, len(pack["timestamps"]) - 1)])
                ax.set_title(f"{name}  t={tq:.2f}s", fontsize=9)
                ax.set_axis_off()
        fig.suptitle(f"{SCENE} {regime}: yellow=GT  color=centroid-track  dashed=global CC", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_track_{regime}.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(3, 3, figsize=(11.2, 8.6))
        for r, name in enumerate(METHODS):
            vis_rows = viz[regime][name]["vis"]
            w = viz[regime][name]["w"]
            for c, row in enumerate(vis_rows[:3]):
                ax = axes[r, c]
                if "e2vid" in row:
                    left = row["e2vid"]
                    pts_a, pts_b, inl = row["sift_e2"]["pts_src"], row["sift_e2"]["pts_rgb"], row["sift_e2"]["inlier"]
                    n_in = row["sift_e2"]["n_inlier_5px"]
                    n_car = row["sift_e2"]["n_inlier_on_target"]
                    tag = "E2VID-SIFT"
                else:
                    left = row["ev"]
                    pts_a, pts_b, inl = row["sift_ev"]["pts_src"], row["sift_ev"]["pts_rgb"], row["sift_ev"]["inlier"]
                    n_in = row["sift_ev"]["n_inlier_5px"]
                    n_car = row["sift_ev"]["n_inlier_on_target"]
                    tag = "event-SIFT"
                left_rgb = cv2.cvtColor(left, cv2.COLOR_GRAY2RGB) if left.ndim == 2 else left
                canvas = np.concatenate([left_rgb, row["rgb"]], axis=1)
                ax.imshow(canvas)
                drawn = 0
                for pa, pb, ok in zip(pts_a, pts_b, inl):
                    if not ok:
                        continue
                    ax.plot([pa[0], pb[0] + w], [pa[1], pb[1]], color=METHODS[name]["color"], lw=0.55, alpha=0.8)
                    drawn += 1
                    if drawn >= 80:
                        break
                ax.set_title(f"{name}  {tag}  inliers={n_in}  on-car={n_car}", fontsize=8)
                ax.set_axis_off()
        fig.suptitle(
            f"{SCENE} {regime}: left=source  right=RGB  lines=SIFT spatial inliers ≤{SPATIAL_PX:.0f}px",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(OUT / f"fig_match_{regime}.png", dpi=160)
        plt.close(fig)

    # summary bars
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))
    names = list(METHODS)
    x = np.arange(len(names))
    for ax, regime, metric, title in (
        (axes[0, 0], "object_only", "trk_mean_iou", "object_only track IoU"),
        (axes[0, 1], "object_only", "target_event_coverage", "object_only target coverage"),
        (axes[1, 0], "co_motion", "trk_mean_iou", "co_motion track IoU"),
        (axes[1, 1], "co_motion", "target_event_coverage", "co_motion target coverage"),
    ):
        vals = [report[regime]["tracking"][n][metric] for n in names]
        ax.bar(x, vals, color=[METHODS[n]["color"] for n in names])
        ax.set_xticks(x, names)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_track_summary.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for ax, regime, title in (
        (axes[0], "co_motion", "co_motion mean SIFT inliers @5px"),
        (axes[1], "object_only", "object_only mean SIFT inliers @5px"),
    ):
        ev = [report[regime]["matching"][n]["event_sift"]["mean_inlier_5px"] for n in names]
        car = [report[regime]["matching"][n]["event_sift"]["mean_inlier_on_target"] for n in names]
        width = 0.35
        ax.bar(x - width / 2, ev, width, label="all inliers", color="#7a7a7a")
        ax.bar(x + width / 2, car, width, label="on-car", color=[METHODS[n]["color"] for n in names])
        if "e2vid_sift" in report[regime]["matching"][names[0]]:
            e2 = [report[regime]["matching"][n]["e2vid_sift"]["mean_inlier_5px"] for n in names]
            ax.plot(x, e2, "k--o", label="E2VID inliers", markersize=5)
        ax.set_xticks(x, names)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_match_summary.png", dpi=160)
    plt.close(fig)

    (OUT / "track_match_stats.json").write_text(json.dumps(report, indent=2))
    print("\nWrote", OUT / "track_match_stats.json", flush=True)


if __name__ == "__main__":
    main()
