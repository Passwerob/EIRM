#!/usr/bin/env python3
"""Car-centric tracking + matching. Dataset object_only masks sit on the skyline;
this script uses a compact orange-body mask of the Aventador as the target.
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
from scripts.evaluate_event_target_tracking import _box, _iou
from scripts.evaluate_eveyes_v2_downstream import _intrinsics, _scene_path

SCENE = "Scene_005"
START = 60
N_FRAMES = 90
OUT = ROOT / "results/ami_next/e2vid_track_match"
E2VID = ROOT / "results/ami_next/e2vid_eirm"
DATA = ROOT / "data/eveyes_comotion_v2_ready"
METHODS = {
    "passive": {"rpm": 0.0, "e2vid": "passive", "color": "#c44e52", "label": "passive"},
    "eirm_180": {"rpm": 180.0, "e2vid": "eirm_180", "color": "#4c72b0", "label": "EIRM 180"},
    "eirm_360": {"rpm": 360.0, "e2vid": "eirm_360", "color": "#55a868", "label": "EIRM 360"},
}


def orange_car_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    active = cv2.inRange(hsv, (8, 120, 90), (28, 255, 255))
    active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats((active > 0).astype(np.uint8))
    h, w = active.shape
    best, best_score = None, -1.0
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < 80 or bh < 12 or bw < 30 or y < 0.35 * h:
            continue
        compactness = area / max(bw * bh, 1)
        aspect = bw / max(bh, 1)
        if compactness < 0.25 or aspect > 6 or aspect < 0.8:
            continue
        score = float(area * compactness)
        if score > best_score:
            best, best_score = i, score
    return labels == best if best is not None else np.zeros(active.shape, dtype=bool)


def pad_box(box, h, w, m=12):
    x, y, bw, bh = [int(v) for v in box]
    x0 = max(x - m, 0)
    y0 = max(y - m, 0)
    x1 = min(x + bw + m, w)
    y1 = min(y + bh + m, h)
    return x0, y0, x1 - x0, y1 - y0


def sift_pair(src, rgb, mask=None, px=5.0):
    a = src if src.ndim == 2 else cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    b = rgb if rgb.ndim == 2 else cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    a = np.clip(a, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    sift = cv2.SIFT_create(nfeatures=800)
    kpa, da = sift.detectAndCompute(a, None)
    kpb, db = sift.detectAndCompute(b, None)
    n_a = 0 if kpa is None else len(kpa)
    n_b = 0 if kpb is None else len(kpb)
    empty = {
        "n_src": n_a,
        "n_rgb": n_b,
        "n_good": 0,
        "n_inlier": 0,
        "n_on_car": 0,
        "pts_src": np.zeros((0, 2), np.float32),
        "pts_rgb": np.zeros((0, 2), np.float32),
        "inlier": np.zeros((0,), bool),
    }
    if da is None or db is None or n_a < 2 or n_b < 2:
        return empty
    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for pair in raw if len(pair) == 2 for m, n in (pair,) if m.distance < 0.75 * n.distance]
    if not good:
        return empty
    pa = np.float32([kpa[m.queryIdx].pt for m in good])
    pb = np.float32([kpb[m.trainIdx].pt for m in good])
    inl = np.linalg.norm(pa - pb, axis=1) <= px
    on = np.zeros(len(good), dtype=bool)
    if mask is not None and mask.any():
        h, w = mask.shape[:2]
        xb = np.clip(np.round(pb[:, 0]).astype(int), 0, w - 1)
        yb = np.clip(np.round(pb[:, 1]).astype(int), 0, h - 1)
        on = mask[yb, xb] & inl
    return {
        "n_src": n_a,
        "n_rgb": n_b,
        "n_good": int(len(good)),
        "n_inlier": int(inl.sum()),
        "n_on_car": int(on.sum()),
        "pts_src": pa,
        "pts_rgb": pb,
        "inlier": inl,
        "on_car": on,
    }


def load_e2vid(name: str):
    recon = E2VID / METHODS[name]["e2vid"] / "reconstruction"
    ts = np.loadtxt(recon / "timestamps.txt")
    frames = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in sorted(recon.glob("frame_*.png"))]
    return frames, np.asarray(ts, dtype=np.float64)


def mil_track(frames, init_box):
    tracker = cv2.TrackerMIL_create()
    bgr0 = cv2.cvtColor(frames[0], cv2.COLOR_GRAY2BGR)
    x, y, w, h = [int(v) for v in init_box]
    tracker.init(bgr0, (x, y, w, h))
    boxes = [init_box]
    for frame in frames[1:]:
        ok, box = tracker.update(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        boxes.append(tuple(box) if ok else None)
    return boxes


def mean_shift_track(windows, init_box, shape, min_mass=25):
    h, w = shape
    box = [float(v) for v in init_box]
    preds = []
    bw0, bh0 = box[2], box[3]
    for stream in windows:
        count = np.zeros((h, w), dtype=np.float32)
        if len(stream):
            xs = np.clip(np.asarray(stream.x, dtype=np.int64), 0, w - 1)
            ys = np.clip(np.asarray(stream.y, dtype=np.int64), 0, h - 1)
            np.add.at(count, (ys, xs), 1)
        x, y, bw, bh = box
        for _ in range(6):
            x0, y0 = int(max(x, 0)), int(max(y, 0))
            x1, y1 = int(min(x + bw, w)), int(min(y + bh, h))
            crop = count[y0:y1, x0:x1]
            mass = float(crop.sum())
            if mass < min_mass:
                break
            yy, xx = np.mgrid[0 : crop.shape[0], 0 : crop.shape[1]]
            cx = x0 + float((xx * crop).sum() / mass)
            cy = y0 + float((yy * crop).sum() / mass)
            x = np.clip(cx - 0.5 * bw0, 0, w - bw0)
            y = np.clip(cy - 0.5 * bh0, 0, h - bh0)
            bw, bh = bw0, bh0
        box = [x, y, bw0, bh0]
        preds.append(tuple(box))
    return preds


def coverage_purity(stream, mask):
    if mask.sum() == 0:
        return 0.0, 0.0
    h, w = mask.shape
    if len(stream) == 0:
        return 0.0, 0.0
    xs = np.clip(np.asarray(stream.x, dtype=np.int64), 0, w - 1)
    ys = np.clip(np.asarray(stream.y, dtype=np.int64), 0, h - 1)
    on = mask[ys, xs]
    hit = np.zeros((h, w), dtype=np.uint8)
    hit[ys, xs] = 1
    cover = float(hit[mask].mean())
    purity = float(on.mean()) if len(on) else 0.0
    return cover, purity


def draw_box(ax, box, color, lw=1.8, ls="-"):
    if box is None:
        return
    ax.add_patch(Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor=color, linewidth=lw, linestyle=ls))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    scene = _scene_path(DATA, "test", SCENE, "co_motion")
    end = START + N_FRAMES
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=end)
    frames = np.clip(frames[START:], 0, 255).astype(np.uint8)
    timestamps = timestamps[START : START + len(frames)]
    car_masks = np.stack([orange_car_mask(img) for img in frames])
    car_boxes = [_box(m) for m in car_masks]
    h, w = frames.shape[1:3]
    report = {"target": "orange Aventador body (color mask; not dataset skyline mask)", "scene": SCENE}

    # ----- co_motion E2VID matching + MIL -----
    report["co_motion"] = {"e2vid_sift": {}, "e2vid_mil": {}}
    time_plot = {}
    vis_match = {}
    vis_mil = {}
    for name, spec in METHODS.items():
        recs, rec_t = load_e2vid(name)
        rows = []
        mil_init = None
        mil_boxes = None
        for j, rec in enumerate(recs):
            k = int(np.argmin(np.abs(timestamps - rec_t[j])))
            rgb = frames[k]
            mask = car_masks[k]
            full = sift_pair(rec, rgb, mask)
            box = car_boxes[k]
            crop = None
            if box is not None:
                x, y, bw, bh = pad_box(box, h, w, 16)
                crop = sift_pair(rec[y : y + bh, x : x + bw], rgb[y : y + bh, x : x + bw], mask[y : y + bh, x : x + bw])
            rows.append(
                {
                    "t": float(rec_t[j]),
                    "frame": int(k),
                    "n_inlier": full["n_inlier"],
                    "n_on_car": full["n_on_car"],
                    "n_crop_inlier": 0 if crop is None else crop["n_inlier"],
                }
            )
            if j == 0:
                vis_match[name] = {"rec": rec, "rgb": rgb, "mask": mask, "full": full, "box": box, "t": float(rec_t[j])}
                mil_init = pad_box(box, h, w, 8) if box is not None else None
        if mil_init is not None:
            mil_boxes = mil_track(recs, mil_init)
            ious = []
            for j, pred in enumerate(mil_boxes):
                k = int(np.argmin(np.abs(timestamps - rec_t[j])))
                ious.append(_iou(pred, car_boxes[k]))
            report["co_motion"]["e2vid_mil"][name] = {
                "mean_iou": float(np.mean(ious)),
                "success_0.5": float(np.mean(np.array(ious) >= 0.5)),
            }
            vis_mil[name] = {"recs": recs, "boxes": mil_boxes, "gt": [car_boxes[int(np.argmin(np.abs(timestamps - t)))] for t in rec_t], "ts": rec_t}
        early = [r for r in rows if r["t"] <= timestamps[0] + 0.25]
        report["co_motion"]["e2vid_sift"][name] = {
            "mean_inlier_all": float(np.mean([r["n_inlier"] for r in rows])),
            "mean_on_car_all": float(np.mean([r["n_on_car"] for r in rows])),
            "mean_crop_inlier_all": float(np.mean([r["n_crop_inlier"] for r in rows])),
            "early_inlier": float(np.mean([r["n_inlier"] for r in early]) if early else 0),
            "early_on_car": float(np.mean([r["n_on_car"] for r in early]) if early else 0),
            "early_crop_inlier": float(np.mean([r["n_crop_inlier"] for r in early]) if early else 0),
            "first_on_car": int(rows[0]["n_on_car"]),
            "first_crop_inlier": int(rows[0]["n_crop_inlier"]),
            "first_inlier": int(rows[0]["n_inlier"]),
            "curve": rows,
        }
        time_plot[name] = rows
        print(
            f"co_motion {name}: first on-car SIFT={rows[0]['n_on_car']} crop={rows[0]['n_crop_inlier']}  "
            f"early on-car={report['co_motion']['e2vid_sift'][name]['early_on_car']:.1f}  "
            f"MIL IoU={report['co_motion']['e2vid_mil'].get(name, {}).get('mean_iou', float('nan')):.3f}",
            flush=True,
        )

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for name, spec in METHODS.items():
        ts = [r["t"] for r in time_plot[name]]
        ax.plot(ts, [r["n_on_car"] for r in time_plot[name]], color=spec["color"], lw=2, label=f"{spec['label']} on-car")
        ax.plot(ts, [r["n_inlier"] for r in time_plot[name]], color=spec["color"], lw=1, ls="--", alpha=0.55)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("SIFT spatial inliers")
    ax.set_title("E2VID↔RGB SIFT (solid=on orange car, dashed=all)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_e2vid_sift_vs_time.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(9.6, 8.4))
    for r, name in enumerate(METHODS):
        pack = vis_match[name]
        rec, rgb, full, box = pack["rec"], pack["rgb"], pack["full"], pack["box"]
        axes[r, 0].imshow(rec, cmap="gray")
        draw_box(axes[r, 0], box, METHODS[name]["color"])
        axes[r, 0].set_title(f"{METHODS[name]['label']} E2VID  t={pack['t']:.2f}s", fontsize=9)
        axes[r, 0].set_axis_off()
        canvas = np.concatenate([cv2.cvtColor(rec, cv2.COLOR_GRAY2RGB), rgb], axis=1)
        axes[r, 1].imshow(canvas)
        drawn = 0
        on = full.get("on_car", full["inlier"])
        for pa, pb, ok in zip(full["pts_src"], full["pts_rgb"], on):
            if not ok:
                continue
            axes[r, 1].plot([pa[0], pb[0] + w], [pa[1], pb[1]], color=METHODS[name]["color"], lw=0.8, alpha=0.85)
            drawn += 1
            if drawn >= 40:
                break
        axes[r, 1].set_title(f"on-car inliers={full['n_on_car']}  all={full['n_inlier']}", fontsize=9)
        axes[r, 1].set_axis_off()
    fig.suptitle("First E2VID window only (before 180/360 auto-HDR collapse)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_e2vid_car_match_t1.png", dpi=160)
    plt.close(fig)

    if vis_mil:
        ids = [0, len(next(iter(vis_mil.values()))["recs"]) // 2, len(next(iter(vis_mil.values()))["recs"]) - 1]
        fig, axes = plt.subplots(3, 3, figsize=(11.0, 8.0))
        for r, name in enumerate(METHODS):
            pack = vis_mil[name]
            for c, j in enumerate(ids):
                axes[r, c].imshow(pack["recs"][j], cmap="gray")
                draw_box(axes[r, c], pack["gt"][j], "#f6c445", lw=2.0)
                draw_box(axes[r, c], pack["boxes"][j], METHODS[name]["color"])
                axes[r, c].set_title(f"{name}  t={pack['ts'][j]:.2f}s", fontsize=8)
                axes[r, c].set_axis_off()
        fig.suptitle("MIL on E2VID (yellow=car GT, color=track). Init from first window.", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "fig_e2vid_mil_track.png", dpi=160)
        plt.close(fig)

    # ----- object_only events on the actual car -----
    print("\n=== object_only car tracking/matching ===", flush=True)
    scene_o = _scene_path(DATA, "test", SCENE, "object_only")
    frames_o, ts_o = load_rgb_sequence(scene_o / "images", scene_o / "timestamps.csv", timestamp_unit="s", max_frames=end)
    frames_o = np.clip(frames_o[START:], 0, 255).astype(np.uint8)
    ts_o = ts_o[START : START + len(frames_o)]
    lum = rgb_to_linear_luminance(frames_o)
    car_o = np.stack([orange_car_mask(img) for img in frames_o])
    boxes_o = [_box(m) for m in car_o]
    valid = np.ones((len(frames_o), h, w), dtype=bool)
    prism = RotatingWedgePrism(_intrinsics(scene_o), 0.5)
    report["object_only"] = {}
    vis_o = {}
    for name, spec in METHODS.items():
        print(f"  simulate {name}", flush=True)
        if spec["rpm"] <= 0:
            windows, _ = _dvs_windows(lum, ts_o)
        else:
            windows, _, _ = _ami_windows(lum, ts_o, valid, prism, spec["rpm"], device=device)
        covers, purities, orb_car, orb_all = [], [], [], []
        for i, stream in enumerate(windows):
            mask = car_o[min(i + 1, len(car_o) - 1)]
            cov, pur = coverage_purity(stream, mask)
            covers.append(cov)
            purities.append(pur)
            rgb = frames_o[min(i + 1, len(frames_o) - 1)]
            lum_i = lum[min(i + 1, len(lum) - 1)]
            orb_all.append(_feature_metrics(lum_i, stream)["spatial_matches"])
            box = boxes_o[min(i + 1, len(boxes_o) - 1)]
            if box is None or len(stream) == 0:
                orb_car.append(0)
                continue
            x, y, bw, bh = pad_box(box, h, w, 10)
            keep = (stream.x >= x) & (stream.x < x + bw) & (stream.y >= y) & (stream.y < y + bh)
            local = EventStream(stream.x[keep] - x, stream.y[keep] - y, stream.t[keep], stream.p[keep]) if np.any(keep) else EventStream.empty()
            orb_car.append(_feature_metrics(lum_i[y : y + bh, x : x + bw], local)["spatial_matches"])
        init = boxes_o[0]
        preds = mean_shift_track(windows, init, (h, w)) if init is not None else [None] * len(windows)
        gt = [boxes_o[min(i + 1, len(boxes_o) - 1)] for i in range(len(windows))]
        ious = np.array([_iou(p, g) for p, g in zip(preds, gt)], dtype=np.float64)
        report["object_only"][name] = {
            "mean_events": float(np.mean([len(s) for s in windows])),
            "car_coverage": float(np.mean(covers)),
            "car_purity": float(np.mean(purities)),
            "trk_mean_iou": float(ious.mean()) if ious.size else 0.0,
            "trk_success_0.5": float((ious >= 0.5).mean()) if ious.size else 0.0,
            "orb_spatial_all": float(np.mean(orb_all)),
            "orb_spatial_car": float(np.mean(orb_car)),
        }
        vis_o[name] = {"rgb": frames_o, "preds": preds, "gt": gt, "ts": ts_o, "windows": windows, "car": car_o}
        print(report["object_only"][name], flush=True)

    fig, axes = plt.subplots(3, 3, figsize=(11.0, 8.0))
    vis_ids = [0, 30, min(len(frames_o) - 2, 88)]
    for r, name in enumerate(METHODS):
        pack = vis_o[name]
        for c, i in enumerate(vis_ids):
            axes[r, c].imshow(pack["rgb"][min(i + 1, len(pack["rgb"]) - 1)])
            draw_box(axes[r, c], pack["gt"][i], "#f6c445", lw=2.0)
            draw_box(axes[r, c], pack["preds"][i], METHODS[name]["color"])
            axes[r, c].set_title(f"{name}  t={pack['ts'][min(i+1,len(pack['ts'])-1)]:.2f}s", fontsize=8)
            axes[r, c].set_axis_off()
    fig.suptitle("object_only: yellow=orange-car GT  color=event mean-shift", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_track_object_only_car.png", dpi=160)
    plt.close(fig)

    names = list(METHODS)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    for ax, key, title, ylim in (
        (axes[0], "car_coverage", "object_only car coverage", (0, 1)),
        (axes[1], "orb_spatial_car", "ORB spatial matches in car ROI", None),
        (axes[2], "trk_mean_iou", "mean-shift IoU on car", (0, 1)),
    ):
        vals = [report["object_only"][n][key] for n in names]
        ax.bar(x, vals, color=[METHODS[n]["color"] for n in names])
        ax.set_xticks(x, [METHODS[n]["label"] for n in names], fontsize=8)
        ax.set_title(title, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_object_only_car_summary.png", dpi=160)
    plt.close(fig)

    # event-image vs RGB car crop at three times
    fig, axes = plt.subplots(3, 3, figsize=(11.0, 8.2))
    for r, name in enumerate(METHODS):
        pack = vis_o[name]
        for c, i in enumerate(vis_ids):
            stream = pack["windows"][i]
            rgb = pack["rgb"][min(i + 1, len(pack["rgb"]) - 1)]
            box = pack["gt"][i]
            if box is None:
                axes[r, c].set_axis_off()
                continue
            x0, y0, bw, bh = pad_box(box, h, w, 12)
            ev = _event_image(stream, (h, w))
            left = ev[y0 : y0 + bh, x0 : x0 + bw]
            right = rgb[y0 : y0 + bh, x0 : x0 + bw]
            left_rgb = cv2.cvtColor(left, cv2.COLOR_GRAY2RGB)
            canvas = np.concatenate([left_rgb, right], axis=1)
            axes[r, c].imshow(canvas)
            axes[r, c].set_title(f"{name}  car ROI", fontsize=8)
            axes[r, c].set_axis_off()
    fig.suptitle("object_only car crop: left=event image  right=RGB", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_match_object_only_car.png", dpi=160)
    plt.close(fig)

    # drop curves from json to keep it small
    for name in METHODS:
        report["co_motion"]["e2vid_sift"][name].pop("curve", None)
    (OUT / "car_track_match_stats.json").write_text(json.dumps(report, indent=2))
    print("Wrote", OUT / "car_track_match_stats.json", flush=True)


if __name__ == "__main__":
    main()
