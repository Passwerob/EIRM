#!/usr/bin/env python3
"""Downstream track + match: Ours vs Fixed 180 vs Fixed 360.

Paper Table 1 controller (eveyes_table1_m0): gears {360, 180}, 180 floor,
confidence 0.5, no IMU gate, dwell 25 ms. Masks score only.
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
import torch
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.motion_net import TinyCausalGearPolicy
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import EventStream
from scripts.evaluate_anchor_feature_matching import _event_image, _feature_metrics
from scripts.evaluate_dl3dv_downstream import _ami_windows, _ours_windows, _visual_motion
from scripts.evaluate_event_target_tracking import _box, _iou
from scripts.evaluate_eveyes_v2_downstream import _imu_fraction, _intrinsics, _scene_path
from scripts.run_car_e2vid_track_match import (
    coverage_purity,
    mil_track,
    orange_car_mask,
    pad_box,
    sift_pair,
)
from scripts.run_e2vid_eirm_compare import _concat, _run_e2vid, _write_e2vid_txt
from scripts.train_inverse_motion_policy import FEATURE_DIM

SCENE = "Scene_005"
START = 60
N_FRAMES = 90
DATA = ROOT / "data/eveyes_comotion_v2_ready"
STUDENT = ROOT / "KEEP_PAPER_WEIGHTS_DO_NOT_DELETE/eveyes_table1_m0/policy.pt"
OUT = ROOT / "results/ami_next/ours_vs_fixed_track_match"
E2VID_FIXED = ROOT / "results/ami_next/e2vid_eirm"
E2VID_ROOT = Path("/data/fcr/rpg_e2vid")
E2VID_WEIGHTS = E2VID_ROOT / "pretrained/E2VID_lightweight.pth.tar"
PYTHON = "/data/fcr/miniconda3/envs/v2e/bin/python"

METHODS = (
    ("fixed_180", "Fixed 180", 180.0, "#4c72b0"),
    ("ours", "Ours", None, "#dd8452"),
    ("fixed_360", "Fixed 360", 360.0, "#55a868"),
)


def _load_policy():
    ckpt = torch.load(STUDENT, map_location="cpu", weights_only=True)
    model = TinyCausalGearPolicy(
        feature_dim=FEATURE_DIM,
        hidden_dim=int(ckpt["hidden_dim"]),
        num_gears=int(ckpt["num_gears"]),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _load_scene(regime: str):
    scene = _scene_path(DATA, "test", SCENE, regime)
    end = START + N_FRAMES
    frames, timestamps = load_rgb_sequence(
        scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=end
    )
    frames = np.clip(frames[START:], 0, 255).astype(np.uint8)
    timestamps = timestamps[START : START + len(frames)]
    luminance = rgb_to_linear_luminance(frames)
    intrinsics = _intrinsics(scene)
    h, w = int(intrinsics.height), int(intrinsics.width)
    valid = np.ones((len(frames), h, w), dtype=bool)
    prism = RotatingWedgePrism(intrinsics, 0.5)
    cars = np.stack([orange_car_mask(img) for img in frames])
    boxes = [_box(m) for m in cars]
    return {
        "scene": scene,
        "frames": frames,
        "timestamps": timestamps,
        "luminance": luminance,
        "intrinsics": intrinsics,
        "valid": valid,
        "prism": prism,
        "cars": cars,
        "boxes": boxes,
        "h": h,
        "w": w,
    }


def _simulate(pack, rpm, device, model=None, ckpt=None):
    lum, ts, valid, prism = pack["luminance"], pack["timestamps"], pack["valid"], pack["prism"]
    if rpm is not None:
        windows, _, _ = _ami_windows(lum, ts, valid, prism, float(rpm), device=device)
        rpms = [float(rpm)] * len(windows)
        return windows, rpms
    visual = _visual_motion(lum, ts)
    imu = _imu_fraction(
        pack["scene"] / "imu.csv",
        ts,
        (float(ckpt["imu_motion_bounds"][0]), float(ckpt["imu_motion_bounds"][1])),
    )
    windows, rpms = _ours_windows(
        lum,
        ts,
        valid,
        prism,
        visual=visual,
        imu=imu,
        model=model,
        checkpoint=ckpt,
        confidence=0.50,
        require_imu=False,
        dwell=0.025,
        device=device,
    )
    return windows, rpms


def _score_events(windows, pack) -> dict:
    cars, boxes, frames, lum = pack["cars"], pack["boxes"], pack["frames"], pack["luminance"]
    h, w = pack["h"], pack["w"]
    covers, purities, orb_all, orb_car = [], [], [], []
    for i, stream in enumerate(windows):
        j = min(i + 1, len(cars) - 1)
        cov, pur = coverage_purity(stream, cars[j])
        covers.append(cov)
        purities.append(pur)
        orb_all.append(int(_feature_metrics(lum[j], stream)["spatial_matches"]))
        box = boxes[j]
        if box is None:
            orb_car.append(0)
            continue
        x, y, bw, bh = pad_box(box, h, w, 10)
        keep = (stream.x >= x) & (stream.x < x + bw) & (stream.y >= y) & (stream.y < y + bh)
        if not np.any(keep):
            orb_car.append(0)
            continue
        local = EventStream(stream.x[keep] - x, stream.y[keep] - y, stream.t[keep], stream.p[keep])
        orb_car.append(int(_feature_metrics(lum[j][y : y + bh, x : x + bw], local)["spatial_matches"]))
    return {
        "mean_events": float(np.mean([len(s) for s in windows])),
        "car_coverage": float(np.mean(covers)),
        "car_purity": float(np.mean(purities)),
        "orb_spatial_all": float(np.mean(orb_all)),
        "orb_spatial_car": float(np.mean(orb_car)),
    }


def _load_recon(folder: Path):
    ts = np.loadtxt(folder / "timestamps.txt")
    frames = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in sorted(folder.glob("frame_*.png"))]
    return frames, np.asarray(ts, dtype=np.float64)


def _e2vid_metrics(recs, rec_t, pack) -> dict:
    frames, ts, cars, boxes = pack["frames"], pack["timestamps"], pack["cars"], pack["boxes"]
    h, w = pack["h"], pack["w"]
    init = None
    crop_first = 0
    on_car_first = 0
    rows = []
    for j, rec in enumerate(recs):
        k = int(np.argmin(np.abs(ts - rec_t[j])))
        full = sift_pair(rec, frames[k], cars[k])
        box = boxes[k]
        crop_n = 0
        if box is not None:
            x, y, bw, bh = pad_box(box, h, w, 16)
            crop_n = sift_pair(rec[y : y + bh, x : x + bw], frames[k][y : y + bh, x : x + bw])["n_inlier"]
            if init is None:
                init = pad_box(box, h, w, 8)
                crop_first = crop_n
                on_car_first = full["n_on_car"]
        rows.append({"t": float(rec_t[j]), "n_inlier": full["n_inlier"], "n_on_car": full["n_on_car"], "n_crop": crop_n})
    mil = mil_track(recs, init) if init is not None else [None] * len(recs)
    ious = []
    for j, pred in enumerate(mil):
        k = int(np.argmin(np.abs(ts - rec_t[j])))
        ious.append(_iou(pred, boxes[k]))
    ious = np.asarray(ious, dtype=np.float64)
    return {
        "first_crop_inlier": int(crop_first),
        "first_on_car": int(on_car_first),
        "mean_crop_inlier": float(np.mean([r["n_crop"] for r in rows])),
        "mil_mean_iou": float(ious.mean()) if ious.size else 0.0,
        "mil_success_0.5": float((ious >= 0.5).mean()) if ious.size else 0.0,
        "mil_boxes": mil,
        "recs": recs,
        "rec_t": rec_t,
        "first_rec": recs[0],
        "first_rgb": frames[int(np.argmin(np.abs(ts - rec_t[0])))],
        "first_box": boxes[int(np.argmin(np.abs(ts - rec_t[0])))],
        "gt_at_rec": [boxes[int(np.argmin(np.abs(ts - t)))] for t in rec_t],
    }


def _draw_box(ax, box, color, lw=1.8):
    if box is None:
        return
    ax.add_patch(Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor=color, linewidth=lw))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = _load_policy()
    print("device", device, "gears", ckpt.get("candidate_rpm"), "floor", ckpt.get("quality_floor"), flush=True)
    report: dict = {
        "scene": SCENE,
        "start": START,
        "n_frames": N_FRAMES,
        "student": str(STUDENT),
        "protocol": "Ours vs Fixed 180 vs Fixed 360; Table 1 m0 policy; orange-car scoring",
        "confidence": 0.5,
        "require_imu": False,
        "dwell": 0.025,
    }

    recon_dirs = {
        "fixed_180": E2VID_FIXED / "eirm_180" / "reconstruction",
        "fixed_360": E2VID_FIXED / "eirm_360" / "reconstruction",
        "ours": OUT / "e2vid_ours" / "reconstruction",
    }

    for regime in ("co_motion", "object_only"):
        print(f"\n=== {regime} ===", flush=True)
        pack = _load_scene(regime)
        report[regime] = {}
        vis = {}
        for key, label, rpm, color in METHODS:
            print(f"  simulate {label}", flush=True)
            windows, rpms = _simulate(pack, rpm, device, model, ckpt)
            ev = _score_events(windows, pack)
            ev["mean_rpm"] = float(np.mean(rpms))
            ev["frac_360"] = float(np.mean([r >= 359 for r in rpms]))
            ev["frac_180"] = float(np.mean([179 <= r <= 181 for r in rpms]))
            print(f"    rpm={ev['mean_rpm']:.1f}  frac360={ev['frac_360']:.2f}  cover={ev['car_coverage']:.3f}  ORB-car={ev['orb_spatial_car']:.1f}", flush=True)
            report[regime][key] = {"events": ev, "rpms": [float(r) for r in rpms]}
            vis[key] = {"windows": windows, "rpms": rpms, "pack": pack, "color": color, "label": label}

            if regime == "co_motion" and key == "ours":
                events = _concat(windows)
                txt = OUT / "ours_events.txt"
                n = _write_e2vid_txt(txt, events, pack["w"], pack["h"])
                print(f"    wrote {n} events -> E2VID", flush=True)
                _run_e2vid(E2VID_ROOT, E2VID_WEIGHTS, txt, OUT / "e2vid_ours", PYTHON)

        if regime == "co_motion":
            for key, label, _, color in METHODS:
                recs, rec_t = _load_recon(recon_dirs[key])
                e2 = _e2vid_metrics(recs, rec_t, pack)
                keep = {k: e2[k] for k in e2 if k not in {"mil_boxes", "recs", "rec_t", "first_rec", "first_rgb", "first_box", "gt_at_rec"}}
                report[regime][key]["e2vid"] = keep
                vis[key]["e2vid"] = e2
                print(
                    f"    {label} MIL IoU={keep['mil_mean_iou']:.3f}  success={keep['mil_success_0.5']:.2f}  "
                    f"first crop SIFT={keep['first_crop_inlier']}",
                    flush=True,
                )

            fig, axes = plt.subplots(3, 3, figsize=(11.2, 8.2))
            ids = [0, len(vis["ours"]["e2vid"]["recs"]) // 2, len(vis["ours"]["e2vid"]["recs"]) - 1]
            for r, (key, label, _, color) in enumerate(METHODS):
                e2 = vis[key]["e2vid"]
                for c, j in enumerate(ids):
                    axes[r, c].imshow(e2["recs"][j], cmap="gray")
                    _draw_box(axes[r, c], e2["gt_at_rec"][j], "#f6c445", lw=2.0)
                    _draw_box(axes[r, c], e2["mil_boxes"][j], color)
                    axes[r, c].set_title(f"{label}  t={e2['rec_t'][j]:.2f}s", fontsize=9)
                    axes[r, c].set_axis_off()
            fig.suptitle("co_motion E2VID + MIL: yellow=car GT  color=track", fontsize=11)
            fig.tight_layout()
            fig.savefig(OUT / "fig_track_e2vid_mil.png", dpi=160)
            plt.close(fig)

            fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))
            for ax, (key, label, _, color) in zip(axes, METHODS):
                e2 = vis[key]["e2vid"]
                ax.imshow(e2["first_rec"], cmap="gray")
                _draw_box(ax, e2["first_box"], color, lw=2.0)
                ax.set_title(f"{label}  t={e2['rec_t'][0]:.2f}s  crop-SIFT={e2['first_crop_inlier']}", fontsize=9)
                ax.set_axis_off()
            fig.suptitle("First E2VID window: Fixed 180 · Ours · Fixed 360", fontsize=11)
            fig.tight_layout()
            fig.savefig(OUT / "fig_e2vid_first_window.png", dpi=160)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7.8, 3.2))
            tmid = 0.5 * (pack["timestamps"][:-1] + pack["timestamps"][1:])
            for key, label, _, color in METHODS:
                ax.step(tmid, vis[key]["rpms"], where="post", color=color, lw=2, label=label)
            ax.set_ylim(0, 400)
            ax.set_xlabel("t (s)")
            ax.set_ylabel("RPM")
            ax.set_title("Causal gear: Ours vs two fixed gears")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(OUT / "fig_rpm_timeline.png", dpi=160)
            plt.close(fig)

        # event-image car crops
        vis_ids = [0, 30, min(len(pack["frames"]) - 2, 88)]
        fig, axes = plt.subplots(3, 3, figsize=(11.0, 8.0))
        h, w = pack["h"], pack["w"]
        for r, (key, label, _, color) in enumerate(METHODS):
            for c, i in enumerate(vis_ids):
                stream = vis[key]["windows"][i]
                rgb = pack["frames"][min(i + 1, len(pack["frames"]) - 1)]
                box = pack["boxes"][min(i + 1, len(pack["boxes"]) - 1)]
                ev = _event_image(stream, (h, w))
                if box is None:
                    axes[r, c].imshow(ev, cmap="gray")
                else:
                    x, y, bw, bh = pad_box(box, h, w, 12)
                    left = cv2.cvtColor(ev[y : y + bh, x : x + bw], cv2.COLOR_GRAY2RGB)
                    right = rgb[y : y + bh, x : x + bw]
                    axes[r, c].imshow(np.concatenate([left, right], axis=1))
                axes[r, c].set_title(f"{label}", fontsize=8)
                axes[r, c].set_axis_off()
        fig.suptitle(f"{regime} car ROI: left=event image  right=RGB", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_match_car_{regime}.png", dpi=160)
        plt.close(fig)

        names = [m[0] for m in METHODS]
        labels = [m[1] for m in METHODS]
        colors = [m[3] for m in METHODS]
        x = np.arange(3)
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
        for ax, metric, title, ylim in (
            (axes[0], "car_coverage", f"{regime} car coverage", (0, 1)),
            (axes[1], "orb_spatial_car", f"{regime} ORB matches in car ROI", None),
            (axes[2], "mean_rpm", f"{regime} mean RPM", (0, 400)),
        ):
            vals = [report[regime][n]["events"][metric] for n in names]
            ax.bar(x, vals, color=colors)
            ax.set_xticks(x, labels, fontsize=8)
            ax.set_title(title, fontsize=9)
            if ylim:
                ax.set_ylim(*ylim)
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_summary_{regime}.png", dpi=160)
        plt.close(fig)

    slim = json.loads(json.dumps(report))
    for regime in ("co_motion", "object_only"):
        for key, _, _, _ in METHODS:
            slim[regime][key].pop("rpms", None)
    (OUT / "ours_vs_fixed_stats.json").write_text(json.dumps(slim, indent=2))
    print("\nWrote", OUT / "ours_vs_fixed_stats.json", flush=True)


if __name__ == "__main__":
    main()
