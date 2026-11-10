#!/usr/bin/env python3
"""Construct one timestamp: Fixed 180 vs 3-gear at 0 RPM.

Paper Ours cannot select 0 ({180,360} + floor). This exhibit time-scales the
same EVEYES Scene_005 camera-only RGB until visual motion exceeds the EVEYES
inverse threshold, then the 3-gear law inverse_rpm(..., floor=0) turns the
prism off. Not Table 1 Ours. Not the 64×64 checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.events import ThresholdEventSimulator
from ami_simulator.inverse_motion import inverse_rpm
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.camera_nfb import pair_visual_motion, sample_path
from scripts.evaluate_eveyes_camera_speed_nfb import DT
from scripts.evaluate_eveyes_timescale_sweep import PRISM_DEG, READY, _rollout, _window
from scripts.evaluate_eveyes_v2_downstream import _intrinsics, _scene_path, _structure_metrics
from scripts.render_eveyes_camera_nfb_event_video import _events_to_map

STUDENT_THRESHOLDS = Path("results/eveyes_v2_plus_dsd_f1_argmax/policy.pt")
RATE = 4.0
SNAPSHOT_PATH = 64.0
FIXED = "#1B4F8A"
GEAR0 = "#15803d"


def _three_gear_rollout(luminance, prism, *, rate: float, device: str, thresholds: tuple[float, float], path_end: float, snapshot_path: float | None = None):
    path_s = 0.0
    last = len(luminance) - 1
    phase = 0.0
    previous_active = None
    sensor_reference = None
    previous_rpm = 180.0
    previous_occupancy = 0.0
    lagged_visual = 0.0
    dvs_simulator = ThresholdEventSimulator(0.2)
    dvs_reference = None
    rows = []
    snap = None
    window = 0
    while path_s < min(path_end, last - 1e-6):
        rpm = inverse_rpm(lagged_visual, thresholds, previous_density=previous_occupancy, previous_overlap=0.0, floor_rpm=0.0)
        if rpm <= 0.0:
            used = float(min(rate, last - path_s))
            lum_a = sample_path(luminance, path_s)
            lum_b = sample_path(luminance, path_s + used)
            pair = np.stack((lum_a, lum_b), axis=0)
            times = np.asarray((window * DT, (window + 1) * DT), dtype=np.float64)
            events, dvs_reference = dvs_simulator.simulate_with_state(pair, times, reference_log=dvs_reference)
            metrics, active, _ = _structure_metrics(events, lum_a, np.ones_like(lum_a, dtype=bool), previous_active)
            step = {
                "metrics": metrics,
                "result": {"reference_events": events, "phase_end": phase},
                "active": active,
                "visual": pair_visual_motion(lum_a, lum_b, DT),
                "used": used,
            }
        else:
            step = _window(
                luminance,
                path_s,
                rate,
                now=window * DT,
                prism=prism,
                rpm=rpm,
                phase=phase,
                previous_active=previous_active,
                previous_rpm=previous_rpm,
                sensor_reference=sensor_reference,
                device=device,
            )
        metrics = step["metrics"]
        row = {
            "window": window,
            "path_s": float(path_s),
            "t": window * DT,
            "rate": float(rate),
            "rpm": float(rpm),
            "visual_motion": float(step["visual"]),
            "occupancy": float(metrics["unique_pixel_ratio"]),
            "f1": float(metrics["edge_f1"]),
            "precision": float(metrics["edge_precision"]),
            "recall": float(metrics["edge_recall"]),
        }
        rows.append(row)
        if snapshot_path is not None and snap is None and path_s + step["used"] >= snapshot_path:
            if rpm > 1.0:
                raise SystemExit(f"snapshot path {path_s:.1f} is {rpm:.0f} RPM, need 0")
            snap = {**row, "events": step["result"]["reference_events"], "used": float(step["used"])}
            break
        path_s = min(last, path_s + step["used"])
        phase = float(step["result"]["phase_end"])
        previous_active = np.asarray(step["active"])
        if rpm > 0.0:
            sensor_reference = np.asarray(step["result"]["sensor_reference_log"])
        previous_rpm = float(rpm)
        previous_occupancy = float(metrics["unique_pixel_ratio"])
        lagged_visual = float(step["visual"])
        window += 1
    return rows, snap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/figures/fig_eveyes_fixed180_vs_gear0_t"))
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    checkpoint = torch.load(STUDENT_THRESHOLDS, map_location="cpu", weights_only=True)
    thresholds = (float(checkpoint["motion_thresholds"][0]), float(checkpoint["motion_thresholds"][1]))
    scene = _scene_path(READY, "test", "Scene_005", "camera_only")
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=240)
    luminance = rgb_to_linear_luminance(frames[: len(timestamps)])
    prism = RotatingWedgePrism(_intrinsics(scene), PRISM_DEG)

    print(f"3-gear inverse  {RATE:g}×  thresholds={thresholds}  {args.device}", flush=True)
    gear_rows, gear_snap = _three_gear_rollout(
        luminance,
        prism,
        rate=RATE,
        device=args.device,
        thresholds=thresholds,
        path_end=SNAPSHOT_PATH + RATE,
        snapshot_path=SNAPSHOT_PATH,
    )
    print(f"native map  {RATE:g}×  Fixed 180  path {SNAPSHOT_PATH:.0f}", flush=True)
    _, fixed_snap = _rollout(
        luminance,
        prism,
        rate=RATE,
        rpm=180.0,
        device=args.device,
        path_end=SNAPSHOT_PATH + RATE,
        snapshot_path=SNAPSHOT_PATH,
        start_path=0.0,
    )
    if gear_snap is None or fixed_snap is None:
        raise SystemExit("missing snapshot")

    height, width = int(prism.intrinsics.height), int(prism.intrinsics.width)
    maps = {
        "180": _events_to_map(fixed_snap["events"], height, width, storm=False),
        "0": _events_to_map(gear_snap["events"], height, width, storm=False),
    }
    frame_id = int(round(float(gear_snap["path_s"])))
    rgb = np.asarray(Image.open(scene / "images" / f"{frame_id:06d}.png").convert("RGB"))

    fig = plt.figure(figsize=(8.8, 6.4), dpi=220)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.45, 1.05, 0.95], hspace=0.34, wspace=0.10, left=0.07, right=0.98, top=0.90, bottom=0.08)
    t_hero = float(gear_snap["t"])
    panels = (
        ("RGB", rgb, "#111827", f"4× playback  ·  path {gear_snap['path_s']:.0f}", None),
        ("Fixed 180 RPM", maps["180"], FIXED, f"occ {fixed_snap['occupancy']:.2f}  F1 {fixed_snap['f1']:.2f}", "180 RPM"),
        ("3-gear", maps["0"], GEAR0, f"occ {gear_snap['occupancy']:.2f}  F1 {gear_snap['f1']:.2f}", "0 RPM"),
    )
    for col, (title, image, color, xlabel, badge) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(image)
        ax.set_title(title, fontsize=10, color=color, pad=4, fontweight="semibold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(xlabel, fontsize=8, color=color)
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.5)
        if badge:
            ax.text(0.03, 0.97, badge, transform=ax.transAxes, va="top", fontsize=8, color="white", bbox={"boxstyle": "round,pad=0.22", "facecolor": color, "edgecolor": "none"})

    ax_o = fig.add_subplot(gs[1, :])
    ax_r = fig.add_subplot(gs[2, :], sharex=ax_o)
    t = np.asarray([row["t"] for row in gear_rows])
    ax_o.plot([t_hero], [fixed_snap["occupancy"]], color=FIXED, marker="s", ms=7, label="Fixed 180 at t*")
    ax_o.plot(t, [row["occupancy"] for row in gear_rows], color=GEAR0, marker="o", lw=1.8, label="3-gear occupancy")
    ax_o.axvline(t_hero, color="#111827", lw=0.9, ls="--")
    ax_o.set_ylabel("Event occupancy")
    ax_o.set_ylim(0.2, 0.85)
    ax_o.legend(frameon=False, loc="upper right", fontsize=8)
    ax_o.grid(True, alpha=0.25, lw=0.5)
    ax_o.spines["top"].set_visible(False)
    ax_o.spines["right"].set_visible(False)
    ax_o.set_title("Constructed 4× path rate  ·  inverse 3-gear {360,180,0}, no 180 floor  ·  not Table 1 Ours", fontsize=8.5, loc="left")

    ax_r.step(t, [row["rpm"] for row in gear_rows], where="post", color=GEAR0, lw=2.0, label="3-gear RPM")
    ax_r.axhline(180.0, color=FIXED, ls="--", lw=1.0)
    ax_r.axhline(0.0, color="#6b7280", ls=":", lw=1.0)
    ax_r.axvline(t_hero, color="#111827", lw=0.9, ls="--")
    ax_r.scatter([t_hero], [0.0], color=GEAR0, s=32, zorder=5)
    ax_r.set_ylim(-30, 390)
    ax_r.set_yticks([0, 180, 360])
    ax_r.set_ylabel("RPM")
    ax_r.set_xlabel("Window time (s)  at 4× playback")
    ax_r.grid(True, alpha=0.25, lw=0.5)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    ax_r.legend(frameon=False, loc="upper right", fontsize=8)

    fig.suptitle("EVEYES 960×540 camera-only  ·  construct speed by time  ·  Fixed 180 vs 3-gear at 0", fontsize=11, y=0.985)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"))
    fig.savefig(args.output.with_suffix(".png"))
    plt.close(fig)

    stats = {
        "dataset": "EVEYES-co-motion-v2",
        "scene": "test/Scene_005/camera_only",
        "not_table1_ours": True,
        "not_the_64x64_procedural_pack": True,
        "controller": "inverse_rpm 3-gear {360,180,0} floor=0",
        "motion_thresholds": list(thresholds),
        "rate": RATE,
        "hero_path_s": float(gear_snap["path_s"]),
        "hero_t": t_hero,
        "gear_rpm": float(gear_snap["rpm"]),
        "occ_180": float(fixed_snap["occupancy"]),
        "occ_0": float(gear_snap["occupancy"]),
        "f1_180": float(fixed_snap["f1"]),
        "f1_0": float(gear_snap["f1"]),
        "visual_motion": float(gear_snap["visual_motion"]),
        "n_windows": len(gear_rows),
        "frac_0": float(np.mean([row["rpm"] < 1.0 for row in gear_rows])),
    }
    args.output.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(args.output) + ".png", **stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
