#!/usr/bin/env python3
"""EVEYES drive-stop-drive: fixed-360 occupancy storm vs paper Ours.

Uses data/eveyes_comotion_v2_ready and the plus_dsd checkpoint traces.
Not the 64x64 procedural pack. Paper Ours is {180,360} with a 180 floor,
so it cannot suppress a fixed-180 storm; the real mismatch is fixed 360.
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
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import EventStream
from scripts.evaluate_dl3dv_downstream import _step
from scripts.evaluate_eveyes_v2_downstream import _intrinsics, _scene_path
from scripts.plot_eveyes_drive_stop_drive import FREEZE
from scripts.render_eveyes_camera_nfb_event_video import _concat, _events_to_map

READY = Path("data/eveyes_comotion_v2_ready")
DSD_JSON = Path("results/ami_next/eveyes_v2_drive_stop_drive_ours_plus_dsd.json")
SCENE = "test/Scene_005/drive_stop_drive"
HERO_T = 6.6
WARMUP = 12
ACCUM = 3
BIN = 6  # 0.1 s at 60 Hz


def _rows(payload: dict, method: str) -> list[dict]:
    selected = [row for row in payload["rows"] if row["scene"] == SCENE and row["method"] == method]
    selected.sort(key=lambda row: int(row["window"]))
    return selected


def _bin(values: np.ndarray, width: int = BIN) -> tuple[np.ndarray, np.ndarray]:
    centers, means = [], []
    for start in range(0, len(values), width):
        chunk = values[start : start + width]
        centers.append((start + 0.5 * len(chunk)) / 60.0)
        means.append(float(chunk.mean()))
    return np.asarray(centers), np.asarray(means)


def _replay(luminance, timestamps, prism, rpms: np.ndarray, device: str) -> list[EventStream]:
    phase = 0.0
    previous_active = None
    sensor_reference = None
    previous_rpm = float(rpms[0])
    ones = np.ones((len(timestamps), luminance.shape[1], luminance.shape[2]), dtype=bool)
    streams = []
    for index, rpm in enumerate(rpms):
        result = _step(
            luminance,
            timestamps,
            ones,
            prism,
            index,
            float(rpm),
            phase,
            previous_active,
            previous_rpm,
            sensor_reference,
            device=device,
        )
        streams.append(result["reference_events"])
        phase = float(result["phase_end"])
        previous_active = np.asarray(result["active"])
        sensor_reference = np.asarray(result["sensor_reference_log"])
        previous_rpm = float(rpm)
    return streams


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/figures/fig_eveyes_drive_stop_drive_storm"))
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    payload = json.loads(DSD_JSON.read_text(encoding="utf-8"))
    if payload.get("dataset") != "EVEYES-co-motion-v2":
        raise SystemExit("expected EVEYES DSD json")
    ours_rows = _rows(payload, "ours")
    rows_180 = _rows(payload, "ami_fixed_180")
    rows_360 = _rows(payload, "ami_fixed_360")
    rows_0 = _rows(payload, "ami_passive")
    t = np.asarray([float(row["t0"]) for row in ours_rows], dtype=np.float64)
    occ = {
        "0": np.asarray([float(row["unique_pixel_ratio"]) for row in rows_0], dtype=np.float64),
        "180": np.asarray([float(row["unique_pixel_ratio"]) for row in rows_180], dtype=np.float64),
        "360": np.asarray([float(row["unique_pixel_ratio"]) for row in rows_360], dtype=np.float64),
        "ours": np.asarray([float(row["unique_pixel_ratio"]) for row in ours_rows], dtype=np.float64),
    }
    rpm_ours = np.asarray([float(row["rpm"]) for row in ours_rows], dtype=np.float64)
    hero = int(np.argmin(np.abs(t - HERO_T)))

    scene = _scene_path(READY, "test", "Scene_005", "drive_stop_drive")
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=480)
    timestamps = timestamps[: len(frames)]
    luminance = rgb_to_linear_luminance(frames)
    prism = RotatingWedgePrism(_intrinsics(scene), float(payload["prism_angle_deg"]))
    start = max(0, hero - WARMUP)
    end = min(len(timestamps) - 1, hero + 1)
    lum = luminance[start : end + 1]
    times = timestamps[start : end + 1]
    local = hero - start
    print(f"simulate windows {start}:{end} hero={hero} t={t[hero]:.3f} device={args.device}", flush=True)
    streams = {
        "360": _replay(lum, times, prism, np.full(end - start, 360.0), args.device),
        "180": _replay(lum, times, prism, np.full(end - start, 180.0), args.device),
        "ours": _replay(lum, times, prism, rpm_ours[start:end], args.device),
    }

    def _map(name: str) -> np.ndarray:
        first = max(0, local - (ACCUM - 1))
        return _events_to_map(_concat(list(streams[name][first : local + 1])), lum.shape[1], lum.shape[2], storm=False)

    rgb = np.asarray(Image.open(scene / "images" / f"{hero:06d}.png").convert("RGB"))
    maps = {"Fixed 360 RPM": _map("360"), "Ours": _map("ours"), "Fixed 180 RPM": _map("180")}

    fig = plt.figure(figsize=(8.8, 6.2), dpi=220)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.45, 1.05, 0.95], hspace=0.32, wspace=0.10, left=0.075, right=0.91, top=0.90, bottom=0.08)
    specs = (
        ("Fixed 360 RPM", maps["Fixed 360 RPM"], "#b45309", f"occupancy {occ['360'][hero]:.2f}", "over-excitation"),
        ("Ours", maps["Ours"], "#15803d", f"occupancy {occ['ours'][hero]:.2f}  ·  {rpm_ours[hero]:.0f} RPM", "holds 180"),
        ("Fixed 180 RPM", maps["Fixed 180 RPM"], "#1B4F8A", f"occupancy {occ['180'][hero]:.2f}", "same floor"),
    )
    for col, (title, image, color, xlabel, badge) in enumerate(specs):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(image)
        ax.set_title(title, fontsize=10, color=color, pad=4, fontweight="semibold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(xlabel, fontsize=8, color=color)
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.5)
        ax.text(0.03, 0.97, badge, transform=ax.transAxes, va="top", fontsize=7.5, color="white", bbox={"boxstyle": "round,pad=0.25", "facecolor": color, "edgecolor": "none"})
        if col == 1:
            inset = ax.inset_axes([0.70, 0.70, 0.28, 0.28])
            inset.imshow(rgb)
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title("RGB", fontsize=6, pad=1, color="white")
            for spine in inset.spines.values():
                spine.set_color("white")

    ax_o = fig.add_subplot(gs[1, :])
    ax_r = fig.add_subplot(gs[2, :], sharex=ax_o)
    for ax in (ax_o, ax_r):
        ax.axvspan(FREEZE[0], FREEZE[1], color="#D7E2EE", lw=0, zorder=0)
        ax.axvline(FREEZE[0], color="#5D7390", lw=0.7, ls="--")
        ax.axvline(FREEZE[1], color="#5D7390", lw=0.7, ls="--")
        ax.axvline(t[hero], color="#111827", lw=0.8, ls="--")
    ax_o.text(np.mean(FREEZE), ax_o.get_ylim()[1] if False else 0.92, "camera frozen", ha="center", fontsize=8, color="#5D7390")
    for key, label, color, lw in (
        ("0", "Fixed 0", "#6b7280", 1.2),
        ("180", "Fixed 180", "#1B4F8A", 1.6),
        ("360", "Fixed 360", "#b45309", 1.6),
        ("ours", "Ours", "#15803d", 1.8),
    ):
        tb, yb = _bin(occ[key])
        ax_o.plot(tb, yb, color=color, lw=lw, label=label)
    ax_o.set_ylabel("Event occupancy")
    ax_o.set_ylim(0, 0.95)
    ax_o.set_xlim(0, 8)
    ax_o.legend(frameon=False, ncol=4, loc="upper left", fontsize=8)
    ax_o.grid(True, alpha=0.25, lw=0.5)
    ax_o.spines["top"].set_visible(False)
    ax_o.spines["right"].set_visible(False)
    ax_o.set_title("EVEYES Scene_005 drive–stop–drive  ·  occupancy from the paper eval json", fontsize=9, loc="left")

    tb, rpm_bin = _bin(rpm_ours)
    ax_r.plot(tb, rpm_bin, color="#15803d", lw=2.0, label="Ours RPM")
    ax_r.axhline(180.0, color="#1B4F8A", ls="--", lw=1.0)
    ax_r.axhline(360.0, color="#b45309", ls=":", lw=1.0)
    ax_r.set_ylim(150, 390)
    ax_r.set_yticks([180, 360])
    ax_r.set_ylabel("RPM")
    ax_r.set_xlabel("Time (s)")
    ax_r.grid(True, alpha=0.25, lw=0.5)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    ax_r.legend(frameon=False, loc="upper left", fontsize=8)

    fig.suptitle("EVEYES 960×540  ·  fixed 360 overfills occupancy; Ours stays on the 180 floor", fontsize=11, y=0.985)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"))
    fig.savefig(args.output.with_suffix(".png"))
    plt.close(fig)
    stats = {
        "dataset": "EVEYES-co-motion-v2",
        "scene": SCENE,
        "json": str(DSD_JSON),
        "prism_angle_deg": payload["prism_angle_deg"],
        "student": payload.get("student"),
        "hero_t": float(t[hero]),
        "occ_0": float(occ["0"][hero]),
        "occ_180": float(occ["180"][hero]),
        "occ_360": float(occ["360"][hero]),
        "occ_ours": float(occ["ours"][hero]),
        "ours_rpm": float(rpm_ours[hero]),
        "not_the_64x64_procedural_pack": True,
    }
    args.output.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(args.output) + ".png", **stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
