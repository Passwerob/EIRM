#!/usr/bin/env python3
"""Single-take drive_stop_drive RPM(t). Caption as one Blender shot, not a splice.

Paper layout: five stills, then 0.5 s Ours RPM, then a shared-x visual track.
Do not draw the raw 16 ms gear square wave.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

READY = Path("data/eveyes_comotion_v2_ready/test/Scene_005/drive_stop_drive")
FREEZE = (2.5, 5.5)
THUMB_TIMES = (0.0, 2.5, 3.25, 4.75, 8.0)
THUMB_LABELS = ("0.0 s", "2.5 s", "3.25 s", "4.75 s", "8.0 s")
RPM_COLOR = "#1B4F8A"
VISUAL_COLOR = "#9A5A2A"
FREEZE_FILL = "#D7E2EE"
DRIVE_EDGE = "#2C3E50"
FREEZE_EDGE = "#5D7390"


def _ours_rpm(rows: list[dict[str, object]], scene: str) -> np.ndarray:
    selected = [row for row in rows if row["scene"] == scene and row["method"] == "ours"]
    selected.sort(key=lambda row: int(row["window"]))
    return np.asarray([float(row["rpm"]) for row in selected], dtype=np.float64)


def _timestamps(path: Path) -> np.ndarray:
    rows = np.genfromtxt(path, delimiter=",", names=True)
    if "timestamp_seconds" in rows.dtype.names:
        return np.asarray(rows["timestamp_seconds"], dtype=np.float64)
    return np.asarray(rows["timestamp_s"], dtype=np.float64)


def _luminance(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB").resize((320, 180), Image.Resampling.BOX), dtype=np.float32) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return np.clip(0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2], 0.0, 1.0)


def _visual_motion(image_dir: Path, timestamps: np.ndarray, n_frames: int) -> np.ndarray:
    log_prev = np.log(np.clip(_luminance(image_dir / "000000.png"), 1e-6, None))
    visual = np.empty(n_frames - 1, dtype=np.float64)
    for index in range(1, n_frames):
        log_now = np.log(np.clip(_luminance(image_dir / f"{index:06d}.png"), 1e-6, None))
        visual[index - 1] = float(np.quantile(np.abs(log_now - log_prev), 0.9) / (timestamps[index] - timestamps[index - 1]))
        log_prev = log_now
    return visual


def _bin(values: np.ndarray, width: int = 30) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    means = []
    for start in range(0, len(values), width):
        chunk = values[start : start + width]
        centers.append((start + 0.5 * len(chunk)) / 60.0)
        means.append(float(chunk.mean()))
    return np.asarray(centers), np.asarray(means)


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "axes.linewidth": 0.55,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _freeze_band(axes) -> None:
    axes.axvspan(FREEZE[0], FREEZE[1], color=FREEZE_FILL, lw=0, zorder=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-json", type=Path, default=Path("results/ami_next/eveyes_v2_drive_stop_drive_ours_m0.json"))
    parser.add_argument("--dataset-root", type=Path, default=READY)
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/figures/eveyes_drive_stop_drive_rpm"))
    args = parser.parse_args()
    _apply_style()
    payload = json.loads(args.ours_json.read_text(encoding="utf-8"))
    rpm = _ours_rpm(payload["rows"], "test/Scene_005/drive_stop_drive")
    timestamps = _timestamps(args.dataset_root / "timestamps.csv")
    visual = _visual_motion(args.dataset_root / "images", timestamps, min(len(timestamps), 480))[: len(rpm)]
    t_bin, rpm_bin = _bin(rpm)
    _, vis_bin = _bin(visual)

    fig = plt.figure(figsize=(7.0, 3.15), dpi=300)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.12, 1.95], hspace=0.28, left=0.09, right=0.985, top=0.93, bottom=0.13)
    thumbs_gs = outer[0].subgridspec(1, 5, wspace=0.07)
    plot_gs = outer[1].subgridspec(2, 1, height_ratios=[2.6, 0.78], hspace=0.03)

    for index, t in enumerate(THUMB_TIMES):
        axis = fig.add_subplot(thumbs_gs[0, index])
        frame = min(int(round(t * 60.0)), 479)
        image = np.asarray(Image.open(args.dataset_root / "images" / f"{frame:06d}.png").convert("RGB"))
        axis.imshow(image)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
        frozen = FREEZE[0] <= t <= FREEZE[1]
        edge = FREEZE_EDGE if frozen else DRIVE_EDGE
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color(edge)
            spine.set_linewidth(1.15 if frozen else 0.45)
        axis.set_xlabel(THUMB_LABELS[index], fontsize=7.0, labelpad=2.5, color=FREEZE_EDGE if frozen else "0.25")
        if index == 0:
            axis.set_title("drive", loc="left", fontsize=7.5, color="0.25", pad=3)
        if index == 2:
            axis.set_title("camera frozen", loc="center", fontsize=7.5, color=FREEZE_EDGE, pad=3)
        if index == 4:
            axis.set_title("drive", loc="right", fontsize=7.5, color="0.25", pad=3)

    rpm_ax = fig.add_subplot(plot_gs[0, 0])
    vis_ax = fig.add_subplot(plot_gs[1, 0], sharex=rpm_ax)
    _freeze_band(rpm_ax)
    _freeze_band(vis_ax)
    rpm_ax.plot(t_bin, rpm_bin, color=RPM_COLOR, lw=2.2, solid_capstyle="round", zorder=3, label="Ours (0.5 s)")
    rpm_ax.axhline(180.0, color="0.78", lw=0.45, zorder=2)
    rpm_ax.axhline(360.0, color="0.78", lw=0.45, zorder=2)
    rpm_ax.set_ylim(155, 385)
    rpm_ax.set_yticks([180, 360])
    rpm_ax.set_ylabel("Prism RPM")
    rpm_ax.legend(loc="upper left", frameon=False, fontsize=7.5, handlelength=1.6, borderaxespad=0.15)
    rpm_ax.text(4.0, 205, "180 RPM hold", ha="center", va="bottom", fontsize=7.5, color="#3E4F61", zorder=4)
    rpm_ax.spines["top"].set_visible(False)
    rpm_ax.tick_params(labelbottom=False)
    rpm_ax.axvline(FREEZE[0], color="#9AAFC3", lw=0.5, zorder=2)
    rpm_ax.axvline(FREEZE[1], color="#9AAFC3", lw=0.5, zorder=2)

    vis_ax.fill_between(t_bin, 0.0, vis_bin, color=VISUAL_COLOR, alpha=0.22, linewidth=0, zorder=1)
    vis_ax.plot(t_bin, vis_bin, color=VISUAL_COLOR, lw=1.1, zorder=3)
    vis_ax.set_ylim(0, 78)
    vis_ax.set_yticks([0, 40])
    vis_ax.set_ylabel("Visual")
    vis_ax.set_xlabel("Time (s)")
    vis_ax.set_xlim(0.0, 8.0)
    vis_ax.set_xticks([0.0, 2.5, 5.5, 8.0])
    vis_ax.spines["top"].set_visible(False)
    vis_ax.axvline(FREEZE[0], color="#9AAFC3", lw=0.5, zorder=2)
    vis_ax.axvline(FREEZE[1], color="#9AAFC3", lw=0.5, zorder=2)
    for axis in (rpm_ax, vis_ax):
        axis.spines["right"].set_visible(False)
        axis.margins(x=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdf = args.output.with_suffix(".pdf")
    png = args.output.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.close(fig)
    freeze = rpm[150:330] if len(rpm) > 330 else rpm
    drive = np.concatenate([rpm[:150], rpm[330:]]) if len(rpm) > 330 else rpm
    print(
        json.dumps(
            {
                "pdf": str(pdf),
                "png": str(png),
                "windows": int(len(rpm)),
                "mean_rpm": float(rpm.mean()) if len(rpm) else None,
                "frac_360": float(np.mean(rpm >= 359.0)) if len(rpm) else None,
                "freeze_mean_rpm": float(freeze.mean()) if len(freeze) else None,
                "freeze_frac_360": float(np.mean(freeze >= 359.0)) if len(freeze) else None,
                "drive_mean_rpm": float(drive.mean()) if len(drive) else None,
                "drive_frac_360": float(np.mean(drive >= 359.0)) if len(drive) else None,
                "visual_freeze_mean": float(visual[150:330].mean()) if len(visual) > 330 else None,
                "visual_drive_mean": float(np.concatenate([visual[:150], visual[330:]]).mean()) if len(visual) > 330 else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
