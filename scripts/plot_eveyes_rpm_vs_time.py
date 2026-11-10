#!/usr/bin/env python3
"""Paper figure: Ours RPM versus time, not a single mean gear.

Saves a three-regime storyboard plus RPM(t). Do not plot the raw 16 ms
square wave in the main paper — 0.5 s bins show the condition change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts.evaluate_phase_continuous_control import GEAR_RPM


READY = Path("data/eveyes_comotion_v2_ready/test/Scene_005")
ORACLE = Path("results/eveyes_v2_regime_f1_oracles")
GEAR_180 = int(np.argmin(np.abs(GEAR_RPM - 180.0)))
# Per-regime stills: do not reuse 1.5/4.0/7.5 on every row.
# object_only at 4.0 s is back at the start pose; camera_only at 1.5 s has already lost the car.
REGIME_THUMBS = {
    "object_only": (0.0, 2.0, 6.0),
    "camera_only": (0.0, 0.5, 7.5),
    "co_motion": (0.0, 2.5, 7.5),
}


def _ours_rpm(rows: list[dict[str, object]], scene: str) -> np.ndarray:
    selected = [row for row in rows if row["scene"] == scene and row["method"] == "ours"]
    selected.sort(key=lambda row: int(row["window"]))
    return np.asarray([float(row["rpm"]) for row in selected], dtype=np.float64)


def _bin(values: np.ndarray, width: int = 30) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    means = []
    for start in range(0, len(values), width):
        chunk = values[start : start + width]
        centers.append((start + 0.5 * len(chunk)) / 60.0)
        means.append(float(chunk.mean()))
    return np.asarray(centers), np.asarray(means)


def _rgb(regime: str, t_seconds: float) -> np.ndarray:
    frame = min(int(round(t_seconds * 60.0)), 479)
    path = READY / regime / "images" / f"{frame:06d}.png"
    return np.asarray(Image.open(path).convert("RGB"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-json", type=Path, default=Path("results/ami_next/eveyes_v2_regimes_test_ours_m0.json"))
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/figures/eveyes_rpm_vs_time"))
    args = parser.parse_args()
    payload = json.loads(args.ours_json.read_text(encoding="utf-8"))
    regimes = (
        ("object_only", "object only (camera frozen)"),
        ("camera_only", "camera only (parked neighbor)"),
        ("co_motion", "co-motion (locked relative pose)"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.2, 7.4), dpi=160)
    outer = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.42)
    for row, (regime, title) in enumerate(regimes):
        times = REGIME_THUMBS[regime]
        block = outer[row].subgridspec(1, 2, width_ratios=[1.15, 1.6], wspace=0.16)
        thumb_gs = block[0, 0].subgridspec(1, len(times), wspace=0.06)
        axes = fig.add_subplot(block[0, 1])
        for index, t in enumerate(times):
            thumb = fig.add_subplot(thumb_gs[0, index])
            image = _rgb(regime, t)
            thumb.imshow(image)
            thumb.set_xticks([])
            thumb.set_yticks([])
            thumb.set_aspect("equal")
            for spine in thumb.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.45)
                spine.set_color("0.35")
            thumb.set_xlabel(f"{t:.1f} s", fontsize=7, labelpad=2)
            if index == 0:
                thumb.set_title(title, loc="left", fontsize=8, pad=3)

        scene = f"test/Scene_005/{regime}"
        rpm = _ours_rpm(payload["rows"], scene)
        visual = np.load(ORACLE / f"test_Scene_005_{regime}.npz")["visual_motion"]
        t = np.arange(len(rpm)) / 60.0
        t_bin, rpm_bin = _bin(rpm)
        _, vis_bin = _bin(visual[: len(rpm)])
        axes.plot(t_bin, rpm_bin, color="#1B4F8A", lw=2.0, label="Ours RPM (0.5 s)")
        axes.set_ylim(150, 390)
        axes.set_xlim(0, 8)
        axes.set_ylabel("RPM")
        axes.set_yticks([180, 270, 360])
        axes.axhline(180, color="0.6", lw=0.6, ls=":")
        axes.axhline(360, color="0.6", lw=0.6, ls=":")
        vis_ax = axes.twinx()
        vis_ax.plot(t_bin, vis_bin, color="#b45309", lw=1.2, ls="--", label="visual motion")
        vis_ax.set_ylabel("visual")
        vis_ax.set_ylim(-2, 90)
        if row == 2:
            axes.set_xlabel("time (s)")
        if row == 0:
            lines, labels = axes.get_legend_handles_labels()
            lines2, labels2 = vis_ax.get_legend_handles_labels()
            axes.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=7, frameon=False)

    fig.suptitle("Causal prism speed follows visual motion; it is not a fixed gear", fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.96, bottom=0.06)
    pdf = args.output.with_suffix(".pdf")
    png = args.output.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(json.dumps({"pdf": str(pdf), "png": str(png)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
