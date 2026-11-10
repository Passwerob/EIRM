#!/usr/bin/env python3
"""Paper qualitative RESULTS figure (not the method pipeline schematic).

Stacks the coverage exhibits the paper needs:
  (a) too little / too much motion
  (b) Scene_005 · four motion forms · DVS / v2e / Ours
  (c) Scene_002/004/005 · object_only + co-motion
  (d) DL3DV in-the-wild · DVS / v2e / Ours
  (e) DL3DV · identical CM-IWE
  (f) accumulated IWE recon observation

Uses already-frozen PNGs. Does not overwrite Table 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PANELS = (
    (
        "a",
        "Relative-motion extremes",
        Path("results/eveyes_scene001/figures/scene001_too_little_too_much.png"),
    ),
    (
        "b",
        "Scene_005 · four motion forms · DVS-Voltmeter / v2e / Ours",
        Path("results/ami_next/figures/paper_motion_regimes_events.png"),
    ),
    (
        "c",
        "Scene_002 / 004 / 005 · object_only + co-motion",
        Path("results/ami_next/figures/paper_multiscene_events.png"),
    ),
    (
        "d",
        "DL3DV in-the-wild · DVS-Voltmeter / v2e / Ours (camera motion)",
        Path("results/ami_next/figures/paper_dl3dv_events.png"),
    ),
    (
        "e",
        "DL3DV · identical CM-IWE velocity grid",
        Path("results/ami_next/figures/paper_dl3dv_cm.png"),
    ),
    (
        "f",
        "Accumulated IWE · observation for 3D reconstruction (not a mesh)",
        Path("results/eveyes_scene001/figures/scene001_recon_iwe.png"),
    ),
)
INK = "#2C3E50"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/figures/paper_qualitative_results"))
    args = parser.parse_args()
    _style()

    images = [_load(path) for _, _, path in PANELS]
    widths = [im.shape[1] for im in images]
    target_w = max(widths)
    # scale each panel to common width, preserve aspect
    scaled = []
    for im in images:
        h, w = im.shape[:2]
        new_h = int(round(h * target_w / w))
        scaled.append(np.asarray(Image.fromarray(im).resize((target_w, new_h), Image.Resampling.LANCZOS)))

    # figure height in inches from total pixel aspect at 7.2" wide
    total_h = sum(im.shape[0] for im in scaled)
    # room for labels between panels
    label_band = 0.28
    fig_w = 7.2
    fig_h = fig_w * (total_h / target_w) + label_band * len(PANELS) + 0.25
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor("white")

    # fractional heights
    pix_heights = [im.shape[0] for im in scaled]
    # convert to figure fractions with label bands
    usable = 0.97 - 0.03 - label_band * len(PANELS) / fig_h
    y = 0.97
    fig.text(
        0.5,
        0.985,
        "Qualitative results: Blender motion regimes + DL3DV in-the-wild transfer",
        ha="center",
        va="top",
        fontsize=9.5,
        color=INK,
        fontweight="bold",
    )
    y = 0.965
    for (tag, title, _), im, ph in zip(PANELS, scaled, pix_heights):
        band = label_band / fig_h
        y -= band
        fig.text(0.015, y + band * 0.35, f"({tag})", fontsize=10, fontweight="bold", color=INK, va="center")
        fig.text(0.055, y + band * 0.35, title, fontsize=8.3, color=INK, va="center")
        panel_h = usable * (ph / total_h)
        y -= panel_h
        ax = fig.add_axes([0.02, y, 0.96, panel_h * 0.98])
        ax.imshow(im)
        ax.axis("off")
        y -= 0.008

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdf = args.output.with_suffix(".pdf")
    png = args.output.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=220)
    plt.close(fig)

    caption = args.output.with_name(args.output.name + "_caption.txt")
    caption.write_text(
        """Figure. Qualitative results of AMI active optical excitation.

(a) Relative-motion extremes on Scene_001. Too little: sharp RGB, sparse DVS
events on the moving car only. Too much: RGB blur and a dense event storm.

(b) Held-out Scene_005 · four motion forms. object_only (camera frozen),
camera_only (ego sweep), co_motion (relative lock), drive_stop_drive (stop
phase). Same RGB window → official DVS-Voltmeter, official v2e, Ours (180 RPM,
inverse-warped). Standalone:
  results/ami_next/figures/paper_motion_regimes_events.{png,pdf}

(c) Multi-scene Blender coverage. Scene_002 (train), Scene_004 (valid),
Scene_005 (test) under object_only and co_motion. Standalone:
  results/ami_next/figures/paper_multiscene_events.{png,pdf}

(d) DL3DV in-the-wild transfer. Five real RGB scenes (alley, field, yard,
temple, statues). Camera motion in the wild — not Blender co-motion — so
passive streams are not empty; the claim is active excitation + inverse warp
under real appearance. Standalone:
  results/ami_next/figures/paper_dl3dv_events.{png,pdf}

(e) DL3DV through identical CM-IWE velocity grid. CM is a representation, not
a fourth generator. Companion:
  results/ami_next/figures/paper_dl3dv_cm.{png,pdf}
  results/ami_next/figures/paper_motion_regimes_cm.{png,pdf}
  results/ami_next/figures/paper_multiscene_cm.{png,pdf}

(f) Accumulated grayscale IWE (the observation a 3D reconstructor would use,
not a mesh).

Companion control exhibits (separate figures; not Table 1 numbers):
  Gear selection {0,180,360}:
    results/comotion_f1_argmax_0_180_360/figures/three_gear_by_regime.{png,pdf}
  Optional motion NFB / storm vs Ours:
    results/eveyes_motion_nfb/figures/camera_nfb_storm_vs_ours_scenes.{png,pdf}

Reproduce panels:
  MPLCONFIGDIR=/tmp/mpl-fcr PYTHONPATH=. \\
    python scripts/plot_paper_dvs_v2e_cm_coverage.py --device cuda:6
  MPLCONFIGDIR=/tmp/mpl-fcr PYTHONPATH=. \\
    python scripts/plot_paper_dl3dv_dvs_v2e_cm.py --device cuda:6
  MPLCONFIGDIR=/tmp/mpl-fcr PYTHONPATH=. python scripts/plot_paper_qualitative_results.py
""",
        encoding="utf-8",
    )
    print({"pdf": str(pdf), "png": str(png), "caption": str(caption)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
