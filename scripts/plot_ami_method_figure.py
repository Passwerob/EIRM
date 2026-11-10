#!/usr/bin/env python3
"""Paper method figure for AMI: rotating wedge, inverse warp, 0/180/360 gears.

Layout follows reports/ami_next/m0_controller_pipeline_draw.txt.
English labels only. Vector PDF with TrueType text. Does not overwrite Table 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Wedge
from PIL import Image

NAVY = "#1B4F8A"
NAVY_FILL = "#E8EEF5"
INK = "#2C3E50"
MUTED = "#5D7390"
WARM = "#9A5A2A"
WARM_FILL = "#F4EEE6"
SENSOR = "#B45309"
REF = "#1B4F8A"
NFB = "#6B7280"
TEACH = "#7A6A4F"
LINE = 0.7
RGB_PATH = Path("data/eveyes_comotion_v2_ready/test/Scene_005/co_motion/images/000060.png")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _rgb(path: Path, width: int = 280) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    height = int(round(image.height * width / image.width))
    return np.asarray(image.resize((width, height), Image.Resampling.BOX))


def _edge_xy(rgb: np.ndarray, n: int = 90, seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    gray = rgb.mean(axis=2).astype(np.float32)
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    ys, xs = np.where(mag >= np.quantile(mag, 0.91))
    rng = np.random.default_rng(seed)
    if len(xs) > n:
        pick = rng.choice(len(xs), size=n, replace=False)
        xs, ys = xs[pick], ys[pick]
    return xs.astype(np.float64), ys.astype(np.float64)


def _rounded(ax, x, y, w, h, *, fc, ec, lw=LINE, ls="-", radius=0.08) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.012,rounding_size={radius}",
            linewidth=lw,
            linestyle=ls,
            facecolor=fc,
            edgecolor=ec,
            zorder=2,
        )
    )


def _arrow(ax, src, dst, *, color=INK, lw=1.15, rad=0.0, ls="-", style="-|>") -> None:
    ax.add_patch(
        FancyArrowPatch(
            src,
            dst,
            arrowstyle=f"{style},head_width=2.4,head_length=3.2",
            mutation_scale=8,
            linewidth=lw,
            linestyle=ls,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=3,
            shrinkA=1.2,
            shrinkB=1.2,
        )
    )


def _text(ax, x, y, s, **kwargs) -> None:
    ax.text(x, y, s, ha=kwargs.pop("ha", "center"), va=kwargs.pop("va", "center"), zorder=4, color=kwargs.pop("color", INK), **kwargs)


def _panel(ax, x, y, label) -> None:
    _text(ax, x, y, label, fontsize=9.5, fontweight="bold", ha="left", va="bottom", color=INK)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, default=RGB_PATH)
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/figures/method"))
    args = parser.parse_args()
    _style()
    rgb = _rgb(args.rgb)
    xs, ys = _edge_xy(rgb)
    h_px, w_px = rgb.shape[:2]
    orbit = 11.0
    xs_s = xs + orbit * np.cos(0.55)
    ys_s = ys + orbit * np.sin(0.55)

    fig = plt.figure(figsize=(7.28, 5.55), dpi=220)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.35)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    _text(
        ax,
        0.18,
        7.18,
        "AMI: active optical excitation with a rotating wedge prism",
        fontsize=10.2,
        fontweight="bold",
        ha="left",
        va="center",
    )
    _text(
        ax,
        0.18,
        6.93,
        "Event camera  +  rotating wedge  +  per-event inverse warp",
        fontsize=7.4,
        ha="left",
        va="center",
        color=MUTED,
    )

    # --- (a) baseline strip ---
    _panel(ax, 0.18, 6.58, "(a)")
    _text(ax, 0.52, 6.62, "Same RGB, four sensing modes", fontsize=8.0, ha="left", va="center")
    baselines = (
        (0.18, "DVS / v2e", "no prism", "#F3F4F6", "#7A8490"),
        (2.58, r"AMI, $\omega=0$", "prism present, never spins", WARM_FILL, WARM),
        (4.98, r"AMI, $\omega=180/360$", "open-loop excitation", NAVY_FILL, NAVY),
        (7.38, "Ours", r"causal $\omega\in\{0,180,360\}$", "#E5EFE8", "#2F6B4F"),
    )
    for x0, title, sub, fc, ec in baselines:
        _rounded(ax, x0, 6.12, 2.28, 0.42, fc=fc, ec=ec, lw=0.85, radius=0.06)
        _text(ax, x0 + 1.14, 6.38, title, fontsize=7.3, fontweight="bold", color=ec)
        _text(ax, x0 + 1.14, 6.22, sub, fontsize=6.2, color=MUTED)

    # --- (b) optical path ---
    _panel(ax, 0.18, 5.82, "(b)")
    _text(ax, 0.52, 5.86, "Optical path: two planes", fontsize=8.0, ha="left", va="center")

    # scene
    _rounded(ax, 0.18, 3.78, 2.22, 1.92, fc="#F8F8F8", ec=INK, radius=0.06)
    ax.imshow(rgb, origin="upper", extent=(0.28, 2.30, 3.98, 5.38), interpolation="bilinear", zorder=3)
    _text(ax, 1.29, 5.52, "3D scene / RGB", fontsize=7.2, fontweight="bold")
    _text(ax, 1.29, 3.88, "reference plane", fontsize=6.0, color=MUTED)

    _arrow(ax, (2.42, 4.72), (2.72, 4.72))
    _rounded(ax, 2.72, 4.28, 1.22, 0.92, fc="#F8F8F8", ec=INK, radius=0.05)
    _text(ax, 3.33, 4.92, "Pinhole", fontsize=7.2, fontweight="bold")
    _text(ax, 3.33, 4.70, "camera", fontsize=7.2, fontweight="bold")
    _text(ax, 3.33, 4.46, r"$+Z$ forward", fontsize=5.8, color=MUTED)

    _arrow(ax, (3.96, 4.72), (4.28, 4.72), color=NAVY, lw=1.35)

    # prism — the method
    _rounded(ax, 4.28, 3.78, 2.08, 1.92, fc=NAVY_FILL, ec=NAVY, lw=1.25, radius=0.08)
    cx, cy, r = 5.32, 4.95, 0.52
    ax.add_patch(Circle((cx, cy), r + 0.16, fill=False, lw=0.9, ls="--", edgecolor=NAVY, zorder=4))
    ax.add_patch(Circle((cx, cy), 0.18, facecolor="white", edgecolor=NAVY, lw=0.9, zorder=5))
    ax.add_patch(Wedge((cx, cy), 0.42, 18, 78, width=0.16, facecolor=NAVY, edgecolor=NAVY, alpha=0.35, zorder=5))
    ax.add_patch(Arc((cx, cy), 1.28, 1.28, theta1=-20, theta2=210, lw=1.05, color=NAVY, zorder=5))
    _arrow(ax, (cx + 0.58, cy + 0.18), (cx + 0.42, cy + 0.46), color=NAVY, lw=1.05, style="-|>")
    _text(ax, 5.32, 5.52, "Rotating wedge", fontsize=7.4, fontweight="bold", color=NAVY)
    _text(ax, 5.32, 4.22, r"$\theta(t)=\theta_0+\omega(t-t_{\mathrm{ref}})$", fontsize=6.1, color=NAVY)
    _text(ax, 5.32, 3.96, r"$0.5^{\circ}$   $\omega\in\{0,180,360\}$ RPM", fontsize=6.3, fontweight="bold", color=NAVY)

    _arrow(ax, (6.38, 5.35), (6.68, 5.55), color=SENSOR, lw=1.15)
    _arrow(ax, (6.38, 4.15), (6.68, 3.98), color=REF, lw=1.15)

    # sensor plane (slightly shifted crop to suggest the prism orbit)
    _rounded(ax, 6.68, 4.92, 3.12, 0.78, fc="#FBF6F0", ec=SENSOR, lw=0.95, radius=0.05)
    ax.imshow(rgb, origin="upper", extent=(6.70, 8.00, 4.98, 5.56), interpolation="bilinear", zorder=3, alpha=0.92)
    ax.scatter(
        6.78 + xs_s / w_px * 1.30,
        5.60 - ys_s / h_px * 0.58,
        s=5.5,
        c=SENSOR,
        linewidths=0,
        zorder=6,
        alpha=0.9,
    )
    _text(ax, 8.95, 5.48, "Sensor plane", fontsize=7.3, fontweight="bold", color=SENSOR, ha="left")
    _text(ax, 8.95, 5.28, "image orbits; events born here", fontsize=5.9, color=MUTED, ha="left")
    _text(ax, 8.95, 5.10, r"forward:  Ref  --prism($\theta$)-->  Sensor", fontsize=5.7, color=SENSOR, ha="left")

    # reference plane
    _rounded(ax, 6.68, 3.78, 3.12, 0.92, fc="#F3F6FB", ec=REF, lw=0.95, radius=0.05)
    ax.imshow(rgb, origin="upper", extent=(6.78, 8.08, 3.90, 4.52), interpolation="bilinear", zorder=3, alpha=0.92)
    ax.scatter(
        6.78 + xs / w_px * 1.30,
        4.52 - ys / h_px * 0.62,
        s=5.5,
        c=REF,
        linewidths=0,
        zorder=6,
        alpha=0.9,
    )
    _text(ax, 8.95, 4.48, "Reference plane", fontsize=7.3, fontweight="bold", color=REF, ha="left")
    _text(ax, 8.95, 4.28, "aligned to RGB; method output", fontsize=5.9, color=MUTED, ha="left")
    _text(ax, 8.95, 4.10, r"inverse: Sensor  --prism($-\theta(t_e)$)-->  Ref", fontsize=5.7, color=REF, ha="left")
    _text(ax, 8.95, 3.92, "per-event timestamp $t_e$", fontsize=5.7, color=MUTED, ha="left")

    # --- (c) pipeline ---
    _panel(ax, 0.18, 3.42, "(c)")
    _text(ax, 0.52, 3.46, "Processing  (optical method)", fontsize=8.0, ha="left", va="center")

    boxes = (
        (0.18, "1. RGB + time", "960×540, 60 Hz"),
        (1.80, "2. Linear Y", "sRGB → luminance"),
        (3.42, "3. Prism warp", r"$\theta(t)$, $\omega$  → sensor"),
        (5.04, "4. Threshold", r"events, $C=0.2$"),
        (6.66, "5. Inverse warp", r"at each $t_e$"),
        (8.28, "6. Ref. events", "IWE / edge F1"),
    )
    for i, (x0, title, sub) in enumerate(boxes):
        fill = NAVY_FILL if i in (2, 4) else "#F7F7F7"
        edge = NAVY if i in (2, 4) else INK
        lw = 1.15 if i in (2, 4) else 0.75
        _rounded(ax, x0, 2.48, 1.52, 0.82, fc=fill, ec=edge, lw=lw, radius=0.06)
        _text(ax, x0 + 0.76, 3.04, title, fontsize=6.9, fontweight="bold", color=edge)
        _text(ax, x0 + 0.76, 2.78, sub, fontsize=5.9, color=MUTED)
        if i < len(boxes) - 1:
            _arrow(ax, (x0 + 1.54, 2.89), (boxes[i + 1][0] - 0.02, 2.89), color=INK if i not in (1, 3) else NAVY, lw=1.05)

    # --- (d) control + optional NFB ---
    _panel(ax, 0.18, 2.18, "(d)")
    _text(ax, 0.52, 2.22, "Ours: prism gears, plus optional motion NFB", fontsize=8.0, ha="left", va="center")

    _rounded(ax, 3.20, 0.28, 2.52, 1.78, fc=NAVY_FILL, ec=NAVY, lw=1.15, radius=0.07)
    _text(ax, 4.46, 1.86, "7. Causal gear policy  (Ours)", fontsize=7.1, fontweight="bold", color=NAVY)
    _text(ax, 4.46, 1.62, r"$\omega\in\{0,\ 180,\ 360\}$ RPM", fontsize=7.6, fontweight="bold", color=NAVY)
    _text(ax, 4.46, 1.36, "0  stop   (enough natural motion)", fontsize=5.9, color=INK)
    _text(ax, 4.46, 1.18, "180  default excitation", fontsize=5.9, color=INK)
    _text(ax, 4.46, 1.00, "360  extra  (too little / co-motion)", fontsize=5.9, color=INK)
    _text(ax, 4.46, 0.76, "storm occupancy: no 360", fontsize=5.8, color=MUTED)
    _text(ax, 4.46, 0.58, "warmup / default: 180", fontsize=5.8, color=MUTED)
    _text(ax, 4.46, 0.40, "does not control exposure or masks", fontsize=5.5, color=MUTED)

    # occupancy feedback from box 6 to gear
    _arrow(ax, (9.04, 2.46), (5.70, 1.95), color=NAVY, lw=0.95, rad=0.18)
    _text(ax, 7.55, 2.08, "occupancy", fontsize=5.8, color=NAVY)

    # gear to prism
    _arrow(ax, (4.46, 2.08), (5.32, 3.76), color=NAVY, lw=1.2)
    _text(ax, 4.05, 2.95, r"$\omega$", fontsize=8.0, fontweight="bold", color=NAVY)

    # teacher
    _rounded(ax, 7.55, 0.28, 2.27, 1.18, fc="#F7F3EA", ec=TEACH, lw=0.85, ls=(0, (3, 1.6)), radius=0.06)
    _text(ax, 8.68, 1.28, "Train only", fontsize=6.5, fontweight="bold", color=TEACH)
    _text(ax, 8.68, 1.08, "Teacher: argmax F1", fontsize=6.2, color=INK)
    _text(ax, 8.68, 0.90, r"among $\{0,180,360\}$", fontsize=6.2, color=INK)
    _text(ax, 8.68, 0.70, "ties → lower RPM", fontsize=5.8, color=MUTED)
    _text(ax, 8.68, 0.52, "storm forbids 360", fontsize=5.8, color=MUTED)
    _text(ax, 8.68, 0.36, "deploy never sees F1", fontsize=5.6, color=MUTED)

    # optional NFB
    _rounded(ax, 0.18, 0.28, 2.78, 1.78, fc="#F4F4F5", ec=NFB, lw=0.85, ls=(0, (3.2, 1.8)), radius=0.07)
    _text(ax, 1.57, 1.86, "8. Motion NFB  (optional)", fontsize=7.0, fontweight="bold", color=NFB)
    _text(ax, 1.57, 1.60, "same occupancy, second actuator", fontsize=5.8, color=MUTED)
    _text(ax, 1.57, 1.34, r"camera rate $\in\{1.0,\ 0.5\}\times$", fontsize=6.3, color=INK)
    _text(ax, 1.57, 1.14, "never stop the camera", fontsize=5.9, color=INK)
    _text(ax, 1.57, 0.92, "dual: also 360 → 180", fontsize=5.9, color=INK)
    _text(ax, 1.57, 0.66, "along recorded path, not a new take", fontsize=5.6, color=MUTED)
    _text(ax, 1.57, 0.42, "inverse warp undoes prism only,", fontsize=5.6, color=WARM)
    _text(ax, 1.57, 0.28, "not camera motion", fontsize=5.6, color=WARM)

    _arrow(ax, (2.96, 1.18), (3.18, 1.18), color=NFB, lw=0.9, ls=(0, (2.5, 1.5)))
    _arrow(ax, (1.57, 2.08), (3.33, 4.26), color=NFB, lw=0.9, ls=(0, (2.5, 1.5)), rad=-0.12)
    _text(ax, 1.95, 3.05, "optional", fontsize=5.7, color=NFB)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdf = args.output.with_suffix(".pdf")
    png = args.output.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=220)
    plt.close(fig)
    print({"pdf": str(pdf), "png": str(png)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
