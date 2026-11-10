#!/usr/bin/env python3
"""Paper qualitative: DVS-Voltmeter / v2e / Ours + identical CM-IWE.

Same-source Scene_005 RGB. Rows are raw polarity events and contrast-maximized
IWE under one frozen translation velocity grid. CM is a representation, not a
fourth event generator. Does not overwrite Table 1 / m0.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ami_simulator.contrast_maximization import contrast_maximization
from ami_simulator.types import EventStream
from scripts.evaluate_dvs_v2e_ours import _read_npz, _slice
from scripts.evaluate_eveyes_v2_downstream import _scene_path
from scripts.plot_eveyes_event_qualitative import (
    CACHE,
    CACHE_WINDOWS,
    READY,
    _ami_window,
    _events_to_map,
)

INK = "#2C3E50"

# Paper-frozen CM velocity grid (pixels / second). Identical for every source.
VX = (-160.0, -80.0, -40.0, -20.0, -10.0, 0.0, 10.0, 20.0, 40.0, 80.0, 160.0)
VY = (-80.0, -40.0, -20.0, -10.0, 0.0, 10.0, 20.0, 40.0, 80.0)
VELOCITY_GRID = tuple((vx, vy) for vx in VX for vy in VY)

# Two regimes that carry the claim: frozen-camera street empty vs co-motion lock.
REGIMES = (
    {
        "name": "object_only",
        "title": "object only · camera frozen",
        "hero": 120,  # t = 2.0 s
        "baseline_dvs": CACHE / "test_Scene_005_object_only_f480_w960_dvs_voltmeter.npz",
        "baseline_v2e": CACHE / "test_Scene_005_object_only_f480_w960_v2e.npz",
        "ami_cache": CACHE_WINDOWS / "Scene_005_object_only_f120_rpm180.npz",
        "ours_note": "city + car",
        "passive_note": "car only",
    },
    {
        "name": "co_motion",
        "title": "co-motion · relative lock",
        "hero": 150,  # t = 2.5 s
        "baseline_dvs": CACHE / "test_Scene_005_f480_w960_dvs_voltmeter.npz",
        "baseline_v2e": CACHE / "test_Scene_005_f480_w960_v2e.npz",
        "ami_cache": CACHE_WINDOWS / "Scene_005_co_motion_f150_rpm180.npz",
        "ours_note": "structure + car",
        "passive_note": "ego clutter",
    },
)


def _timestamps(scene: Path) -> np.ndarray:
    rows = list(csv.DictReader((scene / "timestamps.csv").open()))
    return np.asarray([float(row["timestamp_seconds"]) for row in rows], dtype=np.float64)


def _load_ours(scene: Path, hero: int, ami_cache: Path, device: str) -> EventStream:
    if ami_cache.exists():
        return _read_npz(ami_cache)
    return _ami_window(scene, hero, 180.0, device, plane="reference", n_windows=1)


def _cm_gray(image_2hw: np.ndarray) -> np.ndarray:
    """White-background log IWE from polarity-separated CM image."""
    counts = np.asarray(image_2hw[0] + image_2hw[1], dtype=np.float64)
    positive = counts[counts > 0]
    peak = float(np.percentile(positive, 99.5)) if len(positive) else 1.0
    peak = max(peak, 1.0)
    level = np.clip(np.log1p(counts) / np.log1p(peak), 0.0, 1.0)
    gray = 255.0 - 235.0 * level
    return np.repeat(gray[..., None], 3, axis=-1).astype(np.uint8)


def _panel(axis, image: np.ndarray, xlabel: str | None = None) -> None:
    axis.imshow(image)
    axis.set_xticks([])
    axis.set_yticks([])
    if xlabel:
        axis.set_xlabel(xlabel, fontsize=7.2, color=INK, labelpad=2)
    for spine in axis.spines.values():
        spine.set_linewidth(0.45)
        spine.set_color("0.45")


def _run_regime(regime: dict, dataset_root: Path, device: str) -> dict:
    scene = _scene_path(dataset_root, "test", "Scene_005", regime["name"])
    timestamps = _timestamps(scene)
    hero = int(regime["hero"])
    t0 = float(timestamps[hero])
    t1 = float(timestamps[min(hero + 1, len(timestamps) - 1)])
    rgb = np.asarray(Image.open(scene / "images" / f"{hero:06d}.png").convert("RGB"))
    height, width = rgb.shape[:2]

    dvs = _slice(_read_npz(regime["baseline_dvs"]), t0, t1)
    v2e = _slice(_read_npz(regime["baseline_v2e"]), t0, t1)
    ours = _load_ours(scene, hero, regime["ami_cache"], device)
    # Cached AMI windows already cover [t0, t1]; re-slice defensively.
    ours = _slice(ours, t0 - 1e-6, t1 + 1e-6)

    sources = (
        ("DVS-Voltmeter", dvs, regime["passive_note"]),
        ("v2e", v2e, regime["passive_note"]),
        ("Ours (180, inv-warp)", ours, regime["ours_note"]),
    )
    event_maps = []
    cm_maps = []
    meta_methods = []
    for name, events, note in sources:
        cm = contrast_maximization(events, (height, width), VELOCITY_GRID, reference_time=t1)
        event_maps.append((_events_to_map(events, height, width), f"{len(events.x):,} · {note}"))
        vx, vy = cm["velocity"]  # type: ignore[misc]
        cm_maps.append((_cm_gray(np.asarray(cm["image"])), f"CM ({vx:.0f},{vy:.0f})  var={float(cm['contrast']):.3g}"))
        meta_methods.append(
            {
                "method": name,
                "n_events": int(len(events.x)),
                "cm_velocity": [float(vx), float(vy)],
                "cm_contrast": float(cm["contrast"]),
            }
        )
        print(
            json.dumps(
                {
                    "regime": regime["name"],
                    "method": name,
                    "n": int(len(events.x)),
                    "vx": float(vx),
                    "vy": float(vy),
                    "contrast": float(cm["contrast"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    return {
        "regime": regime,
        "rgb": rgb,
        "t0": t0,
        "t1": t1,
        "hero": hero,
        "event_maps": event_maps,
        "cm_maps": cm_maps,
        "methods": meta_methods,
        "shape": [height, width],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=READY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ami_next/figures/paper_dvs_v2e_cm"),
    )
    args = parser.parse_args()

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

    blocks = [_run_regime(regime, args.dataset_root, args.device) for regime in REGIMES]
    n_reg = len(blocks)
    # columns: RGB + 3 methods; two row-pairs per regime (events, CM)
    fig_w = 7.2
    fig_h = 2.15 * n_reg + 0.55
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Same RGB · DVS-Voltmeter / v2e / Ours · raw events and CM-IWE",
        fontsize=9.5,
        color=INK,
        fontweight="bold",
        y=0.995,
    )

    grid = fig.add_gridspec(
        n_reg * 2,
        4,
        height_ratios=[1.0] * (n_reg * 2),
        width_ratios=[1.05, 1.0, 1.0, 1.0],
        wspace=0.06,
        hspace=0.28,
        left=0.07,
        right=0.995,
        top=0.90,
        bottom=0.04,
    )
    col_titles = ("RGB", "DVS-Voltmeter", "v2e", "Ours (180, inv-warp)")

    for r, block in enumerate(blocks):
        regime = block["regime"]
        row_e = 2 * r
        row_c = 2 * r + 1
        # RGB spans both rows visually by duplicating still (keeps grid simple)
        for row, kind in ((row_e, "events"), (row_c, "CM-IWE")):
            ax = fig.add_subplot(grid[row, 0])
            _panel(ax, block["rgb"], xlabel=f"t={block['t0']:.2f}s" if kind == "events" else None)
            if row == row_e:
                ax.set_title(col_titles[0], fontsize=7.5, color=INK, pad=3)
            ylabel = f"{regime['title']}\n{kind}"
            ax.set_ylabel(ylabel, fontsize=6.8, color=INK)
        for col, ((emap, ecaption), (cmap, ccaption)) in enumerate(
            zip(block["event_maps"], block["cm_maps"]), start=1
        ):
            ax_e = fig.add_subplot(grid[row_e, col])
            _panel(ax_e, emap, xlabel=ecaption)
            if r == 0:
                ax_e.set_title(col_titles[col], fontsize=7.5, color=INK, pad=3)
            ax_c = fig.add_subplot(grid[row_c, col])
            _panel(ax_c, cmap, xlabel=ccaption)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdf = args.output.with_suffix(".pdf")
    png = args.output.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=220)
    plt.close(fig)

    meta = {
        "protocol": "ami_paper_dvs_v2e_cm_qualitative_v1",
        "velocity_grid_px_per_s": {"vx": list(VX), "vy": list(VY), "n": len(VELOCITY_GRID)},
        "cm": "translation-only polarity-separated variance (ami_simulator.contrast_maximization)",
        "window": "same inter-frame [t_hero, t_hero+1] for events and CM",
        "ours": "AMI 180 RPM inverse-warped reference-plane events",
        "regimes": [
            {
                "name": b["regime"]["name"],
                "hero": b["hero"],
                "t0": b["t0"],
                "t1": b["t1"],
                "shape": b["shape"],
                "methods": b["methods"],
            }
            for b in blocks
        ],
    }
    meta_path = args.output.with_name(args.output.name + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    caption = args.output.with_name(args.output.name + "_caption.txt")
    caption.write_text(
        """Figure. Same-source DVS-Voltmeter, v2e, and Ours, each through identical CM-IWE.

Scene_005. Every column receives the same RGB and the same inter-frame window.
Top block: object_only (camera frozen). Passive DVS/v2e fire on the moving car;
the street stays empty. Ours at 180 RPM with per-event inverse warp paints city
structure plus the car.

Bottom block: co-motion (relative lock). Passive streams are ego-motion clutter;
Ours still recovers structure on the reference plane.

CM-IWE row: the same frozen translation velocity grid for all three sources
(ami_simulator.contrast_maximization). CM is a downstream representation, not a
fourth event generator. Subtitles show the selected (vx, vy) in px/s.

Reproduce:
  MPLCONFIGDIR=/tmp/mpl-fcr PYTHONPATH=. \\
    python scripts/plot_eveyes_dvs_v2e_cm_qualitative.py
""",
        encoding="utf-8",
    )
    print({"pdf": str(pdf), "png": str(png), "meta": str(meta_path), "caption": str(caption)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
