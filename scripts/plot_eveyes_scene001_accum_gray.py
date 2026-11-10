#!/usr/bin/env python3
"""Accumulated event grayscale (IWE) as the 3D-reconstruction observation.

Scene_001 camera frozen. DVS accumulation stays empty in the background;
inverse-warped 180 paints a grayscale structure image of the street.
This is the event-count IWE a reconstructor consumes, not E2VID and not a mesh.
Not Table 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from ami_simulator.geometry import RotatingWedgePrism
from scripts.evaluate_dvs_v2e_ours import _read_npz, _slice
from scripts.evaluate_eveyes_v2_downstream import _intrinsics
from scripts.plot_eveyes_scene001_dvs_vs_ours import CACHE, _load
from scripts.render_eveyes_scene001_storm_vs_ours import SCENE, _step

OURS_RPM = 180.0
WINDOWS_S = (0.2, 2.0, 8.0)


def _add_counts(counts: np.ndarray, events) -> int:
    if not len(events.x):
        return 0
    x = np.rint(np.asarray(events.x)).astype(np.int32)
    y = np.rint(np.asarray(events.y)).astype(np.int32)
    valid = (x >= 0) & (x < counts.shape[1]) & (y >= 0) & (y < counts.shape[0])
    np.add.at(counts, (y[valid], x[valid]), 1.0)
    return int(valid.sum())


def _iwe_gray(counts: np.ndarray) -> np.ndarray:
    """White empty, dark where events pile. Independent 99th-percentile stretch."""
    positive = counts[counts > 0]
    peak = float(np.percentile(positive, 99.0)) if len(positive) else 1.0
    peak = max(peak, 1.0)
    level = np.clip(np.log1p(counts) / np.log1p(peak), 0.0, 1.0)
    gray = np.clip(255.0 - 255.0 * level, 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def _stem(window: float) -> str:
    return f"{window:g}s".replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=480)
    parser.add_argument("--output", type=Path, default=Path("results/eveyes_scene001/figures/scene001_accum_gray_recon"))
    parser.add_argument("--dump-dir", type=Path, default=Path("results/eveyes_scene001/recon_gray"))
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("cuda unavailable; AMI 180 IWE on CPU (~9 min for 8 s)", flush=True)
        args.device = "cpu"
    rgb, timestamps, luminance = _load(args.scene, args.max_frames)
    dvs = _read_npz(CACHE)
    prism = RotatingWedgePrism(_intrinsics(args.scene), 0.5)
    height, width = rgb[0].shape[:2]
    ours_counts = np.zeros((height, width), dtype=np.float32)
    ours_n = 0
    state = {"phase": 0.0, "active": None, "ref": None, "rpm": OURS_RPM}
    snapshots: dict[float, dict[str, object]] = {}
    n = len(luminance) - 1
    print(f"accumulate AMI 180 IWE  n={n} device={args.device}", flush=True)
    for index in range(n):
        t0, t1 = float(timestamps[index]), float(timestamps[index + 1])
        ours = _step(
            luminance[index : index + 2],
            np.asarray((t0, t1)),
            prism,
            OURS_RPM,
            state["phase"],
            state["active"],
            state["rpm"],
            state["ref"],
            args.device,
        )
        state.update(phase=float(ours["phase_end"]), active=ours["active"], ref=np.asarray(ours["sensor_reference_log"]), rpm=OURS_RPM)
        ours_n += _add_counts(ours_counts, ours["reference_events"])
        clip_end = float(timestamps[-1])
        for window in WINDOWS_S:
            if window not in snapshots and t1 + 1e-9 >= min(window, clip_end):
                dvs_window = _slice(dvs, 0.0, t1)
                dvs_counts = np.zeros((height, width), dtype=np.float32)
                dvs_n = _add_counts(dvs_counts, dvs_window)
                hero = min(index + 1, len(rgb) - 1)
                snapshots[window] = {
                    "t": t1,
                    "rgb": rgb[hero],
                    "dvs": dvs_counts,
                    "ours": ours_counts.copy(),
                    "dvs_n": dvs_n,
                    "ours_n": ours_n,
                }
                print({"window_s": window, "t": t1, "dvs": dvs_n, "ours": ours_n}, flush=True)
        if index % 40 == 0:
            print(f"  w={index}/{n} t={t1:.2f} ours_events={ours_n}", flush=True)
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, len(WINDOWS_S), figsize=(12.6, 8.6), dpi=140)
    row_titles = (
        "RGB  (camera frozen)",
        "DVS accumulated grayscale",
        "Ours IWE  (180 RPM, inverse-warped)",
    )
    for col, window in enumerate(WINDOWS_S):
        block = snapshots[window]
        dvs_img = _iwe_gray(np.asarray(block["dvs"]))
        ours_img = _iwe_gray(np.asarray(block["ours"]))
        panels = (block["rgb"], dvs_img, ours_img)
        captions = (
            f"t={block['t']:.2f}s",
            f"{block['dvs_n']:,} events  ·  no static scene",
            f"{block['ours_n']/1e6:.2f}M events  ·  street structure",
        )
        for row, (image, caption) in enumerate(zip(panels, captions)):
            axis = axes[row, col]
            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlabel(caption, fontsize=8)
            if col == 0:
                axis.set_ylabel(row_titles[row], fontsize=8)
            for spine in axis.spines.values():
                spine.set_visible(False)
        axes[0, col].set_title(f"accumulate {window:g}s", fontsize=11, pad=6)
        stem = _stem(window)
        Image.fromarray(dvs_img).save(args.dump_dir / f"dvs_{stem}.png")
        Image.fromarray(ours_img).save(args.dump_dir / f"ours_{stem}.png")
        np.savez_compressed(args.dump_dir / f"counts_{stem}.npz", dvs=np.asarray(block["dvs"]), ours=np.asarray(block["ours"]))
    fig.suptitle("Accumulated event grayscale for 3D reconstruction  ·  Scene_001 camera static", fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"))
    fig.savefig(args.output.with_suffix(".png"), dpi=170)
    plt.close(fig)
    print(args.output.with_suffix(".png"), flush=True)
    print(args.dump_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
