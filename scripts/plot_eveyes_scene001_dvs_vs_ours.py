#!/usr/bin/env python3
"""Scene_001: DVS background is empty; AMI 180 paints the static city.

Camera frozen, only the car moves. Official DVS-Voltmeter vs inverse-warped 180.
Not Table 1, not the 1080 storm exhibit.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from scripts.evaluate_dvs_v2e_ours import _read_npz, _slice
from scripts.evaluate_eveyes_v2_downstream import _intrinsics
from scripts.render_eveyes_scene001_storm_vs_ours import SCENE, TIMES, _hero_events, _hero_index
from scripts.render_eveyes_camera_nfb_event_video import _events_to_map

DVS_ROOT = Path("/data/fcr/code/event_vggt/Data_Part/DVS-Voltmeter")
DVS_PYTHON = Path("/data/fcr/miniconda3/envs/dvsvolt/bin/python")
CACHE = Path("results/eveyes_scene001/dvs_voltmeter.npz")


def _load(scene: Path, max_frames: int):
    rgb_dir = scene / "rgb" if (scene / "rgb").exists() else scene / "images"
    frames, timestamps = load_rgb_sequence(rgb_dir, scene / "timestamps.csv", timestamp_unit="us", max_frames=max_frames)
    rgb = np.clip(frames, 0, 255).astype(np.uint8)
    return rgb, timestamps, rgb_to_linear_luminance(frames)


def _dvs_events(luminance: np.ndarray, timestamps: np.ndarray, cache: Path) -> object:
    if cache.exists():
        print(f"reuse {cache}", flush=True)
        return _read_npz(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_dvs_voltmeter_array.py")
    with tempfile.TemporaryDirectory(prefix="scene001_dvs_") as temporary:
        inp = Path(temporary) / "input.npz"
        out = Path(temporary) / "dvs.npz"
        np.savez_compressed(inp, frames=luminance.astype(np.float32), timestamps=timestamps.astype(np.float64))
        cmd = [str(DVS_PYTHON), str(runner), "--input", str(inp), "--output", str(out), "--dvs-root", str(DVS_ROOT), "--seed", "17"]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        events = _read_npz(out)
    np.savez_compressed(cache, x=np.asarray(events.x), y=np.asarray(events.y), t=np.asarray(events.t), p=np.asarray(events.p))
    print(f"wrote {cache}  n={len(events.x)}", flush=True)
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=480)
    parser.add_argument("--output", type=Path, default=Path("results/eveyes_scene001/figures/scene001_dvs_vs_ours_scenes"))
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")
    rgb, timestamps, luminance = _load(args.scene, args.max_frames)
    dvs = _dvs_events(luminance, timestamps, CACHE)
    prism = RotatingWedgePrism(_intrinsics(args.scene), 0.5)
    height, width = rgb[0].shape[:2]
    fig, axes = plt.subplots(3, len(TIMES), figsize=(12.4, 8.4), dpi=140)
    row_titles = (
        "RGB  (camera frozen)",
        "DVS-Voltmeter  (no prism)",
        "Ours  (180 RPM, inverse-warped)",
    )
    for col, t in enumerate(TIMES):
        hero = _hero_index(timestamps, t)
        t0, t1 = float(timestamps[hero]), float(timestamps[min(hero + 1, len(timestamps) - 1)])
        idle = _slice(dvs, t0, t1)
        ours = _hero_events(luminance, timestamps, prism, hero, 180.0, args.device, plane="reference", n_windows=1)
        panels = (rgb[hero], _events_to_map(idle, height, width), _events_to_map(ours, height, width))
        captions = (
            f"t={t0:.2f}s  frame {hero}",
            f"{len(idle.x):,} events  ·  car only",
            f"{len(ours.x)/1e6:.2f}M events  ·  city + car",
        )
        print({"t": t, "hero": hero, "dvs": int(len(idle.x)), "ours": int(len(ours.x))}, flush=True)
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
        axes[0, col].set_title(f"{t:.0f}s" if t >= 1 else f"{t:.1f}s", fontsize=11, pad=6)
    fig.suptitle("Scene_001  ·  camera static  ·  DVS sees only the car; prism paints the street", fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"))
    fig.savefig(args.output.with_suffix(".png"), dpi=170)
    plt.close(fig)
    print(args.output.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
