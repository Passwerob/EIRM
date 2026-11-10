#!/usr/bin/env python3
"""Reconstruct grayscale frames with pretrained E2VID from EIRM vs passive events."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import EventStream
from scripts.evaluate_dl3dv_downstream import _ami_windows, _dvs_windows
from scripts.evaluate_eveyes_v2_downstream import _intrinsics, _scene_path

METHODS = (
    ("passive", "Passive (0 RPM)", 0.0),
    ("eirm_180", "EIRM 180 RPM", 180.0),
    ("eirm_360", "EIRM 360 RPM", 360.0),
)


def _concat(windows: list[EventStream]) -> EventStream:
    if not windows:
        return EventStream.empty()
    return EventStream(
        np.concatenate([np.asarray(w.x) for w in windows]),
        np.concatenate([np.asarray(w.y) for w in windows]),
        np.concatenate([np.asarray(w.t) for w in windows]),
        np.concatenate([np.asarray(w.p) for w in windows]),
    ).sorted_by_time()


def _write_e2vid_txt(path: Path, events: EventStream, width: int, height: int) -> int:
    x = np.asarray(events.x, dtype=np.int32)
    y = np.asarray(events.y, dtype=np.int32)
    t = np.asarray(events.t, dtype=np.float64)
    p = np.asarray(events.p)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, t, p = x[valid], y[valid], t[valid], p[valid]
    pol = np.where(p > 0, 1, 0).astype(np.int8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(f"{width} {height}\n")
        np.savetxt(handle, np.column_stack([t, x, y, pol]), fmt="%.9f %d %d %d")
    return int(len(t))


def _run_e2vid(e2vid_root: Path, weights: Path, events_txt: Path, out_dir: Path, python: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        str(e2vid_root / "run_reconstruction.py"),
        "-c",
        str(weights),
        "-i",
        str(events_txt),
        "--fixed_duration",
        "-T",
        "33.33",
        "--auto_hdr",
        "-o",
        str(out_dir),
        "--dataset_name",
        "reconstruction",
    ]
    print("running", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(e2vid_root))


def _nth_image(folder: Path, index: int) -> Path | None:
    files = sorted(folder.glob("*.png"))
    if not files:
        return None
    return files[min(index, len(files) - 1)]


def _panel(axis, image, title: str) -> None:
    if image.ndim == 2:
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    else:
        axis.imshow(image)
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/eveyes_comotion_v2_ready")
    parser.add_argument("--split", default="test")
    parser.add_argument("--scene", default="Scene_005")
    parser.add_argument("--regime", default="co_motion")
    parser.add_argument("--start-frame", type=int, default=60)
    parser.add_argument("--num-frames", type=int, default=90)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--e2vid-root", type=Path, default=Path("/data/fcr/rpg_e2vid"))
    parser.add_argument("--e2vid-python", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=Path("/data/fcr/rpg_e2vid/pretrained/E2VID_lightweight.pth.tar"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/ami_next/e2vid_eirm")
    args = parser.parse_args()

    scene = _scene_path(args.dataset_root, args.split, args.scene, args.regime)
    end_frame = args.start_frame + args.num_frames
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=end_frame)
    frames = frames[args.start_frame :]
    timestamps = timestamps[args.start_frame : args.start_frame + len(frames)]
    luminance = rgb_to_linear_luminance(frames)
    intrinsics = _intrinsics(scene)
    valid = np.ones((len(frames) - 1, intrinsics.height, intrinsics.width), dtype=bool)
    prism = RotatingWedgePrism(intrinsics, 0.5)

    args.output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for key, _, rpm in METHODS:
        print(f"simulating {key} rpm={rpm}", flush=True)
        if rpm <= 0:
            windows, _ = _dvs_windows(luminance, timestamps)
        else:
            windows, _, _ = _ami_windows(luminance, timestamps, valid, prism, rpm, device=args.device)
        events = _concat(windows)
        txt = args.output / f"{key}_events.txt"
        n = _write_e2vid_txt(txt, events, intrinsics.width, intrinsics.height)
        counts[key] = n
        print(f"  {n} events -> {txt}", flush=True)
        recon_root = args.output / key
        _run_e2vid(args.e2vid_root, args.weights, txt, recon_root, str(args.e2vid_python or sys.executable))

    rgb_indices = [0, len(frames) // 2, len(frames) - 1]
    recon_folders = {key: args.output / key / "reconstruction" for key, _, _ in METHODS}
    fig, axes = plt.subplots(4, 3, figsize=(11.2, 10.4), dpi=140)
    row_titles = ["RGB", "E2VID · passive", "E2VID · EIRM 180", "E2VID · EIRM 360"]
    method_keys = [None, "passive", "eirm_180", "eirm_360"]
    for col, fi in enumerate(rgb_indices):
        t = float(timestamps[fi])
        rgb = np.asarray(frames[fi])
        _panel(axes[0, col], rgb, f"t = {t:.2f} s")
        for row, key in enumerate(method_keys[1:], start=1):
            folder = recon_folders[key]
            path = _nth_image(folder, int(round(col / 2 * max(len(list(folder.glob('*.png'))) - 1, 0))))
            if path is None:
                axes[row, col].set_visible(False)
                continue
            rec = np.asarray(Image.open(path).convert("L"))
            extra = f" · {counts[key]/1e6:.2f}M ev" if col == 0 else ""
            _panel(axes[row, col], rec, f"{row_titles[row]}{extra}" if col == 0 else "")
        if col == 0:
            for row, title in enumerate(row_titles):
                axes[row, 0].set_ylabel(title, fontsize=9)
    fig.suptitle(
        f"Pretrained E2VID on {args.scene} {args.regime} (no fine-tune). "
        "Same 33.3 ms windows; EIRM events are inverse-warped to the RGB plane.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.output / "fig_e2vid_eirm_compare.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", args.output / "fig_e2vid_eirm_compare.png")
    print("event counts", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
