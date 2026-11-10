#!/usr/bin/env python3
"""Paper qualitative: DL3DV in-the-wild RGB · DVS / v2e / Ours (+ CM-IWE).

Real captured scenes (not Blender). Same-source RGB → official DVS-Voltmeter,
official v2e, AMI 180 inverse-warped. Companion CM uses one frozen translation
grid. This is the transfer / in-the-wild exhibit; Table 1 numbers stay on the
frozen 160×90 protocol JSON. Does not overwrite m0 / EVEYES Table 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import EventStream
from scripts.evaluate_dl3dv_downstream import _ami_windows, _intrinsics
from scripts.evaluate_dvs_v2e_ours import _external, _read_npz, _slice
from scripts.plot_eveyes_event_qualitative import _events_to_map
from scripts.plot_paper_dvs_v2e_cm_coverage import (
    CM_MAX_EVENTS,
    INK,
    VELOCITY_GRID,
    VX,
    VY,
    _cm_gray,
    _panel,
    _run_cm,
)

READY = Path("data/dl3dv_large5")
BASELINE_CACHE = Path("results/ami_next/dl3dv_baseline_events")
AMI_CACHE = Path("results/ami_next/dl3dv_qualitative_windows")
WINDOW_CACHE = Path("results/ami_next/dl3dv_qualitative_clip_events")

# Diverse real scenes from the frozen large5 pack.
SCENES = (
    ("screen-000083", "alley / street"),
    ("screen-000087", "open field"),
    ("screen-000089", "cluttered yard"),
    ("screen-000095", "temple court"),
    ("screen-000108", "stone statues"),
)


def _events_to_map_adaptive(events: EventStream, height: int, width: int) -> np.ndarray:
    base = _events_to_map(events, height, width)
    if int(len(events.x)) >= 5000:
        return base
    boost = base.astype(np.float32)
    mask = boost.max(axis=2) > 0
    boost[mask] = np.clip(boost[mask] * 1.35 + 25.0, 0, 255)
    return boost.astype(np.uint8)


def _resize_rgb(frames: np.ndarray, width: int) -> np.ndarray:
    if frames.shape[2] == width:
        return frames
    height = int(round(frames.shape[1] * width / frames.shape[2]))
    return np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames])


def _baseline_window(
    scene_id: str,
    method: str,
    luminance: np.ndarray,
    timestamps: np.ndarray,
    hero: int,
    eval_width: int,
    args_ns: argparse.Namespace,
) -> tuple[EventStream, str]:
    t0 = float(timestamps[hero])
    t1 = float(timestamps[hero + 1])
    window_path = WINDOW_CACHE / f"{scene_id}_f{hero}_w{eval_width}_{method}.npz"
    if window_path.exists():
        return _read_npz(window_path), f"window:{window_path.name}"

    full = BASELINE_CACHE / f"{scene_id}_f{len(timestamps)}_w{eval_width}_{method}.npz"
    # Also accept the frozen Table-1 stem if widths match.
    alt = BASELINE_CACHE / f"{scene_id}_f24_w{eval_width}_{method}.npz"
    for path in (full, alt):
        if path.exists() and path.stat().st_size > 0:
            print(f"reuse full baseline {path.name}", flush=True)
            events = _slice(_read_npz(path), t0, t1)
            window_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(window_path, x=np.asarray(events.x), y=np.asarray(events.y), t=np.asarray(events.t), p=np.asarray(events.p))
            return events, f"full:{path.name}"

    print(f"building {method} for {scene_id} w={eval_width} n={len(timestamps)}", flush=True)
    import tempfile

    with tempfile.TemporaryDirectory(prefix=f"dl3dv_{method}_") as temporary:
        events_all = _external(method, luminance, timestamps, args_ns, Path(temporary))
    full.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(full, x=np.asarray(events_all.x), y=np.asarray(events_all.y), t=np.asarray(events_all.t), p=np.asarray(events_all.p))
    events = _slice(events_all, t0, t1)
    window_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(window_path, x=np.asarray(events.x), y=np.asarray(events.y), t=np.asarray(events.t), p=np.asarray(events.p))
    return events, f"built:{full.name}"


def _ours_window(
    scene: Path,
    luminance: np.ndarray,
    timestamps: np.ndarray,
    hero: int,
    eval_width: int,
    prism_angle_deg: float,
    device: str,
) -> EventStream:
    cache = AMI_CACHE / f"{scene.name}_f{hero}_w{eval_width}_rpm180_reference.npz"
    if cache.exists():
        return _read_npz(cache)
    intrinsics = _intrinsics(scene)
    if eval_width != intrinsics.width:
        intrinsics = intrinsics.resized(eval_width, luminance.shape[1])
    prism = RotatingWedgePrism(intrinsics, prism_angle_deg)
    masks = np.ones((len(luminance), luminance.shape[1], luminance.shape[2]), dtype=bool)
    # Warm a few windows so the sensor reference is settled.
    start = max(0, hero - 8)
    reference, _, _ = _ami_windows(
        luminance[start:],
        timestamps[start:],
        masks[start:],
        prism,
        180.0,
        device=device,
    )
    events = reference[hero - start]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=np.asarray(events.x), y=np.asarray(events.y), t=np.asarray(events.t), p=np.asarray(events.p))
    return events


def _bundle(scene_id: str, tag: str, args: argparse.Namespace) -> dict:
    scene = args.dataset_root / scene_id
    frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=args.max_frames)
    timestamps = timestamps[: len(frames)]
    frames = _resize_rgb(np.asarray(frames), args.eval_width)
    luminance = rgb_to_linear_luminance(frames)
    hero = min(args.hero, len(timestamps) - 2)
    t0 = float(timestamps[hero])
    t1 = float(timestamps[hero + 1])
    rgb = np.clip(frames[hero], 0, 255).astype(np.uint8)
    height, width = rgb.shape[:2]

    # Namespace for _external
    ext = argparse.Namespace(
        dvs_python=args.dvs_python,
        dvs_root=args.dvs_root,
        v2e_python=args.v2e_python,
        v2e_root=args.v2e_root,
        seed=args.seed,
    )
    dvs, dvs_src = _baseline_window(scene_id, "dvs_voltmeter", luminance, timestamps, hero, args.eval_width, ext)
    v2e, v2e_src = _baseline_window(scene_id, "v2e", luminance, timestamps, hero, args.eval_width, ext)
    ours = _ours_window(scene, luminance, timestamps, hero, args.eval_width, args.prism_angle_deg, args.device)
    ours = _slice(ours, t0 - 1e-6, t1 + 1e-6)

    event_maps = []
    cm_maps = []
    methods = []
    for name, events, src in (
        ("DVS-Voltmeter", dvs, dvs_src),
        ("v2e", v2e, v2e_src),
        ("Ours (180, inv-warp)", ours, "ami180"),
    ):
        cm = _run_cm(events, (height, width), t1)
        vx, vy = cm["velocity"]  # type: ignore[misc]
        event_maps.append((_events_to_map_adaptive(events, height, width), f"{len(events.x):,} evt"))
        cm_maps.append((_cm_gray(np.asarray(cm["image"])), f"CM ({vx:.0f},{vy:.0f})"))
        methods.append(
            {
                "method": name,
                "n_events": int(len(events.x)),
                "source": src,
                "cm_velocity": [float(vx), float(vy)],
                "cm_contrast": float(cm["contrast"]),
                "cm_subsampled": bool(cm["subsampled"]),
            }
        )
        print(json.dumps({"scene": scene_id, "method": name, "n": int(len(events.x)), "src": src}, sort_keys=True), flush=True)

    return {
        "scene_id": scene_id,
        "tag": tag,
        "row_label": f"{scene_id}\n{tag}",
        "rgb": rgb,
        "t0": t0,
        "t1": t1,
        "hero": hero,
        "event_maps": event_maps,
        "cm_maps": cm_maps,
        "methods": methods,
        "shape": [height, width],
    }


def _draw_gallery(blocks: list[dict], title: str, output: Path, *, kind: str) -> None:
    n = len(blocks)
    fig = plt.figure(figsize=(7.2, 1.45 * n + 0.45), dpi=200)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=9.2, color=INK, fontweight="bold", y=0.995)
    grid = fig.add_gridspec(n, 4, wspace=0.05, hspace=0.30, left=0.16, right=0.995, top=0.90, bottom=0.03)
    if kind == "events":
        headers = ("RGB", "DVS-Voltmeter", "v2e", "Ours (180, inv-warp)")
        maps_key = "event_maps"
    else:
        headers = ("RGB", "DVS + CM-IWE", "v2e + CM-IWE", "Ours + CM-IWE")
        maps_key = "cm_maps"
    for row, block in enumerate(blocks):
        ax = fig.add_subplot(grid[row, 0])
        _panel(ax, block["rgb"], xlabel=f"t={block['t0']:.2f}s")
        if row == 0:
            ax.set_title(headers[0], fontsize=7.2, color=INK, pad=2)
        ax.set_ylabel(block["row_label"], fontsize=6.2, color=INK)
        for col, (image, caption) in enumerate(block[maps_key], start=1):
            ax = fig.add_subplot(grid[row, col])
            _panel(ax, image, xlabel=caption)
            if row == 0:
                ax.set_title(headers[col], fontsize=7.2, color=INK, pad=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=READY)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--eval-width", type=int, default=480, help="Qualitative width. Table 1 stays at 160.")
    parser.add_argument("--hero", type=int, default=12)
    parser.add_argument("--prism-angle-deg", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--v2e-root", type=Path, default=Path("/data/fcr/vendor/v2e"))
    parser.add_argument("--v2e-python", type=Path, default=Path("/data/fcr/miniconda3/envs/v2e/bin/python"))
    parser.add_argument("--dvs-root", type=Path, default=Path("/data/fcr/code/event_vggt/Data_Part/DVS-Voltmeter"))
    parser.add_argument("--dvs-python", type=Path, default=Path("/data/fcr/miniconda3/envs/dvsvolt/bin/python"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/ami_next/figures"))
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

    blocks = [_bundle(scene_id, tag, args) for scene_id, tag in SCENES]
    out_e = args.out_dir / "paper_dl3dv_events"
    out_c = args.out_dir / "paper_dl3dv_cm"
    _draw_gallery(
        blocks,
        "DL3DV in-the-wild · DVS-Voltmeter / v2e / Ours (camera motion, not co-motion)",
        out_e,
        kind="events",
    )
    _draw_gallery(
        blocks,
        "DL3DV in-the-wild · identical CM-IWE velocity grid",
        out_c,
        kind="cm",
    )

    meta = {
        "protocol": "ami_paper_dl3dv_qualitative_v1",
        "note": "Qualitative native-ish width; frozen Table 1 remains 160x90 in dl3dv_downstream_v2.json",
        "eval_width": args.eval_width,
        "max_frames": args.max_frames,
        "hero": args.hero,
        "prism_angle_deg": args.prism_angle_deg,
        "fps_assumption": 30.0,
        "velocity_grid_px_per_s": {"vx": list(VX), "vy": list(VY), "n": len(VELOCITY_GRID)},
        "cm_max_events": CM_MAX_EVENTS,
        "blocks": [
            {
                "scene": b["scene_id"],
                "tag": b["tag"],
                "t0": b["t0"],
                "t1": b["t1"],
                "shape": b["shape"],
                "methods": b["methods"],
            }
            for b in blocks
        ],
    }
    (out_e.with_name(out_e.name + "_meta.json")).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (out_e.with_name(out_e.name + "_caption.txt")).write_text(
        """Figure. DL3DV in-the-wild transfer qualitative.

Five real RGB scenes from DL3DV-large5 (alley, field, yard, temple, statues).
Same-source columns: official DVS-Voltmeter, official v2e, Ours at 180 RPM
with per-event inverse warp. This is camera motion in the wild — not Blender
co-motion — so passive streams are not empty; the claim is transfer of active
optical excitation + inverse warp under real appearance and ego-motion.

CM companion uses the identical frozen translation velocity grid. Timestamps
are the pack's nominal 30 FPS assumption. Qualitative width here is larger
than the frozen Table 1 slice (160×90); do not cite these panels as Table 1
numbers.

Reproduce:
  MPLCONFIGDIR=/tmp/mpl-fcr PYTHONPATH=. \\
    python scripts/plot_paper_dl3dv_dvs_v2e_cm.py --device cuda:6
""",
        encoding="utf-8",
    )
    print({"events": str(out_e.with_suffix(".png")), "cm": str(out_c.with_suffix(".png"))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
