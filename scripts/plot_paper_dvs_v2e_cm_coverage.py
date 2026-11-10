#!/usr/bin/env python3
"""Paper coverage galleries: many motion regimes × many scenes × DVS/v2e/Ours(+CM).

Produces:
  paper_motion_regimes_events  — Scene_005 · 4 motion forms · polarity events
  paper_motion_regimes_cm      — same windows · identical CM-IWE grid
  paper_multiscene_dvs_v2e_cm  — Scene_002/004/005 · object_only + co_motion

Reuses full-sequence DVS/v2e caches when present; otherwise builds a short
warmup clip cache (documented in meta). AMI 180 inverse-warped windows are
cached under eveyes_event_qualitative_windows. Does not overwrite Table 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ami_simulator.contrast_maximization import contrast_maximization
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import EventStream
from scripts.evaluate_dvs_v2e_ours import _read_npz, _slice
from scripts.evaluate_eveyes_v2_downstream import _scene_path
from scripts.plot_eveyes_event_qualitative import CACHE, CACHE_WINDOWS, READY, _ami_window, _events_to_map

INK = "#2C3E50"
CLIP_CACHE = Path("results/ami_next/eveyes_qualitative_clip_events")
# Slightly coarser than the first Scene_005 exhibit — enough for qualitative CM,
# fast enough for dense co-motion streams.
VX = (-160.0, -80.0, -40.0, -20.0, 0.0, 20.0, 40.0, 80.0, 160.0)
VY = (-80.0, -40.0, -20.0, 0.0, 20.0, 40.0, 80.0)
VELOCITY_GRID = tuple((vx, vy) for vx in VX for vy in VY)
CM_MAX_EVENTS = 80_000

DVS_ROOT = Path("/data/fcr/code/event_vggt/Data_Part/DVS-Voltmeter")
DVS_PYTHON = Path("/data/fcr/miniconda3/envs/dvsvolt/bin/python")
V2E_ROOT = Path("/data/fcr/vendor/v2e")
V2E_PYTHON = Path("/data/fcr/miniconda3/envs/v2e/bin/python")

# Full-sequence baseline stem templates under CACHE.
FULL_STEM = {
    ("test", "Scene_005", "object_only"): "test_Scene_005_object_only_f480_w960_{method}",
    ("test", "Scene_005", "camera_only"): "test_Scene_005_camera_only_f480_w960_{method}",
    ("test", "Scene_005", "co_motion"): "test_Scene_005_f480_w960_{method}",
    ("train", "Scene_002", "co_motion"): "train_Scene_002_f480_w960_{method}",
    ("valid", "Scene_004", "co_motion"): "valid_Scene_004_f480_w960_{method}",
}

REGIME_LABEL = {
    "object_only": "object only · camera frozen",
    "camera_only": "camera only · ego sweep",
    "co_motion": "co-motion · relative lock",
    "drive_stop_drive": "drive–stop–drive · stop phase",
}

# Scene_005 motion-form gallery (hero frames chosen for readable claims).
MOTION_REGIMES = (
    {"split": "test", "scene": "Scene_005", "regime": "object_only", "hero": 120, "passive_note": "car only", "ours_note": "city + car"},
    {"split": "test", "scene": "Scene_005", "regime": "camera_only", "hero": 120, "passive_note": "street ego", "ours_note": "street + car"},
    {"split": "test", "scene": "Scene_005", "regime": "co_motion", "hero": 150, "passive_note": "ego clutter", "ours_note": "structure + car"},
    {"split": "test", "scene": "Scene_005", "regime": "drive_stop_drive", "hero": 195, "passive_note": "near silent", "ours_note": "paints while stopped"},
)

# Multi-scene coverage: train / valid / held-out test.
MULTI_SCENE = (
    {"split": "train", "scene": "Scene_002", "regime": "object_only", "hero": 120, "passive_note": "car only", "ours_note": "city + car"},
    {"split": "valid", "scene": "Scene_004", "regime": "object_only", "hero": 120, "passive_note": "car only", "ours_note": "city + car"},
    {"split": "test", "scene": "Scene_005", "regime": "object_only", "hero": 120, "passive_note": "car only", "ours_note": "city + car"},
    {"split": "train", "scene": "Scene_002", "regime": "co_motion", "hero": 150, "passive_note": "ego clutter", "ours_note": "structure + car"},
    {"split": "valid", "scene": "Scene_004", "regime": "co_motion", "hero": 150, "passive_note": "ego clutter", "ours_note": "structure + car"},
    {"split": "test", "scene": "Scene_005", "regime": "co_motion", "hero": 150, "passive_note": "ego clutter", "ours_note": "structure + car"},
)

AMI_ALIASES = (
    "{parent}_{name}_f{hero}_rpm180_reference_n1.npz",
    "{parent}_{name}_f{hero}_rpm180.npz",
)


def _timestamps(scene: Path) -> np.ndarray:
    rows = list(csv.DictReader((scene / "timestamps.csv").open()))
    return np.asarray([float(row["timestamp_seconds"]) for row in rows], dtype=np.float64)


def _cm_gray(image_2hw: np.ndarray) -> np.ndarray:
    counts = np.asarray(image_2hw[0] + image_2hw[1], dtype=np.float64)
    positive = counts[counts > 0]
    peak = float(np.percentile(positive, 99.5)) if len(positive) else 1.0
    peak = max(peak, 1.0)
    level = np.clip(np.log1p(counts) / np.log1p(peak), 0.0, 1.0)
    gray = 255.0 - 235.0 * level
    return np.repeat(gray[..., None], 3, axis=-1).astype(np.uint8)


def _subsample_events(events: EventStream, max_events: int) -> tuple[EventStream, bool]:
    n = int(len(events.x))
    if n <= max_events:
        return events, False
    idx = np.linspace(0, n - 1, max_events).astype(np.int64)
    return (
        EventStream(events.x[idx], events.y[idx], events.t[idx], events.p[idx]),
        True,
    )


def _run_cm(events: EventStream, shape: tuple[int, int], reference_time: float) -> dict[str, object]:
    """CM for qualitative viz; subsample densest streams so galleries finish."""
    slim, subsampled = _subsample_events(events, CM_MAX_EVENTS)
    result = contrast_maximization(slim, shape, VELOCITY_GRID, reference_time=reference_time)
    result["subsampled"] = subsampled
    result["n_cm_events"] = int(len(slim.x))
    return result


def _panel(axis, image: np.ndarray, xlabel: str | None = None) -> None:
    axis.imshow(image)
    axis.set_xticks([])
    axis.set_yticks([])
    if xlabel:
        axis.set_xlabel(xlabel, fontsize=6.6, color=INK, labelpad=1.5)
    for spine in axis.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("0.4")


def _events_to_map_adaptive(events: EventStream, height: int, width: int) -> np.ndarray:
    """Sparse panels stay bright; dense Ours still reads structure."""
    if not len(events.x):
        return np.zeros((height, width, 3), dtype=np.uint8)
    # Reuse fixed map then optionally lift sparse panels.
    base = _events_to_map(events, height, width)
    n = int(len(events.x))
    if n >= 8000:
        return base
    # Boost visibility for sparse DVS/v2e without inventing events.
    boost = base.astype(np.float32)
    mask = boost.max(axis=2) > 0
    boost[mask] = np.clip(boost[mask] * 1.35 + 25.0, 0, 255)
    return boost.astype(np.uint8)


def _resolve_ours(scene: Path, hero: int, device: str) -> EventStream:
    parent, name = scene.parent.name, scene.name
    for template in AMI_ALIASES:
        path = CACHE_WINDOWS / template.format(parent=parent, name=name, hero=hero)
        if path.exists():
            return _read_npz(path)
    return _ami_window(scene, hero, 180.0, device, plane="reference", n_windows=1)


def _run_external(method: str, luminance: np.ndarray, timestamps: np.ndarray, out: Path) -> EventStream:
    runner = Path(__file__).with_name("run_dvs_voltmeter_array.py" if method == "dvs_voltmeter" else "run_v2e_array.py")
    python = DVS_PYTHON if method == "dvs_voltmeter" else V2E_PYTHON
    root = DVS_ROOT if method == "dvs_voltmeter" else V2E_ROOT
    flag = "--dvs-root" if method == "dvs_voltmeter" else "--v2e-root"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"qual_{method}_") as temporary:
        inp = Path(temporary) / "input.npz"
        tmp_out = Path(temporary) / "out.npz"
        np.savez_compressed(inp, frames=luminance.astype(np.float32), timestamps=timestamps.astype(np.float64))
        cmd = [str(python), str(runner), "--input", str(inp), "--output", str(tmp_out), flag, str(root), "--seed", "17"]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        events = _read_npz(tmp_out)
    np.savez_compressed(out, x=np.asarray(events.x), y=np.asarray(events.y), t=np.asarray(events.t), p=np.asarray(events.p))
    return events


def _slice_npz_window(path: Path, t0: float, t1: float) -> EventStream:
    """Extract one time window without materializing a multi-hundred-M event file."""
    with np.load(path, mmap_mode="r") as data:
        times = np.asarray(data["t"])
        left = int(np.searchsorted(times, t0, side="right"))
        right = int(np.searchsorted(times, t1, side="right"))
        if right <= left:
            return EventStream.empty()
        return EventStream(
            np.asarray(data["x"][left:right]),
            np.asarray(data["y"][left:right]),
            np.asarray(data["t"][left:right]),
            np.asarray(data["p"][left:right]),
        )


def _window_cache_path(spec: dict, method: str, hero: int) -> Path:
    return CLIP_CACHE / f"window_{spec['split']}_{spec['scene']}_{spec['regime']}_f{hero}_{method}.npz"


def _baseline_events(
    spec: dict,
    scene: Path,
    method: str,
    timestamps: np.ndarray,
    hero: int,
    *,
    clip_warmup: int,
) -> tuple[EventStream, str]:
    t0 = float(timestamps[hero])
    t1 = float(timestamps[min(hero + 1, len(timestamps) - 1)])
    window_cache = _window_cache_path(spec, method, hero)
    if window_cache.exists():
        return _read_npz(window_cache), f"window:{window_cache.name}"

    key = (spec["split"], spec["scene"], spec["regime"])
    stem = FULL_STEM.get(key)
    if stem is not None:
        path = CACHE / f"{stem.format(method=method)}.npz"
        if path.exists():
            print(f"mmap-slice {path.name} [{t0:.4f},{t1:.4f}]", flush=True)
            events = _slice_npz_window(path, t0, t1)
            window_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                window_cache,
                x=np.asarray(events.x),
                y=np.asarray(events.y),
                t=np.asarray(events.t),
                p=np.asarray(events.p),
            )
            return events, f"full-window:{path.name}"

    clip = CLIP_CACHE / f"{spec['split']}_{spec['scene']}_{spec['regime']}_f{hero}_w{clip_warmup}_{method}.npz"
    if clip.exists():
        return _slice(_read_npz(clip), t0, t1), f"clip:{clip.name}"
    start = max(0, hero - clip_warmup)
    frames, ts = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=hero + 2)
    ts = ts[: len(frames)]
    lum = rgb_to_linear_luminance(frames[start : hero + 2])
    times = ts[start : hero + 2]
    print(f"building clip baseline {method} {key} frames {start}:{hero + 1}", flush=True)
    all_events = _run_external(method, lum, times, clip)
    events = _slice(all_events, t0, t1)
    np.savez_compressed(
        window_cache,
        x=np.asarray(events.x),
        y=np.asarray(events.y),
        t=np.asarray(events.t),
        p=np.asarray(events.p),
    )
    return events, f"clip:{clip.name}"


def _bundle(spec: dict, dataset_root: Path, device: str, clip_warmup: int) -> dict:
    scene = _scene_path(dataset_root, spec["split"], spec["scene"], spec["regime"])
    timestamps = _timestamps(scene)
    hero = int(spec["hero"])
    t0 = float(timestamps[hero])
    t1 = float(timestamps[min(hero + 1, len(timestamps) - 1)])
    rgb = np.asarray(Image.open(scene / "images" / f"{hero:06d}.png").convert("RGB"))
    height, width = rgb.shape[:2]

    dvs, dvs_src = _baseline_events(spec, scene, "dvs_voltmeter", timestamps, hero, clip_warmup=clip_warmup)
    v2e, v2e_src = _baseline_events(spec, scene, "v2e", timestamps, hero, clip_warmup=clip_warmup)
    ours_all = _resolve_ours(scene, hero, device)
    ours = _slice(ours_all, t0 - 1e-6, t1 + 1e-6)

    methods = []
    event_maps = []
    cm_maps = []
    for name, events, note, src in (
        ("DVS-Voltmeter", dvs, spec["passive_note"], dvs_src),
        ("v2e", v2e, spec["passive_note"], v2e_src),
        ("Ours (180, inv-warp)", ours, spec["ours_note"], "ami180"),
    ):
        cm = _run_cm(events, (height, width), t1)
        vx, vy = cm["velocity"]  # type: ignore[misc]
        event_maps.append((_events_to_map_adaptive(events, height, width), f"{len(events.x):,} · {note}"))
        cm_maps.append((_cm_gray(np.asarray(cm["image"])), f"CM ({vx:.0f},{vy:.0f})"))
        methods.append(
            {
                "method": name,
                "n_events": int(len(events.x)),
                "source": src,
                "cm_velocity": [float(vx), float(vy)],
                "cm_contrast": float(cm["contrast"]),
                "cm_subsampled": bool(cm["subsampled"]),
                "n_cm_events": int(cm["n_cm_events"]),
            }
        )
        print(
            json.dumps(
                {
                    "split": spec["split"],
                    "scene": spec["scene"],
                    "regime": spec["regime"],
                    "method": name,
                    "n": int(len(events.x)),
                    "src": src,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    return {
        "spec": spec,
        "rgb": rgb,
        "t0": t0,
        "t1": t1,
        "hero": hero,
        "label": REGIME_LABEL[spec["regime"]],
        "scene_tag": f"{spec['scene']} · {spec['split']}",
        "event_maps": event_maps,
        "cm_maps": cm_maps,
        "methods": methods,
    }


def _save_meta(path: Path, kind: str, blocks: list[dict]) -> None:
    meta = {
        "protocol": f"ami_paper_{kind}_v1",
        "velocity_grid_px_per_s": {"vx": list(VX), "vy": list(VY), "n": len(VELOCITY_GRID)},
        "window": "inter-frame [t_hero, t_hero+1]",
        "blocks": [
            {
                **{k: b["spec"][k] for k in ("split", "scene", "regime", "hero")},
                "t0": b["t0"],
                "t1": b["t1"],
                "methods": b["methods"],
            }
            for b in blocks
        ],
    }
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _draw_events_gallery(blocks: list[dict], title: str, output: Path, *, row_key: str) -> None:
    n = len(blocks)
    fig_h = 1.55 * n + 0.45
    fig = plt.figure(figsize=(7.2, fig_h), dpi=200)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=9.2, color=INK, fontweight="bold", y=0.995)
    grid = fig.add_gridspec(n, 4, wspace=0.05, hspace=0.32, left=0.16, right=0.995, top=0.90, bottom=0.03)
    headers = ("RGB", "DVS-Voltmeter", "v2e", "Ours (180, inv-warp)")
    for row, block in enumerate(blocks):
        ax = fig.add_subplot(grid[row, 0])
        _panel(ax, block["rgb"], xlabel=f"t={block['t0']:.2f}s")
        if row == 0:
            ax.set_title(headers[0], fontsize=7.2, color=INK, pad=2)
        ax.set_ylabel(block[row_key], fontsize=6.4, color=INK)
        for col, (image, caption) in enumerate(block["event_maps"], start=1):
            ax = fig.add_subplot(grid[row, col])
            _panel(ax, image, xlabel=caption)
            if row == 0:
                ax.set_title(headers[col], fontsize=7.2, color=INK, pad=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=220)
    plt.close(fig)


def _draw_cm_gallery(blocks: list[dict], title: str, output: Path, *, row_key: str) -> None:
    n = len(blocks)
    fig_h = 1.55 * n + 0.45
    fig = plt.figure(figsize=(7.2, fig_h), dpi=200)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=9.2, color=INK, fontweight="bold", y=0.995)
    grid = fig.add_gridspec(n, 4, wspace=0.05, hspace=0.32, left=0.16, right=0.995, top=0.90, bottom=0.03)
    headers = ("RGB", "DVS + CM-IWE", "v2e + CM-IWE", "Ours + CM-IWE")
    for row, block in enumerate(blocks):
        ax = fig.add_subplot(grid[row, 0])
        _panel(ax, block["rgb"], xlabel=f"t={block['t0']:.2f}s")
        if row == 0:
            ax.set_title(headers[0], fontsize=7.2, color=INK, pad=2)
        ax.set_ylabel(block[row_key], fontsize=6.4, color=INK)
        for col, (image, caption) in enumerate(block["cm_maps"], start=1):
            ax = fig.add_subplot(grid[row, col])
            _panel(ax, image, xlabel=caption)
            if row == 0:
                ax.set_title(headers[col], fontsize=7.2, color=INK, pad=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=220)
    plt.close(fig)


def _draw_multiscene(blocks: list[dict], events_out: Path, cm_out: Path) -> None:
    """Six clear rows: object_only×3 scenes, then co_motion×3 scenes."""
    ordered = [b for b in blocks if b["spec"]["regime"] == "object_only"] + [
        b for b in blocks if b["spec"]["regime"] == "co_motion"
    ]
    for block in ordered:
        regime = REGIME_LABEL[block["spec"]["regime"]]
        block["row_label"] = f"{block['spec']['scene']}\n{regime}"
    _draw_events_gallery(
        ordered,
        "Scene_002 / 004 / 005 · object_only + co-motion · DVS / v2e / Ours",
        events_out,
        row_key="row_label",
    )
    _draw_cm_gallery(
        ordered,
        "Scene_002 / 004 / 005 · object_only + co-motion · identical CM-IWE",
        cm_out,
        row_key="row_label",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=READY)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--clip-warmup", type=int, default=48)
    parser.add_argument("--out-dir", type=Path, default=Path("results/ami_next/figures"))
    parser.add_argument("--skip-multiscene", action="store_true")
    parser.add_argument("--skip-regimes", action="store_true")
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

    written: dict[str, str] = {}

    if not args.skip_regimes:
        regime_blocks = [_bundle(spec, args.dataset_root, args.device, args.clip_warmup) for spec in MOTION_REGIMES]
        # Prefer regime label on the y-axis for the motion-form gallery.
        for block in regime_blocks:
            block["row_label"] = block["label"]
        out_e = args.out_dir / "paper_motion_regimes_events"
        out_c = args.out_dir / "paper_motion_regimes_cm"
        _draw_events_gallery(
            regime_blocks,
            "Scene_005 · four motion forms · DVS-Voltmeter / v2e / Ours",
            out_e,
            row_key="row_label",
        )
        _draw_cm_gallery(
            regime_blocks,
            "Scene_005 · four motion forms · identical CM-IWE velocity grid",
            out_c,
            row_key="row_label",
        )
        _save_meta(out_e.with_name(out_e.name + "_meta.json"), "motion_regimes", regime_blocks)
        (out_e.with_name(out_e.name + "_caption.txt")).write_text(
            """Figure. Four motion forms on held-out Scene_005.

Rows: object_only (camera frozen), camera_only (ego sweep), co_motion
(relative lock), drive_stop_drive (stop phase at t≈3.25s). Columns: same RGB,
official DVS-Voltmeter, official v2e, Ours (180 RPM, inverse-warped).

Companion CM figure uses the identical frozen translation velocity grid for
every source (CM is a representation, not a fourth generator).

Reproduce:
  MPLCONFIGDIR=/tmp/mpl-fcr PYTHONPATH=. \\
    python scripts/plot_paper_dvs_v2e_cm_coverage.py --device cuda:6
""",
            encoding="utf-8",
        )
        written["regimes_events"] = str(out_e.with_suffix(".png"))
        written["regimes_cm"] = str(out_c.with_suffix(".png"))

    if not args.skip_multiscene:
        multi_blocks = [_bundle(spec, args.dataset_root, args.device, args.clip_warmup) for spec in MULTI_SCENE]
        out_e = args.out_dir / "paper_multiscene_events"
        out_c = args.out_dir / "paper_multiscene_cm"
        _draw_multiscene(multi_blocks, out_e, out_c)
        _save_meta(out_e.with_name(out_e.name + "_meta.json"), "multiscene", multi_blocks)
        (out_e.with_name(out_e.name + "_caption.txt")).write_text(
            """Figure. Multi-scene coverage under the same DVS / v2e / Ours protocol.

Scenes: Scene_002 (train), Scene_004 (valid), Scene_005 (held-out test).
Top three rows: object_only (camera frozen). Bottom three: co-motion
(relative lock). Companion CM figure uses the identical velocity grid.

Reproduce:
  MPLCONFIGDIR=/tmp/mpl-fcr PYTHONPATH=. \\
    python scripts/plot_paper_dvs_v2e_cm_coverage.py --device cuda:6
""",
            encoding="utf-8",
        )
        written["multiscene_events"] = str(out_e.with_suffix(".png"))
        written["multiscene_cm"] = str(out_c.with_suffix(".png"))

    print(json.dumps(written, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
