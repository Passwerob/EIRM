#!/usr/bin/env python3
"""Export large unlabeled PDFs: Blender / DL3DV / DSEC events and CM-IWE.

Each window writes two PDFs of the same native-resolution 1x4 grid:
  *_events.pdf  RGB | event maps (EIRM)
  *_cm.pdf      RGB | contrast-maximized IWE of those same events (EIRM on the warped counts)

No titles or numbers are burned into the panels. Sidecar JSON holds CM velocity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import imageio.v3 as imageio
import numpy as np
from PIL import Image

from ami_simulator.event_visualization import eirm_counts_from_events, eirm_from_counts, eirm_peak
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import _IMAGE_SUFFIXES, _read_timestamp_file, load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.types import CameraIntrinsics, EventStream
from scripts.evaluate_dl3dv_downstream import _ami_windows, _intrinsics as _dl3dv_intrinsics
from scripts.evaluate_dvs_v2e_ours import _read_npz, _slice
from scripts.evaluate_eveyes_v2_downstream import _intrinsics as _eveyes_intrinsics, _scene_path
from scripts.plot_paper_dvs_v2e_cm_coverage import _run_cm

EVEYES_ROOT = Path("data/eveyes_comotion_v2_ready")
DL3DV_ROOT = Path("data/dl3dv_large5")
DSEC_ROOT = Path("/data/fcr/data/DSEC/thun_00_a_small")
EVEYES_BASELINE = Path("results/ami_next/eveyes_v2_baseline_events")
DL3DV_BASELINE = Path("results/ami_next/dl3dv_baseline_events")
DL3DV_OURS = Path("results/ami_next/dl3dv_qualitative_windows")
CACHE = Path("results/ami_next/qualitative_pdf_cache")
OUT = Path("results/ami_next/figures/qualitative_pdfs")
GUTTER = 16
PAPER_BG = (255, 255, 255)
WARMUP = 10
PRISM_EVEYES = 0.5
PRISM_DL3DV = 2.0
PRISM_DSEC = 0.5


def _grid(panels: list[np.ndarray], scale: int = 1) -> np.ndarray:
    resized = []
    for panel in panels:
        image = panel[..., :3]
        if scale != 1:
            interp = cv2.INTER_AREA if image.mean() > 40 else cv2.INTER_NEAREST
            image = cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale), interpolation=interp)
        resized.append(image)
    height, width = resized[0].shape[:2]
    canvas = np.full((height, len(resized) * width + (len(resized) - 1) * GUTTER * scale, 3), PAPER_BG, dtype=np.uint8)
    for index, panel in enumerate(resized):
        left = index * (width + GUTTER * scale)
        canvas[:, left : left + width] = panel
    return canvas


def _save_pdf(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, "PDF", resolution=72.0)
    imageio.imwrite(path.with_suffix(".png"), image)


def _eirm_row(streams: list[EventStream], height: int, width: int) -> list[np.ndarray]:
    counts = [eirm_counts_from_events(stream, height, width) for stream in streams]
    peak = (max(eirm_peak(item[0]) for item in counts), max(eirm_peak(item[1]) for item in counts))
    return [eirm_from_counts(*item, peak=peak) for item in counts]


def _cm_row(streams: list[EventStream], height: int, width: int, tref: float) -> tuple[list[np.ndarray], list[dict]]:
    images = []
    meta = []
    counts = []
    for stream in streams:
        result = _run_cm(stream, (height, width), tref)
        image = np.asarray(result["image"])
        counts.append((image[1], image[0]))
        vx, vy = result["velocity"]
        meta.append({"vx": float(vx), "vy": float(vy), "contrast": float(result["contrast"]), "n_events": int(len(stream)), "subsampled": bool(result["subsampled"])})
    peak = (max(eirm_peak(item[0]) for item in counts), max(eirm_peak(item[1]) for item in counts))
    images = [eirm_from_counts(*item, peak=peak) for item in counts]
    return images, meta


def _write_pair(stem: Path, rgb: np.ndarray, streams: list[EventStream], tref: float, scale: int, extra: dict) -> None:
    event_pdf = stem.parent / f"{stem.name}_events.pdf"
    if event_pdf.exists() and event_pdf.with_name(stem.name + "_cm.pdf").exists():
        print(f"  skip existing {stem.name}", flush=True)
        return
    height, width = rgb.shape[:2]
    events = _eirm_row(streams, height, width)
    cms, cm_meta = _cm_row(streams, height, width, tref)
    _save_pdf(_grid([rgb, *events], scale=scale), stem.parent / f"{stem.name}_events.pdf")
    _save_pdf(_grid([rgb, *cms], scale=scale), stem.parent / f"{stem.name}_cm.pdf")
    (stem.parent / f"{stem.name}.json").write_text(json.dumps({"tref": tref, "shape": [height, width], "scale": scale, "cm": cm_meta, **extra}, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {stem.name}_events.pdf / _cm.pdf", flush=True)


def _cache_npz(path: Path, events: EventStream) -> EventStream:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=np.asarray(events.x), y=np.asarray(events.y), t=np.asarray(events.t), p=np.asarray(events.p))
    return events


def _load_or_ami(cache: Path, luminance, timestamps, prism, rpm: float, hero: int, device: str) -> EventStream:
    if cache.exists():
        return _read_npz(cache)
    start = max(0, hero - WARMUP)
    masks = np.ones((len(timestamps) - start, luminance.shape[1], luminance.shape[2]), dtype=bool)
    reference, _, _ = _ami_windows(luminance[start:], timestamps[start:], masks, prism, rpm, device=device)
    events = reference[hero - start]
    return _cache_npz(cache, events)


def _rgb_at(directory: Path, index: int) -> np.ndarray:
    path = directory / f"{index:06d}.png"
    if not path.exists():
        paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
        path = paths[index]
    return np.asarray(imageio.imread(path)[..., :3])


def _resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def export_blender(device: str) -> list[Path]:
    jobs = [
        ("test", "Scene_005", "object_only", 60),
        ("test", "Scene_005", "object_only", 120),
        ("test", "Scene_005", "object_only", 240),
        ("test", "Scene_005", "camera_only", 60),
        ("test", "Scene_005", "camera_only", 120),
        ("test", "Scene_005", "camera_only", 240),
        ("test", "Scene_005", "co_motion", 90),
        ("test", "Scene_005", "co_motion", 150),
        ("test", "Scene_005", "co_motion", 240),
        ("test", "Scene_005", "drive_stop_drive", 60),
        ("test", "Scene_005", "drive_stop_drive", 195),
        ("test", "Scene_005", "drive_stop_drive", 400),
        ("train", "Scene_002", "object_only", 120),
        ("train", "Scene_002", "co_motion", 150),
        ("train", "Scene_002", "camera_only", 120),
        ("valid", "Scene_004", "object_only", 120),
        ("valid", "Scene_004", "co_motion", 150),
        ("valid", "Scene_004", "camera_only", 120),
    ]
    out_dir = OUT / "blender"
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = []
    for split, scene_name, regime, hero in jobs:
        scene = _scene_path(EVEYES_ROOT, split, scene_name, regime)
        timestamps = _read_timestamp_file(scene / "timestamps.csv")
        if hero >= len(timestamps) - 1:
            print(f"skip {scene} hero {hero}", flush=True)
            continue
        print(f"blender {split}/{scene_name}/{regime} f{hero}", flush=True)
        start = max(0, hero - WARMUP)
        frames, times = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=hero + 2)
        times = times[: len(frames)]
        luminance = rgb_to_linear_luminance(frames)
        rgb = np.clip(frames[hero], 0, 255).astype(np.uint8)
        prism = RotatingWedgePrism(_eveyes_intrinsics(scene), PRISM_EVEYES)
        gears = {}
        for rpm in (0.0, 180.0, 360.0):
            cache = CACHE / f"eveyes_{split}_{scene_name}_{regime}_f{hero}_rpm{int(rpm)}.npz"
            gears[rpm] = _load_or_ami(cache, luminance, times, prism, rpm, hero, device)
        t0, t1 = float(times[hero]), float(times[hero + 1])
        stem = out_dir / f"{split}_{scene_name}_{regime}_f{hero:04d}_gears"
        _write_pair(stem, rgb, [gears[0.0], gears[180.0], gears[360.0]], t1, 1, {"dataset": "EVEYES-Blender", "scene": f"{split}/{scene_name}/{regime}", "hero": hero, "t0": t0, "t1": t1, "columns": ["RGB", "0 RPM", "180 RPM", "360 RPM"]})
        stems.append(stem)

        template = {
            ("test", "Scene_005", "co_motion"): "test_Scene_005_f480_w960_{m}.npz",
            ("test", "Scene_005", "object_only"): "test_Scene_005_object_only_f480_w960_{m}.npz",
            ("test", "Scene_005", "camera_only"): "test_Scene_005_camera_only_f480_w960_{m}.npz",
            ("train", "Scene_002", "co_motion"): "train_Scene_002_f480_w960_{m}.npz",
            ("valid", "Scene_004", "co_motion"): "valid_Scene_004_f480_w960_{m}.npz",
        }.get((split, scene_name, regime))
        if template:
            dvs = _slice(_read_npz(EVEYES_BASELINE / template.format(m="dvs_voltmeter")), t0, t1)
            v2e = _slice(_read_npz(EVEYES_BASELINE / template.format(m="v2e")), t0, t1)
            stem_b = out_dir / f"{split}_{scene_name}_{regime}_f{hero:04d}_dvs_v2e_ours"
            _write_pair(stem_b, rgb, [dvs, v2e, gears[180.0]], t1, 1, {"dataset": "EVEYES-Blender", "scene": f"{split}/{scene_name}/{regime}", "hero": hero, "t0": t0, "t1": t1, "columns": ["RGB", "DVS-Voltmeter", "v2e", "Ours 180"]})
            stems.append(stem_b)
    return stems


def export_dl3dv(device: str) -> list[Path]:
    scenes = (
        ("screen-000083", "alley"),
        ("screen-000087", "field"),
        ("screen-000089", "yard"),
        ("screen-000095", "temple"),
        ("screen-000108", "statues"),
    )
    heroes = (8, 12, 16)
    out_dir = OUT / "dl3dv"
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = []
    width = 480
    for scene_id, tag in scenes:
        scene = DL3DV_ROOT / scene_id
        frames, timestamps = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=24)
        timestamps = timestamps[: len(frames)]
        stacked = np.stack([_resize_rgb(np.asarray(frame)[..., :3], width, int(round(frame.shape[0] * width / frame.shape[1]))) for frame in frames])
        luminance = rgb_to_linear_luminance(stacked)
        height = stacked.shape[1]
        prism = RotatingWedgePrism(_dl3dv_intrinsics(scene).resized(width, height), PRISM_DL3DV)
        for hero in heroes:
            if hero >= len(timestamps) - 1:
                continue
            print(f"dl3dv {scene_id} {tag} f{hero}", flush=True)
            rgb = stacked[hero].astype(np.uint8)
            t0, t1 = float(timestamps[hero]), float(timestamps[hero + 1])
            ours_cache = DL3DV_OURS / f"{scene_id}_f{hero}_w{width}_rpm180_reference.npz"
            ours = _load_or_ami(ours_cache, luminance, timestamps, prism, 180.0, hero, device)
            ours = _slice(ours, t0 - 1e-6, t1 + 1e-6)
            dvs_path = DL3DV_BASELINE / f"{scene_id}_f24_w{width}_dvs_voltmeter.npz"
            v2e_path = DL3DV_BASELINE / f"{scene_id}_f24_w{width}_v2e.npz"
            streams = [ours]
            columns = ["RGB", "Ours 180"]
            if dvs_path.exists() and v2e_path.exists():
                streams = [_slice(_read_npz(dvs_path), t0, t1), _slice(_read_npz(v2e_path), t0, t1), ours]
                columns = ["RGB", "DVS-Voltmeter", "v2e", "Ours 180"]
            stem = out_dir / f"{scene_id}_{tag}_f{hero:04d}"
            _write_pair(stem, rgb, streams, t1, 2, {"dataset": "DL3DV-large5", "scene": scene_id, "tag": tag, "hero": hero, "t0": t0, "t1": t1, "columns": columns})
            stems.append(stem)
    return stems


def _read_dsec_chunk(path: Path) -> EventStream:
    with h5py.File(path, "r") as handle:
        group = handle["events"]
        x = np.asarray(group["x"])
        y = np.asarray(group["y"])
        t = np.asarray(group["t"], dtype=np.float64)
        p = np.asarray(group["p"])
    if t.size and t.max() > 1e6:
        t = t * 1e-6
    if p.size and np.all(np.isin(p, (0, 1))):
        p = p * 2 - 1
    return EventStream(x, y, t, p.astype(np.int8, copy=False))


def export_dsec(device: str) -> list[Path]:
    root = DSEC_ROOT
    image_dir = root / "images"
    paths = sorted(image_dir.glob("*.png"))
    raw = np.stack([imageio.imread(path)[..., :3] for path in paths])
    width, height = 640, 480
    frames = np.stack([_resize_rgb(frame, width, height) for frame in raw])
    source_times = np.loadtxt(root / "thun_00_a_image_timestamps.txt")[: len(frames)] * 1e-6
    source_times = source_times - source_times[0]
    luminance = rgb_to_linear_luminance(frames.astype(np.float32))
    intrinsics = CameraIntrinsics(width, height, 569.7632987676102, 569.7632987676102, 335.0999870300293, 221.23667526245117)
    prism = RotatingWedgePrism(intrinsics, PRISM_DSEC)
    out_dir = OUT / "dsec"
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = []
    heroes = [4, 8, 12, 16, 20, 24, 28]
    for hero in heroes:
        if hero >= len(frames) - 1:
            continue
        print(f"dsec thun_00_a f{hero}", flush=True)
        rgb = frames[hero].astype(np.uint8)
        t0, t1 = float(source_times[hero]), float(source_times[hero + 1])
        native_path = root / "events" / f"{hero:06d}.h5"
        native = _read_dsec_chunk(native_path) if native_path.exists() else EventStream.empty()
        ami = {}
        for rpm in (0.0, 180.0, 360.0):
            cache = CACHE / f"dsec_thun00a_f{hero}_rpm{int(rpm)}.npz"
            ami[rpm] = _load_or_ami(cache, luminance, source_times, prism, rpm, hero, device)
        stem_g = out_dir / f"thun_00_a_f{hero:04d}_gears"
        _write_pair(stem_g, rgb, [ami[0.0], ami[180.0], ami[360.0]], t1, 2, {"dataset": "DSEC", "scene": "thun_00_a", "hero": hero, "t0": t0, "t1": t1, "columns": ["RGB", "0 RPM", "180 RPM", "360 RPM"]})
        stems.append(stem_g)
        stem_n = out_dir / f"thun_00_a_f{hero:04d}_native_vs_ours"
        _write_pair(stem_n, rgb, [native, ami[180.0], ami[360.0]], t1, 2, {"dataset": "DSEC", "scene": "thun_00_a", "hero": hero, "t0": t0, "t1": t1, "columns": ["RGB", "DSEC native events", "Ours 180", "Ours 360"]})
        stems.append(stem_n)
    return stems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--skip-blender", action="store_true")
    parser.add_argument("--skip-dl3dv", action="store_true")
    parser.add_argument("--skip-dsec", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    stems: list[Path] = []
    if not args.skip_blender:
        stems.extend(export_blender(args.device))
    if not args.skip_dl3dv:
        stems.extend(export_dl3dv(args.device))
    if not args.skip_dsec:
        stems.extend(export_dsec(args.device))
    index = {"n": len(stems), "each": "two PDFs: *_events.pdf (EIRM events) and *_cm.pdf (CM-IWE of the same events, EIRM on warped counts)", "stems": [str(path) for path in stems]}
    (OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "n_stems": len(stems)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
