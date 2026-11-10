#!/usr/bin/env python3
"""Render an original-resolution EVEYES co-motion event comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio.v3 as imageio
import numpy as np
import torch

from ami_simulator.events import ThresholdEventSimulator
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance
from ami_simulator.rotation import ConstantRotation
from ami_simulator.types import CameraIntrinsics, EventStream
from ami_simulator.warp import PrismWarper, compensate_events


def _read_npz(path: Path) -> EventStream:
    with np.load(path) as data:
        return EventStream(data["x"], data["y"], data["t"], data["p"]).sorted_by_time()


def _slice(events: EventStream, start: float, end: float) -> EventStream:
    keep = (events.t > start) & (events.t <= end)
    return EventStream(events.x[keep], events.y[keep], events.t[keep], events.p[keep])


def _event_image(events: EventStream, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    output = np.zeros((height, width, 3), dtype=np.uint8)
    if len(events):
        valid = (events.x >= 0) & (events.x < width) & (events.y >= 0) & (events.y < height)
        positive = valid & (events.p > 0)
        negative = valid & (events.p < 0)
        output[events.y[positive], events.x[positive]] = (40, 70, 255)
        output[events.y[negative], events.x[negative]] = (255, 220, 40)
    return output


def _label(image: np.ndarray, title: str, detail: str) -> np.ndarray:
    output = image.copy()
    scale = max(output.shape[1] / 960.0, 1.0)
    top = int(round(28 * scale))
    cv2.rectangle(output, (0, 0), (output.shape[1], top), (8, 8, 8), thickness=-1)
    cv2.putText(output, title, (int(8 * scale), int(20 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (245, 245, 245), max(1, int(round(scale))), cv2.LINE_AA)
    cv2.putText(output, detail, (int(8 * scale), output.shape[0] - int(10 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, (235, 235, 235), max(1, int(round(scale))), cv2.LINE_AA)
    return output


def _external(method: str, luminance: np.ndarray, timestamps: np.ndarray, args: argparse.Namespace, root: Path) -> EventStream:
    input_path, output_path = root / "frames.npz", root / f"{method}.npz"
    np.savez_compressed(input_path, frames=luminance.astype(np.float32), timestamps=timestamps.astype(np.float64))
    if method == "DVS-Voltmeter":
        command = [str(args.dvs_python), str(Path(__file__).with_name("run_dvs_voltmeter_array.py")), "--input", str(input_path), "--output", str(output_path), "--dvs-root", str(args.dvs_root), "--seed", str(args.seed)]
    else:
        command = [str(args.v2e_python), str(Path(__file__).with_name("run_v2e_array.py")), "--input", str(input_path), "--output", str(output_path), "--v2e-root", str(args.v2e_root), "--seed", str(args.seed)]
    subprocess.run(command, check=True)
    return _read_npz(output_path)


def _ours(luminance: np.ndarray, timestamps: np.ndarray, intrinsics: CameraIntrinsics, rpm: float, threshold: float) -> EventStream:
    prism = RotatingWedgePrism(intrinsics, 0.5)
    rotation = ConstantRotation(rpm)
    phase_step = np.deg2rad(7.5)
    phase = 0.0
    chunks: list[EventStream] = []
    for index, (start, end) in enumerate(zip(timestamps[:-1], timestamps[1:])):
        delta = abs(rotation.omega_rad_s * float(end - start))
        subdivisions = max(1, int(np.ceil(delta / phase_step)))
        pair_times = np.linspace(float(start), float(end), subdivisions + 1, dtype=np.float64)
        fractions = (pair_times - float(start)) / max(float(end - start), 1e-12)
        values = luminance[index][None] * (1.0 - fractions[:, None, None]) + luminance[index + 1][None] * fractions[:, None, None]
        warper = PrismWarper(prism, rotation, height=luminance.shape[1], width=luminance.shape[2], reference_time=float(start), theta0=phase, padding_mode="border")
        with torch.inference_mode():
            sensor, _ = warper.warp(torch.from_numpy(values.astype(np.float32)).unsqueeze(1), pair_times)
        raw = ThresholdEventSimulator(threshold).simulate(sensor[:, 0].numpy(), pair_times)
        if len(raw):
            mapped = compensate_events(raw, prism, rotation, theta0=phase, reference_time=float(start))
            chunks.append(EventStream(np.rint(mapped.extras["x_reference"]), np.rint(mapped.extras["y_reference"]), mapped.t, mapped.p))
        phase = float(np.remainder(phase + rotation.omega_rad_s * float(end - start), 2.0 * np.pi))
    if not chunks:
        return EventStream(np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float64), np.empty(0, np.int8))
    return EventStream(np.concatenate([item.x for item in chunks]), np.concatenate([item.y for item in chunks]), np.concatenate([item.t for item in chunks]), np.concatenate([item.p for item in chunks])).sorted_by_time()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("data/eveyes_comotion_ready/test/Scene_003"))
    parser.add_argument("--output", type=Path, default=Path("results/eveyes_comotion_comparison.png"))
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--rpm", type=float, default=180.0)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--dvs-root", type=Path, default=Path("/data/fcr/code/event_vggt/Data_Part/DVS-Voltmeter"))
    parser.add_argument("--dvs-python", type=Path, default=Path("/data/fcr/miniconda3/envs/dvsvolt/bin/python"))
    parser.add_argument("--v2e-root", type=Path, default=Path("/data/fcr/vendor/v2e"))
    parser.add_argument("--v2e-python", type=Path, default=Path("/data/fcr/miniconda3/envs/v2e/bin/python"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    frames, timestamps = load_rgb_sequence(args.scene / "images", args.scene / "timestamps.csv", timestamp_unit="s", max_frames=args.max_frames)
    luminance = rgb_to_linear_luminance(frames)
    calibration = json.loads((args.scene / "camera_intrinsics.json").read_text(encoding="utf-8"))
    intrinsics = CameraIntrinsics(frames.shape[2], frames.shape[1], float(calibration["fx"]), float(calibration["fy"]), float(calibration["cx"]), float(calibration["cy"]))
    with tempfile.TemporaryDirectory(prefix="eveyes_comotion_render_") as temporary:
        temporary_path = Path(temporary)
        dvs = _external("DVS-Voltmeter", luminance, timestamps, args, temporary_path)
        v2e = _external("v2e", luminance, timestamps, args, temporary_path)
    ours = _ours(luminance, timestamps, intrinsics, args.rpm, args.threshold)
    selected = list(range(8, min(56, len(timestamps) - 1), 12))
    panels: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for index in selected:
        start, end = float(timestamps[index]), float(timestamps[index + 1])
        streams = {"DVS-Voltmeter": _slice(dvs, start, end), "v2e": _slice(v2e, start, end), "Ours prism": _slice(ours, start, end)}
        rows.append({"window": index, "start": start, "end": end, **{f"{name}_events": len(stream) for name, stream in streams.items()}})
        base = frames[index]
        row = [_label(base.copy(), "RGB", f"frame {index}")]
        for name, stream in streams.items():
            detail = f"N={len(stream):,}" if name != "Ours prism" else f"N={len(stream):,} RPM={args.rpm:g}"
            row.append(_label(_event_image(stream, base.shape[:2]), name, detail))
        panels.append(np.concatenate(row, axis=1))
    montage = np.concatenate(panels, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.output, montage)
    args.output.with_suffix(".json").write_text(json.dumps({"scene": str(args.scene), "source_resolution": [frames.shape[2], frames.shape[1]], "methods": ["DVS-Voltmeter", "v2e", "Ours prism"], "ours_controller": "fixed 180 RPM anchor; learned pilot policy is reported separately", "rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
