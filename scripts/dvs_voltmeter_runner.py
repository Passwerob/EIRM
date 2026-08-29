#!/usr/bin/env python3
"""Bridge AMI's array contract to the official DVS-Voltmeter EventSim API."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from ami_sim.events import write_events_h5
from ami_sim.types import EventStream


_SENSOR_PARAMETERS = {
    "DVS346": [0.00018 * 29250, 20, 0.0001, 1e-7, 5e-9, 0.00001],
    "DVS240": [0.000094 * 47065, 23, 0.0002, 1e-7, 5e-8, 0.00001],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official DVS-Voltmeter on AMI intensity arrays.")
    parser.add_argument("--frames", required=True, type=Path, help="Directory of ordered float32 [0,1] .npy luminance frames.")
    parser.add_argument("--timestamps", required=True, type=Path, help="Float64 seconds .npy timeline matching --frames.")
    parser.add_argument("--output", required=True, type=Path, help="Canonical HDF5 event output path.")
    parser.add_argument("--dvs-root", type=Path, default=Path(__file__).resolve().parents[1] / "DVS-Voltmeter")
    parser.add_argument("--camera-type", choices=sorted(_SENSOR_PARAMETERS), default="DVS346")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_frames(directory: Path) -> list[np.ndarray]:
    paths = sorted(directory.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy frames found in {directory}.")
    frames = [np.asarray(np.load(path), dtype=np.float32) for path in paths]
    shape = frames[0].shape
    if len(shape) != 2 or any(frame.shape != shape for frame in frames):
        raise ValueError("DVS-Voltmeter input must be a non-empty, fixed-resolution grayscale sequence.")
    return frames


def main() -> int:
    args = _parse_args()
    root = args.dvs_root.resolve()
    if not (root / "src" / "simulator.py").is_file():
        raise FileNotFoundError(f"DVS-Voltmeter checkout is missing its EventSim source: {root}")
    sys.path.insert(0, str(root))
    from src.config import cfg  # type: ignore[import-not-found]
    from src.simulator import EventSim  # type: ignore[import-not-found]

    frames = _load_frames(args.frames)
    timestamps = np.asarray(np.load(args.timestamps), dtype=np.float64).reshape(-1)
    if len(frames) != len(timestamps):
        raise ValueError("Frame count does not match timestamp count.")
    if len(timestamps) < 2 or not np.all(np.diff(timestamps) > 0):
        raise ValueError("DVS-Voltmeter timestamps must contain at least two strictly increasing samples.")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("DVS-Voltmeter timestamps must be finite.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg.SENSOR.CAMERA_TYPE = args.camera_type
    cfg.SENSOR.K = _SENSOR_PARAMETERS[args.camera_type]
    simulator = EventSim(cfg=cfg, output_folder=str(args.output.parent), video_name=args.output.stem)

    generated: list[np.ndarray] = []
    for frame, timestamp in zip(frames, timestamps):
        # The released model is calibrated for OpenCV-style uint8 grayscale
        # inputs and microsecond timestamps. It does not define a [0,1] mode.
        image_u8 = np.rint(np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        event = simulator.generate_events(image_u8, int(np.rint(timestamp * 1e6)))
        if event is not None and len(event):
            generated.append(event)
    simulator.reset()

    if generated:
        events_us_xy_p = np.concatenate(generated, axis=0)
        order = np.argsort(events_us_xy_p[:, 0], kind="stable")
        events_us_xy_p = events_us_xy_p[order]
        events = EventStream(
            x=events_us_xy_p[:, 1],
            y=events_us_xy_p[:, 2],
            t=events_us_xy_p[:, 0].astype(np.float64) * 1e-6,
            p=np.where(events_us_xy_p[:, 3] > 0, 1, -1),
        )
    else:
        events = EventStream.empty()
    write_events_h5(
        args.output,
        events,
        {
            "event_backend": "official_dvs_voltmeter",
            "dvs_voltmeter_root": str(root),
            "camera_type": args.camera_type,
            "sensor_parameters": cfg.SENSOR.K,
            "input_intensity_range": "[0,1] converted to uint8 [0,255]",
            "input_timestamp_unit": "seconds converted to rounded microseconds",
            "official_polarity": "0/1 converted to -1/+1",
            "random_seed": args.seed,
        },
    )
    print(f"DVS-Voltmeter wrote {len(events)} events to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
