#!/usr/bin/env python3
"""Scene_001 video: DVS (car only) vs AMI 180 (city + car). Camera frozen."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from ami_simulator.geometry import RotatingWedgePrism
from scripts.evaluate_dvs_v2e_ours import _read_npz, _slice
from scripts.evaluate_eveyes_v2_downstream import _intrinsics
from scripts.plot_eveyes_scene001_dvs_vs_ours import CACHE, _load
from scripts.render_eveyes_camera_nfb_event_video import _events_to_map, _hud
from scripts.render_eveyes_scene001_storm_vs_ours import SCENE, _step

OURS_RPM = 180.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("results/eveyes_scene001/figures/scene001_dvs_vs_ours.mp4"))
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")
    rgb, timestamps, luminance = _load(args.scene, args.max_frames)
    dvs = _read_npz(CACHE)
    prism = RotatingWedgePrism(_intrinsics(args.scene), 0.5)
    height, width = rgb[0].shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = args.output.with_name(args.output.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width * 2, height * 2))
    if not writer.isOpened():
        raise SystemExit(f"failed to open {raw}")
    state = {"phase": 0.0, "active": None, "ref": None, "rpm": OURS_RPM}
    n = len(luminance) - 1
    for index in range(n):
        t0, t1 = float(timestamps[index]), float(timestamps[index + 1])
        idle = _slice(dvs, t0, t1)
        ours = _step(luminance[index : index + 2], np.asarray((t0, t1)), prism, OURS_RPM, state["phase"], state["active"], state["rpm"], state["ref"], args.device)
        state.update(phase=float(ours["phase_end"]), active=ours["active"], ref=np.asarray(ours["sensor_reference_log"]), rpm=OURS_RPM)
        left_rgb = _hud(rgb[index], ["RGB  camera frozen", f"t={t0:.2f}s  only the car moves"])
        right_rgb = _hud(rgb[index], ["RGB  camera frozen", f"t={t0:.2f}s  only the car moves"])
        left_evt = _hud(_events_to_map(idle, height, width), ["DVS  no prism", f"{len(idle.x):.0f} events  background empty"])
        right_evt = _hud(_events_to_map(ours["reference_events"], height, width), ["OURS  180 RPM inverse-warped", f"{len(ours['reference_events'].x):.0f} events  city painted"])
        frame = np.concatenate((np.concatenate((left_rgb, right_rgb), axis=1), np.concatenate((left_evt, right_evt), axis=1)), axis=0)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if index % 40 == 0:
            print(f"  frame={index}/{n} dvs={len(idle.x)} ours={len(ours['reference_events'].x)}", flush=True)
    writer.release()
    print(raw, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
