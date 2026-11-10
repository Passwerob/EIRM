#!/usr/bin/env python3
"""Render a co-motion visual comparison: official DVS versus learned Ours."""
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

from ami_simulator.event_quality import compute_reference_edge_mask, event_quality_metrics
from ami_simulator.flow import estimate_dense_flow, summarize_motion_clusters
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.motion_net import CausalAnchorQGRUPolicy
from ami_simulator.types import EventStream
from scripts.evaluate_anchor_residual_policy import CLUSTER_FEATURE_DIM, _state
from scripts.evaluate_phase_continuous_control import _load_scene
from scripts.run_control_headroom_stateful import RolloutState, _transition


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
        # Red = ON, cyan = OFF; repeated events saturate at the same pixel.
        output[events.y[positive], events.x[positive]] = (40, 70, 255)
        output[events.y[negative], events.x[negative]] = (255, 220, 40)
    return output


def _overlay(rgb: np.ndarray, events: EventStream) -> np.ndarray:
    base = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    event = _event_image(events, base.shape[:2])
    return cv2.addWeighted(base, 0.62, event, 0.95, 0.0)


def _label(image: np.ndarray, title: str, detail: str) -> np.ndarray:
    output = image.copy()
    scale = max(output.shape[1] / 64.0, 1.0)
    top = int(round(23 * scale))
    cv2.rectangle(output, (0, 0), (output.shape[1], top), (8, 8, 8), thickness=-1)
    cv2.putText(output, title, (int(4 * scale), int(16 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, (245, 245, 245), max(1, int(round(scale / 2))), cv2.LINE_AA)
    cv2.putText(output, detail, (int(3 * scale), output.shape[0] - int(5 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.28 * scale, (235, 235, 235), max(1, int(round(scale / 2))), cv2.LINE_AA)
    return output


def _large(image: np.ndarray, scale: int = 4) -> np.ndarray:
    return cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale), interpolation=cv2.INTER_NEAREST)


def _run_ours(scene, oracle, checkpoint: dict[str, object], *, margin: float, confidence: float, dwell: float) -> list[EventStream]:
    luminance, timestamps, masks, intrinsics = scene
    model = CausalAnchorQGRUPolicy(CLUSTER_FEATURE_DIM, int(checkpoint["hidden_dim"]), int(checkpoint["num_actions"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    prism = RotatingWedgePrism(intrinsics, 2.0)
    clusters = np.stack([summarize_motion_clusters(estimate_dense_flow(luminance[i], luminance[i + 1]), clusters=3) for i in range(len(timestamps) - 1)]).astype(np.float32)
    state = RolloutState(0.0, None, None, 180.0, float(timestamps[0]) - dwell)
    history: list[np.ndarray] = []
    streams: list[EventStream] = []
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    window = int(checkpoint["window"])
    objective = {"event_efficiency_weight": 0.0, "coverage_efficiency_weight": 0.0, "overlap_weight": 0.5, "density_weight": 0.1, "invalid_fraction_weight": 0.25, "rpm_effort_weight": 0.0, "rpm_change_weight": 0.0}
    for index in range(len(timestamps) - 1):
        feature = _state(oracle, index=index, previous=state.previous_metrics, phase=state.phase, dwell_seconds=float(timestamps[index] - state.last_switch_time), previous_rpm=state.previous_rpm, cluster_features=clusters)
        history.append(feature)
        rpm = 180.0
        if index >= window - 1:
            values = np.clip((np.stack(history[-window:]) - mean) / std, -float(checkpoint.get("feature_clip", 5.0)), float(checkpoint.get("feature_clip", 5.0)))
            with torch.inference_mode():
                q = model(torch.from_numpy(values[None].astype(np.float32)))[0].numpy()
            best = int(np.argmax(q))
            probability = float(torch.softmax(torch.from_numpy(q), dim=0)[best])
            ready = float(timestamps[index] - state.last_switch_time) >= dwell
            if best != 1 and float(q[best] - q[1]) > margin and probability > confidence and ready:
                rpm = (0.0, 180.0, 360.0)[best]
        result, state = _transition(scene, oracle, objective, prism, state, index, rpm, dwell=dwell)
        streams.append(result["reference_events"])
    return streams


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("data/comotion_controlled_v2/test_seed09_co_motion"))
    parser.add_argument("--student", type=Path, default=Path("results/comotion_horizon2_q_policy/policy.pt"))
    parser.add_argument("--dvs-root", type=Path, default=Path("/data/fcr/code/event_vggt/Data_Part/DVS-Voltmeter"))
    parser.add_argument("--dvs-python", type=Path, default=Path("/data/fcr/miniconda3/envs/dvsvolt/bin/python"))
    parser.add_argument("--output", type=Path, default=Path("results/ami_next/comotion_dvs_ours.png"))
    parser.add_argument("--start-window", type=int, default=20)
    parser.add_argument("--num-windows", type=int, default=4)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--dwell", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    manifest = json.loads(Path("results/comotion_v2_learned/manifest.json").read_text(encoding="utf-8"))
    record = next(item for item in manifest["oracles"] if item["scene"] == args.scene.name)
    luminance, timestamps, masks, intrinsics, oracle = _load_scene(args.scene, Path(str(record["output"])))
    try:
        rgb = np.stack([imageio.imread(path) for path in sorted((args.scene / "images").glob("*.png"))])
        with tempfile.TemporaryDirectory(prefix="ami_comotion_dvs_") as temporary:
            input_path = Path(temporary) / "frames.npz"
            dvs_path = Path(temporary) / "dvs.npz"
            np.savez_compressed(input_path, frames=luminance.astype(np.float32), timestamps=timestamps.astype(np.float64))
            command = [str(args.dvs_python), str(Path(__file__).with_name("run_dvs_voltmeter_array.py")), "--input", str(input_path), "--output", str(dvs_path), "--dvs-root", str(args.dvs_root), "--seed", str(args.seed)]
            subprocess.run(command, check=True)
            dvs = _read_npz(dvs_path)
        checkpoint = torch.load(args.student, map_location="cpu", weights_only=True)
        ours = _run_ours((luminance, timestamps, masks, intrinsics), oracle, checkpoint, margin=args.margin, confidence=args.confidence, dwell=args.dwell)
        selected = list(range(args.start_window, min(args.start_window + args.num_windows, len(timestamps) - 1)))
        panels: list[np.ndarray] = []
        rows: list[dict[str, object]] = []
        for index in selected:
            start, end = float(timestamps[index]), float(timestamps[index + 1])
            dvs_window = _slice(dvs, start, end)
            ours_window = ours[index]
            reference = luminance[index]
            edge = compute_reference_edge_mask(reference, edge_quantile=0.75) & masks[index]
            dvs_metrics = event_quality_metrics(dvs_window, reference_edge_mask=edge, image_shape=reference.shape)
            ours_metrics = event_quality_metrics(ours_window, reference_edge_mask=edge, image_shape=reference.shape)
            rows.append({"window": index, "start": start, "end": end, "dvs_events": len(dvs_window), "ours_events": len(ours_window), "dvs_edge_f1": float(dvs_metrics["edge_f1"]), "ours_edge_f1": float(ours_metrics["edge_f1"])})
            base = cv2.cvtColor(rgb[index], cv2.COLOR_RGB2BGR)
            dvs_img = _event_image(dvs_window, base.shape[:2])
            ours_img = _event_image(ours_window, base.shape[:2])
            panels.append(np.concatenate((_label(_large(base), "RGB", f"window {index}"), _label(_large(dvs_img), "DVS-Voltmeter", f"N={len(dvs_window)} F1={dvs_metrics['edge_f1']:.3f}"), _label(_large(ours_img), "Ours + prism", f"N={len(ours_window)} F1={ours_metrics['edge_f1']:.3f}"), _label(_large(_overlay(rgb[index], ours_window)), "Ours overlay", "red=ON cyan=OFF")), axis=1))
        montage = np.concatenate(panels, axis=0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(args.output, montage)
        args.output.with_suffix(".json").write_text(json.dumps({"scene": str(args.scene), "student": str(args.student), "methods": ["DVS-Voltmeter", "Ours"], "rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "rows": rows}, indent=2, sort_keys=True))
    finally:
        oracle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
