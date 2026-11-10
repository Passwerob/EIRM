#!/usr/bin/env python3
"""Append privileged GT object/background flow to a horizon-2 teacher dataset.

The added 16 dimensions are an observability upper bound. They use simulator
object masks and must not be presented as a deployable controller input.
Existing teacher returns ``q_h2`` are copied unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as imageio
import numpy as np

from ami_simulator.flow import CAUSAL_BASE_FEATURE_DIM, PRIVILEGED_ROI_FEATURE_NAMES, PRIVILEGED_ROI_Q_FEATURE_DIM, privileged_roi_sequence
from ami_simulator.pipeline import load_rgb_sequence, rgb_to_linear_luminance


def _scene_roi_features(scene: Path, windows: int) -> np.ndarray:
    frames, _ = load_rgb_sequence(scene / "images", scene / "timestamps.csv", timestamp_unit="s", max_frames=windows + 1)
    mask_paths = sorted((scene / "object_masks").glob("*.png"))[: len(frames)]
    if len(frames) != windows + 1 or len(mask_paths) != len(frames):
        raise ValueError(f"{scene} needs {windows + 1} RGB frames and object masks.")
    masks = np.stack([np.asarray(imageio.imread(path)) > 0 for path in mask_paths])
    return privileged_roi_sequence(rgb_to_linear_luminance(frames), masks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for record in manifest["records"]:
        source = Path(str(record["output"]))
        destination = args.output_root / str(record["split"]) / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with np.load(source) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        base = np.asarray(arrays["features"], dtype=np.float32)
        if base.ndim != 2 or base.shape[1] < CAUSAL_BASE_FEATURE_DIM:
            raise ValueError(f"Expected at least {CAUSAL_BASE_FEATURE_DIM}-D features in {source}.")
        roi = _scene_roi_features(args.dataset_root / str(record["scene"]), len(base))
        arrays["features"] = np.concatenate((base[:, :CAUSAL_BASE_FEATURE_DIM], roi), axis=1)
        if arrays["features"].shape != (len(base), PRIVILEGED_ROI_Q_FEATURE_DIM):
            raise ValueError(f"Privileged feature shape mismatch in {source}: {arrays['features'].shape}.")
        np.savez_compressed(destination, **arrays)
        records.append({**record, "output": str(destination), "feature_dim": PRIVILEGED_ROI_Q_FEATURE_DIM})
    output = {
        **manifest,
        "dataset": "stateful horizon-2 teacher with privileged ROI optical flow",
        "feature_dim": PRIVILEGED_ROI_Q_FEATURE_DIM,
        "privileged": True,
        "causal_base_feature_dim": CAUSAL_BASE_FEATURE_DIM,
        "privileged_roi_feature_names": list(PRIVILEGED_ROI_FEATURE_NAMES),
        "limitations": [
            "Object masks are simulator ground truth and are an observability upper bound, not a deployable input.",
            "Teacher returns are copied from the source dataset and are still privileged horizon-2 edge-F1 search.",
        ],
        "records": records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"records": len(records), "feature_dim": PRIVILEGED_ROI_Q_FEATURE_DIM, "output_root": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
