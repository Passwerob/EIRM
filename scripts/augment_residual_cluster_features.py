#!/usr/bin/env python3
"""Append mask-free dense-flow motion-cluster features to residual datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as imageio
import numpy as np

from ami_simulator.flow import MOTION_CLUSTER_FEATURE_NAMES, estimate_dense_flow, summarize_motion_clusters


def _cluster_sequence(scene: Path, windows: int) -> np.ndarray:
    images = sorted((scene / "images").glob("*.png"))
    if len(images) < windows + 1:
        raise ValueError(f"{scene} has {len(images)} images but needs {windows + 1}.")
    values = np.zeros((windows, len(MOTION_CLUSTER_FEATURE_NAMES)), dtype=np.float32)
    for index in range(1, windows):
        previous = imageio.imread(images[index - 1])
        current = imageio.imread(images[index])
        values[index] = summarize_motion_clusters(estimate_dense_flow(previous, current), clusters=3)
    return values


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
        base = arrays["features"].astype(np.float32)
        cluster = _cluster_sequence(args.dataset_root / str(record["scene"]), len(base))
        if base.shape[1] == 20:
            arrays["features"] = np.concatenate((base, cluster), axis=1)
        elif base.shape[1] != 47:
            raise ValueError(f"Unexpected feature dimension {base.shape[1]} in {source}")
        np.savez_compressed(destination, **arrays)
        records.append({**record, "output": str(destination), "feature_dim": int(arrays["features"].shape[1])})
    output = {**manifest, "dataset": "AMI anchor residual with mask-free motion clusters", "feature_dim": 47, "cluster_feature_names": list(MOTION_CLUSTER_FEATURE_NAMES), "cluster_feature_alignment": "feature row i uses pre-prism RGB flow pair i-1; row 0 is zero", "records": records}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"records": len(records), "feature_dim": 47, "output_root": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
