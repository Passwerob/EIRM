# Minimal Migration Scope

This repository is a source-focused migration of the Project simulator.

## Included

- `ami_sim/`: simulation pipeline, prism models, event interfaces, CUDA source,
  metrics, validation helpers, and visualization code;
- `scripts/`: simulator entry point, CUDA benchmark, event-quality checker,
  official DVS bridge, validation entry point, and external RIFE fetch helper;
- `configs/baseline.yaml`: fixed-rotation baseline configuration;
- `tests/`: physics, event-quality, optional DVS, and optional CUDA coverage;
- `docs/` and `results/`: current technical report and compact CSV/JSON tables;
- `ami_simulator/`, additional `scripts/` and `configs/`, and `NEW/`: the
  AMI-Simulator research package and active-sensing pilot, without datasets;
- `weights/` and `KEEP_PAPER_WEIGHTS_DO_NOT_DELETE/`: Practical-RIFE and E2VID
  checkpoints that fit in git, plus the paper policy checkpoints.

## Excluded

- Practical-RIFE and DVS-Voltmeter source checkouts;
- V2E/ESIM installations, DL3DV data, camera calibration assets, and private
  experiment inputs;
- raw or compensated event HDF5 streams, images, videos, and large output
  directories;
- historical workspaces unrelated to this simulator.

## External Components

The code provides adapters for third-party components but does not vendor or
modify them. Their source URLs, required interfaces, and unit conversions are
recorded in [DEPENDENCIES.md](DEPENDENCIES.md). Install those components under
their own licenses before using the corresponding backends.

## Result Tables

The files in `results/` are small, derived summaries for the technical report.
They are not a replacement for the original datasets or complete generated
event streams. They retain the evaluated protocol values without including
input data or large artifacts.
