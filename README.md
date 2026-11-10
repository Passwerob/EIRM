# EIRM

EIRM is the minimal, reproducible implementation package for **Project**, a
physically ordered simulator for rotating-prism AMI event cameras. It applies
the prism image warp before DVS event generation and optionally maps each
event back to a reference plane using its individual timestamp.

```text
RGB + timestamps -> VFI -> linear luminance -> rotating prism warp
                 -> DVS events -> per-event inverse compensation
```

This repository contains the minimal `ami_sim` package, the research
`ami_simulator` code, tests, configuration, documentation, compact result
tables, and the small checkpoints listed below. It does not redistribute
third-party simulator source trees, DL3DV data, event HDF5 files, or large
generated media.

## Install

```bash
python -m pip install -e .
```

The deterministic smoke path has no external model checkout:

```bash
eirm \
  --rgb-dir /path/to/images \
  --timestamps /path/to/timestamps.csv \
  --vfi debug_linear \
  --event-simulator debug_threshold \
  --prism-model global-circle \
  --output outputs/smoke
```

For the research configuration, install the external Practical-RIFE and
DVS-Voltmeter checkouts yourself, then supply their paths with
`--practical-rife-root` and `--dvs-root`. See
[external dependencies](docs/DEPENDENCIES.md). The Practical-RIFE checkpoint
and the E2VID lightweight checkpoint are already in `weights/`.

## Research code

The AMI-Simulator research tree is included next to the minimal package:

- `ami_simulator/`: experiment simulator, controllers, and metrics (`ami-sim`)
- `scripts/` and `configs/`: experiment entry points and profiles
- `NEW/`: active-sensing pilot. Its dataset directory is not included
- `KEEP_PAPER_WEIGHTS_DO_NOT_DELETE/`: paper policy checkpoints, about 21 KB each

## Weights

| Path | Size | Origin |
| --- | --- | --- |
| `weights/practical_rife/train_log/flownet.pkl` | 24 MB | [Practical-RIFE](https://github.com/hzwer/Practical-RIFE), MIT |
| `weights/e2vid/E2VID_lightweight.pth.tar` | 41 MB | [rpg_e2vid](https://github.com/uzh-rpg/rpg_e2vid) pretrained checkpoint, GPL-3 |
| `KEEP_PAPER_WEIGHTS_DO_NOT_DELETE/*/policy.pt` | 21 KB | paper controllers trained for this project |

After `scripts/fetch_practical_rife.sh`, copy or link the checkpoint into that checkout:

```bash
cp weights/practical_rife/train_log/flownet.pkl Practical-RIFE/train_log/flownet.pkl
```

Point E2VID reconstruction at the bundled file with
`--weights weights/e2vid/E2VID_lightweight.pth.tar`. The E2VID source tree is
not included.

## Verify

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest
```

Tests that require an external DVS-Voltmeter checkout or CUDA are skipped when
those optional dependencies are unavailable.

## Contents

- [Technical report (Chinese)](docs/PROJECT_TECHNICAL_REPORT_CN.md)
- [Migration scope](docs/MIGRATION_SCOPE.md)
- [External dependencies](docs/DEPENDENCIES.md)
- [Compact result tables](results/)

## Naming

`Project` is the public placeholder used in documents. `EIRM` is the package
and command-line name. The minimal import path is `ami_sim`. The research
import path is `ami_simulator`, exposed as the `ami-sim` command.
