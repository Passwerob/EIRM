# EIRM

EIRM is the minimal, reproducible implementation package for **Project**, a
physically ordered simulator for rotating-prism AMI event cameras. It applies
the prism image warp before DVS event generation and optionally maps each
event back to a reference plane using its individual timestamp.

```text
RGB + timestamps -> VFI -> linear luminance -> rotating prism warp
                 -> DVS events -> per-event inverse compensation
```

This repository intentionally contains source code, tests, configuration,
documentation, and compact result tables only. It does not redistribute model
weights, third-party simulator source trees, DL3DV data, event HDF5 files, or
large generated media.

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
[external dependencies](docs/DEPENDENCIES.md).

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
and command-line name. The retained Python import path is `ami_sim`.
