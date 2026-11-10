# External dependencies

The adapters keep external model code outside this package. This avoids copying or
silently modifying research implementations and permits reproducible metadata.

## Practical-RIFE

Use the official repository:

```text
https://github.com/hzwer/Practical-RIFE
```

The provided `scripts/fetch_practical_rife.sh` clones that URL into the local
EIRM `Practical-RIFE/` directory without overwriting an existing checkout.
Network access is intentionally not attempted by the simulator itself.

The released `train_log/flownet.pkl` checkpoint (24 MB) is stored at
`weights/practical_rife/train_log/flownet.pkl`. Copy it into a checkout before
loading the model:

```bash
cp weights/practical_rife/train_log/flownet.pkl Practical-RIFE/train_log/flownet.pkl
```

Place a checkout anywhere and invoke the simulator with
`--practical-rife-root /path/to/Practical-RIFE`. The adapter loads `model.RIFE.Model`
and calls its direct timestep API when that checkout exposes one. An older checkout
that only supports a midpoint is rejected for arbitrary samples instead of using
hidden recursive 2x interpolation. The run metadata records this as
`recursive_interpolation: false`.

## GIMM-VFI

The official GIMM-VFI checkout should be exposed through a small callable with the
signature:

```python
def interpolate(frame0: numpy.ndarray, frame1: numpy.ndarray,
                target_timestamps: numpy.ndarray) -> numpy.ndarray:
    ...  # returns [N, H, W, 3] for normalized t in (0, 1)
```

Pass it as `--gimm-adapter package.module:interpolate`. This contract deliberately
queries every requested timestep directly.

## DVS-Voltmeter and v2e

Both event adapters run a supplied command without a shell. The command may contain
`{frames_dir}`, `{timestamps_path}`, and `{output_path}` placeholders. It receives
float32 `.npy` luminance frames in `[0, 1]` plus float64-second timestamps and must
write HDF5, NPZ, NPY, or CSV events containing `x`, `y`, `t`, and `p`.

For example:

```bash
--event-command 'python /path/to/runner.py --frames {frames_dir} --timestamps {timestamps_path} --output {output_path}'
```

Set `--event-output-time-unit ns`, `us`, `ms`, or `s` to preserve the official
simulator's timing exactly while normalizing internal timestamps to float64 seconds.
No in-package threshold simulator is exposed under the DVS-Voltmeter name.

### Official DVS-Voltmeter checkout

The official checkout used for the DL3DV comparison is
`DVS-Voltmeter/` at commit `9d86353`:
`https://github.com/Lynn0306/DVS-Voltmeter`.
Its released `EventSim` API consumes uint8 grayscale frames and microsecond
timestamps, so the repository bridge converts AMI's linear `[0, 1]` luminance
arrays and second timestamps explicitly:

```bash
PYTHONPATH=. python scripts/dvs_voltmeter_runner.py \
  --frames /path/to/frames \
  --timestamps /path/to/timestamps.npy \
  --output /path/to/events.h5 \
  --dvs-root /path/to/DVS-Voltmeter \
  --camera-type DVS346 --seed 0
```

The bridge maps official polarity `0/1` to the canonical `-1/+1` form and
stores the conversion details in the HDF5 attributes.

### Official V2E checkout

The downstream baseline uses SensorsINI/v2e at commit `cac6384`. Its core
`v2ecore.emulator.EventEmulator` is invoked by
`scripts/v2e_runner.py`; the runner converts the normalized `[0, 1]` luminance
frames to the official uint8 input format and writes canonical HDF5 events.
For the clean baseline, use `C+ = C- = 0.2`, zero threshold variation, and
disable leak, shot, and photoreceptor noise:

```bash
PYTHONPATH=. python scripts/v2e_runner.py \
  --frames /path/to/interpolated_intensity \
  --timestamps /path/to/timestamps.npy \
  --output /path/to/events.h5 \
  --v2e-root /path/to/v2e --device cpu
```

`dv-processing` is only required by V2E's optional AEDAT-4 writer. The runner
uses an import-only compatibility stub when that writer is not selected.

## E2VID checkpoint

The lightweight reconstruction checkpoint is included at
`weights/e2vid/E2VID_lightweight.pth.tar` (41 MB). It is the pretrained file
released with [rpg_e2vid](https://github.com/uzh-rpg/rpg_e2vid) (GPL-3). The
E2VID source tree itself is not vendored. Pass the bundled file with
`--weights weights/e2vid/E2VID_lightweight.pth.tar`. The seed-42 mixed-training
fine-tune is `weights/e2vid/mixed_seed42/best.pt`, with its config and loss
history in the same directory.

### ESIM reference port

The native ESIM simulator is a ROS/C++ package. Because those build/runtime
dependencies are not present in this environment, `scripts/esim_reference_runner.py`
implements the event core from `event_simulator.cpp` with vectorized NumPy and
records `native_binary=false` in HDF5 metadata. This is suitable for a
controlled algorithm comparison, but a paper should distinguish it from a
native ESIM executable and repeat the check in a ROS environment when possible.
