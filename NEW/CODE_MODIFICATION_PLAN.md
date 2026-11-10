# Code Modification Plan

## 0. Design principle

The root package `ami_simulator` is the established simulator and remains
unchanged during the first implementation pass. It already provides the
required central-ray prism model, timestamp-aware warping, CPU/CUDA threshold
event backends, inverse event coordinates, and useful-event metrics.

The new work is an experiment/controller layer under `NEW/`. To avoid Python
import shadowing between `NEW/ami_simulator` and the existing root package, the
implementation package should be named `ami_active_sensing` (the logical module
names below retain the requested `motion/`, `control/`, and `evaluation/`
layout). It imports the root simulator explicitly.

```text
NEW/
  README.md
  EXPERIMENT_PLAN.md
  CODE_MODIFICATION_PLAN.md
  ami_active_sensing/
    __init__.py
    motion/
      __init__.py
      optical_flow.py
      motion_features.py
      state.py
    control/
      __init__.py
      trajectory.py
      adaptive_controller.py
      policy_interface.py
    evaluation/
      __init__.py
      perception_metrics.py
      trajectory_sweep.py
      reports.py
    data/
      __init__.py
      scene_protocol.py
      regimes.py
    configs/
      pilot.yaml
      paper_scale.yaml
    scripts/
      extract_motion_features.py
      run_trajectory_sweep.py
      evaluate_feature_matching.py
      run_ablation.py
    tests/
      test_optical_flow.py
      test_motion_features.py
      test_trajectory.py
      test_adaptive_controller.py
      test_perception_metrics.py
```

No root simulator file is copied into `NEW/`. If the later implementation
needs a small compatibility change, it must be made in a separate, reviewed
patch to the root API and covered by the existing tests.

## 1. Public interfaces to implement

### 1.1 `motion/optical_flow.py`

```python
class OpticalFlowBackend(Protocol):
    def estimate(self, previous_rgb: np.ndarray, current_rgb: np.ndarray) -> np.ndarray:
        """Return flow [H, W, 2] in pixel units (dx, dy)."""

    def metadata(self) -> dict[str, object]: ...

class FarnebackFlowBackend:
    ...
```

The first backend may use the repository's existing OpenCV dependency when
available. A clear optional-dependency error is required; do not silently fall
back to a fabricated flow field. The backend must validate RGB shape, dtype,
resolution, finite output, and pixel-unit convention.

### 1.2 `motion/motion_features.py`

```python
FLOW_FEATURE_NAMES = (...)
IMU_FEATURE_NAMES = (...)
EVENT_FEATURE_NAMES = (...)

def extract_flow_features(
    flow_xy: np.ndarray,
    *,
    low_motion_threshold_px_s: float,
    valid_mask: np.ndarray | None = None,
    normalization: FeatureNormalization | None = None,
) -> dict[str, float]: ...

def extract_imu_features(imu_window: ImuStream | None) -> dict[str, float]: ...

def extract_event_features(metrics: Mapping[str, float | int]) -> dict[str, float]: ...

def build_state_vector(
    flow: Mapping[str, float],
    event: Mapping[str, float],
    imu: Mapping[str, float] | None,
    previous_trajectory: TrajectoryParameters,
) -> tuple[np.ndarray, tuple[str, ...]]: ...
```

Feature names and ordering are serialized with every run. Normalization bounds
are fitted on training scenes only and then frozen for validation/test. The
state builder must expose a `use_flow` switch for the ablation rather than
changing the feature dimensionality implicitly.

### 1.3 `control/trajectory.py`

```python
@dataclass(frozen=True)
class TrajectoryParameters:
    waveform: Literal["constant", "sinusoid", "triangle", "chirp", "piecewise"]
    amplitude_rad: float = 0.0
    frequency_hz: float = 0.0
    phase_rad: float = 0.0
    direction: int = 1
    duration_s: float = 0.0
    angular_speed_rad_s: float | None = None

class OpticalTrajectory(Protocol):
    def theta_values(self, timestamps: np.ndarray, *, theta0: float, reference_time: float) -> np.ndarray: ...
    def omega_values(self, timestamps: np.ndarray) -> np.ndarray: ...
    def metadata(self) -> dict[str, object]: ...

def make_trajectory(parameters: TrajectoryParameters, *, start_time: float) -> OpticalTrajectory: ...
def sample_candidate_trajectories(config: TrajectorySearchConfig, *, seed: int) -> list[TrajectoryParameters]: ...
```

Implement phase-continuous trajectory classes or adapters for constant,
sinusoidal, triangular, chirp, and piecewise schedules. Candidate generation
must enforce speed, acceleration, phase-step, duration, and direction bounds.
The root `PrismWarper` should consume these objects without modification.

Important semantic tests:

- `theta_values` is continuous across control windows;
- `omega_values` agrees with finite differences within tolerance;
- a constant trajectory matches root `ConstantRotation`;
- reversing direction changes the sign of angular velocity;
- invalid parameters fail early;
- all candidate metadata contains the complete parameter record.

### 1.4 `control/adaptive_controller.py`

```python
class TrajectoryController(Protocol):
    def reset(self, *, timestamp: float, trajectory: TrajectoryParameters | None = None) -> None: ...
    def select(self, state: Mapping[str, float], *, timestamp: float) -> TrajectoryParameters: ...
    def metadata(self) -> dict[str, object]: ...

class RuleBasedAdaptiveController:
    ...
```

The rule-based controller scores a precomputed candidate set using a named
deficit/objective. It applies minimum dwell, switching margin, one-window
latency, acceleration, and phase-continuity constraints. Its `select` method
must be deterministic for a fixed seed/configuration and must log both the
unconstrained winner and the feasible action.

Do not add a neural controller in the first pass. A later learned implementation
must satisfy this same interface and be evaluated in autonomous rollout.

### 1.5 `data/scene_protocol.py` and `data/regimes.py`

Define a scene-level manifest rather than teaching the controller about a
specific dataset:

```text
SceneRecord:
    scene_id, split, regime, seed
    rgb_paths, timestamps, intrinsics
    optional imu_path, flow_ground_truth_path, correspondence_path
    image_shape, event_window_seconds
    provenance and synthetic/real flags
```

Regime generation must preserve paired scene identity and timestamp alignment.
The manifest records whether correspondence is supplied by synthetic generation,
known transform, or an external calibration. No pose recovery is performed by
the controller.

## 2. Evaluation modules

### 2.1 `evaluation/trajectory_sweep.py`

Responsibilities:

1. Load one `SceneRecord` and compute luminance/flow state once.
2. Instantiate each candidate `OpticalTrajectory`.
3. Call root `PrismWarper`, threshold simulator, and `compensate_events`.
4. Convert events to reference-plane and sensor-plane metric records.
5. Run the frozen feature detector/matcher adapter.
6. Write one row per scene/window/trajectory/seed, including parameters and
   feasibility diagnostics.

The sweep must never reset phase at RGB pair boundaries. It must use the same
root simulator config, interpolation, event window, and intrinsics for all
trajectories in a comparison.

### 2.2 `evaluation/perception_metrics.py`

Provide a detector/matcher-neutral adapter and a reference implementation with
the simplest dependency already available in the environment. The evaluator
must expose:

```python
def feature_matching_metrics(
    reference_representation: np.ndarray,
    candidate_representation: np.ndarray,
    correspondence: Correspondence,
    *,
    detector_config: Mapping[str, object],
) -> dict[str, float | int]: ...
```

Metrics are repeatability, matching precision/recall, inlier ratio, median
localization error, and failure rate. Detector settings and geometric
tolerances are serialized. If an external detector is unavailable, fail with a
clear message instead of changing the benchmark definition.

### 2.3 `evaluation/reports.py`

Aggregate per-window records by scene with scene-bootstrap confidence intervals.
Generate the required CSV/JSON tables and figures:

- policy x regime event/task table;
- scene-wise trajectory winner histogram;
- event-count versus useful-quality scatter;
- Pareto frontier of matching quality, bandwidth, and actuator effort;
- flow/no-flow, RPM/trajectory, and count/useful-objective ablation tables.

## 3. Root simulator integration

Use the existing public root APIs:

- `ami_simulator.pipeline.run_simulation` for end-to-end artifact writing when
  a single trajectory is evaluated;
- `ami_simulator.warp.PrismWarper` for sweep loops that must retain per-window
  state and phase;
- `ami_simulator.events.ThresholdEventSimulator` (then CUDA backends) for
  identical event generation;
- `ami_simulator.event_quality.event_quality_metrics` and
  `event_quality_score` for provisional useful-event signals;
- `ami_simulator.rotation.ScheduledRotation` only where a piecewise constant
  speed schedule is sufficient; use the new trajectory adapter for non-RPM
  waveforms;
- existing `CameraIntrinsics`, `EventStream`, and `ImuStream` types.

Do not reinterpret `compensate_events` as camera-motion compensation. It is a
reference-plane coordinate attachment used to measure structural support.

Potential root API change, only if required: broaden type annotations from
`object` to a small `RotationProtocol` with `theta_values`, `omega_values`, and
`metadata`. This is non-functional and must preserve all current rotation
classes and tests.

## 4. Configurations and scripts

Create config-driven experiments, with no source edits between ablations:

`configs/pilot.yaml` freezes a five-scene smoke matrix, low resolution, CPU
backend, and a small trajectory grid. `configs/paper_scale.yaml` contains the
scene-disjoint split, event window, detector, actuator bounds, seeds, and all
trajectory families.

The scripts should be resumable and write a manifest:

1. `extract_motion_features.py`: RGB/IMU/event windows -> feature NPZ/JSON;
2. `run_trajectory_sweep.py`: five policies plus validation trajectory grid;
3. `evaluate_feature_matching.py`: frozen task evaluator over existing sweep
   outputs;
4. `run_ablation.py`: executes the predeclared ablation matrix and aggregates
   reports.

Every script records command line, git revision, environment, seed, and config
hash. Test scenes must not be used for trajectory-family or objective-weight
selection.

## 5. Test plan before any large run

### Unit tests

- flow feature formulas on constant, translational, rotational, and invalid
  flow fields;
- entropy, low-motion, spatial-consistency, and normalization edge cases;
- IMU feature units and empty optional IMU behavior;
- trajectory continuity, derivatives, direction, dwell, and constraints;
- controller causality, deterministic selection, hysteresis, and actuator
  feasibility;
- feature matching perfect/noisy/duplicate/no-correspondence cases;
- scene manifest timestamp and split validation.

### Integration tests

- run a tiny RGB sequence through each trajectory family and root threshold
  simulator;
- verify event HDF5 output, reference-plane extras, valid masks, and trajectory
  metadata;
- verify no-prism reproduces `ConstantRotation(0.0)`;
- verify repeated windows preserve global phase;
- verify a sweep is invariant to candidate ordering;
- verify no future state is read by a causal controller;
- verify all five required policies produce comparable rows under one config.

### Smoke acceptance

The first milestone is complete only when a CPU smoke command creates a
machine-readable sweep manifest, task metrics, and plots for at least two
scenes and all five conditions. No neural policy training is part of this
milestone.

## 6. Implementation order and gates

1. Scaffold `NEW/ami_active_sensing`, config loading, manifests, and tests.
2. Implement trajectory abstraction and integration with root `PrismWarper`.
3. Implement optical-flow and state features, including train-only
   normalization.
4. Implement the trajectory sweep and event-level report.
5. Implement the frozen feature-matching evaluator.
6. Run Stage 0 and inspect trajectory-winner heterogeneity (Gate B precursor).
7. Implement the rule-based adaptive controller only if Stage 0 supports the
   need for adaptation.
8. Run the full ablation matrix and decision gates.
9. Add a learned policy only after the analytical controller passes validation.

At no point should the root simulator be rewritten, the prism gain be treated
as depth, or IMU be wired directly into a prism command.
