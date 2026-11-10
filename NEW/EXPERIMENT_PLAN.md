# Adaptive Optical Excitation for Event Cameras

## 0. Scope and execution status

This document freezes the first experiment protocol for the redesigned project.
It is deliberately written before implementation and before any controller
training. The existing simulator under `../ami_simulator` remains the optical
and event-generation backend; this experiment layer changes the scientific
question and the control interface.

The central claim to test is:

> When natural relative motion is insufficient, a single-DoF rotating wedge
> prism can be controlled as an active temporal signal generator. A trajectory
> selected from motion and event-quality state can expose more useful,
> non-redundant temporal structure than a fixed trajectory under the same
> sensing budget.

The prism is not a motion compensator or a depth sensor. Camera pose, object
pose, SLAM, and monocular metric depth are outside the primary contribution.
IMU, when present, is only a motion prior.

## 1. Falsifiable hypotheses

### H1: Natural-motion deficit predicts event-perception deficit

For a fixed scene and event sensor configuration, low natural temporal
excitation (low flow magnitude/entropy and low useful event support) should
correlate with lower feature repeatability and matching recall in the no-prism
condition.

### H2: The best optical trajectory is scene-dependent

The validation-optimal trajectory should vary with natural motion, texture
density, spatial flow consistency, and event history. If one fixed speed or one
fixed waveform wins across all held-out scenes and regimes, adaptive trajectory
control is not justified as the main contribution.

### H3: Adaptive useful-event control beats fixed and random control

After matching event window, event-rate/bandwidth, actuator limits, and compute,
the adaptive trajectory should improve the primary feature-matching metrics
over no prism, fixed slow/fast prism, and random trajectories on unseen scenes.

### H4: Event-count optimization is an invalid objective

A controller optimized for event density alone should produce more repeated or
off-structure events and should not reliably improve feature matching. A
motion-aware objective using useful event quality should perform better under
the same budget.

## 2. Frozen benchmark protocol

### 2.1 Scene and motion regimes

Use scene-disjoint splits. The target paper-scale split is 60/10/20 scenes for
train/validation/test; a five-scene pilot is allowed only as a feasibility gate
and must not be reported as final evidence.

Each scene must provide RGB frames with monotonic timestamps and calibrated
intrinsics. The benchmark must contain the following paired regimes generated
from the same underlying scene and timeline:

1. `low_relative_motion`: static or near-static image motion;
2. `camera_only`: natural image motion without object/camera cancellation;
3. `object_only`: object motion with a stationary camera;
4. `co_motion`: camera and object motion cancel, creating an intentional
   no-prism temporal-information deficit;
5. `high_relative_motion` (optional stress condition): natural motion already
   supplies strong temporal excitation.

The co-motion condition is mandatory for the main claim. A scene package must
record how the regime was produced, the RGB timeline, calibration, masks or
correspondence ground truth used by the task, and whether IMU is measured or
pose-derived synthetic data. A dataset without independent object/background
control can be used for the pilot stratification, but cannot by itself prove
the co-motion claim.

### 2.2 Sensor and simulation constants

Freeze these values before comparing policies:

- prism optical model: existing `RotatingWedgePrism` central-ray warp;
- prism axes: one rotating DoF only;
- event model: existing deterministic log-threshold backend for the first
  protocol, with threshold/noise robustness as a later ablation;
- RGB-to-luminance, timestamp interpolation, padding, resolution, and event
  window length;
- intrinsics and prism apex angle per scene;
- random seeds for scene generation, trajectory sampling, and detector;
- maximum phase step and actuator envelope (speed, acceleration, latency,
  minimum dwell, and valid image-border fraction).

Practical-RIFE may be used for timestamp-faithful intermediate frames. Its FPS
is numerical sampling, not a second physical motion variable. Debug linear VFI
is allowed only for unit tests and smoke checks.

### 2.3 Required trajectory conditions

Every scene/regime/seed must evaluate exactly these named conditions:

1. `no_prism`: `theta(t) = 0`;
2. `fixed_slow`: constant rotation at the predeclared low speed;
3. `fixed_fast`: constant rotation at the predeclared high speed;
4. `random_trajectory`: seeded random choice of valid trajectory parameters,
   filtered by the same actuator constraints and dwell policy;
5. `adaptive_trajectory`: the analytical controller in Stage 1.

The sweep must additionally include a validation-only grid of constant RPMs,
so that `best_fixed` is selected on validation and evaluated once on test. The
grid must not be selected from test-scene results. To identify whether
waveform matters, run a trajectory-family grid over:

- constant rotation;
- sinusoid `A sin(2*pi*f*t + phase)`;
- triangular sweep;
- chirp;
- piecewise adaptive dwell schedule.

Direction is the sign of angular motion. No second prism axis may be added.

## 3. Controller state and action contract

At control time `t`, construct a causal state from the current/previous window:

```text
s_t = [flow statistics,
       optional IMU motion prior,
       event quality statistics,
       previous trajectory parameters]
```

No future RGB frame, event, or oracle trajectory may enter `s_t`.

### 3.1 Optical-flow features

Compute flow on RGB frames or a documented event reconstruction using one frozen
backend. Store both raw and normalized values:

- mean, median, p90, and maximum flow magnitude;
- flow variance and robust spatial variance;
- histogram entropy of flow magnitude and direction;
- spatial consistency (fraction agreeing with the dominant flow vector);
- low-motion pixel fraction under a threshold frozen from training scenes;
- valid-flow fraction and edge-weighted flow magnitude.

The extractor must return a named feature vector, a validity mask, and metadata
for backend, resolution, threshold, and normalization bounds. Flow is a state
signal, not a prism command.

### 3.2 Optional IMU features

Use only as a prior describing natural motion:

- gyro norm mean/p90/max and integrated rotational energy;
- acceleration norm summary;
- dominant-axis and motion-category encoding.

Never map an IMU sample directly to prism RPM or angle. Report experiments with
`flow_only`, `flow_plus_imu`, and `no_flow_imu_only` separately.

### 3.3 Event features

Use the existing `event_quality_metrics` contract and add only features needed
by the controller:

- event density and active-pixel ratio;
- edge coverage, edge precision/F1, and structural ratio;
- event efficiency and repeat-event fraction;
- temporal overlap with the preceding window;
- polarity imbalance;
- valid reference-plane fraction and border/invalid fraction;
- timestamp-bin occupancy/diversity.

Sensor-plane bandwidth metrics and reference-plane structural metrics must stay
separate. Event count is a diagnostic and budget variable, never an information
proxy.

### 3.4 Action representation

Expose a trajectory object with continuous phase and a common `theta_values(t)`
and `omega_values(t)` interface. The public parameter record is:

```text
TrajectoryParameters:
    waveform: constant | sinusoid | triangle | chirp | piecewise
    amplitude_rad: float
    frequency_hz: float
    phase_rad: float
    direction: -1 or +1
    duration_s: float
    angular_speed_rad_s: float | None   # required for constant/piecewise segments
```

`duration_s` is the dwell duration. Family-specific fields are explicit rather
than silently overloaded; `angular_speed_rad_s` is derived to RPM only for
reporting. All trajectories must satisfy the same speed, acceleration,
phase-step, latency, and border-validity constraints.

## 4. Two-stage control study

### Stage 0: Controlled trajectory sweep (required before learning)

For each scene window, evaluate the full candidate trajectory grid offline,
recording event and task metrics. Report the per-scene best trajectory and the
distribution of winners. This directly tests H2 and provides a causal oracle
without training a neural controller.

### Stage 1: Rule-based adaptive controller

Implement a transparent controller that estimates a temporal-information
deficit and selects a feasible candidate trajectory:

```text
deficit = w_flow * low_motion_fraction
        + w_event * (1 - useful_event_quality)
        + w_overlap * temporal_overlap
        + w_border * invalid_fraction
```

Select the candidate maximizing predicted useful quality minus bandwidth,
effort, switching, and invalid-border penalties. Apply minimum dwell,
acceleration, phase continuity, and one-window observation latency. Weights are
fit or frozen on training/validation only.

### Stage 2: Learned policy (conditional)

Only if Stage 0 demonstrates heterogeneous winners and Stage 1 beats `best_fixed`
on validation, train a causal policy that predicts trajectory parameters. The
policy must be evaluated in autonomous rollout, not teacher forcing, and must
retain the same feasibility wrapper. A negative learned result is reported as
evidence against learned control, not hidden by adding more features.

## 5. Metrics and analysis

### 5.1 Event-level metrics

For every scene, condition, and policy report:

- event count and event density (sensor bandwidth);
- active-pixel ratio and edge coverage;
- event efficiency, repeat-event fraction, polarity imbalance;
- temporal diversity/overlap;
- reference-plane valid fraction and border loss.

Also report event-rate-normalized and matched-bandwidth results. A high event
count without improved structural support is explicitly marked as failure.

### 5.2 Primary task: event feature detection and matching

Use one frozen detector/descriptor and matcher for all policies. Input is the
same event representation (for example a polarity-aware time surface or voxel
window); RGB may be used only in a separately labelled RGB+event condition.

For adjacent windows with known synthetic correspondence or a frozen geometric
transform, report:

- detector repeatability: fraction of reference keypoints recovered within a
  fixed pixel tolerance;
- matching precision: correct matches / all accepted matches;
- matching recall: correct matches / reference correspondences;
- RANSAC inlier ratio and median localization error;
- failure rate for windows with fewer than the minimum valid matches.

The detector, thresholds, spatial tolerance, RANSAC settings, event window,
resolution, and compute budget are frozen before test evaluation. Bootstrap
confidence intervals are computed over scenes, not individual events.

### 5.3 Secondary analyses

- correlation between provisional event-quality score and feature metrics on a
  held-out scene set;
- Pareto plots of useful information versus event bandwidth and actuator effort;
- winner entropy and per-regime trajectory selection;
- sensitivity to contrast threshold, timestamp/angle perturbation, and noise;
- optional tracking/reconstruction only after the matching benchmark is stable.

## 6. Required ablations

1. Motion information: `no_flow` versus `flow`; optional `flow_plus_imu` and
   `imu_only` diagnostic.
2. Control space: RPM-only constant schedules versus full trajectory
   parameters (waveform, amplitude, frequency, phase, direction, dwell).
3. Objective: event-count maximization versus useful-event quality and the
   downstream matching objective.
4. Prism trajectory family: constant, sinusoid, triangle, chirp, piecewise.
5. Sensor realism: deterministic threshold versus threshold variation,
   timing jitter, background activity, and photon/read noise.
6. Actuator realism: unconstrained versus speed/acceleration/latency/dwell
   constraints.
7. Natural-motion regime: low, co-motion, ordinary, and high relative motion.
8. Matched event-rate/bandwidth versus unconstrained output.

## 7. Decision gates and stop rules

- **Gate A, excitation need:** in co-motion, prism conditions must improve a
  useful structural or feature metric over no prism. If not, stop the adaptive
  claim and retain only a simulator/sensing study.
- **Gate B, adaptation need:** the best validation trajectory must vary across
  scenes/regimes, and adaptive control must beat `best_fixed` on held-out test.
  Otherwise keep optical excitation but remove adaptive control as the primary
  contribution.
- **Gate C, objective validity:** event-count optimization must not be allowed
  to stand in for useful information if it fails the held-out score/task
  correlation gate.
- **Gate D, robustness:** the direction must survive sensor and actuator
  perturbations. If not, weaken the physical claim and document the valid
  operating envelope.
- **Gate E, hardware:** absent a matched prism recording, use simulation-only
  wording. A minimal hardware trend check is validation, not a prerequisite for
  implementing the simulation plan.

## 8. Reproducibility and artifacts

Every run stores the git revision, configuration, seed, dataset split, scene
and regime identifiers, timestamps, intrinsics, prism metadata, trajectory
parameters, event backend metadata, flow backend/normalization, and controller
state/action logs. Each table is generated from machine-readable per-window
CSV/NPZ/JSON records; plots never become the source of truth.

The first implementation milestone is therefore not a learned model. It is a
reproducible trajectory sweep and a feature-matching evaluator that can accept
the five required policies.
