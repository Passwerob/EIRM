# Pilot Results

These numbers are engineering-gate results from the deterministic synthetic
pilot, not paper evidence. The pilot uses four textures, four translation
regimes, eight candidate trajectory families, 128x128 frames, 120 FPS, and
0.1 s control windows. The existing root prism and threshold event simulator
are unchanged.

## Stage 0: trajectory heterogeneity

`scene_motion_sweep.json` evaluates every candidate offline. The event-count
winner is `constant_30` in all 16 conditions, while the useful-event winner
varies across `constant_0`, `chirp`, and `constant_reverse_14`. Every texture
has more than one useful winner. This passes the heterogeneity gate and rejects
event count as a sufficient objective. Two high-motion conditions exceed the
hard 15 events/pixel/window budget and are explicitly marked infeasible.

## Stage 1: causal rule-based controller

`adaptive_pilot_final.json` is the flow-aware rollout after the controller
deficit was gated by low natural motion and the switching penalty was reduced.
The controller sees only completed prior windows and retains prism phase,
event threshold, and actuator state.

| evaluation | result |
| --- | ---: |
| mean task utility | 0.62296 |
| mean useful-event utility | 0.56309 |
| mean regret to per-condition task oracle | 0.04640 |
| conditions beating per-condition task oracle | 0 / 16 |
| adaptive minus global fixed `constant_30` task utility | +0.00938 |

The positive delta over the single global fixed policy is a sanity signal, not
a validation claim: this is a small pilot with no held-out scene split.

## Flow ablation

`adaptive_pilot_no_flow_v2.json` uses the same controller and seed while
masking optical-flow features with a conservative `low_motion_fraction=1`
missing-flow prior. Mean task utility is `0.60920` versus `0.62296` with flow;
mean oracle regret is `0.06017` versus `0.04640`. The older
`adaptive_pilot_no_flow_final.json` used an invalid zero-flow encoding and is
kept only as a historical artifact, not as an ablation result. This remains a
controller-state ablation, not a claim that Farneback is the only valid flow
backend.

## Decision

Stage 0 is ready for a scene-disjoint benchmark. Stage 1 is useful as a
transparent baseline and confirms that motion-aware state matters, but it does
not yet justify learned-policy training. The next experiment should fit
controller weights on training scenes, select `best_fixed` on validation, and
evaluate autonomous rollout on held-out scenes with repeated seeds and
confidence intervals.

## EveYES IMU export audit

The extracted `eveyes_single_car_dataset_final_imu_v2` contains four train
scenes and one test scene, with three motion modes per scene (15 sequences in
total). Each sequence has 480 RGB frames over 8 seconds and 8001 IMU samples.
All sequences have finite, strictly increasing timestamps and `imu_valid=true`;
14/15 have a nonzero gyro norm at every sample and the remaining sequence has
99.975% nonzero samples. The audit also shows a useful mode separation in
sampled RGB flow: camera-moving modes average about 14.6--14.9 px/s, while
camera-static/object-moving averages about 1.0 px/s. Gyro magnitude is nearly
constant across modes (about 0.00185 rad/s), so IMU is retained as an
optional prior and is not treated as a camera/object-motion label.

The machine-readable audit is [eveyes_imu_audit.json](results/eveyes_imu_audit.json).

## Synthetic-rendered EveYES event probe

On the first 31 frames (0.501 s) of the test driving sequence, resized to
128x72, the existing prism/event backend produced:

| policy | events | edge coverage | event efficiency |
| --- | ---: | ---: | ---: |
| no prism | 143,196 | 0.931 | 0.115 |
| fixed slow (6 rad/s) | 168,514 | 0.994 | 0.104 |
| fixed fast (20 rad/s) | 306,378 | 1.000 | 0.059 |

The probe demonstrates the intended bandwidth/utility trade-off on the new
rendered RGB data. It is not a feature-matching result because this export does not
provide a dense correspondence or depth map for the rendered frames. The raw
output is [eveyes_event_probe.json](results/eveyes_event_probe.json).

## Synthetic-rendered EveYES adaptive probe

The causal controller was then run for one second at 256x144 over all three
`Scene_005` test modes, with IMU features enabled and a useful-objective speed
cap of 8 rad/s calibrated from the train sweep. It selected no prism after the
initial exploratory action for both camera-moving modes, while it selected
`constant_6` for camera-static/object-moving. The latter has nearly zero RGB
flow (`0.006 px/s`) and therefore triggers moderate active excitation.

| mode | selected indices | event density | edge coverage | efficiency |
| --- | --- | ---: | ---: | ---: |
| camera moving + object moving | 1,0,0,0,0 | 10.52 | 0.942 | 0.0856 |
| camera moving + object static | 1,0,0,0,0 | 10.87 | 0.943 | 0.0830 |
| camera static + object moving | 1,1,1,1,1 | 6.83 | 0.990 | 0.1209 |

This is still a synthetic event-quality/control diagnostic. Before the
train-calibrated speed cap, the same mode selected `constant_30` and produced
21.85 density with 0.0567 efficiency; the cap reduces that flooding while
preserving nearly identical edge coverage. The output is
[eveyes_adaptive_probe.json](results/eveyes_adaptive_probe.json); matching
fields are not used as evidence without correspondence/depth.

## Synthetic-rendered EveYES trajectory sweep

The short 128x72 sweep evaluates all eight trajectory families under the same
15 events/pixel/window budget. On the 12 train sequences, `constant_30` wins
event count in all 12, while the useful-event winner is `constant_6` in 8 and
`chirp` in 4. On the held-out Scene_005 test sequences, `constant_30` still
wins count in all 3 and `constant_6` wins useful quality in all 3. This is a
promising bandwidth/quality pattern, but not yet a real-world or
scene-generalization claim:
the train/test textures are not independent real-world scenes at this stage,
and the sweep has no task correspondence.

The machine-readable outputs are [eveyes_trajectory_sweep_train.json](results/eveyes_trajectory_sweep_train.json)
and [eveyes_trajectory_sweep_test.json](results/eveyes_trajectory_sweep_test.json).

These EveYES files are rendered Blender scenes, not real event-camera
recordings. Their IMU is generated from trajectory derivatives plus deterministic
road vibration, as recorded in `motion_config.json`. They are suitable for
controlled simulation validation and interface testing, but not for claiming
real-world robustness.

The short IMU ablation outputs are
[eveyes_adaptive_probe_imu_prior_small.json](results/eveyes_adaptive_probe_imu_prior_small.json),
[eveyes_adaptive_probe_no_imu_small.json](results/eveyes_adaptive_probe_no_imu_small.json),
and [eveyes_adaptive_probe_no_flow_imu_small_v2.json](results/eveyes_adaptive_probe_no_flow_imu_small_v2.json).
They show that nonzero IMU changes the deficit and can change the selected
waveform, but IMU alone does not reliably identify the natural temporal deficit.

## Fair adaptive comparison

The calibrated adaptive policy was compared with fixed policies at identical
256x144 resolution, one-second duration, five windows, event threshold, and
prism model. This removes the resolution mismatch from earlier probes.

| mode | policy | density | edge F1 | efficiency |
| --- | --- | ---: | ---: | ---: |
| camera moving + object moving | constant_6 | 12.34 | 0.413 | 0.0767 |
| camera moving + object moving | adaptive | 10.52 | 0.410 | 0.0856 |
| camera moving + object static | constant_6 | 12.70 | 0.412 | 0.0747 |
| camera moving + object static | adaptive | 10.87 | 0.410 | 0.0830 |
| camera static + object moving | constant_6 | 6.89 | 0.470 | 0.1199 |
| camera static + object moving | adaptive | 6.83 | 0.471 | 0.1209 |

Adaptive therefore reduces bandwidth and improves efficiency relative to the
fixed useful baseline in this synthetic test, while preserving nearly the same
edge F1. The result is an engineering validation, not real-world evidence.
The complete table is [eveyes_policy_comparison.json](results/eveyes_policy_comparison.json).

## Clip-level stability

To check whether the effect depends on the first second, the same comparison
was repeated on seven temporal clips from each of the three test sequences
(21 clips total, 128x72, 0.5 s per clip). Adaptive used the train-calibrated
8 rad/s speed cap and the same IMU/flow state.

| statistic over 21 clips | value |
| --- | ---: |
| adaptive minus constant_6 event-density mean | -0.6051 |
| adaptive minus constant_6 event-density median | -0.9043 |
| adaptive minus constant_6 efficiency mean | +0.0144 |
| clips with positive efficiency delta | 21 / 21 |

The camera-moving/object-static sequence switched to constant_6 in one late
clip; all camera-moving/object-moving clips selected no prism after the initial
exploration, and all camera-static/object-moving clips selected constant_6.
This is stronger than a single-clip sanity check, but it remains synthetic
rendered evidence without a task ground truth. See
[eveyes_policy_stability.json](results/eveyes_policy_stability.json).
