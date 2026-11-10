# NEW: Adaptive Optical Excitation Experiment Layer

This directory contains the redesign plan and, after approval, the new
active-sensing experiment code. The existing simulator in the parent directory
is retained and reused.

Current status: Stage-1 analytical-controller pilot. The new layer includes
single-DoF trajectory generation, actuator limiting, Farneback flow features,
a rule-based controller, polarity-aware event time surfaces, ORB matching
metrics, deterministic trajectory sweeps, and causal adaptive rollouts. This
is pipeline validation, not paper evidence.

- [Experiment plan](EXPERIMENT_PLAN.md): hypotheses, benchmark, metrics,
  ablations, and decision gates.
- [Code modification plan](CODE_MODIFICATION_PLAN.md): proposed modules,
  interfaces, integration points, tests, and implementation order.
- [Pilot results](RESULTS.md): reproducible Stage-0/Stage-1 gate outcomes and
  the current decision about learned control.
- `ami_active_sensing.data.eveyes_dataset`: lazy scene adapter for the
  scene-disjoint EveYES export, including canonical SI-unit IMU loading.

The first implementation milestone is a reproducible controlled trajectory
sweep and feature-matching evaluator. Neural controller training is explicitly
deferred until the sweep demonstrates that different scenes require different
optical excitation.

Run the unit and integration tests from the repository root:

```bash
PYTHONPATH=NEW:. python -m unittest discover -s NEW/tests -v
```

Run the five-policy synthetic smoke experiment:

```bash
PYTHONPATH=NEW:. python NEW/scripts/run_smoke_experiment.py \
  --output NEW/results/smoke_experiment.json
```

Test the scene-dependent-trajectory hypothesis across four textures, four
natural-motion regimes, and eight trajectory families:

```bash
PYTHONPATH=NEW:. python NEW/scripts/run_scene_motion_sweep.py \
  --output NEW/results/scene_motion_sweep.json
```

Run the rule-based controller in causal autonomous rollout on the same pilot
conditions (the controller only sees completed prior windows):

```bash
PYTHONPATH=NEW:. python NEW/scripts/run_adaptive_pilot.py \
  --sweep NEW/results/scene_motion_sweep.csv \
  --output NEW/results/adaptive_pilot.json
```

Audit the extracted RGB/timestamp/IMU contracts without loading the full
dataset into memory:

```bash
PYTHONPATH=NEW:. python NEW/scripts/audit_eveyes_dataset.py \
  --dataset NEW/data/eveyes_single_car_dataset_final_imu_v2 \
  --output NEW/results/eveyes_imu_audit.json \
  --sample-stride 60
```

Run the short synthetic-rendered EveYES event-quality probe (RGB is resized; no task-level
matching claim is made without correspondence/depth):

```bash
PYTHONPATH=NEW:. python NEW/scripts/run_eveyes_event_probe.py \
  --dataset NEW/data/eveyes_single_car_dataset_final_imu_v2 \
  --output NEW/results/eveyes_event_probe.json
```

Run the causal adaptive probe on all three synthetic test motion modes (IMU is included
in the state by default). The default 8 rad/s useful-speed cap is calibrated
from the train trajectory sweep:

```bash
PYTHONPATH=NEW:. python NEW/scripts/run_eveyes_adaptive_probe.py \
  --dataset NEW/data/eveyes_single_car_dataset_final_imu_v2 \
  --output NEW/results/eveyes_adaptive_probe.json
```

Sweep all eight trajectory families on a selected synthetic EveYES split. This
is an event-quality-only evaluation until correspondence/depth is available:

```bash
PYTHONPATH=NEW:. python NEW/scripts/run_eveyes_trajectory_sweep.py \
  --dataset NEW/data/eveyes_single_car_dataset_final_imu_v2 \
  --split train \
  --output NEW/results/eveyes_trajectory_sweep_train.json
```

Compare fixed and calibrated adaptive policies under identical RGB, timing,
resolution, and event-simulator settings:

```bash
PYTHONPATH=NEW:. python NEW/scripts/compare_eveyes_policies.py \
  --dataset NEW/data/eveyes_single_car_dataset_final_imu_v2 \
  --output NEW/results/eveyes_policy_comparison.json
```

Check stability across multiple temporal clips of every synthetic test
sequence:

```bash
PYTHONPATH=NEW:. python NEW/scripts/run_eveyes_policy_stability.py \
  --dataset NEW/data/eveyes_single_car_dataset_final_imu_v2 \
  --output NEW/results/eveyes_policy_stability.json
```

The pilot report separates two claims: per-condition oracle regret (adaptive
versus the best trajectory selected after seeing all results), and the mean
delta versus each global fixed trajectory. The latter is a useful sanity check
for a deployable controller, but neither is paper evidence without held-out
scenes and repeated seeds. The current synthetic pilot is intended to decide
whether that larger evaluation is warranted.
