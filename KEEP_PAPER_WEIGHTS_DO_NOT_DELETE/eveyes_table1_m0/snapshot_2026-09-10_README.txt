Snapshot of the three-regime F1-argmax margin-0 controller, before adding CRUISE IMU=0 data.

Checkpoint: policy/policy.pt
Test eval: eveyes_v2_regimes_test_ours_m0.json
  object_only  F1 0.652  mean RPM 180  frac_360 0.00
  camera_only  F1 0.580  mean RPM 273  frac_360 0.52
  co_motion    F1 0.580  mean RPM 262  frac_360 0.46
Teacher: f1-argmax, upshift_margin 0.0
Eval: confidence 0.50, no-require-imu, use_temporal_overlap false
Date: 2026-09-10
