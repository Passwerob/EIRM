# KEEP — EIRM / AMI-Simulator paper weights. DO NOT DELETE

Copied on 2026-10-28 09:23  so cleanup of `results/` cannot drop the paper controllers.

These are tiny policy checkpoints (~21 KB), not Night3R-sized geometry weights.
`.pt` files are hard links when the filesystem allows it.

## Primary paper controller (use this)

`eveyes_table1_m0/policy.pt`

- EVEYES-native F1-argmax, margin 0
- Gears: {360, 180} with 180 quality floor
- Dataset tag: `EVEYES-co-motion-v2-regime-f1-oracles`
- Frozen as **Paper Table 1 / qualitative m0** (snapshots 2026-09-10 and 2026-09-12)
- Original: `results/eveyes_v2_regime_f1_argmax_m0/policy.pt`
- Eval protocol used with it: confidence 0.5, no IMU gate, dwell 0.025 s, prism 0.5 deg

Byte-identical copies already exist under:
- `results/snapshots/2026-09-10_eveyes_v2_m0/policy/policy.pt`
- `results/snapshots/2026-09-12_paper_event_qualitative/policy/policy.pt`

## Also kept (different experiments, do not mix)

`eveyes_timescale_dsd/policy.pt`
- Extra drive-stop-drive training data; used for the timescale-180 figure, **not** Table 1.
- Original: `results/eveyes_v2_plus_dsd_f1_argmax/policy.pt`

`procedural_64x64_main_table/policy.pt`
- 64x64 procedural pack (`comotion_controlled`). Older `MAIN_EXPERIMENT_RESULTS.txt` Table 1.
- **Not** the EVEYES photorealistic Table 1. Do not evaluate it on EVEYES native.
- Original: `results/comotion_f1_argmax_policy/policy.pt`
