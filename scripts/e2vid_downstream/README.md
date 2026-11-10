# EIRM → E2VID downstream reconstruction

Zero-shot original E2VID, then **one shared** mixed-trained E2VID per seed
(`42 / 123 / 3407`) on Fixed-180 + Fixed-360 + EIRM. No Prism is evaluation-only.
The EIRM controller is not modified.

## What this is / is not

- Downstream **image reconstruction** of already-generated event streams.
- Events for AMI conditions are **inverse-warped to the RGB reference plane**.
  No Prism is native DVS on that same plane. PSNR is not claimed on raw
  sensor-plane prism orbits.
- Target = linear luminance of the RGB frame at each window end.
- PSNR / SSIM / LPIPS use a **per-image affine (scale+shift)** fit. E2VID is
  not photometrically calibrated; unaligned PSNR is stored only as a diagnostic.
- Zero-shot vs mixed-trained is **not** a training-matched comparison.

## Data

- Dataset: `data/eveyes_comotion_v2_ready` (960×540, 60 Hz, scene-level split).
- Train `Scene_002` / valid `Scene_004` / test `Scene_005`.
- Regimes: `object_only` (weak), `camera_only` (moderate), `co_motion` (mixed),
  `drive_stop_drive` (mixed with motion-regime changes).
- Controller: `KEEP_PAPER_WEIGHTS_DO_NOT_DELETE/eveyes_table1_m0/policy.pt`
  (confidence 0.5, dwell 25 ms, prism 0.5°).
- IMU ablation (same E2VID weights, different event streams):
  - `eirm_imu`: IMU features ON + IMU gate ON
  - `eirm`: paper setting — IMU features ON, gate OFF
  - `eirm_no_imu`: IMU features zeroed, gate OFF

Do not point this pipeline at `data/comotion_controlled_v2` (64×64 toy pack).

## Reproduce

```bash
cd /data/fcr/AMI-Simulator
export PYTHON=/data/fcr/miniconda3/envs/v2e/bin/python
export CUDA_VISIBLE_DEVICES=4

# 1. Cache reference-plane events (resume-safe)
$PYTHON scripts/e2vid_downstream/simulate_cache.py --device cuda

# 2. Zero-shot original E2VID
$PYTHON scripts/e2vid_downstream/eval.py \
  --weights /data/fcr/rpg_e2vid/pretrained/E2VID_lightweight.pth.tar \
  --output results/ami_next/e2vid_downstream/zero_shot \
  --splits test --save-recon --tag zeroshot

# 3. Three mixed-training seeds (same hparams, different seed)
for seed in 42 123 3407; do
  $PYTHON scripts/e2vid_downstream/train.py \
    --config configs/e2vid_downstream/mixed_seed${seed}.json \
    --device cuda
  $PYTHON scripts/e2vid_downstream/eval.py \
    --weights results/ami_next/e2vid_downstream/mixed_seed${seed}/best.pt \
    --output results/ami_next/e2vid_downstream/eval_seed${seed} \
    --splits test --save-recon --tag $seed
done

# 4. Tables + figures
$PYTHON scripts/e2vid_downstream/report.py \
  --zero-shot results/ami_next/e2vid_downstream/zero_shot/metrics.json \
  --mixed results/ami_next/e2vid_downstream/eval_seed42/metrics.json \
          results/ami_next/e2vid_downstream/eval_seed123/metrics.json \
          results/ami_next/e2vid_downstream/eval_seed3407/metrics.json \
  --cache results/ami_next/e2vid_downstream/cache \
  --output results/ami_next/e2vid_downstream/report
```

Smoke (short clips, two optimization steps):

```bash
$PYTHON scripts/e2vid_downstream/simulate_cache.py \
  --scene-ids train/Scene_002/object_only valid/Scene_004/object_only \
              test/Scene_005/object_only test/Scene_005/drive_stop_drive \
  --max-frames 16 --device cuda \
  --output results/ami_next/e2vid_downstream/cache_smoke

$PYTHON scripts/e2vid_downstream/train.py \
  --config configs/e2vid_downstream/mixed_seed42.json \
  --cache results/ami_next/e2vid_downstream/cache_smoke \
  --output results/ami_next/e2vid_downstream/mixed_seed42_smoke \
  --smoke --device cuda
```

## Outputs

| Path | Contents |
|---|---|
| `results/ami_next/e2vid_downstream/cache/` | Event npz + luminance, `index.json` |
| `mixed_seed{42,123,3407}/best.pt` | Fine-tuned checkpoints (do not overwrite pretrained) |
| `zero_shot/metrics.json` | Table 1 |
| `eval_seed*/metrics.json` | Table 2/3 per-frame rows |
| `report/REPORT.md` | Markdown tables + figure captions |
