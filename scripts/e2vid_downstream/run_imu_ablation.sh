#!/usr/bin/env bash
# IMU ablation: EIRM + IMU gate  vs  paper EIRM (feature, no gate)  vs  EIRM no IMU.
# Shared E2VID weights. Does not retrain. Does not overwrite the main mixed-training cache.
set -euo pipefail
ROOT=/data/fcr/AMI-Simulator
PYTHON=/data/fcr/miniconda3/envs/v2e/bin/python
CACHE=$ROOT/results/ami_next/e2vid_downstream/cache
WEIGHTS=/data/fcr/rpg_e2vid/pretrained/E2VID_lightweight.pth.tar
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
cd "$ROOT"

echo "caching eirm_imu (IMU gate ON, IMU features ON)"
$PYTHON scripts/e2vid_downstream/simulate_cache.py \
  --conditions eirm_imu \
  --splits train valid test \
  --device cuda \
  --output "$CACHE"

wait_cond() {
  local cond=$1 need=$2
  while true; do
    n=$(find "$CACHE" -name "*__${cond}.npz" | wc -l)
    echo "$(date -Is) ${cond}=$n / $need"
    if [ "$n" -ge "$need" ]; then
      return 0
    fi
    sleep 60
  done
}

wait_cond eirm_imu 12
wait_cond eirm_no_imu 12

echo "zero-shot eval IMU ablation"
$PYTHON scripts/e2vid_downstream/eval.py \
  --weights "$WEIGHTS" \
  --cache "$CACHE" \
  --output $ROOT/results/ami_next/e2vid_downstream/zero_shot_imu_ablation \
  --splits test \
  --conditions eirm_imu eirm eirm_no_imu \
  --save-recon --tag imu_ablation --device cuda

$PYTHON scripts/e2vid_downstream/report.py \
  --zero-shot $ROOT/results/ami_next/e2vid_downstream/zero_shot_imu_ablation/metrics.json \
  --output $ROOT/results/ami_next/e2vid_downstream/report_imu_ablation \
  --hero-scene test/Scene_005/drive_stop_drive

echo done imu ablation
