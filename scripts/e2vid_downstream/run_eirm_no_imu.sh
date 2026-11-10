#!/usr/bin/env bash
# Cache EIRM with IMU features zeroed, then zero-shot eval that condition only.
set -euo pipefail
ROOT=/data/fcr/AMI-Simulator
PYTHON=/data/fcr/miniconda3/envs/v2e/bin/python
CACHE=$ROOT/results/ami_next/e2vid_downstream/cache
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
cd "$ROOT"

$PYTHON scripts/e2vid_downstream/simulate_cache.py \
  --conditions eirm_no_imu \
  --splits train valid test \
  --device cuda \
  --output "$CACHE"

need=12
echo "waiting for $need eirm_no_imu files"
while true; do
  n=$(find "$CACHE" -name '*__eirm_no_imu.npz' | wc -l)
  echo "$(date -Is) eirm_no_imu=$n / $need"
  if [ "$n" -ge "$need" ]; then
    break
  fi
  sleep 60
done

$PYTHON scripts/e2vid_downstream/eval.py \
  --weights /data/fcr/rpg_e2vid/pretrained/E2VID_lightweight.pth.tar \
  --cache "$CACHE" \
  --output $ROOT/results/ami_next/e2vid_downstream/zero_shot_eirm_no_imu \
  --splits test \
  --conditions eirm_no_imu \
  --save-recon --tag eirm_no_imu --device cuda
echo done eirm_no_imu
