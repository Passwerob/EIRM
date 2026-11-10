#!/usr/bin/env bash
# Wait for the event cache, then train three seeds and evaluate.
set -euo pipefail
ROOT=/data/fcr/AMI-Simulator
PYTHON=/data/fcr/miniconda3/envs/v2e/bin/python
CACHE=$ROOT/results/ami_next/e2vid_downstream/cache
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
cd "$ROOT"

need=48
echo "waiting for $need cached condition files in $CACHE"
while true; do
  n=$(find "$CACHE" -name '*__*.npz' ! -name '*luminance.npz' | wc -l)
  echo "$(date -Is) cached=$n / $need"
  if [ "$n" -ge "$need" ]; then
    break
  fi
  sleep 60
done

$PYTHON scripts/e2vid_downstream/eval.py \
  --weights /data/fcr/rpg_e2vid/pretrained/E2VID_lightweight.pth.tar \
  --cache "$CACHE" \
  --output $ROOT/results/ami_next/e2vid_downstream/zero_shot \
  --splits test --save-recon --tag zeroshot --device cuda

for seed in 42 123 3407; do
  $PYTHON scripts/e2vid_downstream/train.py \
    --config configs/e2vid_downstream/mixed_seed${seed}.json \
    --cache "$CACHE" \
    --output $ROOT/results/ami_next/e2vid_downstream/mixed_seed${seed} \
    --device cuda
  $PYTHON scripts/e2vid_downstream/eval.py \
    --weights $ROOT/results/ami_next/e2vid_downstream/mixed_seed${seed}/best.pt \
    --cache "$CACHE" \
    --output $ROOT/results/ami_next/e2vid_downstream/eval_seed${seed} \
    --splits test --save-recon --tag $seed --device cuda
done

$PYTHON scripts/e2vid_downstream/report.py \
  --zero-shot $ROOT/results/ami_next/e2vid_downstream/zero_shot/metrics.json \
  --mixed $ROOT/results/ami_next/e2vid_downstream/eval_seed42/metrics.json \
          $ROOT/results/ami_next/e2vid_downstream/eval_seed123/metrics.json \
          $ROOT/results/ami_next/e2vid_downstream/eval_seed3407/metrics.json \
  --cache "$CACHE" \
  --output $ROOT/results/ami_next/e2vid_downstream/report
echo done
