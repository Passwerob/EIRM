#!/usr/bin/env python3
"""Fine-tune one shared E2VID on mixed Fixed-180 / Fixed-360 / EIRM.

No Prism is held out of training. Three seeds share architecture, data,
and hyperparameters; only the RNG seed changes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from data import load_sequence, sample_crop, sequence_item  # noqa: E402
from metrics import ssim_torch  # noqa: E402
from model_io import forward_frame, freeze_batchnorm, load_e2vid  # noqa: E402
from protocol import (  # noqa: E402
    BATCH_SIZE,
    CROP,
    EPOCHS,
    EVAL_CONDITIONS,
    L1_WEIGHT,
    LR,
    OUT_ROOT,
    SEQ_LEN,
    SSIM_WEIGHT,
    STEPS_PER_EPOCH,
    TRAIN_CONDITIONS,
    VAL_STRIDE,
    WARMUP_FRAMES,
    WEIGHT_DECAY,
    scene_ids,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pool(cache_dir: Path, split: str, conditions: tuple[str, ...], wanted: list[str] | None = None) -> dict[str, object]:
    pool = {}
    for scene_id in wanted or scene_ids(split):
        if scene_id.split("/")[0] != split:
            continue
        for condition in conditions:
            path = cache_dir / f"{scene_id.replace('/', '__')}__{condition}.npz"
            if not path.exists():
                print(f"skip missing {path.name}", flush=True)
                continue
            pool[f"{scene_id}::{condition}"] = load_sequence(cache_dir, scene_id, condition)
    if not pool:
        raise FileNotFoundError(f"no cached sequences for split={split} conditions={conditions}")
    return pool


def make_batch(pool: dict, rng: np.random.Generator, *, batch_size: int, crop: int, seq_len: int, conditions: tuple[str, ...]):
    keys_by_condition = {c: [k for k in pool if k.endswith(f"::{c}")] for c in conditions}
    voxels, targets, meta = [], [], []
    for _ in range(batch_size):
        condition = str(rng.choice(conditions))
        key = str(rng.choice(keys_by_condition[condition]))
        cache = pool[key]
        max_start = cache.n_windows - seq_len
        if max_start <= 0:
            raise RuntimeError(f"{key} too short for seq_len={seq_len}")
        start = int(rng.integers(0, max_start))
        x0, y0 = sample_crop(cache.width, cache.height, crop, rng)
        item = sequence_item(cache, start, crop=crop, x0=x0, y0=y0, seq_len=seq_len)
        voxels.append(item["voxel"])
        targets.append(item["target"])
        meta.append({"scene_id": cache.scene_id, "condition": condition, "start": start})
    voxel = torch.from_numpy(np.stack(voxels))  # [B, T, C, H, W]
    target = torch.from_numpy(np.stack(targets)).unsqueeze(2)  # [B, T, 1, H, W]
    return voxel, target, meta


def sequence_loss(model: nn.Module, voxel: torch.Tensor, target: torch.Tensor, *, warmup: int) -> dict[str, torch.Tensor]:
    batch, seq, _, _, _ = voxel.shape
    states = None
    preds = []
    for t in range(seq):
        recon, states = forward_frame(model, voxel[:, t], states)
        preds.append(recon)
    pred = torch.stack(preds, dim=1)
    use = pred[:, warmup:]
    gt = target[:, warmup:]
    l1 = torch.mean(torch.abs(use - gt))
    ssim = ssim_torch(use.flatten(0, 1), gt.flatten(0, 1))
    loss = L1_WEIGHT * l1 + SSIM_WEIGHT * (1.0 - ssim)
    return {"loss": loss, "l1": l1, "ssim": ssim}


@torch.no_grad()
def validate(model: nn.Module, pool: dict, device: torch.device, *, stride: int, crop: int, seq_len: int) -> dict[str, float]:
    model.eval()
    losses = []
    for cache in pool.values():
        for start in range(0, cache.n_windows - seq_len + 1, stride):
            x0 = max((cache.width - crop) // 2, 0)
            y0 = max((cache.height - crop) // 2, 0)
            item = sequence_item(cache, start, crop=min(crop, cache.width), x0=x0, y0=y0, seq_len=seq_len)
            voxel = torch.from_numpy(item["voxel"][None]).to(device)
            target = torch.from_numpy(item["target"][None, :, None]).to(device)
            out = sequence_loss(model, voxel, target, warmup=WARMUP_FRAMES)
            losses.append(float(out["loss"].item()))
    model.train()
    freeze_batchnorm(model)
    return {"val_loss": float(np.mean(losses)) if losses else float("nan"), "n": float(len(losses))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=OUT_ROOT / "cache")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--scene-ids", nargs="*", default=None)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    set_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = args.output or (OUT_ROOT / f"mixed_seed{seed}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    train_pool = load_pool(args.cache, "train", TRAIN_CONDITIONS, args.scene_ids)
    valid_pool = load_pool(args.cache, "valid", TRAIN_CONDITIONS, args.scene_ids)
    model = load_e2vid(device=device)
    freeze_batchnorm(model)
    model.train()
    freeze_batchnorm(model)
    opt = Adam((p for p in model.parameters() if p.requires_grad), lr=float(cfg.get("lr", LR)), weight_decay=float(cfg.get("weight_decay", WEIGHT_DECAY)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(seed)
    epochs = int(args.epochs or (1 if args.smoke else cfg.get("epochs", EPOCHS)))
    steps = int(args.steps or (2 if args.smoke else cfg.get("steps_per_epoch", STEPS_PER_EPOCH)))
    history = []
    best = None
    for epoch in range(epochs):
        model.train()
        freeze_batchnorm(model)
        running = []
        for step in range(steps):
            voxel, target, _ = make_batch(
                train_pool,
                rng,
                batch_size=int(cfg.get("batch_size", BATCH_SIZE)),
                crop=int(cfg.get("crop", CROP)),
                seq_len=int(cfg.get("seq_len", SEQ_LEN)),
                conditions=TRAIN_CONDITIONS,
            )
            voxel = voxel.to(device)
            target = target.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out_loss = sequence_loss(model, voxel, target, warmup=int(cfg.get("warmup_frames", WARMUP_FRAMES)))
            scaler.scale(out_loss["loss"]).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running.append(float(out_loss["loss"].item()))
            if args.smoke or step % 20 == 0:
                print(f"epoch {epoch} step {step}/{steps} loss={out_loss['loss'].item():.4f} l1={out_loss['l1'].item():.4f} ssim={out_loss['ssim'].item():.4f}", flush=True)
        val = validate(
            model,
            valid_pool,
            device,
            stride=max(VAL_STRIDE, 32 if args.smoke else VAL_STRIDE),
            crop=int(cfg.get("crop", CROP)),
            seq_len=int(cfg.get("seq_len", SEQ_LEN)),
        )
        row = {"epoch": epoch, "train_loss": float(np.mean(running)), **val}
        history.append(row)
        print("val", row, flush=True)
        ckpt = {
            "arch": "E2VIDRecurrent",
            "state_dict": model.state_dict(),
            "model": {
                "num_bins": 5,
                "skip_type": "sum",
                "recurrent_block_type": "convlstm",
                "num_encoders": 3,
                "base_num_channels": 32,
                "num_residual_blocks": 2,
                "use_upsample_conv": False,
                "norm": "BN",
            },
            "seed": seed,
            "epoch": epoch,
            "val_loss": row["val_loss"],
            "config": cfg,
        }
        torch.save(ckpt, out / "last.pt")
        if best is None or row["val_loss"] < best:
            best = row["val_loss"]
            torch.save(ckpt, out / "best.pt")
            print(f"  saved best val_loss={best:.4f}", flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("done", out, "eval_conditions_for_later", list(EVAL_CONDITIONS), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
