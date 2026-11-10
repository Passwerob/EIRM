"""Frozen protocol for the EIRM → E2VID downstream benchmark.

Do not change EIRM gears, dwell, prism angle, or the paper controller.
Event streams are simulated with the existing AMI / Ours pipeline and
inverse-warped to the RGB reference plane before reconstruction.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E2VID_ROOT = Path("/data/fcr/rpg_e2vid")
E2VID_WEIGHTS = E2VID_ROOT / "pretrained/E2VID_lightweight.pth.tar"
POLICY_PATH = ROOT / "KEEP_PAPER_WEIGHTS_DO_NOT_DELETE/eveyes_table1_m0/policy.pt"
DATASET_ROOT = ROOT / "data/eveyes_comotion_v2_ready"
OUT_ROOT = ROOT / "results/ami_next/e2vid_downstream"

# Scene-level split is applied *before* building modulation variants.
PAPER_SCENES = {
    "train": ("Scene_002",),
    "valid": ("Scene_004",),
    "test": ("Scene_005",),
}
PAPER_REGIMES = ("object_only", "camera_only", "co_motion", "drive_stop_drive")
# Extra test-only mixed-switch sequences. Same Scene_005, never used for training.
EXTRA_TEST_REGIMES = ("full_switch", "gear_switch_native")

TRAIN_CONDITIONS = ("fixed_180", "fixed_360", "eirm")
EVAL_CONDITIONS = ("no_prism", "fixed_180", "fixed_360", "eirm")
CONDITION_LABELS = {
    "no_prism": "No Prism",
    "fixed_180": "Fixed-180",
    "fixed_360": "Fixed-360",
    "eirm": "EIRM",
    "eirm_imu": "EIRM + IMU",
    "eirm_no_imu": "EIRM no IMU",
}
IMU_ABLATION = ("eirm_imu", "eirm", "eirm_no_imu")
SEEDS = (42, 123, 3407)

# Paper Table-1 controller protocol (KEEP_PAPER_WEIGHTS README).
PRISM_ANGLE_DEG = 0.5
CONFIDENCE = 0.50
REQUIRE_IMU = False
DWELL_S = 0.025
CONTRAST_THRESHOLD = 0.2

NUM_BINS = 5
NUM_ENCODERS = 3  # E2VID_lightweight
SEQ_LEN = 8
WARMUP_FRAMES = 2
CROP = 256
BATCH_SIZE = 2
LR = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 12
STEPS_PER_EPOCH = 160
L1_WEIGHT = 1.0
SSIM_WEIGHT = 0.5
VAL_STRIDE = 8

MOTION_REGIME_MAP = {
    "object_only": "weak",
    "camera_only": "moderate",
    "co_motion": "mixed",
    "drive_stop_drive": "mixed_change",
    "full_switch": "mixed_change",
    "gear_switch_native": "mixed_change",
}


def scene_ids(split: str, *, extra_test: bool = False) -> list[str]:
    scenes = PAPER_SCENES[split]
    regimes = list(PAPER_REGIMES)
    if extra_test and split == "test":
        regimes.extend(EXTRA_TEST_REGIMES)
    return [f"{split}/{scene}/{regime}" for scene in scenes for regime in regimes]


def cache_stem(scene_id: str, condition: str) -> str:
    return scene_id.replace("/", "__") + f"__{condition}"


def cache_path(cache_dir: Path, scene_id: str, condition: str) -> Path:
    return cache_dir / f"{cache_stem(scene_id, condition)}.npz"
