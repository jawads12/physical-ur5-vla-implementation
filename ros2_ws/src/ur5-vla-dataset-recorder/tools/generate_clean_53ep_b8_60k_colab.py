"""Generate the A100 batch-8, 60k-step SmolVLA Colab notebook with W&B."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks/train_ur5_cube_clean_53ep_smolvla_80000_wandb_colab.ipynb"
OUTPUT = ROOT / "notebooks/train_ur5_cube_clean_53ep_smolvla_b8_60000_wandb_colab.ipynb"


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict, text: str) -> None:
    cell["source"] = text.strip().splitlines(keepends=True)


def find_cell(notebook: dict, marker: str) -> dict:
    matches = [cell for cell in notebook["cells"] if marker in source(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cell containing {marker!r}, found {len(matches)}")
    return matches[0]


notebook = json.loads(BASE.read_text())

for cell in notebook["cells"]:
    text = source(cell)
    text = text.replace("80,000 steps", "60,000 steps")
    text = text.replace("80,000-step", "60,000-step")
    text = text.replace("TRAIN_STEPS = 80000", "TRAIN_STEPS = 60000")
    text = text.replace("BATCH_SIZE = 1", "BATCH_SIZE = 8")
    text = text.replace(
        'RUN_NAME = f"ur5_cube_clean_53ep_smolvla_{TRAIN_STEPS}_steps"',
        'RUN_NAME = f"ur5_cube_clean_53ep_smolvla_b{BATCH_SIZE}_{TRAIN_STEPS}_steps"',
    )
    text = text.replace("smolvla_53ep_80k", "smolvla_53ep_b8_60k")
    text = text.replace("005000` through `080000", "005000` through `060000")
    text = text.replace("080000", "060000")
    cell["source"] = text.splitlines(keepends=True)

set_source(
    notebook["cells"][0],
    """
# UR5 cube-to-bowl SmolVLA training — 53 episodes / A100 batch 8 / 60,000 steps

This is a fresh SmolVLA fine-tuning run for `ur5_cube_clean_53ep`. It is configured for an A100, processes eight samples per optimizer step, performs 60,000 optimizer steps (about 7.1 dataset passes), logs live metrics to W&B, and automatically copies every complete 5,000-step checkpoint to Google Drive. Run every cell from top to bottom. It never contacts ROS or the robot and never uploads to Hugging Face.

Before starting, add `WANDB_API_KEY` to Colab Secrets, keep the Drive folders restricted, and select an A100 GPU runtime.
""",
)

set_source(
    find_cell(notebook, "TRAIN_STEPS = 60000"),
    """
import json, os, random
import numpy as np, torch

TRAIN_STEPS = 60000
SAVE_FREQ = 5000
KEEP_LOCAL_CHECKPOINTS = 2
BATCH_SIZE = 8
SEED = 42
WANDB_PROJECT = "smolvla_53ep_b8_60k"
RUN_NAME = f"ur5_cube_clean_53ep_smolvla_b{BATCH_SIZE}_{TRAIN_STEPS}_steps"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
TRAIN_OUTPUT = Path("/content/outputs") / RUN_NAME
LOCAL_REPORT_DIR = Path("/content/training_reports")
DRIVE_RUN_DIR = CHECKPOINT_ROOT / RUN_NAME
DRIVE_CHECKPOINTS_DIR = DRIVE_RUN_DIR / "checkpoints"
LOCAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
assert not TRAIN_OUTPUT.exists(), f"Local output already exists: {TRAIN_OUTPUT}. Restart the runtime or change RUN_NAME."
assert not DRIVE_RUN_DIR.exists(), f"Drive output already exists: {DRIVE_RUN_DIR}. Change RUN_NAME to preserve it."

TRAIN_ARGS = [
    "lerobot-train", "--policy.type=smolvla", "--policy.pretrained_path=lerobot/smolvla_base",
    "--policy.device=cuda", "--policy.use_amp=true", "--policy.push_to_hub=false",
    f"--dataset.repo_id={REPO_ID}", f"--dataset.root={LOCAL_DATASET}", "--dataset.video_backend=pyav",
    f"--batch_size={BATCH_SIZE}", f"--seed={SEED}", "--num_workers=4",
    "--wandb.enable=true", f"--wandb.project={WANDB_PROJECT}", "--wandb.mode=online",
    "--save_checkpoint=true", "--eval_freq=0", f"--steps={TRAIN_STEPS}", f"--save_freq={SAVE_FREQ}",
    "--log_freq=10", f"--output_dir={TRAIN_OUTPUT}", f"--job_name={RUN_NAME}",
]
resolved_training = {
    "command": TRAIN_ARGS, "train_steps": TRAIN_STEPS, "save_frequency": SAVE_FREQ,
    "drive_checkpoint_retention": "all", "local_checkpoint_retention": KEEP_LOCAL_CHECKPOINTS,
    "batch_size": BATCH_SIZE, "seed": SEED, "approximate_dataset_passes": TRAIN_STEPS * BATCH_SIZE / 67621,
    "dataset_episodes": 53, "dataset_frames": 67621, "rgb_only": True,
    "real_camera_views": 2, "empty_camera_slots": 0, "task": TASK,
    "state_semantics": "six measured UR5 joints plus estimated pneumatic command state",
    "action_semantics": "six absolute joint targets plus binary gripper command",
    "wandb_project": WANDB_PROJECT,
}
(LOCAL_REPORT_DIR / "resolved_training_config.json").write_text(json.dumps(resolved_training, indent=2))
print(json.dumps(resolved_training, indent=2))
""",
)

set_source(
    find_cell(notebook, "from torch.utils.data import DataLoader"),
    """
import gc, psutil, time, torch
from torch.utils.data import DataLoader
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

started = time.time()
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
PRETRAINED_MODEL = "lerobot/smolvla_base"
cfg = SmolVLAConfig(pretrained_path=Path(PRETRAINED_MODEL), device="cuda", use_amp=True, push_to_hub=False)
delta_timestamps = resolve_delta_timestamps(cfg, dataset.meta)
training_dataset = LeRobotDataset(
    REPO_ID, root=LOCAL_DATASET, delta_timestamps=delta_timestamps, video_backend="pyav"
)
batch = next(iter(DataLoader(training_dataset, batch_size=BATCH_SIZE, num_workers=0)))
assert tuple(batch["action"].shape) == (BATCH_SIZE, cfg.chunk_size, 7), batch["action"].shape
policy = make_policy(cfg, ds_meta=training_dataset.meta).train()
preprocessor, _ = make_pre_post_processors(
    cfg,
    pretrained_path=PRETRAINED_MODEL,
    dataset_stats=training_dataset.meta.stats,
    preprocessor_overrides={
        "device_processor": {"device": "cuda"},
        "normalizer_processor": {
            "stats": training_dataset.meta.stats,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
        "rename_observations_processor": {"rename_map": {}},
    },
)
model_batch = {key: (value.cuda() if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}
model_batch["task"] = [TASK] * BATCH_SIZE
optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=1e-5)
optimizer.zero_grad(set_to_none=True)
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss, _ = policy.forward(preprocessor(model_batch))
assert torch.isfinite(loss), f"Non-finite batch-8 compatibility loss: {loss}"
loss.backward(); optimizer.step(); torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated()
total = torch.cuda.get_device_properties(0).total_memory
assert peak < total * 0.90, f"Batch 8 leaves insufficient VRAM headroom: peak={peak/2**30:.1f} GiB total={total/2**30:.1f} GiB"
compatibility_report = {
    "pass": True, "batch_size": BATCH_SIZE, "loss": float(loss.detach()),
    "elapsed_seconds": time.time() - started, "cpu_ram_bytes": psutil.Process().memory_info().rss,
    "peak_gpu_vram_bytes": peak, "total_gpu_vram_bytes": total,
    "peak_gpu_fraction": peak / total,
}
(LOCAL_REPORT_DIR / "compatibility_report.json").write_text(json.dumps(compatibility_report, indent=2))
print("A100 batch-8 forward/backward/optimizer preflight PASS", json.dumps(compatibility_report, indent=2))
del optimizer, policy, preprocessor, model_batch, batch, training_dataset
gc.collect(); torch.cuda.empty_cache()
""",
)

# Make the explanatory text exact for this experiment.
for cell in notebook["cells"]:
    text = source(cell)
    text = text.replace("Configure the 60,000-step training run", "Configure the A100 batch-8 / 60,000-step training run")
    text = text.replace(
        "This notebook intentionally uses 60,000 optimizer steps.",
        "This notebook uses 60,000 optimizer steps with batch size 8, approximately 7.1 dataset passes.",
    )
    text = text.replace(
        "The single training cell automatically retains checkpoints `005000` through `060000`",
        "The single training cell automatically retains all 12 checkpoints from `005000` through `060000`",
    )
    cell["source"] = text.splitlines(keepends=True)

notebook["metadata"].setdefault("colab", {})["name"] = OUTPUT.name
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(OUTPUT)
