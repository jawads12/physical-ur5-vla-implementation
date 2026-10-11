"""Generate the 28-accepted-episode, 40k-step SmolVLA Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks/train_ur5_smolvla_colab.ipynb"
OUTPUT = ROOT / "notebooks/train_ur5_cube_30ep_smolvla_40000_colab.ipynb"


def source(cell: dict) -> str:
    return "".join(cell["source"])


def set_source(cell: dict, text: str) -> None:
    cell["source"] = text.strip().splitlines(keepends=True)


def find_cell(notebook: dict, marker: str) -> dict:
    matches = [cell for cell in notebook["cells"] if marker in source(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cell containing {marker!r}, found {len(matches)}")
    return matches[0]


notebook = json.loads(BASE.read_text())

set_source(
    notebook["cells"][0],
    """
# UR5 cube-to-bowl SmolVLA training — 28 episodes / 40,000 steps

This is a fresh training run for the converted physical-UR5 dataset `ur5_cube_30ep`. It copies the unpacked LeRobot dataset from Google Drive to fast Colab storage, validates it, fine-tunes SmolVLA for 40,000 steps, and copies verified resumable checkpoints and final weights back to Drive. Run every cell from top to bottom. It never contacts ROS or the robot and never uploads to Hugging Face.

Before starting, keep the shared Drive folders **Restricted**, add the Colab account as Editor, and ensure `pHYSICAL UR5` is available under My Drive. A GPU runtime is required.
""",
)

set_source(
    find_cell(notebook, "DATASET_ARCHIVE ="),
    """
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/pHYSICAL UR5")
DRIVE_DATASET = DRIVE_ROOT / "datasets/lerobot/ur5_cube_30ep"
CHECKPOINT_ROOT = DRIVE_ROOT / "checkpoints/ur5_smolvla_trained"
REPORT_ROOT = DRIVE_ROOT / "reports/training"
DEPLOYMENT_ROOT = DRIVE_ROOT / "deployment/downloaded_checkpoints"

assert DRIVE_ROOT.is_dir(), f"Drive shortcut/path not found: {DRIVE_ROOT}"
assert DRIVE_DATASET.is_dir(), f"Converted dataset folder missing: {DRIVE_DATASET}"
for required in ("meta/info.json", "meta/stats.json", "meta/tasks.parquet", "data", "videos"):
    assert (DRIVE_DATASET / required).exists(), f"Dataset item missing: {DRIVE_DATASET / required}"
for directory in (CHECKPOINT_ROOT, REPORT_ROOT, DEPLOYMENT_ROOT):
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / ".write_test"; probe.write_text("ok"); probe.unlink()
(REPORT_ROOT / "runtime_report.json").write_text(json.dumps(runtime_report, indent=2))
print("Drive dataset:", DRIVE_DATASET)
print("Drive output:", CHECKPOINT_ROOT)
""",
)

set_source(
    find_cell(notebook, "## D — Copy, verify, and safely extract the dataset"),
    """
## D — Copy and verify the converted dataset

The dataset is already converted and unpacked in Drive. Copy it to `/content` before training; training directly against mounted Drive is too slow and fragile. `rsync` prints transfer progress, and the next cells perform structural, numerical, task, and video-decode validation.
""",
)

set_source(
    find_cell(notebook, "LOCAL_ARCHIVE ="),
    """
import shutil, subprocess

LOCAL_DATASET = Path("/content/ur5_cube_30ep")
usage = shutil.disk_usage("/content")
drive_bytes = sum(path.stat().st_size for path in DRIVE_DATASET.rglob("*") if path.is_file())
assert usage.free > drive_bytes + 25 * 1024**3, (
    f"Insufficient Colab disk: need dataset size plus 25 GiB; "
    f"dataset={drive_bytes / 1024**3:.1f} GiB free={usage.free / 1024**3:.1f} GiB"
)
assert not LOCAL_DATASET.exists(), f"Local dataset already exists: {LOCAL_DATASET}. Restart the runtime or remove only that exact directory."
LOCAL_DATASET.mkdir(parents=True)
subprocess.run(
    ["rsync", "-ah", "--info=progress2", f"{DRIVE_DATASET}/", f"{LOCAL_DATASET}/"],
    check=True,
)

def file_inventory(root):
    return {str(path.relative_to(root)): path.stat().st_size for path in root.rglob("*") if path.is_file()}

drive_inventory = file_inventory(DRIVE_DATASET)
local_inventory = file_inventory(LOCAL_DATASET)
assert local_inventory == drive_inventory, "Drive-to-local dataset copy verification failed"
print("Dataset copy PASS:", len(local_inventory), "files,", sum(local_inventory.values()), "bytes")
""",
)

validation = find_cell(notebook, 'REPO_ID = "local/ur5_orange_cube_v1"')
validation_text = source(validation)
validation_text = validation_text.replace('local/ur5_orange_cube_v1', 'local/ur5_cube_30ep')
validation_text = validation_text.replace(
    'Pick up the orange cube and place it in the tray',
    'Pick up the orange cube and place it in the bowl',
)
validation_text = validation_text.replace(
    'dataset.meta.total_episodes == 10 and dataset.meta.total_frames == 20583',
    'dataset.meta.total_episodes == 28 and dataset.meta.total_frames == 47204',
)
set_source(validation, validation_text)

set_source(
    find_cell(notebook, "fig, axes = plt.subplots(10, 2"),
    """
import matplotlib.pyplot as plt
episode_rows = dataset.meta.episodes
fig, axes = plt.subplots(28, 2, figsize=(12, 84))
representative_samples = []
for episode in range(28):
    start, stop = int(episode_rows[episode]["dataset_from_index"]), int(episode_rows[episode]["dataset_to_index"])
    index = (start + stop - 1) // 2
    sample = dataset[index]; representative_samples.append(sample)
    for column, key in enumerate(("observation.images.base", "observation.images.wrist")):
        image = sample[key].permute(1, 2, 0).cpu().numpy()
        axes[episode, column].imshow(image)
        axes[episode, column].set_title(f"ep={episode} frame={int(sample['frame_index'])} {key}")
        axes[episode, column].axis("off")
    print({"episode": episode, "frame": int(sample["frame_index"]), "timestamp": float(sample["timestamp"]),
           "task": TASK, "state": sample["observation.state"].tolist(), "action": sample["action"].tolist(),
           "gripper_state": float(sample["observation.state"][6]), "gripper_command": float(sample["action"][6])})
plt.tight_layout(); plt.show()
""",
)

set_source(
    find_cell(notebook, "## E — Choose the training length"),
    """
## E — Configure the 40,000-step training run

This notebook intentionally uses 40,000 optimizer steps. A complete resumable checkpoint is saved every 5,000 steps, and every checkpoint is retained in Drive. More steps do not guarantee better robot behavior: compare checkpoints offline and in ROS shadow mode before any physical execution.
""",
)

set_source(
    find_cell(notebook, "TRAIN_STEPS = 5000"),
    """
import json, os, random
import numpy as np, torch

TRAIN_STEPS = 40000
SAVE_FREQ = 5000
KEEP_LOCAL_CHECKPOINTS = 2
BATCH_SIZE = 1
SEED = 42
RUN_NAME = f"ur5_cube_30ep_smolvla_{TRAIN_STEPS}_steps"

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
    f"--batch_size={BATCH_SIZE}", f"--seed={SEED}", "--num_workers=2", "--wandb.enable=false",
    "--save_checkpoint=true", "--eval_freq=0", f"--steps={TRAIN_STEPS}", f"--save_freq={SAVE_FREQ}",
    "--log_freq=10", f"--output_dir={TRAIN_OUTPUT}", f"--job_name={RUN_NAME}",
]
resolved_training = {
    "command": TRAIN_ARGS, "train_steps": TRAIN_STEPS, "save_frequency": SAVE_FREQ,
    "drive_checkpoint_retention": "all", "local_checkpoint_retention": KEEP_LOCAL_CHECKPOINTS,
    "batch_size": BATCH_SIZE, "seed": SEED,
    "dataset_episodes": 28, "dataset_frames": 47204, "rgb_only": True,
    "real_camera_views": 2, "empty_camera_slots": 0, "task": TASK,
    "state_semantics": "six measured UR5 joints plus estimated pneumatic command state",
    "action_semantics": "six absolute joint targets plus binary gripper command",
}
(LOCAL_REPORT_DIR / "resolved_training_config.json").write_text(json.dumps(resolved_training, indent=2))
print(json.dumps(resolved_training, indent=2))
""",
)

set_source(
    find_cell(notebook, "## G — Train SmolVLA"),
    """
## G — Train SmolVLA and continuously protect progress

This is the only long-running cell. Every complete 5,000-step checkpoint is copied automatically to Drive through a temporary `.copying` directory and verified before it becomes visible. All Drive checkpoints are retained permanently. To protect limited Colab storage, older local copies are removed only after their verified Drive copies exist. When training finishes, this same cell automatically creates the final `pretrained_model`, training reports, downloadable tar archive, and SHA-256 checksum in Drive. No later copying cell is required.
""",
)

sync_cell = find_cell(notebook, "def sync_complete_checkpoints():")
h_markdown = find_cell(notebook, "## H — Copy the complete trained weights to Drive")
h_code = find_cell(notebook, "local_models = sorted")
i_markdown = find_cell(notebook, "## I — Package the trained weights for download")
i_code = find_cell(notebook, "export_archive = DEPLOYMENT_ROOT")

sync_text = source(sync_cell)
sync_text = sync_text.replace(
    '    if copied:\n        shutil.copy2(TRAIN_LOG, DRIVE_RUN_DIR / "training_in_progress.log")\n    return copied',
    '''    if copied:
        shutil.copy2(TRAIN_LOG, DRIVE_RUN_DIR / "training_in_progress.log")
        local_complete = sorted(
            path for path in (TRAIN_OUTPUT / "checkpoints").glob("*")
            if path.name != "last" and checkpoint_is_complete(path)
        )
        for old in local_complete[:-KEEP_LOCAL_CHECKPOINTS]:
            shutil.rmtree(old)
            print("PRUNED VERIFIED LOCAL COPY; DRIVE CHECKPOINT RETAINED:", old)
    return copied''',
)
if "DRIVE CHECKPOINT RETAINED" not in sync_text:
    raise RuntimeError("Failed to inject safe local checkpoint cleanup")
sync_text += "\n\n# Automatic final model copy and packaging; no later cell is required.\n"
sync_text += source(h_code) + "\n\n" + source(i_code)
set_source(sync_cell, sync_text)

set_source(
    notebook["cells"][-1],
    """
## Finished

The single training cell automatically retains checkpoints `005000` through `040000`, creates the final complete weights under `checkpoints/ur5_smolvla_trained/ur5_cube_30ep_smolvla_40000_steps/pretrained_model`, and writes a downloadable archive under `deployment/downloaded_checkpoints`. No separate copy/package cell is required. Training for 40,000 steps does not by itself prove deployment quality and does not authorize robot motion; evaluate multiple checkpoints offline and then in ROS shadow mode before a guarded physical test.
""",
)

# Final copy and packaging are now part of the training cell. Remove the old
# follow-up sections so users cannot mistake them for required manual steps.
removed = {id(h_markdown), id(h_code), id(i_markdown), id(i_code)}
notebook["cells"] = [cell for cell in notebook["cells"] if id(cell) not in removed]

notebook["metadata"]["colab"]["name"] = OUTPUT.name
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(OUTPUT)
