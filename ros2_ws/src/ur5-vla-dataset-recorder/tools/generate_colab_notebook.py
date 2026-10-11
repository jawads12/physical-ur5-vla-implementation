"""Generate the checked-in UR5 SmolVLA Colab notebook without notebook dependencies."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip().splitlines(keepends=True)}


cells = [
markdown("""
# UR5 SmolVLA training on Google Colab Pro

This notebook is one linear workflow: load the converted physical-UR5 dataset from Drive, validate it, train SmolVLA, and copy the complete trained weights back to Drive. Run every cell from top to bottom. It never contacts ROS or the robot, never trains directly from mounted Drive, and never uploads to Hugging Face.

Before starting, change the shared Drive folder from **Anyone with the link: Editor** to **Restricted**, explicitly add the Colab account as Editor, and add a shortcut to the folder in My Drive.
"""),
markdown("""## How this differs from the earlier simulation experiment

The earlier `smolVLATraining.ipynb` is a reference only. It used a different cube task, CSV/raw schema, simulated UR5 state/action meaning, camera configuration, hard-coded Drive paths, batch size 8, 30k–50k steps, direct Drive training, WandB, and mixed LeRobot CLI generations. This notebook instead uses the validated physical UR5 LeRobot v3 dataset: two real cameras, measured six-joint state plus estimated gripper state, six absolute MoveIt Servo joint targets plus binary gripper command, batch size 1, RGB only, local `/content` training, and the LeRobot 0.4.3 `--policy.pretrained_path` API. LeRobot derives the two physical camera features directly from this dataset.
"""),
markdown("""## A — Basic runtime verification

Capture Python, OS, and the assigned GPU without importing PyTorch yet. PyTorch must not be imported before the pinned installation in Section C. Stop if Colab did not assign a GPU.

If you previously ran an older version of this notebook, first use **Runtime → Disconnect and delete runtime**, reconnect with a GPU, and begin again from this cell."""),
code("""
import json, os, platform, subprocess, sys, time
from pathlib import Path

def command_output(command):
    return subprocess.run(command, text=True, capture_output=True, check=False).stdout.strip()

runtime_report = {
    "python": sys.version,
    "os": platform.platform(),
    "nvidia_smi": command_output(["nvidia-smi"]),
}
print(json.dumps(runtime_report, indent=2))
assert runtime_report["nvidia_smi"], "No NVIDIA GPU assigned: Runtime > Change runtime type > GPU"
assert "torch" not in sys.modules and "torchvision" not in sys.modules, "Torch/Torchvision already imported. Disconnect and delete the runtime, reconnect, and restart at Section A."
"""),
markdown("""## B — Mount and configure Google Drive

Edit only `DRIVE_ROOT` if your shortcut has a different name. The checks deliberately stop before any missing archive is used."""),
code("""
from google.colab import drive
drive.mount("/content/drive")
"""),
code("""
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/pHYSICAL UR5")
DATASET_ARCHIVE = DRIVE_ROOT / "datasets/lerobot/ur5_orange_cube_v1.tar"
DATASET_CHECKSUM = Path(f"{DATASET_ARCHIVE}.sha256")
DATASET_PARTS = sorted(DATASET_ARCHIVE.parent.glob(f"{DATASET_ARCHIVE.name}.part-*"))
DATASET_PARTS_CHECKSUM = Path(f"{DATASET_ARCHIVE}.parts.sha256")
CHECKPOINT_ROOT = DRIVE_ROOT / "checkpoints/ur5_smolvla_trained"
REPORT_ROOT = DRIVE_ROOT / "reports/training"
DEPLOYMENT_ROOT = DRIVE_ROOT / "deployment/downloaded_checkpoints"

assert DRIVE_ROOT.is_dir(), f"Drive shortcut/path not found: {DRIVE_ROOT}"
assert DATASET_ARCHIVE.is_file() or DATASET_PARTS, f"Dataset archive or split parts missing beside: {DATASET_ARCHIVE}"
assert DATASET_CHECKSUM.is_file(), f"Checksum missing: {DATASET_CHECKSUM}"
if not DATASET_ARCHIVE.is_file(): assert DATASET_PARTS_CHECKSUM.is_file(), "Split-part checksum file missing"
for directory in (CHECKPOINT_ROOT, REPORT_ROOT, DEPLOYMENT_ROOT):
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / ".write_test"; probe.write_text("ok"); probe.unlink()
(REPORT_ROOT / "runtime_report.json").write_text(json.dumps(runtime_report, indent=2))
print("Drive configuration verified")
"""),
markdown("""## C — Install and inspect the pinned training environment

Install the official matching PyTorch 2.7.1 / Torchvision 0.22.1 CUDA 12.6 wheels before LeRobot 0.4.3. PyAV is the explicit video backend. This ordering prevents Colab from retaining a partially initialized Torchvision module."""),
code("""
%pip install -q --upgrade pip
%pip install -q --index-url https://download.pytorch.org/whl/cu126 "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1"
%pip install -q "lerobot[smolvla]==0.4.3" "av>=12" psutil matplotlib pytest num2words
!ffmpeg -version | head -2
"""),
code("""
import importlib.metadata as md, subprocess, sys
for package in ("lerobot", "torch", "torchvision", "av", "transformers", "accelerate"):
    try: print(package, md.version(package))
    except md.PackageNotFoundError: print(package, "NOT INSTALLED")
import torch, torchvision
assert torch.__version__.startswith("2.7.1"), torch.__version__
assert torchvision.__version__.startswith("0.22.1"), torchvision.__version__
assert torchvision.extension._has_ops(), "Torchvision native operators did not load"
assert torch.cuda.is_available(), "Pinned PyTorch installed, but CUDA is unavailable"
runtime_report.update({
    "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(), "gpu_name": torch.cuda.get_device_name(0),
    "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
    "gpu_free_total_bytes": torch.cuda.mem_get_info(),
})
(REPORT_ROOT / "runtime_report.json").write_text(json.dumps(runtime_report, indent=2))
print("Torch/Torchvision/CUDA import PASS", json.dumps(runtime_report, indent=2))
print(subprocess.run(["lerobot-train", "--help"], text=True, capture_output=True).stdout[:12000])
try:
    import torchcodec
    print("TorchCodec import PASS", md.version("torchcodec"))
except Exception as exc:
    print("TorchCodec unavailable/incompatible; using supported PyAV fallback:", repr(exc))
import av
print("PyAV PASS", av.__version__)
print("CUDA package consistency:")
print(subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True).stdout)
"""),
markdown("""### Optional CUDA bindings repair

The simulation runtime once required `cuda-python==12.9.6` and `cuda-bindings==12.9.6`. Do **not** force that pin unless the model-loading test reports a CUDA bindings import error and the runtime CUDA version is compatible. This cell is disabled by default."""),
code("""
REPAIR_CUDA_BINDINGS = False
if REPAIR_CUDA_BINDINGS:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "cuda-python", "cuda-bindings"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "cuda-python==12.9.6", "cuda-bindings==12.9.6"], check=True)
    print("Restart the runtime, then rerun from Section A.")
"""),
markdown("""## D — Copy, verify, and safely extract the dataset

Training occurs under `/content`, never on mounted Drive. SHA-256 is checked before extraction; absolute paths, traversal, and links are rejected."""),
code("""
import hashlib, shutil, tarfile
from pathlib import Path, PurePosixPath

LOCAL_ARCHIVE = Path("/content/ur5_orange_cube_v1.tar")
LOCAL_DATASET = Path("/content/ur5_orange_cube_v1")
if DATASET_ARCHIVE.is_file():
    shutil.copy2(DATASET_ARCHIVE, LOCAL_ARCHIVE)
else:
    expected_part_hashes={line.split()[1]:line.split()[0] for line in DATASET_PARTS_CHECKSUM.read_text().splitlines() if line.strip()}
    with LOCAL_ARCHIVE.open("wb") as assembled:
        for part in DATASET_PARTS:
            part_digest=hashlib.sha256(part.read_bytes()).hexdigest()
            assert part_digest==expected_part_hashes[part.name], f"part SHA-256 mismatch: {part.name}"
            with part.open("rb") as source: shutil.copyfileobj(source,assembled,length=8*1024*1024)
    print("Reassembled split Drive upload into",LOCAL_ARCHIVE)
expected_sha = DATASET_CHECKSUM.read_text().split()[0].lower()
digest = hashlib.sha256()
with LOCAL_ARCHIVE.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024): digest.update(chunk)
actual_sha = digest.hexdigest()
assert actual_sha == expected_sha, f"SHA-256 mismatch: {actual_sha} != {expected_sha}"
if LOCAL_DATASET.exists(): shutil.rmtree(LOCAL_DATASET)
with tarfile.open(LOCAL_ARCHIVE, "r:") as archive:
    members = archive.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        assert not path.is_absolute() and ".." not in path.parts, f"unsafe path: {member.name}"
        assert not member.issym() and not member.islnk(), f"archive link rejected: {member.name}"
    archive.extractall("/content", members=members, filter="data")
assert LOCAL_DATASET.is_dir()
print("Archive verified and extracted:", LOCAL_DATASET)
"""),
markdown("""### Load and validate through LeRobot

Validate metadata and every state/action row, then decode representative synchronized views from every episode."""),
code("""
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "local/ur5_orange_cube_v1"
TASK = "Pick up the orange cube and place it in the tray"
dataset = LeRobotDataset(REPO_ID, root=LOCAL_DATASET, video_backend="pyav")
features = dataset.meta.features
assert dataset.meta.total_episodes == 10 and dataset.meta.total_frames == 20583
assert features["observation.images.base"]["dtype"] == "video"
assert features["observation.images.wrist"]["dtype"] == "video"
assert tuple(features["observation.state"]["shape"]) == (7,)
assert tuple(features["action"]["shape"]) == (7,)
assert "source_episode_time_s" in features and TASK in dataset.meta.tasks.index
for index in range(len(dataset)):
    row = dataset.hf_dataset[index]
    state, action = np.asarray(row["observation.state"]), np.asarray(row["action"])
    assert np.isfinite(state).all() and np.isfinite(action).all()
    assert min(abs(float(state[6])), abs(float(state[6])-1)) < 1e-5
    assert min(abs(float(action[6])), abs(float(action[6])-1)) < 1e-5
print("Dataset metadata/vector validation PASS", dataset.meta.total_episodes, dataset.meta.total_frames)
"""),
markdown("""### Display one synchronized sample from every episode"""),
code("""
import matplotlib.pyplot as plt
episode_rows = dataset.meta.episodes
fig, axes = plt.subplots(10, 2, figsize=(12, 35))
representative_samples = []
for episode in range(10):
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
"""),
markdown("""## E — Choose the training length

This is the only training choice. `TRAIN_STEPS=5000` performs one 5,000-step fine-tuning run. You can change it before running the cell. Checkpoints are saved every 500 steps and the final complete weights are copied to Drive automatically."""),
code("""
import json, os, random
import numpy as np, torch

TRAIN_STEPS = 5000
SAVE_FREQ = 500
BATCH_SIZE = 1
SEED = 42
RUN_NAME = f"ur5_orange_cube_smolvla_{TRAIN_STEPS}_steps"

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
    "batch_size": BATCH_SIZE, "seed": SEED, "rgb_only": True,
    "real_camera_views": 2, "empty_camera_slots": 0, "task": TASK,
    "state_semantics": "six measured UR5 joints plus estimated pneumatic command state",
    "action_semantics": "six absolute joint targets plus binary gripper command",
}
(LOCAL_REPORT_DIR / "resolved_training_config.json").write_text(json.dumps(resolved_training, indent=2))
print(json.dumps(resolved_training, indent=2))
"""),
markdown("""## F — Check that one batch fits the pretrained model

This does not train a separate model. It only catches camera, tensor-shape, CUDA-memory, or preprocessing errors before the long training cell."""),
code("""
import psutil, time, torch
from torch.utils.data import DataLoader
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

started = time.time(); torch.cuda.reset_peak_memory_stats()
PRETRAINED_MODEL = "lerobot/smolvla_base"
cfg = SmolVLAConfig(pretrained_path=Path(PRETRAINED_MODEL), device="cuda", use_amp=True, push_to_hub=False)
delta_timestamps = resolve_delta_timestamps(cfg, dataset.meta)
training_dataset = LeRobotDataset(
    REPO_ID, root=LOCAL_DATASET, delta_timestamps=delta_timestamps, video_backend="pyav"
)
batch = next(iter(DataLoader(training_dataset, batch_size=1, num_workers=0)))
assert tuple(batch["action"].shape) == (1, cfg.chunk_size, 7), batch["action"].shape
policy = make_policy(cfg, ds_meta=training_dataset.meta).eval()
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
model_batch["task"] = [TASK]
with torch.inference_mode():
    loss, _ = policy.forward(preprocessor(model_batch))
assert torch.isfinite(loss), f"Non-finite compatibility loss: {loss}"
compatibility_report = {
    "pass": True, "loss": float(loss), "elapsed_seconds": time.time() - started,
    "cpu_ram_bytes": psutil.Process().memory_info().rss,
    "peak_gpu_vram_bytes": torch.cuda.max_memory_allocated(),
}
(LOCAL_REPORT_DIR / "compatibility_report.json").write_text(json.dumps(compatibility_report, indent=2))
print("Compatibility check PASS", json.dumps(compatibility_report, indent=2))
del policy, preprocessor, model_batch, batch; torch.cuda.empty_cache()
"""),
markdown("""## G — Train SmolVLA

Run this cell once and leave the browser connected. Training remains on fast local `/content` storage. After each complete 500-step checkpoint is closed, the entire checkpoint—including resumable training state—is copied to Drive through a temporary `.copying` directory and then renamed into place."""),
code("""
import shutil, subprocess

TRAIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
TRAIN_LOG = LOCAL_REPORT_DIR / "training.log"
DRIVE_RUN_DIR.mkdir(parents=True, exist_ok=False)
DRIVE_CHECKPOINTS_DIR.mkdir()
shutil.copy2(LOCAL_REPORT_DIR / "resolved_training_config.json", DRIVE_RUN_DIR / "resolved_training_config.json")
shutil.copy2(LOCAL_REPORT_DIR / "compatibility_report.json", DRIVE_RUN_DIR / "compatibility_report.json")

def checkpoint_is_complete(checkpoint_dir):
    required = (
        checkpoint_dir / "pretrained_model/model.safetensors",
        checkpoint_dir / "pretrained_model/config.json",
        checkpoint_dir / "training_state/optimizer_state.safetensors",
        checkpoint_dir / "training_state/training_step.json",
    )
    return checkpoint_dir.is_dir() and not checkpoint_dir.is_symlink() and all(path.is_file() for path in required)

def file_sizes(root):
    return {str(path.relative_to(root)): path.stat().st_size for path in root.rglob("*") if path.is_file()}

def sync_complete_checkpoints():
    copied = []
    for source in sorted((TRAIN_OUTPUT / "checkpoints").glob("*")):
        if source.name == "last" or not checkpoint_is_complete(source):
            continue
        target = DRIVE_CHECKPOINTS_DIR / source.name
        if target.exists():
            continue
        staging = DRIVE_CHECKPOINTS_DIR / f".{source.name}.copying"
        try:
            if staging.exists():
                shutil.rmtree(staging)
            shutil.copytree(source, staging)
            assert file_sizes(staging) == file_sizes(source), f"Drive checkpoint verification failed: {source.name}"
            staging.rename(target)
            copied.append(source.name)
            print("SYNCED COMPLETE CHECKPOINT TO DRIVE:", target)
        except Exception as exc:
            print("DRIVE CHECKPOINT SYNC WARNING:", source.name, repr(exc))
            if staging.exists():
                shutil.rmtree(staging)
    if copied:
        shutil.copy2(TRAIN_LOG, DRIVE_RUN_DIR / "training_in_progress.log")
    return copied

print("Starting training:", " ".join(map(str, TRAIN_ARGS)))
with TRAIN_LOG.open("w") as log:
    process = subprocess.Popen(TRAIN_ARGS, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    for line in process.stdout:
        print(line, end="")
        log.write(line); log.flush()
        sync_complete_checkpoints()
    return_code = process.wait()
sync_complete_checkpoints()
shutil.copy2(TRAIN_LOG, DRIVE_RUN_DIR / "training.log")
assert return_code == 0, f"Training failed with return code {return_code}. Read {TRAIN_LOG}"
print("TRAINING COMPLETE:", TRAIN_OUTPUT)
print("DRIVE CHECKPOINTS:", sorted(path.name for path in DRIVE_CHECKPOINTS_DIR.iterdir() if not path.name.startswith(".")))
"""),
markdown("""## H — Copy the complete trained weights to Drive

This selects the final local checkpoint (or the latest already-synced Drive checkpoint after a recovered session) and creates the convenient final `pretrained_model` directory. Do not copy only `model.safetensors`; the other files are required to load the policy."""),
code("""
import shutil

local_models = sorted(TRAIN_OUTPUT.glob("checkpoints/*/pretrained_model"), key=lambda path: path.parent.name)
drive_models = sorted(DRIVE_CHECKPOINTS_DIR.glob("*/pretrained_model"), key=lambda path: path.parent.name)
models = local_models or drive_models
assert models, f"No trained checkpoint found locally or under {DRIVE_CHECKPOINTS_DIR}"
FINAL_MODEL = models[-1]
assert (FINAL_MODEL / "model.safetensors").is_file()
assert (FINAL_MODEL / "config.json").is_file()

final_target = DRIVE_RUN_DIR / "pretrained_model"
staging = DRIVE_RUN_DIR / ".pretrained_model.copying"
if not final_target.exists():
    if staging.exists(): shutil.rmtree(staging)
    shutil.copytree(FINAL_MODEL, staging)
    assert file_sizes(staging) == file_sizes(FINAL_MODEL), "Final Drive model verification failed"
    staging.rename(final_target)
assert (final_target / "model.safetensors").is_file() and (final_target / "config.json").is_file()
shutil.copy2(LOCAL_REPORT_DIR / "resolved_training_config.json", DRIVE_RUN_DIR / "resolved_training_config.json")
shutil.copy2(LOCAL_REPORT_DIR / "compatibility_report.json", DRIVE_RUN_DIR / "compatibility_report.json")
shutil.copy2(TRAIN_LOG, DRIVE_RUN_DIR / "training.log")
print("TRAINED WEIGHTS SAVED TO DRIVE:", DRIVE_RUN_DIR / "pretrained_model")
print("Files:", sorted(path.name for path in (DRIVE_RUN_DIR / "pretrained_model").iterdir()))
"""),
markdown("""## I — Package the trained weights for download

The archive and SHA-256 checksum are written to `deployment/downloaded_checkpoints` in Drive."""),
code("""
import hashlib, tarfile

export_archive = DEPLOYMENT_ROOT / f"{RUN_NAME}_complete_checkpoint.tar"
with tarfile.open(export_archive, "w") as archive:
    archive.add(DRIVE_RUN_DIR / "pretrained_model", arcname="pretrained_model")
digest = hashlib.sha256()
with export_archive.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024): digest.update(chunk)
checksum = Path(f"{export_archive}.sha256")
checksum.write_text(f"{digest.hexdigest()}  {export_archive.name}\\n")
shutil.copy2(LOCAL_REPORT_DIR / "resolved_training_config.json", REPORT_ROOT / f"{RUN_NAME}_config.json")
shutil.copy2(LOCAL_REPORT_DIR / "compatibility_report.json", REPORT_ROOT / f"{RUN_NAME}_compatibility.json")
shutil.copy2(TRAIN_LOG, REPORT_ROOT / f"{RUN_NAME}_training.log")
print("READY TO DOWNLOAD:", export_archive)
print("SHA-256:", digest.hexdigest())
"""),
markdown("""## Finished

Your complete trained weights are now in `checkpoints/ur5_smolvla_trained/<RUN_NAME>/pretrained_model`, with a downloadable archive under `deployment/downloaded_checkpoints`. Training does not authorize robot motion; evaluate the checkpoint offline before any ROS shadow-mode or physical test."""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "train_ur5_smolvla_colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path(__file__).resolve().parents[1] / "notebooks/train_ur5_smolvla_colab.ipynb"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(notebook, indent=1) + "\n")
print(output)
