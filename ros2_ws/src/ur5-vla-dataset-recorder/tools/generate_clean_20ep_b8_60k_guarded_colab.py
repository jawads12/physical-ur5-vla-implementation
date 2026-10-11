#!/usr/bin/env python3
"""Generate the guarded A100 SmolVLA notebook for the clean 20-episode dataset."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/train_ur5_cube_clean_20ep_smolvla_b8_60000_wandb_guarded_colab.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown("""# UR5 cube-to-bowl SmolVLA — clean 20 episodes, A100 batch 8, guarded 60k run

This notebook fine-tunes `lerobot/smolvla_base` on the new validated physical-UR5 dataset containing **20 episodes and 53,084 frames**. It uses new W&B/checkpoint names, saves every complete 5,000-step checkpoint directly to Google Drive, and automatically exports the final model.

Every operation that can wait on an external resource has a finite timeout. The training cell has both a 36-hour total deadline and a 30-minute no-output watchdog. Run cells from top to bottom in a fresh A100 runtime. This notebook never contacts ROS and never commands the robot.
"""),
    markdown("""## 1 — Runtime and bounded-command helpers

Start with **Runtime → Disconnect and delete runtime**, reconnect, and select an A100 GPU. Do not import Torch before the pinned installation cell.
"""),
    code("""import json, os, platform, signal, subprocess, sys, time
from contextlib import contextmanager
from pathlib import Path

def run_checked(command, *, timeout_s, **kwargs):
    print("RUN:", " ".join(map(str, command)), flush=True)
    try:
        return subprocess.run(command, check=True, timeout=timeout_s, **kwargs)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command exceeded {timeout_s}s: {command}") from exc

@contextmanager
def deadline(seconds, label):
    def alarm_handler(signum, frame):
        raise TimeoutError(f"{label} exceeded {seconds}s")
    previous = signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

def command_output(command, timeout_s=60):
    return subprocess.run(command, text=True, capture_output=True, check=False, timeout=timeout_s).stdout.strip()

runtime_report = {
    "python": sys.version,
    "os": platform.platform(),
    "nvidia_smi": command_output(["nvidia-smi"]),
}
print(json.dumps(runtime_report, indent=2))
assert runtime_report["nvidia_smi"], "No NVIDIA GPU assigned"
assert "torch" not in sys.modules and "torchvision" not in sys.modules, (
    "Torch was imported before installation. Delete the runtime and start again."
)
"""),
    markdown("""## 2 — Mount Drive and define the new run names

Upload the entire local `ur5_cube_clean_20ep` directory under the Drive dataset path below. Edit only `DRIVE_ROOT` if your shortcut uses another name.
"""),
    code("""from google.colab import drive

drive.mount("/content/drive", timeout_ms=180000)
"""),
    code("""DRIVE_ROOT = Path("/content/drive/MyDrive/pHYSICAL UR5")
DATASET_NAME = "ur5_cube_clean_20ep"
REPO_ID = "local/ur5_cube_clean_20ep"
TASK = "Pick up the orange cube and place it in the bowl"

# These names are intentionally new and do not reuse the earlier 20ep or 53ep runs.
WANDB_PROJECT = "smolvla_20ep_clean_b8_60k"
RUN_NAME = "ur5_cube_clean_20ep_smolvla_b8_60000_guarded_v1"

DRIVE_DATASET = DRIVE_ROOT / "datasets/lerobot" / DATASET_NAME
CHECKPOINT_ROOT = DRIVE_ROOT / "checkpoints/ur5_smolvla_trained"
REPORT_ROOT = DRIVE_ROOT / "reports/training"
DEPLOYMENT_ROOT = DRIVE_ROOT / "deployment/downloaded_checkpoints"
DRIVE_RUN_DIR = CHECKPOINT_ROOT / RUN_NAME
DRIVE_CHECKPOINTS_DIR = DRIVE_RUN_DIR / "checkpoints"

assert DRIVE_ROOT.is_dir(), f"Drive root missing: {DRIVE_ROOT}"
assert DRIVE_DATASET.is_dir(), f"Upload the converted dataset here: {DRIVE_DATASET}"
for required in ("meta/info.json", "meta/stats.json", "meta/tasks.parquet", "data", "videos"):
    assert (DRIVE_DATASET / required).exists(), f"Dataset item missing: {DRIVE_DATASET / required}"
assert not DRIVE_RUN_DIR.exists(), f"Run output already exists; change RUN_NAME: {DRIVE_RUN_DIR}"
for directory in (CHECKPOINT_ROOT, REPORT_ROOT, DEPLOYMENT_ROOT):
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / ".write_test"
    probe.write_text("ok")
    probe.unlink()
print("Dataset:", DRIVE_DATASET)
print("New Drive run:", DRIVE_RUN_DIR)
print("New W&B project:", WANDB_PROJECT)
"""),
    markdown("""## 3 — Install the pinned environment

The installs have 30-minute limits. If a timeout occurs, delete the runtime and retry rather than continuing with a partial environment.
"""),
    code("""run_checked([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"], timeout_s=600)
run_checked([
    sys.executable, "-m", "pip", "install", "-q",
    "--index-url", "https://download.pytorch.org/whl/cu126",
    "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
], timeout_s=1800)
run_checked([
    sys.executable, "-m", "pip", "install", "-q",
    "lerobot[smolvla]==0.4.3", "av>=12", "psutil", "matplotlib", "wandb", "num2words",
], timeout_s=1800)
run_checked(["ffmpeg", "-version"], timeout_s=30, stdout=subprocess.DEVNULL)
print("Installation PASS")
"""),
    code("""import importlib.metadata as md
with deadline(300, "Torch/Torchvision import"):
    import torch, torchvision

assert torch.__version__.startswith("2.7.1"), torch.__version__
assert torchvision.__version__.startswith("0.22.1"), torchvision.__version__
assert torchvision.extension._has_ops(), "Torchvision native operators did not load"
assert torch.cuda.is_available(), "CUDA unavailable"
gpu_name = torch.cuda.get_device_name(0)
assert "A100" in gpu_name, f"This configuration expects an A100, received: {gpu_name}"
runtime_report.update({
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu_name": gpu_name,
    "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
})
(REPORT_ROOT / f"{RUN_NAME}_runtime.json").write_text(json.dumps(runtime_report, indent=2))
print("Environment PASS", json.dumps(runtime_report, indent=2))
"""),
    markdown("""## 4 — Copy and validate the dataset

Training uses fast Colab-local storage. The Drive-to-local copy has a 30-minute limit and is verified file-for-file. Validation is also executed in a separate process with a 30-minute limit.
"""),
    code("""import shutil

LOCAL_DATASET = Path("/content") / DATASET_NAME
with deadline(300, "Drive dataset size scan"):
    drive_bytes = sum(path.stat().st_size for path in DRIVE_DATASET.rglob("*") if path.is_file())
free_bytes = shutil.disk_usage("/content").free
assert free_bytes > drive_bytes + 25 * 1024**3, (
    f"Insufficient Colab disk: dataset={drive_bytes/2**30:.2f}GiB free={free_bytes/2**30:.2f}GiB"
)
assert not LOCAL_DATASET.exists(), f"Local path already exists; delete the runtime: {LOCAL_DATASET}"
LOCAL_DATASET.mkdir(parents=True)
run_checked([
    "rsync", "-ah", "--info=progress2", f"{DRIVE_DATASET}/", f"{LOCAL_DATASET}/"
], timeout_s=1800)

def file_sizes(root):
    return {str(path.relative_to(root)): path.stat().st_size for path in root.rglob("*") if path.is_file()}

with deadline(300, "Drive/local inventory verification"):
    assert file_sizes(LOCAL_DATASET) == file_sizes(DRIVE_DATASET), "Drive/local file inventory mismatch"
print(f"Dataset copy PASS: {drive_bytes/2**30:.2f} GiB")
"""),
    code("""VALIDATION_CODE = r'''\
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path("/content/ur5_cube_clean_20ep")
task = "Pick up the orange cube and place it in the bowl"
dataset = LeRobotDataset("local/ur5_cube_clean_20ep", root=root, video_backend="pyav")
assert dataset.meta.total_episodes == 20, dataset.meta.total_episodes
assert dataset.meta.total_frames == 53084, dataset.meta.total_frames
assert dataset.meta.fps == 20, dataset.meta.fps
assert task in dataset.meta.tasks.index
features = dataset.meta.features
assert features["observation.images.base"]["dtype"] == "video"
assert features["observation.images.wrist"]["dtype"] == "video"
assert tuple(features["observation.state"]["shape"]) == (7,)
assert tuple(features["action"]["shape"]) == (7,)
for row in dataset.hf_dataset:
    state = np.asarray(row["observation.state"])
    action = np.asarray(row["action"])
    assert np.isfinite(state).all() and np.isfinite(action).all()
    assert min(abs(float(state[6])), abs(float(state[6]) - 1.0)) < 1e-5
    assert min(abs(float(action[6])), abs(float(action[6]) - 1.0)) < 1e-5
for episode in range(dataset.meta.total_episodes):
    row = dataset.meta.episodes[episode]
    index = (int(row["dataset_from_index"]) + int(row["dataset_to_index"]) - 1) // 2
    sample = dataset[index]
    assert tuple(sample["observation.images.base"].shape) == (3, 480, 640)
    assert tuple(sample["observation.images.wrist"].shape) == (3, 480, 640)
print("DATASET VALIDATION PASS: 20 episodes / 53084 frames / both cameras decoded")
'''
run_checked([sys.executable, "-c", VALIDATION_CODE], timeout_s=1800)
"""),
    markdown("""## 5 — Authenticate W&B securely

Create a Colab secret named `WANDB_API_KEY` and enable notebook access. The key is never stored in the notebook or Drive.
"""),
    code("""from google.colab import userdata
import wandb

with deadline(120, "W&B login"):
    wandb_key = userdata.get("WANDB_API_KEY")
    assert wandb_key, "Add WANDB_API_KEY in Colab Secrets and enable access"
    os.environ["WANDB_API_KEY"] = wandb_key
    wandb.login(key=wandb_key, relogin=True)
del wandb_key
print("W&B PASS; project:", WANDB_PROJECT)
"""),
    markdown("""## 6 — Configure training

This run uses 60,000 optimizer steps at batch size 8 (about 9.0 sample-equivalent dataset passes). It creates checkpoints every 5,000 steps and retains all complete checkpoints in Drive.
"""),
    code("""import random
import numpy as np

TRAIN_STEPS = 60000
SAVE_FREQ = 5000
BATCH_SIZE = 8
SEED = 42
KEEP_LOCAL_CHECKPOINTS = 2
TRAIN_TIMEOUT_HOURS = 36
NO_OUTPUT_TIMEOUT_MINUTES = 30
CHECKPOINT_COPY_TIMEOUT_S = 1800

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

TRAIN_OUTPUT = Path("/content/outputs") / RUN_NAME
LOCAL_REPORT_DIR = Path("/content/training_reports") / RUN_NAME
TRAIN_LOG = LOCAL_REPORT_DIR / "training.log"
LOCAL_REPORT_DIR.mkdir(parents=True, exist_ok=False)
assert not TRAIN_OUTPUT.exists(), f"Training output exists: {TRAIN_OUTPUT}"

TRAIN_ARGS = [
    "lerobot-train",
    "--policy.type=smolvla",
    "--policy.pretrained_path=lerobot/smolvla_base",
    "--policy.device=cuda",
    "--policy.use_amp=true",
    "--policy.push_to_hub=false",
    f"--dataset.repo_id={REPO_ID}",
    f"--dataset.root={LOCAL_DATASET}",
    "--dataset.video_backend=pyav",
    f"--batch_size={BATCH_SIZE}",
    f"--seed={SEED}",
    "--num_workers=4",
    "--wandb.enable=true",
    f"--wandb.project={WANDB_PROJECT}",
    "--wandb.mode=online",
    "--save_checkpoint=true",
    "--eval_freq=0",
    f"--steps={TRAIN_STEPS}",
    f"--save_freq={SAVE_FREQ}",
    "--log_freq=10",
    f"--output_dir={TRAIN_OUTPUT}",
    f"--job_name={RUN_NAME}",
]

resolved_training = {
    "command": TRAIN_ARGS,
    "dataset": DATASET_NAME,
    "episodes": 20,
    "frames": 53084,
    "task": TASK,
    "train_steps": TRAIN_STEPS,
    "batch_size": BATCH_SIZE,
    "approximate_sample_passes": TRAIN_STEPS * BATCH_SIZE / 53084,
    "save_frequency": SAVE_FREQ,
    "wandb_project": WANDB_PROJECT,
    "run_name": RUN_NAME,
    "training_timeout_hours": TRAIN_TIMEOUT_HOURS,
    "no_output_timeout_minutes": NO_OUTPUT_TIMEOUT_MINUTES,
}
(LOCAL_REPORT_DIR / "resolved_training_config.json").write_text(json.dumps(resolved_training, indent=2))
print(json.dumps(resolved_training, indent=2))
"""),
    markdown("""## 7 — Bounded model compatibility test

This separate process downloads/loads the base model and executes one batch-8 forward/backward optimizer step. It is terminated after 45 minutes if model loading or CUDA becomes stuck.
"""),
    code("""PREFLIGHT_CODE = r'''\
import json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

root = Path("/content/ur5_cube_clean_20ep")
repo_id = "local/ur5_cube_clean_20ep"
task = "Pick up the orange cube and place it in the bowl"
batch_size = 8
started = time.time()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
meta_dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")
cfg = SmolVLAConfig(pretrained_path=Path("lerobot/smolvla_base"), device="cuda", use_amp=True, push_to_hub=False)
delta = resolve_delta_timestamps(cfg, meta_dataset.meta)
dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=delta, video_backend="pyav")
batch = next(iter(DataLoader(dataset, batch_size=batch_size, num_workers=0)))
assert tuple(batch["action"].shape) == (batch_size, cfg.chunk_size, 7), batch["action"].shape
policy = make_policy(cfg, ds_meta=dataset.meta).train()
preprocessor, _ = make_pre_post_processors(
    cfg,
    pretrained_path="lerobot/smolvla_base",
    dataset_stats=dataset.meta.stats,
    preprocessor_overrides={
        "device_processor": {"device": "cuda"},
        "normalizer_processor": {
            "stats": dataset.meta.stats,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
        "rename_observations_processor": {"rename_map": {}},
    },
)
model_batch = {key: (value.cuda() if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}
model_batch["task"] = [task] * batch_size
optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=1e-5)
optimizer.zero_grad(set_to_none=True)
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss, _ = policy.forward(preprocessor(model_batch))
assert torch.isfinite(loss), loss
loss.backward()
optimizer.step()
torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated()
total = torch.cuda.get_device_properties(0).total_memory
assert peak < total * 0.90, f"Insufficient VRAM headroom: {peak/2**30:.1f}/{total/2**30:.1f} GiB"
report = {
    "pass": True,
    "batch_size": batch_size,
    "loss": float(loss.detach()),
    "elapsed_seconds": time.time() - started,
    "peak_gpu_vram_bytes": peak,
    "total_gpu_vram_bytes": total,
}
Path("/content/training_reports/ur5_cube_clean_20ep_smolvla_b8_60000_guarded_v1/compatibility_report.json").write_text(json.dumps(report, indent=2))
print("MODEL PREFLIGHT PASS", json.dumps(report, indent=2))
'''
run_checked([sys.executable, "-c", PREFLIGHT_CODE], timeout_s=2700)
"""),
    markdown("""## 8 — Train, monitor, and save directly to Drive

This is the only long cell. It cannot wait forever:

- absolute deadline: 36 hours;
- no-output watchdog: 30 minutes;
- each Drive checkpoint copy: 30 minutes maximum;
- every completed 5,000-step checkpoint is atomically published to Drive;
- only Drive-verified checkpoints are removed from local storage;
- final model packaging is automatic.
"""),
    code("""import hashlib, queue, shutil, threading

TRAIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
DRIVE_RUN_DIR.mkdir(parents=True, exist_ok=False)
DRIVE_CHECKPOINTS_DIR.mkdir()
shutil.copy2(LOCAL_REPORT_DIR / "resolved_training_config.json", DRIVE_RUN_DIR / "resolved_training_config.json")
shutil.copy2(LOCAL_REPORT_DIR / "compatibility_report.json", DRIVE_RUN_DIR / "compatibility_report.json")

def checkpoint_is_complete(path):
    required = (
        path / "pretrained_model/model.safetensors",
        path / "pretrained_model/config.json",
        path / "training_state/optimizer_state.safetensors",
        path / "training_state/training_step.json",
    )
    return path.is_dir() and not path.is_symlink() and all(item.is_file() for item in required)

def bounded_rsync(source, target, timeout_s):
    target.mkdir(parents=True, exist_ok=False)
    try:
        run_checked(["rsync", "-a", "--delete", f"{source}/", f"{target}/"], timeout_s=timeout_s)
        with deadline(300, f"Copy verification for {source.name}"):
            assert file_sizes(source) == file_sizes(target), f"Copy verification failed: {source}"
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise

def sync_complete_checkpoints():
    copied = []
    checkpoint_root = TRAIN_OUTPUT / "checkpoints"
    if not checkpoint_root.exists():
        return copied
    for source in sorted(checkpoint_root.iterdir()):
        if source.name == "last" or not checkpoint_is_complete(source):
            continue
        target = DRIVE_CHECKPOINTS_DIR / source.name
        if target.exists():
            continue
        staging = DRIVE_CHECKPOINTS_DIR / f".{source.name}.copying"
        try:
            if staging.exists():
                shutil.rmtree(staging)
            bounded_rsync(source, staging, CHECKPOINT_COPY_TIMEOUT_S)
            staging.rename(target)
            copied.append(source.name)
            print("SYNCED TO DRIVE:", target, flush=True)
        except Exception as exc:
            print("CHECKPOINT SYNC WARNING:", source.name, repr(exc), flush=True)
            shutil.rmtree(staging, ignore_errors=True)
    if copied:
        shutil.copy2(TRAIN_LOG, DRIVE_RUN_DIR / "training_in_progress.log")
        local_complete = sorted(
            path for path in checkpoint_root.iterdir()
            if path.name != "last" and checkpoint_is_complete(path)
        )
        for old in local_complete[:-KEEP_LOCAL_CHECKPOINTS]:
            drive_copy = DRIVE_CHECKPOINTS_DIR / old.name
            if checkpoint_is_complete(drive_copy) and file_sizes(old) == file_sizes(drive_copy):
                shutil.rmtree(old)
                print("PRUNED DRIVE-VERIFIED LOCAL CHECKPOINT:", old.name, flush=True)
    return copied

line_queue = queue.Queue()

def read_training_output(stream):
    try:
        for line in iter(stream.readline, ""):
            line_queue.put(line)
    finally:
        line_queue.put(None)

print("Starting guarded training:", " ".join(TRAIN_ARGS), flush=True)
process = subprocess.Popen(
    TRAIN_ARGS,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=1,
)
reader = threading.Thread(target=read_training_output, args=(process.stdout,), daemon=True)
reader.start()
started = time.monotonic()
last_output = started
last_sync = 0.0
failure = None

with TRAIN_LOG.open("w") as log:
    while True:
        now = time.monotonic()
        if now - started > TRAIN_TIMEOUT_HOURS * 3600:
            failure = f"Training exceeded {TRAIN_TIMEOUT_HOURS} hours"
            break
        # The reader thread keeps draining stdout while a Drive copy is in
        # progress. Do not declare inactivity while buffered output is waiting.
        if line_queue.empty() and now - last_output > NO_OUTPUT_TIMEOUT_MINUTES * 60:
            failure = f"Training produced no output for {NO_OUTPUT_TIMEOUT_MINUTES} minutes"
            break
        try:
            line = line_queue.get(timeout=30)
            if line is None:
                if process.poll() is not None:
                    break
                continue
            print(line, end="")
            log.write(line)
            log.flush()
            last_output = time.monotonic()
        except queue.Empty:
            print(f"HEARTBEAT: training alive for {(now-started)/3600:.2f}h", flush=True)
        if time.monotonic() - last_sync >= 60:
            sync_complete_checkpoints()
            last_sync = time.monotonic()
        if process.poll() is not None and line_queue.empty():
            break

if failure:
    process.terminate()
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)
    sync_complete_checkpoints()
    shutil.copy2(TRAIN_LOG, DRIVE_RUN_DIR / "training_stopped.log")
    raise RuntimeError(failure)

return_code = process.wait(timeout=120)
sync_complete_checkpoints()
shutil.copy2(TRAIN_LOG, DRIVE_RUN_DIR / "training.log")
assert return_code == 0, f"Training failed with return code {return_code}; see {DRIVE_RUN_DIR / 'training.log'}"

drive_models = sorted(DRIVE_CHECKPOINTS_DIR.glob("*/pretrained_model"), key=lambda path: path.parent.name)
assert drive_models, "No complete Drive checkpoint exists"
final_source = drive_models[-1]
final_target = DRIVE_RUN_DIR / "pretrained_model"
final_staging = DRIVE_RUN_DIR / ".pretrained_model.copying"
bounded_rsync(final_source, final_staging, CHECKPOINT_COPY_TIMEOUT_S)
final_staging.rename(final_target)
assert (final_target / "model.safetensors").is_file()
assert (final_target / "config.json").is_file()

archive = DEPLOYMENT_ROOT / f"{RUN_NAME}_pretrained_model.tar"
run_checked([
    "tar", "-cf", str(archive), "-C", str(DRIVE_RUN_DIR), "pretrained_model"
], timeout_s=3600)
sha = run_checked(["sha256sum", str(archive)], timeout_s=1800, text=True, capture_output=True).stdout.strip()
checksum = Path(f"{archive}.sha256")
checksum.write_text(sha + "\\n")

print("TRAINING COMPLETE")
print("W&B project:", WANDB_PROJECT)
print("All checkpoints:", DRIVE_CHECKPOINTS_DIR)
print("Final pretrained model:", final_target)
print("Download archive:", archive)
print("SHA-256:", sha)
"""),
    markdown("""## 9 — After training

Do not assume the last checkpoint is best. Compare at least 20k, 30k, 40k, 50k, and 60k using offline inference and ROS shadow mode. Only then perform a guarded low-speed physical trial from the recorded standard pose, with unchanged cameras/workspace and the command-safety rejection layer enabled.

The model is trained only when the long cell prints `TRAINING COMPLETE`. A W&B run appearing by itself is not proof that all weights reached Drive.
"""),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(OUTPUT)
