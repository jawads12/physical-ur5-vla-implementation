#!/usr/bin/env python3
"""Generate the reproducible Colab notebook for UR5 pi0 fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/train_ur5_cube_clean_73ep_v2_pi0_a100_colab.ipynb"


def lines(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


cells = [
    markdown(
        r"""
# UR5 73-episode π0 training — Colab A100 40 GB

This notebook fine-tunes `lerobot/pi0_base` with the published LeRobot 0.6.1 release on the validated physical UR5 dataset `ur5_cube_clean_73ep_v2` (two RGB cameras, 7D measured state, and 7D absolute action: six joint targets plus gripper).

The default **expert** profile trains π0's action expert and input/output projections while keeping the large vision-language backbone frozen. That is the reliable profile for a single A100 40 GB. Set `FINETUNE_MODE = "full"` in the configuration cell only if the batch probe confirms that full-model optimizer state fits. Both profiles use bfloat16, gradient checkpointing, automatic batch selection, local-disk training, W&B logging, periodic Drive checkpoint copies, and a deployable final archive.

Run every cell from top to bottom. The notebook accepts either the uploaded LeRobot folder or the earlier `.tar` plus `.sha256` pair. It creates a timestamped run name, so rerunning does not collide with an old report directory.
"""
    ),
    code(
        r"""
# 1. Verify the Colab GPU. This intentionally does not reject a runtime merely because torch was imported earlier.
import json, os, platform, subprocess, sys, time
from pathlib import Path

def output(command):
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return result.stdout.strip()

gpu_report = output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
print(gpu_report)
assert gpu_report, "No NVIDIA GPU assigned. In Colab choose Runtime > Change runtime type > A100 GPU."
gpu_name, memory_mib, driver = [item.strip() for item in gpu_report.splitlines()[0].split(",", 2)]
assert "A100" in gpu_name, f"Expected an A100 runtime, received {gpu_name}"
assert int(memory_mib) >= 38000, f"Expected an A100 40 GB (or larger), received {memory_mib} MiB"
runtime_report = {
    "python": sys.version,
    "platform": platform.platform(),
    "gpu": gpu_name,
    "memory_mib": int(memory_mib),
    "driver": driver,
    "torch_was_already_imported": "torch" in sys.modules,
}
print(json.dumps(runtime_report, indent=2))
"""
    ),
    code(
        r"""
# 2. Mount Drive and define paths. Edit DRIVE_ROOT only if your Drive folder moved.
from datetime import datetime, timezone
from google.colab import drive
drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive/VLA_Thesis/Sorted VLA Backup Final/pHYSICAL UR5")
DATASET_NAME = "ur5_cube_clean_73ep_v2"
REPO_ID = f"local/{DATASET_NAME}"
TASK = "Pick up the orange cube and place it in the bowl"
DATASET_DRIVE_FOLDER = DRIVE_ROOT / "datasets" / "lerobot" / DATASET_NAME
DATASET_ARCHIVE = DRIVE_ROOT / "datasets" / "lerobot" / f"{DATASET_NAME}.tar"
DATASET_CHECKSUM = Path(f"{DATASET_ARCHIVE}.sha256")
LOCAL_DATASET = Path("/content") / DATASET_NAME
LOCAL_ARCHIVE = Path("/content") / DATASET_ARCHIVE.name
CHECKPOINT_ROOT = DRIVE_ROOT / "checkpoints" / "ur5_pi0_trained"
REPORT_ROOT = DRIVE_ROOT / "reports" / "training"
DEPLOYMENT_ROOT = DRIVE_ROOT / "deployment" / "downloaded_checkpoints"

assert DRIVE_ROOT.is_dir(), f"Drive root missing: {DRIVE_ROOT}"
folder_upload = DATASET_DRIVE_FOLDER.is_dir()
archive_upload = DATASET_ARCHIVE.is_file() and DATASET_CHECKSUM.is_file()
assert folder_upload or archive_upload, (
    f"Dataset not found as folder {DATASET_DRIVE_FOLDER} or verified archive {DATASET_ARCHIVE}"
)
if folder_upload:
    for required in ("data", "meta", "videos"):
        assert (DATASET_DRIVE_FOLDER / required).is_dir(), f"Uploaded dataset folder is missing {required}"
for directory in (CHECKPOINT_ROOT, REPORT_ROOT, DEPLOYMENT_ROOT):
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / ".write_test"
    probe.write_text("ok")
    probe.unlink()
print("Drive paths verified; dataset source:", "folder" if folder_upload else "archive")
"""
    ),
    markdown(
        r"""
## Install the pinned π0 software stack

If Colab says packages already loaded must be restarted, restart the runtime once and rerun from the top. The notebook itself does not require a clean module table.
"""
    ),
    code(
        r"""
%pip install -q --upgrade pip
%pip install -q "lerobot[pi]==0.6.1" "wandb>=0.19" "av==15.1.0" psutil matplotlib pandas num2words

import importlib.metadata as md
import av, torch, torchvision

assert torch.cuda.is_available(), "PyTorch cannot see the A100"
assert md.version("lerobot") == "0.6.1", md.version("lerobot")
assert av.__version__ == "15.1.0" and hasattr(av, "option"), (
    f"Expected PyAV 15.1.0 with av.option, but the runtime still has {av.__version__}. "
    "Restart the Colab runtime once, then run all cells from the top."
)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
runtime_report.update({
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "lerobot": md.version("lerobot"),
    "pyav": av.__version__,
    "cuda": torch.version.cuda,
})
print(json.dumps(runtime_report, indent=2))
"""
    ),
    code(
        r"""
# 3. Copy the dataset to fast local Colab storage, safely extracting only a verified archive.
import hashlib, shutil, tarfile
from pathlib import PurePosixPath

if LOCAL_DATASET.exists():
    shutil.rmtree(LOCAL_DATASET)
if folder_upload:
    print("Copying the LeRobot folder from Drive to local storage ...")
    shutil.copytree(DATASET_DRIVE_FOLDER, LOCAL_DATASET)
else:
    expected_sha = DATASET_CHECKSUM.read_text().split()[0].lower()
    digest = hashlib.sha256()
    with DATASET_ARCHIVE.open("rb") as source, LOCAL_ARCHIVE.open("wb") as target:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
            target.write(chunk)
    assert digest.hexdigest() == expected_sha, f"Dataset SHA-256 mismatch: {digest.hexdigest()}"
    with tarfile.open(LOCAL_ARCHIVE, "r:") as archive:
        members = archive.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            assert not path.is_absolute() and ".." not in path.parts, f"Unsafe archive member: {member.name}"
            assert not member.issym() and not member.islnk(), f"Archive link rejected: {member.name}"
        archive.extractall("/content", members=members, filter="data")

assert LOCAL_DATASET.is_dir(), f"Dataset copy did not create {LOCAL_DATASET}"
for required in ("data", "meta", "videos"):
    assert (LOCAL_DATASET / required).is_dir(), f"Local dataset is missing {required}"
print("Local dataset ready:", LOCAL_DATASET)
"""
    ),
    code(
        r"""
# 4. Validate metadata and all 7D state/action values before spending GPU time.
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(REPO_ID, root=LOCAL_DATASET, video_backend="pyav")
features = dataset.meta.features
assert dataset.meta.total_episodes == 73, dataset.meta.total_episodes
assert tuple(features["observation.state"]["shape"]) == (7,)
assert tuple(features["action"]["shape"]) == (7,)
for camera in ("observation.images.base", "observation.images.wrist"):
    assert features[camera]["dtype"] == "video", (camera, features[camera])
assert TASK in dataset.meta.tasks.index, list(dataset.meta.tasks.index)

for index in range(len(dataset)):
    row = dataset.hf_dataset[index]
    state = np.asarray(row["observation.state"], dtype=np.float32)
    action = np.asarray(row["action"], dtype=np.float32)
    assert state.shape == action.shape == (7,)
    assert np.isfinite(state).all() and np.isfinite(action).all(), f"Non-finite value at frame {index}"
    assert min(abs(float(state[6])), abs(float(state[6]) - 1.0)) < 1e-5, f"Bad state gripper at {index}"
    assert min(abs(float(action[6])), abs(float(action[6]) - 1.0)) < 1e-5, f"Bad action gripper at {index}"
print({
    "episodes": dataset.meta.total_episodes,
    "frames": dataset.meta.total_frames,
    "cameras": ["base", "wrist"],
    "state/action": "7D absolute joints + binary gripper",
    "validation": "PASS",
})
"""
    ),
    code(
        r"""
# 5. Authenticate and prove gated PaliGemma access before starting a long model/batch probe.
from google.colab import userdata
from huggingface_hub import HfApi, hf_hub_download, login as hf_login
import wandb

try:
    hf_token = userdata.get("HF_TOKEN")
except Exception:
    hf_token = None
assert hf_token, (
    "HF_TOKEN is missing. First accept access at https://huggingface.co/google/paligemma-3b-pt-224, "
    "then create a Hugging Face read token, add it to Colab Secrets as HF_TOKEN, and enable notebook access."
)
hf_login(token=hf_token, add_to_git_credential=False)
try:
    hf_identity = HfApi().whoami(token=hf_token)
    paligemma_config = hf_hub_download(
        repo_id="google/paligemma-3b-pt-224",
        filename="config.json",
        token=hf_token,
    )
except Exception as exc:
    raise RuntimeError(
        "HF_TOKEN is present but cannot access google/paligemma-3b-pt-224. "
        "Log into the same Hugging Face account, accept the repository terms, and replace the Colab secret."
    ) from exc
print("Hugging Face gated-model access: PASS for", hf_identity.get("name", "authenticated user"))

try:
    wandb_key = userdata.get("WANDB_API_KEY")
except Exception:
    wandb_key = None
wandb.login(key=wandb_key) if wandb_key else wandb.login()
WANDB_PROJECT = "pi0_ur5_cube_clean_73ep_v2_a100"
print("W&B project:", WANDB_PROJECT)
"""
    ),
    markdown(
        r"""
## Training configuration and A100 batch probe

`expert` is the recommended A100 40 GB profile. It trains the action expert and projections and avoids allocating optimizer state for the entire vision-language backbone. `full` is available for experimentation, but its candidate batches are intentionally small. Actions remain absolute because that is how this dataset was recorded.
"""
    ),
    code(
        r"""
# 6. Edit only this small block when changing the experiment.
FINETUNE_MODE = "expert"       # "expert" (recommended on 40 GB) or "full"
TRAIN_STEPS = 20_000
SAVE_FREQ = 2_500
SEED = 42
# The pretrained pi0 checkpoint names three cameras. Map our two real cameras to
# its base and left-wrist slots; pi0 masks/pads the absent right-wrist slot.
CAMERA_RENAME_MAP = {
    "observation.images.base": "observation.images.base_0_rgb",
    "observation.images.wrist": "observation.images.left_wrist_0_rgb",
}
NUM_WORKERS = max(4, min(12, (os.cpu_count() or 8) - 2))
BENCHMARK_STEPS = 4
GPU_MEMORY_LIMIT_MIB = int(runtime_report["memory_mib"] * 0.92)
BATCH_CANDIDATES = [4, 8, 16, 24] if FINETUNE_MODE == "expert" else [1, 2, 4]

assert FINETUNE_MODE in {"expert", "full"}
assert TRAIN_STEPS > 0 and SAVE_FREQ > 0 and TRAIN_STEPS % SAVE_FREQ == 0
TRAIN_EXPERT_ONLY = FINETUNE_MODE == "expert"
FREEZE_VISION_ENCODER = FINETUNE_MODE == "expert"
print({
    "mode": FINETUNE_MODE,
    "steps": TRAIN_STEPS,
    "candidate_batches": BATCH_CANDIDATES,
    "workers": NUM_WORKERS,
    "absolute_actions": True,
    "camera_rename_map": CAMERA_RENAME_MAP,
})
"""
    ),
    code(
        r"""
# 7. Probe short runs and select the fastest successful batch below the VRAM reserve.
import re, statistics, threading

BENCHMARK_ROOT = Path("/content/pi0_a100_batch_benchmarks")
shutil.rmtree(BENCHMARK_ROOT, ignore_errors=True)
BENCHMARK_ROOT.mkdir(parents=True)

def gpu_sample():
    text = output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
    memory, utilization = [int(float(value.strip())) for value in text.splitlines()[0].split(",")[:2]]
    return memory, utilization

def policy_arguments(compile_model):
    return [
        "--policy.path=lerobot/pi0_base",
        "--policy.device=cuda",
        "--policy.dtype=bfloat16",
        "--policy.use_relative_actions=false",
        "--policy.gradient_checkpointing=true",
        f"--policy.compile_model={str(compile_model).lower()}",
        "--policy.compile_mode=max-autotune",
        f"--policy.freeze_vision_encoder={str(FREEZE_VISION_ENCODER).lower()}",
        f"--policy.train_expert_only={str(TRAIN_EXPERT_ONLY).lower()}",
        "--policy.push_to_hub=false",
    ]

def benchmark(batch_size):
    run_dir = BENCHMARK_ROOT / f"batch_{batch_size}"
    command = ["lerobot-train", *policy_arguments(False),
        f"--dataset.repo_id={REPO_ID}", f"--dataset.root={LOCAL_DATASET}", "--dataset.video_backend=pyav",
        "--dataset.image_transforms.enable=false",
        f"--rename_map={json.dumps(CAMERA_RENAME_MAP, separators=(',', ':'))}",
        f"--batch_size={batch_size}", f"--num_workers={NUM_WORKERS}", f"--seed={SEED}",
        "--wandb.enable=false", "--save_checkpoint=false", "--env_eval_freq=0", "--eval_steps=0",
        f"--steps={BENCHMARK_STEPS}", "--log_freq=1", f"--output_dir={run_dir}",
        f"--job_name=pi0_{FINETUNE_MODE}_bench_b{batch_size}",
    ]
    samples, log_lines, stop = [], [], threading.Event()
    def monitor():
        while not stop.wait(0.25):
            try: samples.append(gpu_sample())
            except Exception: pass
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    started = time.time()
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    for line in process.stdout:
        print(f"[batch {batch_size}] {line}", end="")
        log_lines.append(line)
    return_code = process.wait()
    stop.set(); monitor_thread.join(timeout=2)
    text = "".join(log_lines)
    update_times = [float(value) for value in re.findall(r"updt_s:([0-9.]+)", text)]
    stable_times = update_times[max(1, len(update_times) // 2):]
    peak_memory = max((sample[0] for sample in samples), default=0)
    throughput = batch_size / statistics.median(stable_times) if stable_times and return_code == 0 else 0.0
    result = {
        "batch_size": batch_size,
        "success": return_code == 0,
        "peak_memory_mib": peak_memory,
        "median_gpu_utilization": statistics.median([x[1] for x in samples]) if samples else 0,
        "examples_per_second": throughput,
        "elapsed_s": time.time() - started,
        "oom": "out of memory" in text.lower(),
    }
    if return_code != 0:
        result["error_tail"] = "\n".join(text.splitlines()[-50:])
    shutil.rmtree(run_dir, ignore_errors=True)
    torch.cuda.empty_cache()
    return result

benchmark_results = []
for candidate in BATCH_CANDIDATES:
    result = benchmark(candidate)
    benchmark_results.append(result)
    print(json.dumps(result, indent=2))
    if not result["success"]:
        break

eligible = [r for r in benchmark_results if r["success"] and r["peak_memory_mib"] <= GPU_MEMORY_LIMIT_MIB]
assert eligible, (
    "No batch succeeded. This was not necessarily an OOM. The final subprocess traceback is included below:\n"
    + "\n\n".join(result.get("error_tail", json.dumps(result, indent=2)) for result in benchmark_results)
)
selected = max(eligible, key=lambda r: r["examples_per_second"])
BATCH_SIZE = selected["batch_size"]
print("SELECTED BATCH:", BATCH_SIZE, json.dumps(selected, indent=2))
"""
    ),
    code(
        r"""
# 8. Resolve a collision-proof run and the exact command that will be preserved with the model.
import random

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_NAME = f"{DATASET_NAME}_pi0_{FINETUNE_MODE}_a100_b{BATCH_SIZE}_{TRAIN_STEPS}_{timestamp}"
TRAIN_OUTPUT = Path("/content/outputs") / RUN_NAME
LOCAL_REPORT_DIR = Path("/content/training_reports") / RUN_NAME
DRIVE_RUN_DIR = CHECKPOINT_ROOT / RUN_NAME
DRIVE_CHECKPOINTS_DIR = DRIVE_RUN_DIR / "checkpoints"
LOCAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
assert not TRAIN_OUTPUT.exists(), TRAIN_OUTPUT
assert not DRIVE_RUN_DIR.exists(), DRIVE_RUN_DIR

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
TRAIN_ARGS = ["lerobot-train", *policy_arguments(True),
    f"--dataset.repo_id={REPO_ID}", f"--dataset.root={LOCAL_DATASET}", "--dataset.video_backend=pyav",
    "--dataset.image_transforms.enable=false",
    f"--rename_map={json.dumps(CAMERA_RENAME_MAP, separators=(',', ':'))}",
    f"--batch_size={BATCH_SIZE}", f"--num_workers={NUM_WORKERS}", f"--seed={SEED}",
    "--wandb.enable=true", f"--wandb.project={WANDB_PROJECT}",
    "--save_checkpoint=true", "--env_eval_freq=0", "--eval_steps=0", f"--steps={TRAIN_STEPS}", f"--save_freq={SAVE_FREQ}",
    "--log_freq=10", "--use_policy_training_preset=true",
    f"--output_dir={TRAIN_OUTPUT}", f"--job_name={RUN_NAME}",
]
resolved = {
    "run_name": RUN_NAME,
    "dataset": DATASET_NAME,
    "episodes": dataset.meta.total_episodes,
    "frames": dataset.meta.total_frames,
    "policy": "pi0",
    "base_model": "lerobot/pi0_base",
    "finetune_mode": FINETUNE_MODE,
    "train_expert_only": TRAIN_EXPERT_ONLY,
    "freeze_vision_encoder": FREEZE_VISION_ENCODER,
    "action_mode": "absolute",
    "camera_rename_map": CAMERA_RENAME_MAP,
    "missing_policy_camera": "observation.images.right_wrist_0_rgb (masked zero image)",
    "dtype": "bfloat16",
    "batch_size": BATCH_SIZE,
    "steps": TRAIN_STEPS,
    "save_freq": SAVE_FREQ,
    "workers": NUM_WORKERS,
    "benchmark_results": benchmark_results,
    "command": TRAIN_ARGS,
}
(LOCAL_REPORT_DIR / "resolved_training_config.json").write_text(json.dumps(resolved, indent=2))
print(json.dumps(resolved, indent=2))
"""
    ),
    markdown(
        r"""
## Train, watch live progress, and sync checkpoints

Leave the next cell running. W&B provides the loss graphs; the inline line reports percentage and ETA whenever LeRobot logs a step. Completed checkpoints and logs are copied to Drive during the run, not only at the end.
"""
    ),
    code(
        r"""
# 9. Run training with live percentage, ETA, GPU logging, and completed-checkpoint sync.
import csv
from IPython.display import clear_output, display

TRAIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
TRAIN_LOG = LOCAL_REPORT_DIR / "training.log"
GPU_LOG = LOCAL_REPORT_DIR / "gpu_metrics.csv"
DRIVE_RUN_DIR.mkdir(parents=True)
DRIVE_CHECKPOINTS_DIR.mkdir()
shutil.copy2(LOCAL_REPORT_DIR / "resolved_training_config.json", DRIVE_RUN_DIR / "resolved_training_config.json")

def model_complete(model_dir):
    return (model_dir / "config.json").is_file() and any(model_dir.glob("*.safetensors"))

def checkpoint_complete(path):
    training_state = path / "training_state" / "training_step.json"
    return path.is_dir() and training_state.is_file() and model_complete(path / "pretrained_model")

def sync_checkpoints():
    source_root = TRAIN_OUTPUT / "checkpoints"
    if not source_root.exists():
        return
    for source in sorted(source_root.iterdir()):
        if source.name == "last" or not checkpoint_complete(source):
            continue
        target = DRIVE_CHECKPOINTS_DIR / source.name
        if target.exists():
            continue
        staging = DRIVE_CHECKPOINTS_DIR / f".{source.name}.copying"
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(source, staging)
        staging.rename(target)
        print("SYNCED CHECKPOINT:", target)

monitor_stop = threading.Event()
def gpu_monitor():
    with GPU_LOG.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "memory_mib", "utilization_percent"])
        started = time.time()
        while not monitor_stop.wait(2.0):
            try:
                memory, utilization = gpu_sample()
                writer.writerow([time.time() - started, memory, utilization])
                stream.flush()
            except Exception:
                pass

monitor_thread = threading.Thread(target=gpu_monitor, daemon=True)
monitor_thread.start()
started = time.time()
last_step = 0
line_count = 0
step_pattern = re.compile(r"(?:step[:= ]+)(\d+)", re.IGNORECASE)
with TRAIN_LOG.open("w") as log:
    process = subprocess.Popen(TRAIN_ARGS, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    for line in process.stdout:
        print(line, end="")
        log.write(line); log.flush(); line_count += 1
        matches = step_pattern.findall(line)
        if matches:
            last_step = max(last_step, int(matches[-1]))
            elapsed = time.time() - started
            eta = elapsed * (TRAIN_STEPS - last_step) / max(last_step, 1)
            print(f"PROGRESS {100 * last_step / TRAIN_STEPS:6.2f}% | step {last_step}/{TRAIN_STEPS} | ETA {eta/3600:.2f} h")
        if line_count % 25 == 0:
            sync_checkpoints()
        if line_count % 100 == 0:
            shutil.copy2(TRAIN_LOG, DRIVE_RUN_DIR / "training_in_progress.log")
            if GPU_LOG.exists(): shutil.copy2(GPU_LOG, DRIVE_RUN_DIR / "gpu_metrics_in_progress.csv")
    return_code = process.wait()

monitor_stop.set(); monitor_thread.join(timeout=3); sync_checkpoints()
shutil.copy2(TRAIN_LOG, DRIVE_RUN_DIR / "training.log")
shutil.copy2(GPU_LOG, DRIVE_RUN_DIR / "gpu_metrics.csv")
assert return_code == 0, f"Training failed with code {return_code}; inspect {TRAIN_LOG}"
print("TRAINING COMPLETE:", RUN_NAME)
"""
    ),
    code(
        r"""
# 10. Plot actual A100 use and copy the last complete pretrained model to its stable Drive location.
import matplotlib.pyplot as plt
import pandas as pd

metrics = pd.read_csv(GPU_LOG)
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].plot(metrics.time_s / 60, metrics.utilization_percent)
axes[0].set_ylabel("GPU utilization %"); axes[0].set_ylim(0, 105); axes[0].grid()
axes[1].plot(metrics.time_s / 60, metrics.memory_mib / 1024)
axes[1].set_ylabel("VRAM GiB"); axes[1].set_xlabel("Minutes"); axes[1].grid()
plt.tight_layout(); plt.show()
summary = {
    "median_gpu_utilization": float(metrics.utilization_percent.median()),
    "p95_gpu_utilization": float(metrics.utilization_percent.quantile(0.95)),
    "peak_vram_gib": float(metrics.memory_mib.max() / 1024),
}
print(json.dumps(summary, indent=2))

models = [p for p in TRAIN_OUTPUT.glob("checkpoints/*/pretrained_model") if model_complete(p)]
models.sort(key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else -1)
assert models, "No complete local π0 checkpoint found"
FINAL_MODEL = models[-1]
final_target = DRIVE_RUN_DIR / "pretrained_model"
staging = DRIVE_RUN_DIR / ".pretrained_model.copying"
shutil.rmtree(staging, ignore_errors=True)
shutil.copytree(FINAL_MODEL, staging)
assert model_complete(staging), f"Incomplete final model: {staging}"
staging.rename(final_target)
(DRIVE_RUN_DIR / "gpu_summary.json").write_text(json.dumps(summary, indent=2))
print("FINAL MODEL:", final_target)
"""
    ),
    code(
        r"""
# 11. Validate π0 metadata, then package the complete checkpoint and SHA-256 for the robot PC.
config_path = final_target / "config.json"
model_config = json.loads(config_path.read_text())
policy_type = model_config.get("type", model_config.get("policy_type"))
assert policy_type == "pi0", (policy_type, model_config)
assert model_complete(final_target)

export_archive = DEPLOYMENT_ROOT / f"{RUN_NAME}_complete_checkpoint.tar"
with tarfile.open(export_archive, "w") as archive:
    archive.add(final_target, arcname="pretrained_model")
digest = hashlib.sha256()
with export_archive.open("rb") as stream:
    while chunk := stream.read(16 * 1024 * 1024):
        digest.update(chunk)
checksum = Path(f"{export_archive}.sha256")
checksum.write_text(f"{digest.hexdigest()}  {export_archive.name}\n")
shutil.copy2(LOCAL_REPORT_DIR / "resolved_training_config.json", REPORT_ROOT / f"{RUN_NAME}_config.json")
shutil.copy2(TRAIN_LOG, REPORT_ROOT / f"{RUN_NAME}_training.log")
shutil.copy2(GPU_LOG, REPORT_ROOT / f"{RUN_NAME}_gpu_metrics.csv")
print("VALIDATION: PASS")
print("READY TO DOWNLOAD:", export_archive)
print("SHA-256:", digest.hexdigest())
"""
    ),
    markdown(
        r"""
## Finished

Download both the `_complete_checkpoint.tar` and its `.sha256` from `pHYSICAL UR5/deployment/downloaded_checkpoints`. Keep `resolved_training_config.json` with the experiment record.

This π0 checkpoint requires a π0/LeRobot 0.6.1 inference service. It is not a drop-in replacement for the earlier SmolVLA policy server. Before physical execution, validate observation feature names, joint ordering, image ordering, normalization, and 7D output semantics in shadow mode.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "A100",
            "name": OUTPUT.name,
            "provenance": [],
        },
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(OUTPUT)
