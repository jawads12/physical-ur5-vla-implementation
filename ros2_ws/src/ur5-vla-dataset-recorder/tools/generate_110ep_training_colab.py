"""Adapt the previous 73-episode A100 notebook for the reviewed 110-episode dataset."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'notebooks/train_ur5_cube_clean_73ep_v2_smolvla_a100_autotuned_colab.ipynb'
OUTPUT=ROOT/'notebooks/train_ur5_cube_110ep_v3_smolvla_a100_colab.ipynb'
DATA=ROOT.parents[2]/'data/dataset/lerobot/ur5_cube_110ep_v3'
notebook=json.loads(BASE.read_text())

def put(index,text):
    notebook['cells'][index]['source']=text.strip().splitlines(keepends=True)

for cell in notebook['cells']:
    text=''.join(cell['source']).replace('ur5_cube_clean_73ep_v2','ur5_cube_110ep_v3').replace('73-episode','110-episode')
    cell['source']=text.splitlines(keepends=True)
    if cell['cell_type']=='code':cell.update(execution_count=None,outputs=[])
    cell['metadata']={}

put(0,'''# UR5 110-episode SmolVLA training — A100 Colab

This notebook adapts the previous A100 training workflow to **110 episodes / 285,204 frames at 20 Hz** in `ur5_cube_110ep_v3`. Inputs are base/wrist RGB and seven-dimensional state; outputs are six absolute joint targets plus an absolute binary gripper command. The task is **Pick up the orange cube and place it in the bowl**.

Upload the complete converted dataset folder to:
`MyDrive/pHYSICAL UR5/datasets/lerobot/ur5_cube_110ep_v3/`
The folder must contain `data`, `meta`, and `videos`. A `.tar` with a `.tar.sha256` sidecar at the same parent location is also supported. The archive must contain a top-level `ur5_cube_110ep_v3` folder. Edit `DRIVE_ROOT` in Cell 2 if your Drive layout differs.

Select an **A100 GPU** and run cells top to bottom in a fresh Colab runtime. Training uses local `/content` storage. W&B logging, batch autotuning, complete checkpoint copies to Drive, and final model packaging follow the previous workflow. The default budget is approximately **eight dataset passes**: with batch 96, this means **24,000 steps**, not the previous dataset's 16,500.

The dataset includes the original 73 plus 37 accepted new episodes. The original four-transition episode is retained as agreed for the 110-episode conversion. All episodes are in the training split; this notebook does not claim held-out evaluation or physical success. It does not connect to the robot. This is a fresh fine-tune from `lerobot/smolvla_base`, not a resume of an earlier run.''')

put(2,'''# 2. Mount Drive and configure the new dataset.
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = Path('/content/drive/MyDrive/pHYSICAL UR5')
DATASET_NAME = 'ur5_cube_110ep_v3'
REPO_ID = f'local/{DATASET_NAME}'
EXPECTED_EPISODES = 110
EXPECTED_FRAMES = 285204
TASK = 'Pick up the orange cube and place it in the bowl'
TARGET_EPOCHS = 8.0
USE_WANDB = True
RUN_TAG = time.strftime('%Y%m%d_%H%M%S')
DATASET_DRIVE_FOLDER = DRIVE_ROOT / 'datasets/lerobot' / DATASET_NAME
DATASET_ARCHIVE = DRIVE_ROOT / 'datasets/lerobot' / f'{DATASET_NAME}.tar'
DATASET_CHECKSUM = Path(f'{DATASET_ARCHIVE}.sha256')
LOCAL_ARCHIVE = Path('/content') / DATASET_ARCHIVE.name
LOCAL_DATASET = Path('/content') / DATASET_NAME
CHECKPOINT_ROOT = DRIVE_ROOT / 'checkpoints/ur5_smolvla_trained'
REPORT_ROOT = DRIVE_ROOT / 'reports/training'
DEPLOYMENT_ROOT = DRIVE_ROOT / 'deployment/downloaded_checkpoints'
assert DRIVE_ROOT.is_dir(), f'Drive root missing: {DRIVE_ROOT}'
folder_upload = DATASET_DRIVE_FOLDER.is_dir()
archive_upload = DATASET_ARCHIVE.is_file() and DATASET_CHECKSUM.is_file()
assert folder_upload or archive_upload, f'Upload {DATASET_DRIVE_FOLDER}, or {DATASET_ARCHIVE} plus its .sha256 file'
if folder_upload:
    for name in ('data', 'meta', 'videos'):
        assert (DATASET_DRIVE_FOLDER / name).is_dir(), f'Incomplete upload: missing {name}'
for directory in (CHECKPOINT_ROOT, REPORT_ROOT, DEPLOYMENT_ROOT):
    directory.mkdir(parents=True, exist_ok=True)
print('Input:', DATASET_DRIVE_FOLDER if folder_upload else DATASET_ARCHIVE)
print('Dataset:', EXPECTED_EPISODES, 'episodes;', EXPECTED_FRAMES, 'frames')
print('W&B enabled:', USE_WANDB, '| Target dataset passes:', TARGET_EPOCHS)
''')

put(3,'''## Install the previous training stack
Use a fresh runtime. If Torch or Torchvision has already been imported before installation, choose **Runtime → Disconnect and delete runtime**, reconnect with an A100, and start again. The notebook intentionally keeps the earlier Torch 2.7.1 / Torchvision 0.22.1 / LeRobot 0.4.3 training versions. The dataset was converted locally with LeRobot 0.5.2 in the same v3 data format; the next cells verify that the training loader can read it before any benchmark or training starts.''')
put(4,'''# Install before importing Torch; do not replace libraries in an already-imported runtime.
assert 'torch' not in sys.modules and 'torchvision' not in sys.modules, 'Disconnect and delete runtime, then run from Cell 1.'
%pip install -q --upgrade pip
%pip install -q --index-url https://download.pytorch.org/whl/cu126 "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1"
%pip install -q "lerobot[smolvla]==0.4.3" "av==15.1.0" "wandb>=0.19" psutil matplotlib num2words
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import importlib.metadata as md, torch, torchvision
assert torch.cuda.is_available()
assert torch.__version__.startswith('2.7.1')
assert torchvision.__version__.startswith('0.22.1')
assert md.version('lerobot') == '0.4.3'
assert torchvision.extension._has_ops()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
runtime_report.update({'torch': torch.__version__, 'torchvision': torchvision.__version__, 'lerobot': md.version('lerobot'), 'cuda': torch.version.cuda})
print(json.dumps(runtime_report, indent=2))
''')
# Explicitly refuse an imported runtime in the first cell as well.
text=''.join(notebook['cells'][1]['source'])
start=text.index('torch_already_imported =')
stop=text.index('runtime_report =',start)
text=text[:start]+"assert 'torch' not in sys.modules and 'torchvision' not in sys.modules, 'Start a fresh runtime before installing the pinned stack.'\n"+text[stop:]
put(1,text)

expected_meta={str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA/'meta/info.json',DATA/'meta/ur5_vla_conversion.json']}
put(5,'''# 3. Copy to local disk, verify every copied file, and retain a fingerprint report.
import hashlib, shutil, tarfile
from pathlib import PurePosixPath

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def tree_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob('*')) if p.is_file()}

def copy_verified_tree(source, destination):
    source, destination = Path(source), Path(destination)
    before = tree_hashes(source)
    assert before, f'No files in {source}'
    if destination.exists():
        assert tree_hashes(destination) == before, f'Destination exists with different content: {destination}'
        return before
    staging = destination.with_name('.' + destination.name + '.copying')
    assert not staging.exists(), f'Partial copy exists: {staging}. Inspect it before retrying.'
    shutil.copytree(source, staging)
    assert tree_hashes(staging) == before, f'Copy checksum failed: {staging}'
    assert tree_hashes(source) == before, f'Source changed during copy: {source}'
    staging.rename(destination)
    return before

assert shutil.disk_usage('/content').free >= 35 * 1024**3, 'Need at least 35 GiB free local disk before staging/training.'
if folder_upload:
    print('Copying dataset folder; verifying SHA-256 of every file ...')
    dataset_file_hashes = copy_verified_tree(DATASET_DRIVE_FOLDER, LOCAL_DATASET)
else:
    expected_sha = DATASET_CHECKSUM.read_text().split()[0].lower()
    assert len(expected_sha) == 64 and all(c in '0123456789abcdef' for c in expected_sha)
    if not LOCAL_ARCHIVE.exists():
        shutil.copy2(DATASET_ARCHIVE, LOCAL_ARCHIVE)
    assert sha256_file(LOCAL_ARCHIVE) == expected_sha, 'Archive SHA-256 mismatch'
    assert not LOCAL_DATASET.exists(), f'Local dataset already exists: {LOCAL_DATASET}; inspect before extracting.'
    staging = Path('/content') / ('.' + DATASET_NAME + '.extracting')
    staging.mkdir(exist_ok=False)
    with tarfile.open(LOCAL_ARCHIVE, 'r:*') as archive:
        members = archive.getmembers()
        for member in members:
            relative = PurePosixPath(member.name)
            assert relative.parts and relative.parts[0] == DATASET_NAME, f'Unexpected archive root: {member.name}'
            assert not relative.is_absolute() and '..' not in relative.parts
            assert member.isfile() or member.isdir(), f'Unsupported archive entry: {member.name}'
        archive.extractall(staging, members=members, filter='data')
    (staging / DATASET_NAME).rename(LOCAL_DATASET)
    staging.rmdir()
    dataset_file_hashes = tree_hashes(LOCAL_DATASET)
EXPECTED_METADATA_SHA256 = '''+repr(expected_meta)+'''
for relative, expected in EXPECTED_METADATA_SHA256.items():
    assert sha256_file(LOCAL_DATASET / relative) == expected, f'Dataset metadata differs from the reviewed 110-episode artifact: {relative}'
for name in ('data', 'meta', 'videos'):
    assert (LOCAL_DATASET / name).is_dir(), f'Missing local {name}'
DATASET_HASH_REPORT = Path('/content') / f'{DATASET_NAME}_file_hashes.json'
DATASET_HASH_REPORT.write_text(json.dumps(dataset_file_hashes, indent=2))
print('COPY VERIFIED:', len(dataset_file_hashes), 'files;', LOCAL_DATASET)
''')

put(6,'''# 4. Validate all state/action rows and decode three paired video samples per episode.
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(REPO_ID, root=LOCAL_DATASET, video_backend='pyav')
assert dataset.meta.total_episodes == EXPECTED_EPISODES
assert dataset.meta.total_frames == EXPECTED_FRAMES == len(dataset)
assert dataset.meta.fps == 20
assert list(dataset.meta.tasks.index) == [TASK]
features = dataset.meta.features
JOINT_ORDER = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
for key in ('observation.images.base', 'observation.images.wrist'):
    assert features[key]['dtype'] == 'video' and tuple(features[key]['shape']) == (480, 640, 3)
assert tuple(features['observation.state']['shape']) == (7,)
assert tuple(features['action']['shape']) == (7,)
assert features['observation.state']['names'] == JOINT_ORDER + ['estimated_gripper_state']
assert features['action']['names'] == JOINT_ORDER + ['gripper_command']
conversion = json.loads((LOCAL_DATASET / 'meta/ur5_vla_conversion.json').read_text())
assert conversion['action_semantics'] == 'six absolute joint-position targets plus absolute binary gripper command'
assert conversion['action_mode'] == 'absolute' and conversion['joint_unit'] == 'radian'
assert conversion['joint_order'] == JOINT_ORDER
assert conversion['gripper_closed_value'] == 0.0 and conversion['gripper_open_value'] == 1.0
assert conversion['gripper_is_measured'] is False
assert len(conversion['source_episode_names']) == len(set(conversion['source_episode_names'])) == EXPECTED_EPISODES
assert not {'episode_20261001_175654', 'episode_20261001_182413'} & set(conversion['source_episode_names'])

rows_checked, episode_ids = 0, set()
vectors = dataset.hf_dataset.select_columns(['observation.state', 'action', 'episode_index', 'task_index'])
for batch in vectors.iter(batch_size=8192):
    state = np.asarray(batch['observation.state'], dtype=np.float32)
    action = np.asarray(batch['action'], dtype=np.float32)
    assert state.ndim == 2 and state.shape == action.shape and state.shape[1] == 7
    assert np.isfinite(state).all() and np.isfinite(action).all()
    for gripper in (state[:, 6], action[:, 6]):
        assert np.all(np.minimum(np.abs(gripper), np.abs(gripper - 1)) < 1e-5)
    assert np.all(np.asarray(batch['task_index']) == 0)
    episode_ids.update(np.asarray(batch['episode_index']).reshape(-1).tolist())
    rows_checked += len(state)
assert rows_checked == EXPECTED_FRAMES and episode_ids == set(range(EXPECTED_EPISODES))

sample_count = 0
for ep in range(EXPECTED_EPISODES):
    info = dataset.meta.episodes[ep]
    start, stop = int(info['dataset_from_index']), int(info['dataset_to_index'])
    assert 0 <= start < stop <= EXPECTED_FRAMES
    for index in sorted({start, (start + stop - 1) // 2, stop - 1}):
        sample = dataset[index]
        for key in ('observation.images.base', 'observation.images.wrist'):
            image = np.asarray(sample[key])
            assert image.shape == (3, 480, 640) and np.isfinite(image).all()
        sample_count += 1
    if (ep + 1) % 10 == 0:
        print(f'Video validation: {ep + 1}/{EXPECTED_EPISODES} episodes')
validation_report = {'episodes': EXPECTED_EPISODES, 'frames': rows_checked, 'fps': 20, 'task': TASK, 'paired_video_samples': sample_count, 'action_mode': 'absolute', 'validation': 'PASS', 'held_out_split': False}
print(json.dumps(validation_report, indent=2))
Path('/content/dataset_validation.json').write_text(json.dumps(validation_report, indent=2))
# Show representative paired inputs before training.
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for row, index in enumerate((0, EXPECTED_FRAMES - 1)):
    sample = dataset[index]
    for col, key in enumerate(('observation.images.base', 'observation.images.wrist')):
        axes[row, col].imshow(np.asarray(sample[key]).transpose(1, 2, 0))
        axes[row, col].set_title(f'{key.rsplit(".", 1)[-1]} — frame {index}')
        axes[row, col].axis('off')
plt.tight_layout(); plt.show()
''')
put(7,'''# 5. W&B login uses its secure prompt; no credentials are saved in notebook source.
WANDB_PROJECT = 'smolvla_ur5_cube_110ep_v3_a100'
if USE_WANDB:
    import wandb
    wandb.login()
print('W&B:', WANDB_PROJECT if USE_WANDB else 'disabled')
''')

# Keep the previous batch benchmark, adding observable-failure requirements.
text=''.join(notebook['cells'][9]['source'])
text=text.replace('max(4, min(12, (os.cpu_count() or 8) - 2))','max(1, min(12, (os.cpu_count() or 2) - 2))')
text=text.replace('Path("/content/a100_batch_benchmarks")','Path("/content") / f"{DATASET_NAME}_benchmarks_{RUN_TAG}"')
text=text.replace('if BENCHMARK_ROOT.exists(): shutil.rmtree(BENCHMARK_ROOT)','assert not BENCHMARK_ROOT.exists(), "Benchmark directory already exists; inspect before rerunning"')
text=text.replace('r"updt_s:([0-9.]+)"','r"updt_s:\\s*([0-9.]+)"')
text=text.replace('    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)\n    for line in process.stdout:\n        print(f"[batch {batch_size}] {line}", end="")\n        lines.append(line)\n    code = process.wait(); stop.set(); thread.join(timeout=2)', '''    process = None
    try:
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        for line in process.stdout:
            print(f"[batch {batch_size}] {line}", end="")
            lines.append(line)
        code = process.wait()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        stop.set(); thread.join(timeout=2)
        (BENCHMARK_ROOT / f"batch_{batch_size}.log").write_text("".join(lines))''')
text=text.replace('    shutil.rmtree(run_dir, ignore_errors=True)\n    return result','    if code != 0:\n        result["error_tail"] = "\\n".join(text.splitlines()[-40:])\n    (BENCHMARK_ROOT / f"batch_{batch_size}.log").write_text(text)\n    shutil.rmtree(run_dir, ignore_errors=True)\n    return result')
text=text.replace('item["success"] and item["peak_memory_mib"] <= GPU_MEMORY_LIMIT_MIB','item["success"] and 0 < item["peak_memory_mib"] <= GPU_MEMORY_LIMIT_MIB and item["examples_per_second"] > 0')
text=text.replace('assert eligible, f"No A100 batch candidate succeeded: {benchmark_results}"','assert eligible, f"No measured batch met the reserve/throughput requirements. Inspect benchmark errors: {benchmark_results}"')
put(9,text)
text=''.join(notebook['cells'][10]['source']).replace('# 7. Preserve the previous successful run\'s sample exposure instead of blindly keeping 60k steps.','# 7. Set the training budget from the NEW dataset size and measured batch size.')
text=text.replace('TARGET_EPOCHS = 8.0\n','')
text=text.replace('_{TRAIN_STEPS}_steps"','_{TRAIN_STEPS}_steps_{RUN_TAG}"')
text=text.replace('"--wandb.enable=true"','f"--wandb.enable={str(USE_WANDB).lower()}"')
text=text.replace('"action_mode": "absolute",','"action_mode": "absolute", "seed": SEED, "runtime": runtime_report, "drive_checkpoint_retention": "all", "validation": validation_report,')
text+='\n(LOCAL_REPORT_DIR / "dataset_file_hashes.json").write_text(DATASET_HASH_REPORT.read_text())\n(LOCAL_REPORT_DIR / "pip_freeze.txt").write_text(output([sys.executable, "-m", "pip", "freeze"]))\nshutil.copytree(BENCHMARK_ROOT, LOCAL_REPORT_DIR / "batch_benchmarks")\n'
put(10,text)

put(11,'''## Train, sync, and export automatically
Leave the next cell running. It streams the training log, monitors GPU use, preserves every completed checkpoint on Drive, and packages the final model automatically. Checkpoints are eligible only after LeRobot's `last` pointer confirms saving completed. Every copied file is SHA-256 verified before the staging folder is renamed. If training fails or is interrupted, completed checkpoint artifacts and logs are retained; this notebook does not silently resume or overwrite an earlier run. The final plotting cell is optional.''')

put(12,'''# 8. Train and automatically preserve complete artifacts and final export.
import csv
TRAIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
TRAIN_LOG = LOCAL_REPORT_DIR / 'training.log'
GPU_LOG = LOCAL_REPORT_DIR / 'gpu_metrics.csv'
DRIVE_RUN_DIR.mkdir(parents=True, exist_ok=False)
DRIVE_CHECKPOINTS_DIR.mkdir()
for filename in ('resolved_training_config.json', 'dataset_file_hashes.json', 'pip_freeze.txt'):
    shutil.copy2(LOCAL_REPORT_DIR / filename, DRIVE_RUN_DIR / filename)

REQUIRED_CHECKPOINT_FILES = (
    'pretrained_model/model.safetensors', 'pretrained_model/config.json',
    'pretrained_model/train_config.json', 'pretrained_model/policy_preprocessor.json',
    'pretrained_model/policy_postprocessor.json', 'training_state/training_step.json',
    'training_state/optimizer_state.safetensors', 'training_state/optimizer_param_groups.json',
    'training_state/rng_state.safetensors', 'training_state/scheduler_state.json',
)

def checkpoint_complete(path):
    return path.is_dir() and all((path / name).is_file() and (path / name).stat().st_size > 0 for name in REQUIRED_CHECKPOINT_FILES)

synced_steps = set()
def sync_checkpoints():
    checkpoint_dir = TRAIN_OUTPUT / 'checkpoints'
    last = checkpoint_dir / 'last'
    if not last.is_symlink() or not last.exists():
        return
    latest_name = last.resolve().name
    if not latest_name.isdigit():
        raise RuntimeError(f'Unexpected completed-checkpoint pointer: {last}')
    for source in sorted(checkpoint_dir.iterdir()):
        if not source.name.isdigit() or int(source.name) > int(latest_name) or source.name in synced_steps:
            continue
        if not checkpoint_complete(source):
            raise RuntimeError(f'Completed checkpoint is missing required artifacts: {source}')
        target = DRIVE_CHECKPOINTS_DIR / source.name
        hashes = copy_verified_tree(source, target)
        (DRIVE_RUN_DIR / f'checkpoint_{source.name}_sha256.json').write_text(json.dumps(hashes, indent=2))
        synced_steps.add(source.name)
        print('SYNCED COMPLETE CHECKPOINT TO DRIVE:', target, flush=True)

monitor_stop = threading.Event()
def gpu_monitor():
    with GPU_LOG.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_s', 'memory_mib', 'utilization_percent'])
        started = time.time()
        while not monitor_stop.wait(2.0):
            try:
                memory, utilization = gpu_sample()
                writer.writerow([time.time() - started, memory, utilization]); stream.flush()
            except Exception:
                pass

monitor_thread = threading.Thread(target=gpu_monitor, daemon=True)
monitor_thread.start()
process = None
try:
    with TRAIN_LOG.open('w') as log:
        process = subprocess.Popen(TRAIN_ARGS, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        for line_count, line in enumerate(process.stdout, 1):
            print(line, end=''); log.write(line); log.flush()
            sync_checkpoints()
            if line_count % 100 == 0:
                shutil.copy2(TRAIN_LOG, DRIVE_RUN_DIR / 'training_in_progress.log')
        return_code = process.wait()
finally:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
    monitor_stop.set(); monitor_thread.join(timeout=5)
    for filename in ('training.log', 'gpu_metrics.csv'):
        source = LOCAL_REPORT_DIR / filename
        if source.exists():
            shutil.copy2(source, DRIVE_RUN_DIR / filename)
    sync_checkpoints()
assert return_code == 0, f'Training failed ({return_code}); inspect {TRAIN_LOG}'
models = [p for p in (TRAIN_OUTPUT / 'checkpoints').iterdir() if p.name.isdigit() and checkpoint_complete(p)]
assert models, 'No complete numbered checkpoint found'
final_checkpoint = max(models, key=lambda p: int(p.name))
assert int(final_checkpoint.name) == TRAIN_STEPS, f'Final step missing: {final_checkpoint}'
FINAL_MODEL = final_checkpoint / 'pretrained_model'
final_target = DRIVE_RUN_DIR / 'pretrained_model'
model_hashes = copy_verified_tree(FINAL_MODEL, final_target)
(DRIVE_RUN_DIR / 'pretrained_model_sha256.json').write_text(json.dumps(model_hashes, indent=2))

export_archive = DEPLOYMENT_ROOT / f'{RUN_NAME}_complete_checkpoint.tar'
assert not export_archive.exists(), f'Export already exists: {export_archive}'
export_staging = export_archive.with_name('.' + export_archive.name + '.copying')
assert not export_staging.exists(), f'Partial export exists: {export_staging}'
with tarfile.open(export_staging, 'w') as archive:
    archive.add(final_target, arcname='pretrained_model')
# Verify every archived regular file against the model directory before finalizing.
with tarfile.open(export_staging, 'r:') as archive:
    archive_hashes = {}
    for member in archive.getmembers():
        if not member.isfile():
            continue
        relative = PurePosixPath(member.name).relative_to('pretrained_model').as_posix()
        digest = hashlib.sha256()
        with archive.extractfile(member) as stream:
            while chunk := stream.read(16 * 1024 * 1024):
                digest.update(chunk)
        archive_hashes[relative] = digest.hexdigest()
assert archive_hashes == model_hashes, 'Export file verification failed'
export_staging.rename(export_archive)
archive_sha256 = sha256_file(export_archive)
Path(f'{export_archive}.sha256').write_text(f'{archive_sha256}  {export_archive.name}\\n')
copy_verified_tree(LOCAL_REPORT_DIR, REPORT_ROOT / RUN_NAME)
print('TRAINING COMPLETE:', RUN_NAME)
print('TRAINED WEIGHTS SAVED TO DRIVE:', final_target)
print('READY TO DOWNLOAD:', export_archive)
print('SHA-256:', archive_sha256)
''')
put(13,'''# 9. Optional: inspect actual GPU utilization after the automatic export.
import matplotlib.pyplot as plt, pandas as pd
metrics = pd.read_csv(GPU_LOG)
if len(metrics):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(metrics.time_s / 60, metrics.utilization_percent)
    axes[0].set_ylabel('GPU utilization %'); axes[0].set_ylim(0, 105); axes[0].grid()
    axes[1].plot(metrics.time_s / 60, metrics.memory_mib / 1024)
    axes[1].set_ylabel('VRAM GiB'); axes[1].set_xlabel('Minutes'); axes[1].grid()
    plt.tight_layout(); plt.show()
    gpu_summary = {'median_gpu_utilization': float(metrics.utilization_percent.median()), 'p95_gpu_utilization': float(metrics.utilization_percent.quantile(0.95)), 'peak_vram_gib': float(metrics.memory_mib.max() / 1024)}
    (DRIVE_RUN_DIR / 'gpu_summary.json').write_text(json.dumps(gpu_summary, indent=2))
    print(json.dumps(gpu_summary, indent=2))
else:
    print('Training finished before a GPU monitor sample was recorded.')
''')
put(14,'''# 10. Locate the exported model and its checksum. No extra copy cell is required.
print('Final model:', DRIVE_RUN_DIR / 'pretrained_model')
print('All retained numbered checkpoints:', DRIVE_CHECKPOINTS_DIR)
print('Download archive:', export_archive)
print('Checksum file:', Path(f'{export_archive}.sha256'))
assert (DRIVE_RUN_DIR / 'pretrained_model/model.safetensors').is_file()
assert export_archive.is_file() and Path(f'{export_archive}.sha256').is_file()
''')
put(15,'''## Finished
Drive now contains the complete final model, all completed numbered checkpoints including optimizer state, dataset fingerprints, software versions, resolved training command, and training/GPU logs. W&B contains metrics if enabled. Download the complete archive and checksum for offline evaluation on the workstation. The notebook has no robot command interface. Evaluate the checkpoint offline and in shadow mode before a separately controlled physical trial.''')
notebook['metadata']['colab']['name']=OUTPUT.name
notebook['metadata']['accelerator']='GPU'
notebook['metadata']['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
for i,cell in enumerate(notebook['cells']):cell['id']=f'ur5-110-{i:02d}'
OUTPUT.write_text(json.dumps(notebook,indent=2)+'\n')
print(OUTPUT)
