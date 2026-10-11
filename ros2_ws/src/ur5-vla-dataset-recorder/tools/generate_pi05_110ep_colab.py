"""Build the separate Pi0.5 notebook using the verified 110-episode workflow."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'notebooks/train_ur5_cube_110ep_v3_pi05_a100_colab.ipynb'
notebook = json.loads((ROOT / 'notebooks/train_ur5_cube_110ep_v3_smolvla_a100_colab.ipynb').read_text())


def source(i):
    return ''.join(notebook['cells'][i]['source'])


def put(i, text):
    notebook['cells'][i]['source'] = (text.strip() + '\n').splitlines(keepends=True)


put(0, '''# UR5 110-episode π0.5 training — A100 Colab

Fine-tune **`lerobot/pi05_base`** on **110 episodes / 285,204 frames at 20 Hz** from `ur5_cube_110ep_v3`. This is a separate Pi0.5 experiment, initialized from its base model. Inputs are base/wrist RGB, seven-dimensional state, and the task instruction. Targets remain six absolute joint positions in radians plus the absolute binary gripper command (0 closed, 1 open).

Use a **fresh A100 40 GB or larger runtime**, with Python 3.12 or 3.13. Run cells top to bottom. The default **expert** profile freezes the vision-language backbone and trains the action expert/projections using bfloat16 and gradient checkpointing. An actual optimizer-step probe selects a batch that fits; fitting is not guaranteed until this probe passes. Full fine-tuning can be selected in configuration but needs more memory.

The starting budget is **20,000 optimizer steps**, with checkpoints every **2,500**, following the previous Pi0 experiment. The notebook reports the resulting number of dataset passes and estimates training time after the batch probe. This is not matched to the SmolVLA notebook's eight-pass budget and does not constitute a controlled model comparison. You can edit the step budget before training.

The dataset is already in Drive under `MyDrive/VLA_Thesis/Sorted VLA Backup Final/pHYSICAL UR5/datasets/lerobot/ur5_cube_110ep_v3/`. The notebook accepts that folder or a tar/checksum pair. It preserves its quantile statistics, checks every state/action row, and decodes 330 paired camera samples. All episodes are training data; no held-out evaluation is enabled. The retained original four-transition episode is unchanged.

Before running: accept the [PaliGemma tokenizer license](https://huggingface.co/google/paligemma-3b-pt-224) and create a [Hugging Face read token](https://huggingface.co/settings/tokens). Add it to **Colab Secrets → HF_TOKEN** with notebook access, or enter it in the authentication cell's hidden prompt. An existing Hugging Face login is also accepted. W&B is optional. Keep sufficient Drive capacity for eight large checkpoints plus final exports (many tens of GB). Local staging requires at least 60 GiB free; only the latest two verified local checkpoints are retained. Drive retains all completed checkpoints for this run.

References: [LeRobot Pi0.5 training](https://huggingface.co/docs/lerobot/main/en/pi05), [base model](https://huggingface.co/lerobot/pi05_base), [pinned LeRobot 0.6.1](https://pypi.org/project/lerobot/0.6.1/), [TorchCodec compatibility](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions).
''')
put(1, source(1) + "\nassert (3, 12) <= sys.version_info[:2] <= (3, 13), 'This pinned stack expects Python 3.12 or 3.13.'\n")
put(2, source(2).replace("/content/drive/MyDrive/pHYSICAL UR5", "/content/drive/MyDrive/VLA_Thesis/Sorted VLA Backup Final/pHYSICAL UR5")
    .replace("TARGET_EPOCHS = 8.0", "TRAIN_STEPS = 20_000\nSAVE_FREQ = 2_500\nFINETUNE_MODE = 'expert'  # 'expert' or 'full'\nLOCAL_CHECKPOINTS_TO_KEEP = 2")
    .replace('ur5_smolvla_trained', 'ur5_pi05_trained')
    .replace("print('W&B enabled:', USE_WANDB, '| Target dataset passes:', TARGET_EPOCHS)", "assert FINETUNE_MODE in {'expert', 'full'}\nassert TRAIN_STEPS > 0 and SAVE_FREQ > 0\nprint('W&B:', USE_WANDB, '| Fine-tune mode:', FINETUNE_MODE, '| Steps:', TRAIN_STEPS)"))
put(3, '''## Install the Pi0.5 training stack

Use a fresh runtime, separate from the SmolVLA runtime. This installs **LeRobot 0.6.1 with both `pi` and `training` extras**, Torch 2.7.1 / Torchvision 0.22.1, compatible TorchCodec 0.3.0, Transformers 5.4.0, and PyAV 15.1.0. Decoding explicitly uses PyAV. Version checks and model imports run before copying data or downloading weights. The exact resolved environment is saved with the run.
''')
put(4, '''# Install before importing Torch or Transformers.
assert 'torch' not in sys.modules and 'transformers' not in sys.modules, 'Start a fresh runtime and run from the top.'
%pip install -q --upgrade pip
%pip install -q --index-url https://download.pytorch.org/whl/cu126 "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1"
%pip install -q "lerobot[pi,training]==0.6.1" "torch==2.7.1" "torchvision==0.22.1" "torchcodec==0.3.0" "transformers==5.4.0" "av==15.1.0" matplotlib psutil
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import importlib.metadata as md, torch, torchvision, av
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
assert torch.__version__.startswith('2.7.1') and torchvision.__version__.startswith('0.22.1')
assert md.version('lerobot') == '0.6.1' and md.version('transformers') == '5.4.0'
assert av.__version__ == '15.1.0' and hasattr(av, 'option')
assert torchvision.extension._has_ops()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
runtime_report.update({name: md.version(name) for name in ('torch', 'torchvision', 'lerobot', 'transformers', 'accelerate', 'av', 'torchcodec')})
runtime_report['cuda'] = torch.version.cuda
print(json.dumps(runtime_report, indent=2))
''')
put(5, source(5).replace('>= 35 * 1024**3', '>= 60 * 1024**3').replace('35 GiB', '60 GiB'))
put(6, source(6) + '''
# Pi0.5's tokenizer expects quantile-normalized state. Keep the reviewed stats.
stats = dataset.meta.stats
quantile_report = {}
for key in ('observation.state', 'action'):
    assert 'q01' in stats[key] and 'q99' in stats[key], f'Quantile stats missing: {key}'
    low, high = (np.asarray(stats[key][name]).reshape(-1) for name in ('q01', 'q99'))
    assert low.shape == high.shape == (7,)
    assert np.isfinite(low).all() and np.isfinite(high).all() and np.all(high >= low)
    # Histogram-estimated quantiles can differ slightly from exact binary endpoints.
    assert np.isclose(low[6], 0, atol=1e-5, rtol=0) and np.isclose(high[6], 1, atol=1e-5, rtol=0), 'Both gripper states must be represented in the quantiles'
    quantile_report[key] = {'q01': low.tolist(), 'q99': high.tolist()}
validation_report['normalization'] = 'QUANTILES for STATE/ACTION; IDENTITY for VISUAL'
validation_report['quantiles'] = quantile_report
print('Pi0.5 quantile statistics verified')
Path('/content/dataset_validation.json').write_text(json.dumps(validation_report, indent=2))
''')
put(7, '''# Authenticate without embedding tokens in notebook source or reports.
from google.colab import userdata
from getpass import getpass
from huggingface_hub import HfApi, get_token, login as hf_login, snapshot_download
from transformers import AutoTokenizer

def authenticate_hugging_face():
    try:
        token = userdata.get('HF_TOKEN')
    except Exception as error:
        print(f'Colab secret unavailable ({type(error).__name__}); checking existing login.')
        token = None
    token = (token or get_token() or '').strip()
    if not token:
        print('Create a read token at https://huggingface.co/settings/tokens')
        print('Accept tokenizer access at https://huggingface.co/google/paligemma-3b-pt-224 using the same account.')
        token = getpass('Hugging Face read token (hidden input): ').strip()
    if not token:
        raise ValueError('No token entered. Add HF_TOKEN in Colab Secrets or rerun this cell and use the hidden prompt.')
    hf_login(token=token, add_to_git_credential=False)
    # Downloads tokenizer files only. Access must work before the large model download.
    try:
        tokenizer = AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224', token=token)
    except OSError:
        raise RuntimeError('Tokenizer download failed. Confirm this token can read google/paligemma-3b-pt-224, accept its license with the same account, and check connectivity. Then rerun this cell.') from None
    assert tokenizer('test')['input_ids'], 'Tokenizer returned no tokens'
    print('Hugging Face login and PaliGemma tokenizer verified.')

authenticate_hugging_face()
BASE_MODEL = 'lerobot/pi05_base'
MODEL_REVISION = HfApi().model_info(BASE_MODEL).sha
assert MODEL_REVISION
print('Tokenizer access verified. Caching base weights at revision:', MODEL_REVISION)
# Resolve a fixed revision once, then reuse this exact local snapshot in every subprocess.
BASE_MODEL_PATH = snapshot_download(BASE_MODEL, revision=MODEL_REVISION, allow_patterns=['*.json', '*.safetensors'])
for name in ('config.json', 'model.safetensors', 'policy_preprocessor.json', 'policy_postprocessor.json'):
    assert (Path(BASE_MODEL_PATH) / name).is_file(), f'Missing base model artifact: {name}'
assert json.loads((Path(BASE_MODEL_PATH) / 'config.json').read_text())['type'] == 'pi05'
WANDB_PROJECT = 'pi05_ur5_cube_110ep_v3_a100'
if USE_WANDB:
    import wandb
    wandb.login()
print('W&B:', WANDB_PROJECT if USE_WANDB else 'disabled')
''')
put(8, '''## Probe actual optimizer steps before training

The probe uses the same Pi0.5 settings as training: bfloat16, gradient checkpointing, quantile normalization, and absolute actions. Two real camera streams are inferred from the dataset; one masked empty camera supplies the third slot used in pretraining. No camera rename or 32-dimensional target replacement is needed: the model pads seven-dimensional vectors internally.

Compilation stays disabled for both probe and training. Select the fastest successful candidate with at least 12% measured VRAM reserve. Non-memory errors stop the probe with its traceback. The 20,000-step budget is configurable in the Drive/configuration cell.
''')
policy_helpers = '''
SEED = 42
TRAIN_EXPERT_ONLY = FINETUNE_MODE == 'expert'
NORMALIZATION = {'STATE': 'QUANTILES', 'ACTION': 'QUANTILES', 'VISUAL': 'IDENTITY'}
def policy_arguments():
    return [
        '--policy.type=pi05', f'--policy.pretrained_path={BASE_MODEL_PATH}',
        '--policy.device=cuda', '--policy.dtype=bfloat16', '--policy.use_amp=false',
        '--policy.use_relative_actions=false', '--policy.gradient_checkpointing=true',
        '--policy.compile_model=false', '--policy.push_to_hub=false',
        '--policy.chunk_size=50', '--policy.n_action_steps=50', '--policy.empty_cameras=1',
        f'--policy.train_expert_only={str(TRAIN_EXPERT_ONLY).lower()}',
        f'--policy.freeze_vision_encoder={str(TRAIN_EXPERT_ONLY).lower()}',
        '--policy.normalization_mapping=' + json.dumps(NORMALIZATION, separators=(',', ':')),
    ]
def dataset_arguments():
    return [f'--dataset.repo_id={REPO_ID}', f'--dataset.root={LOCAL_DATASET}',
            '--dataset.video_backend=pyav', '--dataset.image_transforms.enable=false',
            '--dataset.eval_split=0.0', '--env_eval_freq=0', '--eval_steps=0']
'''
bench = source(9)
bench = bench.replace('BATCH_CANDIDATES = [16, 32, 48, 64, 96, 128]', "BATCH_CANDIDATES = [1, 2, 4, 8, 16, 24] if FINETUNE_MODE == 'expert' else [1, 2, 4]")
bench = bench.replace('BENCHMARK_STEPS = 12', 'BENCHMARK_STEPS = 6').replace('min(12,', 'min(8,')
start, end = bench.index('    command = ['), bench.index('    samples, lines, stop')
bench = bench[:start] + '''    command = [
        'lerobot-train', *policy_arguments(), *dataset_arguments(),
        f'--batch_size={batch_size}', f'--num_workers={NUM_WORKERS}', f'--seed={SEED}',
        '--wandb.enable=false', '--save_checkpoint=false',
        f'--steps={BENCHMARK_STEPS}', '--log_freq=1', f'--output_dir={run_dir}',
        f'--job_name=pi05_bench_b{batch_size}',
    ]
''' + bench[end:]
bench = bench.replace('    if not result["success"]: break', '''    if not result["success"]:
        error = result.get('error_tail', '').lower()
        if 'out of memory' in error:
            break
        raise RuntimeError('Pi0.5 probe failed for a non-memory reason: ' + result.get('error_tail', 'No log'))''')
put(9, policy_helpers + bench)
put(10, '''# Resolve the run budget and preserve reproducibility information.
import random
WARMUP_STEPS = min(1000, max(1, TRAIN_STEPS // 20))
APPROX_DATASET_PASSES = TRAIN_STEPS * BATCH_SIZE / dataset.meta.total_frames
ESTIMATED_TRAIN_HOURS = TRAIN_STEPS * BATCH_SIZE / selected['examples_per_second'] / 3600
RUN_NAME = f'{DATASET_NAME}_pi05_{FINETUNE_MODE}_a100_b{BATCH_SIZE}_{TRAIN_STEPS}_steps_{RUN_TAG}'
TRAIN_OUTPUT = Path('/content/outputs') / RUN_NAME
LOCAL_REPORT_DIR = Path('/content/training_reports') / RUN_NAME
DRIVE_RUN_DIR = CHECKPOINT_ROOT / RUN_NAME
DRIVE_CHECKPOINTS_DIR = DRIVE_RUN_DIR / 'checkpoints'
assert not TRAIN_OUTPUT.exists() and not DRIVE_RUN_DIR.exists(), 'Run already exists; choose a new run tag.'
assert shutil.disk_usage('/content').free >= 40 * 1024**3, 'Need 40 GiB local headroom after staging data and model.'
LOCAL_REPORT_DIR.mkdir(parents=True, exist_ok=False)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
TRAIN_ARGS = [
    'lerobot-train', *policy_arguments(), *dataset_arguments(),
    f'--policy.scheduler_warmup_steps={WARMUP_STEPS}', f'--policy.scheduler_decay_steps={TRAIN_STEPS}',
    f'--batch_size={BATCH_SIZE}', f'--num_workers={NUM_WORKERS}', f'--seed={SEED}',
    f'--wandb.enable={str(USE_WANDB).lower()}', f'--wandb.project={WANDB_PROJECT}',
    '--save_checkpoint=true', f'--steps={TRAIN_STEPS}', f'--save_freq={SAVE_FREQ}',
    '--log_freq=10', f'--output_dir={TRAIN_OUTPUT}', f'--job_name={RUN_NAME}',
]
resolved = {'policy': 'pi05', 'base_model': BASE_MODEL, 'model_revision': MODEL_REVISION,
    'dataset': DATASET_NAME, 'episodes': dataset.meta.total_episodes, 'frames': dataset.meta.total_frames,
    'finetune_mode': FINETUNE_MODE, 'batch_size': BATCH_SIZE, 'steps': TRAIN_STEPS, 'save_freq': SAVE_FREQ,
    'approx_dataset_passes': APPROX_DATASET_PASSES, 'estimated_compute_hours': ESTIMATED_TRAIN_HOURS,
    'action_mode': 'absolute', 'normalization': NORMALIZATION, 'seed': SEED, 'runtime': runtime_report,
    'validation': validation_report, 'benchmark_results': benchmark_results, 'command': TRAIN_ARGS,
    'drive_checkpoint_retention': 'all', 'local_checkpoint_retention': LOCAL_CHECKPOINTS_TO_KEEP}
(LOCAL_REPORT_DIR / 'resolved_training_config.json').write_text(json.dumps(resolved, indent=2))
(LOCAL_REPORT_DIR / 'dataset_file_hashes.json').write_text(DATASET_HASH_REPORT.read_text())
(LOCAL_REPORT_DIR / 'dataset_stats.json').write_text((LOCAL_DATASET / 'meta/stats.json').read_text())
(LOCAL_REPORT_DIR / 'pip_freeze.txt').write_text(output([sys.executable, '-m', 'pip', 'freeze']))
shutil.copytree(BENCHMARK_ROOT, LOCAL_REPORT_DIR / 'batch_benchmarks')
print(json.dumps(resolved, indent=2))
print(f'Estimated training compute: {ESTIMATED_TRAIN_HOURS:.1f} h; checkpoint copies add time. Colab sessions can end before completion.')
''')
put(11, '''## Train, save completed checkpoints, and export

The next cell starts training. Each completed checkpoint is copied to Drive and SHA-256 verified before older local copies are pruned, retaining the latest two locally. All Drive checkpoints remain available. Checkpoints include optimizer state; the final model includes preprocessing/normalization state. The final tar and checksum are produced automatically.

If the runtime disconnects, completed Drive checkpoints remain, but this notebook starts fresh runs and does not automatically resume. Record the run name before reconnecting. The printed time estimate excludes startup and Drive copies. No robot motion is started by this notebook.
''')
train = source(12)
train = train.replace("('resolved_training_config.json', 'dataset_file_hashes.json', 'pip_freeze.txt')", "('resolved_training_config.json', 'dataset_file_hashes.json', 'dataset_stats.json', 'pip_freeze.txt')")
train = train.replace("synced_steps = set()", '''def prune_verified_local_checkpoints(checkpoint_dir):
    completed = sorted((p for p in checkpoint_dir.iterdir() if p.name.isdigit() and p.name in synced_steps), key=lambda p: int(p.name))
    for old in completed[:-LOCAL_CHECKPOINTS_TO_KEEP]:
        # Verify the persistent copy again before deleting this run's local copy.
        assert tree_hashes(old) == tree_hashes(DRIVE_CHECKPOINTS_DIR / old.name), f'Drive copy changed: {old.name}'
        shutil.rmtree(old)

synced_steps = set()''')
train = train.replace("        print('SYNCED COMPLETE CHECKPOINT TO DRIVE:', target, flush=True)", "        print('SYNCED COMPLETE CHECKPOINT TO DRIVE:', target, flush=True)\n    prune_verified_local_checkpoints(checkpoint_dir)")
put(12, train)
put(15, '''## Finished

The completed Pi0.5 model and numbered checkpoints are under `checkpoints/ur5_pi05_trained/<run-name>/`. A model tar and checksum are under `deployment/downloaded_checkpoints/`. Reports record the model revision, package versions, source dataset hashes, quantile statistics, mode, batch size, training steps, and approximate dataset passes.

This is a Pi0.5 **expert fine-tune** by default, not full-model fine-tuning. All 110 episodes were used for training; loss alone is not a success-rate measurement. Use the saved Pi0.5 pre/post-processors for offline evaluation, including its quantile normalization and discretized state prompt. The notebook contains no physical robot commands.
''')
notebook['metadata']['colab']['name'] = OUTPUT.name
for i, cell in enumerate(notebook['cells']):
    cell['id'] = f'ur5-pi05-110-{i:02d}'
    if cell['cell_type'] == 'code':
        cell.update(outputs=[], execution_count=None)
OUTPUT.write_text(json.dumps(notebook, indent=2) + '\n')
print(OUTPUT)
