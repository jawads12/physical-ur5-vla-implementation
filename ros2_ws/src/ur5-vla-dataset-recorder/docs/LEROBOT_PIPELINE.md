# LeRobot v3 and SmolVLA Pipeline

The raw staging tree is immutable input. Dataset/model commands run in a separate environment from ROS 2 Humble. Ten demonstrations are enough for a pipeline smoke test, not a reliable autonomous policy.

## Architecture

```text
immutable raw episodes ──validate──> local LeRobot v3 (RGB + state/action + task)
                                            │
                                            ├── converted validator
                                            ├── SmolVLA smoke training
                                            └── offline inference checker
                                                        │
real sensors ──ROS Python 3.10──> shadow executor <──── checkpoint process
                                      │
                                      └── no publishers (development default)

future live request ──deadman + latched safety supervisor──> reviewed control adapter
                                                        └── MoveIt Servo architecture
```

The policy must never publish directly to the UR motor controller. The checked-in executor hard-locks live mode until the site-specific adapter, controller/readiness checks, workspace checks, stop contract, and manual-enable UI are reviewed.

## Separate training environment

Do not run ROS nodes from Conda. ROS Humble uses Ubuntu Python 3.10.

```bash
conda create -n ur5-lerobot python=3.11 -y
conda activate ur5-lerobot
python -m pip install --upgrade pip
python -m pip install "lerobot[smolvla]==0.4.3" pytest
```

The inspected workstation currently has LeRobot 0.4.3 (dataset codebase v3.0), PyTorch 2.7.1+cu126, and four RTX 2080 Ti GPUs with 11,264 MiB each. Pin versions in a deployment lock file after the smoke test succeeds.

## Convert and validate

```bash
cd /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/ros2_ws/src/ur5-vla-dataset-recorder
conda activate ur5-lerobot

python -m ur5_vla_dataset_recorder.convert_lerobot \
  --input-root /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/staging/episodes \
  --output-root /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/lerobot/ur5_orange_cube_v1 \
  --repo-id local/ur5_orange_cube_v1 \
  --task "Pick up the orange cube and place it in the tray" \
  --depth none \
  --storage video

python -m ur5_vla_dataset_recorder.validate_lerobot \
  --root /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/lerobot/ur5_orange_cube_v1 \
  --repo-id local/ur5_orange_cube_v1 \
  --expected-episodes 10 \
  --expected-frames 20583 \
  --task "Pick up the orange cube and place it in the tray" \
  --report /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/lerobot/ur5_orange_cube_v1/validation_report.md
```

Existing output is refused. Use `--overwrite` only after reviewing the exact output path. `--episodes episode_name ...` selects a subset. Default `--depth none` is intentionally RGB-only for SmolVLA. `--depth meters` stores metric float depth and validity masks; raw values 0 and 65535 are masked to zero. Never present raw uint16 depth as RGB.

Inspect metadata and one sample:

```bash
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/ur5_orange_cube_v1", root="/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/lerobot/ur5_orange_cube_v1")
print(ds.meta.features)
print(ds.meta.total_episodes, ds.meta.total_frames, ds.meta.tasks.index.tolist())
sample = ds[0]
print({key: getattr(value, "shape", value) for key, value in sample.items()})
PY
```

## Optional private upload

Do this only after explicit approval and authentication. Never upload by default.

```bash
huggingface-cli login
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("YOUR_ORG/ur5_orange_cube_v1", root="/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/lerobot/ur5_orange_cube_v1")
ds.push_to_hub(private=True)
PY
```

## SmolVLA smoke test

The installed 0.4.3 CLI uses `--policy.pretrained_path`, not `--policy.path`.

```bash
CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/ur5_orange_cube_v1 \
  --dataset.root=$HOME/ur5_vla_dataset/lerobot/ur5_orange_cube_v1 \
  --batch_size=4 \
  --steps=5000 \
  --output_dir=outputs/ur5_smolvla_test \
  --job_name=ur5_smolvla_test \
  --wandb.enable=false
```

Checkpoints are written below `outputs/ur5_smolvla_test/checkpoints/`; inspect the training log for the final numbered checkpoint. If 11 GB VRAM is insufficient, reduce batch size to 1 and enable the version-supported memory options. Do not interpret smoke-test loss as task reliability.

## Offline inference only

```bash
python -m ur5_vla_dataset_recorder.offline_inference \
  --dataset-root /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/lerobot/ur5_orange_cube_v1 \
  --repo-id local/ur5_orange_cube_v1 \
  --checkpoint outputs/ur5_smolvla_test/checkpoints/LATEST/pretrained_model \
  --output-csv outputs/ur5_smolvla_test/offline_actions.csv \
  --max-frames 100 --device cuda --video-backend pyav
```

This checks `(batch, horizon, 7)` output, finite values, predicted ranges, binary-target behavior, and sampled consecutive jumps. It has no ROS imports or publishers.

On the inspected machine, TorchCodec is installed but cannot load the available FFmpeg shared libraries. The commands therefore select PyAV explicitly. Repair and compatibility-test TorchCodec before choosing `--video-backend torchcodec`.

## Shadow mode

Run only from the ROS Python 3.10 environment. Model inference should be isolated behind a reviewed bridge because the installed model environment uses a different Python runtime.

```bash
conda deactivate
source /opt/ros/humble/setup.bash
source /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/ros2_ws/install/setup.bash
ros2 run ur5_vla_dataset_recorder policy_executor \
  --checkpoint /absolute/checkpoint/path \
  --dataset-root /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/lerobot/ur5_orange_cube_v1 \
  --repo-id local/ur5_orange_cube_v1
```

Shadow mode creates no command publishers. `--live` is currently hard-locked and exits. Disabling or any fault clears queued actions; faults latch and cannot auto-resume.
