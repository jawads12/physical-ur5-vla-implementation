# Pi0 Training Pipeline — UR5 Blue Cube

Complete training setup for fine-tuning **Pi0** (from HuggingFace LeRobot) on your **dataset_raw_cube** UR5 data.

## Folder Structure

```
VLA_script_training/
├── README.md           ← this file
├── convert_data.py     ← converts dataset_raw_cube → LeRobot format (local/ur5)
├── train_pi0.sh        ← full automated pipeline (clone → install → convert → train)
```

## Quick Start (Automated)

```bash
cd ~/Desktop/VLA_script_training
chmod +x train_pi0.sh
./train_pi0.sh
```

## Step-by-Step (Manual)

### 1. Clone & Install LeRobot

```bash
cd ~/Desktop/VLA_script_training
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[pi]"
pip install wandb pandas
```

### 2. Convert Your Dataset

```bash
cd ~/Desktop/VLA_script_training
python convert_data.py
```

This reads `~/Desktop/dataset_raw_cube/episode_*` and creates a HuggingFace-compatible dataset at `local/ur5`.

### 3. Login to WandB (for graphical metrics)

```bash
wandb login
# paste your API key when prompted
```

### 4. Launch Training

```bash
cd ~/Desktop/VLA_script_training/lerobot

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id local/ur5 \
  --policy.type pi0 \
  --policy.repo_id local/pi0-ur5-blue-cube \
  --policy.push_to_hub false \
  --policy.device cuda \
  --batch_size 4 \
  --policy.gradient_checkpointing true \
  --steps 5000 \
  --save_freq 1000 \
  --eval.n_episodes 0 \
  --eval.batch_size 0 \
  --save_checkpoint true \
  --wandb.enable true \
  --policy.dtype bfloat16 \
  --policy.pretrained_path lerobot/pi0_base \
  --policy.compile_model true \
  --output_dir ../lerobot_output_final
```

## Key Training Parameters

| Parameter | Value | Purpose |
|---|---|---|
| `policy.type` | `pi0` | Pi0 vision-language-action model |
| `policy.pretrained_path` | `lerobot/pi0_base` | Pre-trained foundation weights |
| `policy.device` | `cuda` | Use NVIDIA GPU |
| `policy.dtype` | `bfloat16` | Save VRAM with mixed precision |
| `policy.gradient_checkpointing` | `true` | Further VRAM savings |
| `batch_size` | `4` | Adjust to 2 or 8 based on your VRAM |
| `steps` | `5000` | Total training steps |
| `save_freq` | `1000` | Checkpoint every 1000 steps |
| `wandb.enable` | `true` | Stream live loss/LR graphs to browser |

## Dataset Info

- **Source**: `~/Desktop/dataset_raw_cube` (74 episodes)
- **Image resolution**: 720 × 1280 RGB
- **State dims**: 5 joints (`shoulder_pan`, `shoulder_lift`, `elbow`, `wrist_1`, `wrist_3`)
- **Action dims**: 5 joints + 1 gripper command
- **Task**: "pick up the blue cube"

## Output

Checkpoints saved to `~/Desktop/VLA_script_training/lerobot_output_final/`
