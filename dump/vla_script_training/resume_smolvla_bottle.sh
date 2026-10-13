#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# resume_smolvla_bottle.sh — Resume SmolVLA Training on Bottle Dataset
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export HF_HOME="${PROJECT_ROOT}/data/huggingface_cache"
export HF_LEROBOT_HOME="${PROJECT_ROOT}/data/hf_lerobot_cache"
LEROBOT_DIR="${SCRIPT_DIR}/lerobot"
OUTPUT_DIR="${SCRIPT_DIR}/lerobot_output_smolvla_bottle"

echo ""
echo "══════════════════════════════════════════════════"
echo "  Resuming SmolVLA Training — Bottle Dataset"
echo "  Resuming from: ${OUTPUT_DIR}"
echo "  Target steps: 10000"
echo "══════════════════════════════════════════════════"
echo ""

cd "${LEROBOT_DIR}"

export LD_LIBRARY_PATH="/home/grasplab/miniconda3/envs/pi0/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="/home/grasplab/miniconda3/envs/pi0/bin:${PATH}"

# Detect number of GPUs
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "  Detected ${NUM_GPUS} GPU(s)"

if [ "${NUM_GPUS}" -gt 1 ]; then
    LAUNCH_CMD="torchrun --nproc_per_node=${NUM_GPUS} -m lerobot.scripts.lerobot_train"
else
    LAUNCH_CMD="python -m lerobot.scripts.lerobot_train"
fi

${LAUNCH_CMD} \
  --dataset.repo_id local/ur5 \
  --policy.type smolvla \
  --policy.repo_id local/smolvla-ur5-bottle \
  --policy.push_to_hub false \
  --policy.device cuda \
  --batch_size 16 \
  --steps 10000 \
  --save_freq 1000 \
  --eval.n_episodes 0 \
  --eval.batch_size 0 \
  --save_checkpoint true \
  --wandb.enable true \
  --policy.compile_model false \
  --output_dir "${OUTPUT_DIR}" \
  --resume true \
  --config_path="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"

echo ""
echo "══════════════════════════════════════════════════"
echo "  ✅ Training complete (10,000 steps target)!"
echo "══════════════════════════════════════════════════"
