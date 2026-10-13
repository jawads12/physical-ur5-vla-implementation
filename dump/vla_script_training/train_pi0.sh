#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# train_pi0.sh — Full Pi0 Training Pipeline for UR5 Blue Cube
# ═══════════════════════════════════════════════════════════════
#
# Steps:
#   1. Clone lerobot repo (if needed)
#   2. Install lerobot[pi] + wandb
#   3. Convert dataset_raw_cube → local/ur5
#   4. Launch Pi0 fine-tuning with wandb logging
#
# Usage:
#   chmod +x train_pi0.sh
#   ./train_pi0.sh
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export HF_HOME="${PROJECT_ROOT}/data/huggingface_cache"
export HF_LEROBOT_HOME="${PROJECT_ROOT}/data/hf_lerobot_cache"
LEROBOT_DIR="${SCRIPT_DIR}/lerobot"

echo ""
echo "══════════════════════════════════════════════════"
echo "  Pi0 Training Pipeline — UR5 Blue Cube"
echo "══════════════════════════════════════════════════"
echo ""

# ── Step 1: Clone LeRobot ─────────────────────────────────────
if [ ! -d "${LEROBOT_DIR}" ]; then
    echo "📦 Cloning LeRobot repository…"
    git clone https://github.com/huggingface/lerobot.git "${LEROBOT_DIR}"
else
    echo "✔ LeRobot repo already exists at ${LEROBOT_DIR}"
fi

# ── Step 2: Install dependencies ─────────────────────────────
echo ""
echo "📦 Installing lerobot[pi,dataset] + wandb…"
cd "${LEROBOT_DIR}"
pip install -e ".[pi,dataset]"
pip install wandb pandas

# ── Step 3: Convert dataset ──────────────────────────────────
echo ""
echo "🔄 Converting dataset_raw_cube → local/ur5…"
cd "${SCRIPT_DIR}"
python convert_data.py

# ── Step 4: Train Pi0 ────────────────────────────────────────
echo ""
echo "🚀 Starting Pi0 training…"
echo "   View live metrics at your WandB dashboard URL (printed below)"
echo ""
cd "${LEROBOT_DIR}"

# Detect number of GPUs
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "   Detected ${NUM_GPUS} GPU(s)"

if [ "${NUM_GPUS}" -gt 1 ]; then
    # Multi-GPU: use torchrun for distributed training
    LAUNCH_CMD="torchrun --nproc_per_node=${NUM_GPUS} -m lerobot.scripts.lerobot_train"
else
    LAUNCH_CMD="python -m lerobot.scripts.lerobot_train"
fi

${LAUNCH_CMD} \
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
  --policy.dtype float16 \
  --policy.pretrained_path lerobot/pi0_base \
  --policy.compile_model true \
  --output_dir "${SCRIPT_DIR}/lerobot_output_final"

echo ""
echo "══════════════════════════════════════════════════"
echo "  ✅ Training complete!"
echo "  📁 Checkpoints: ${SCRIPT_DIR}/lerobot_output_final"
echo "══════════════════════════════════════════════════"
