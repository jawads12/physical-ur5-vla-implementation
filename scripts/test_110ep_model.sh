#!/usr/bin/env bash
# Tests the final 18,000-step SmolVLA checkpoint. No motion commands.
set -eo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo="$project_root/ros2_ws/src/ur5-vla-dataset-recorder"
checkpoint="$project_root/data/models/checkpoints/ur5_cube_110ep_v3_smolvla_a100_b128_18000_steps_20261003_180950/pretrained_model"
dataset="$project_root/data/dataset/lerobot/ur5_cube_110ep_v3"
reports="$project_root/data/models/evaluation/smolvla_110ep_18000"
conda_exe=/home/grasplab/miniconda3/bin/conda
mode=${1:-help}
cd "$repo"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$project_root/data/huggingface_cache"
export HF_LEROBOT_HOME="$project_root/data/hf_lerobot_cache"
export PYTHONPATH="$repo"

case "$mode" in
  offline|server)
    if ! nvidia-smi >/dev/null 2>&1; then
      echo 'GPU unavailable. Resolve the NVIDIA driver/library mismatch before GPU testing.' >&2
      echo 'A restart is pending; stop robot operation safely before restarting the workstation.' >&2
      exit 1
    fi
    ;;
esac

case "$mode" in
  offline)
    mkdir -p "$reports"
    stamp=$(date +%Y%m%d_%H%M%S)
    exec "$conda_exe" run --no-capture-output -n pi0 python -u \
      -m ur5_vla_dataset_recorder.offline_inference \
      --checkpoint "$checkpoint" --dataset-root "$dataset" \
      --repo-id local/ur5_cube_110ep_v3 \
      --output-csv "$reports/offline_gpu_${stamp}.csv" \
      --samples-per-episode 3 --inference-seed 42 --device cuda \
      --max-target-delta-rad 0.012566
    ;;
  server)
    exec "$conda_exe" run --no-capture-output -n pi0 python -u \
      -m ur5_vla_dataset_recorder.policy_inference_server \
      --checkpoint "$checkpoint" --host 127.0.0.1 --port 8765 \
      --device cuda --inference-seed 42
    ;;
  shadow)
    source /opt/ros/humble/setup.bash
    source "$project_root/hardware_ws/install/setup.bash"
    source "$project_root/ros2_ws/install/setup.bash"
    exec /usr/bin/python3 /opt/ros/humble/bin/ros2 run ur5_vla_dataset_recorder policy_executor \
      --checkpoint "$checkpoint" --dataset-root "$dataset" \
      --repo-id local/ur5_cube_110ep_v3 \
      --task 'Pick up the orange cube and place it in the bowl' \
      --server-host 127.0.0.1 --server-port 8765 --inference-period 1.0 \
      --max-target-delta 0.012566 --max-prediction-jitter 0.005 --stable-predictions 100
    ;;
  help|--help|-h)
    echo "Usage: bash $project_root/scripts/test_110ep_model.sh offline|server|shadow"
    echo 'offline: check 330 saved observations across all 110 episodes on GPU.'
    echo 'server: start the model server in terminal 1.'
    echo 'shadow: read live robot observations in terminal 2, without sending motion.'
    ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
