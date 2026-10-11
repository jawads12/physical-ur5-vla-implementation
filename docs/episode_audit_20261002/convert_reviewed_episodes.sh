#!/usr/bin/env bash
set -euo pipefail

UR5_AUDIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UR5_PROJECT_ROOT=$(cd "$UR5_AUDIT_DIR/../.." && pwd)
UR5_REPO="$UR5_PROJECT_ROOT/ros2_ws/src/ur5-vla-dataset-recorder"
UR5_RAW_ROOT="$UR5_PROJECT_ROOT/data/dataset/staging/episodes"

case "${1:---help}" in
  new)
    UR5_SELECTION="$UR5_AUDIT_DIR/accepted_new_episodes.txt"
    UR5_DATASET_NAME=ur5_cube_oct01_37ep
    UR5_EXPECTED_EPISODES=37
    UR5_EXPECTED_FRAMES=88497
    ;;
  combined)
    UR5_SELECTION="$UR5_AUDIT_DIR/combined_episodes.txt"
    UR5_DATASET_NAME=ur5_cube_110ep_v3
    UR5_EXPECTED_EPISODES=110
    UR5_EXPECTED_FRAMES=285204
    ;;
  --help|-h)
    echo 'Usage: bash convert_reviewed_episodes.sh new|combined'
    echo 'new: 37 accepted October 1 episodes, 88,497 frames.'
    echo 'combined: original 73 plus accepted 37, 110 episodes / 285,204 frames.'
    echo 'combined preserves the historical four-transition episode; see REPORT.md.'
    exit 0
    ;;
  *) echo 'Expected new or combined; use --help.' >&2; exit 2 ;;
esac

# These lists are written only after every new episode finishes full validation.
mapfile -t UR5_EPISODES < "$UR5_SELECTION"
if (( ${#UR5_EPISODES[@]} != UR5_EXPECTED_EPISODES )); then
  echo 'Selection count does not match the reviewed manifest; refusing conversion.' >&2
  exit 1
fi
for UR5_EPISODE in "${UR5_EPISODES[@]}"; do
  [[ "$UR5_EPISODE" =~ ^episode_[0-9]{8}_[0-9]{6}$ ]] || { echo 'Invalid episode name.' >&2; exit 1; }
done
UR5_OUTPUT_ROOT="$UR5_PROJECT_ROOT/data/dataset/lerobot/$UR5_DATASET_NAME"
if [[ -e "$UR5_OUTPUT_ROOT" ]]; then
  echo "Output already exists; refusing overwrite: $UR5_OUTPUT_ROOT" >&2
  exit 1
fi
cd "$UR5_REPO"
echo "Converting $UR5_EXPECTED_EPISODES episodes to $UR5_OUTPUT_ROOT"
/home/grasplab/miniconda3/bin/conda run --no-capture-output -n pi0 \
  python -m ur5_vla_dataset_recorder.convert_lerobot \
  --input-root "$UR5_RAW_ROOT" \
  --output-root "$UR5_OUTPUT_ROOT" \
  --repo-id "local/$UR5_DATASET_NAME" \
  --task 'Pick up the orange cube and place it in the bowl' \
  --episodes "${UR5_EPISODES[@]}" \
  --depth none --storage video --action-mode absolute

/home/grasplab/miniconda3/bin/conda run --no-capture-output -n pi0 \
  python -m ur5_vla_dataset_recorder.validate_lerobot \
  --root "$UR5_OUTPUT_ROOT" \
  --repo-id "local/$UR5_DATASET_NAME" \
  --expected-episodes "$UR5_EXPECTED_EPISODES" \
  --expected-frames "$UR5_EXPECTED_FRAMES" \
  --task 'Pick up the orange cube and place it in the bowl' \
  --video-backend pyav \
  --report "$UR5_OUTPUT_ROOT/validation_report.md"
