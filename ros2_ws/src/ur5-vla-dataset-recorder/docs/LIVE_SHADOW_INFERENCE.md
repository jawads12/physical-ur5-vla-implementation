# Physical UR5 Live Shadow Inference

This procedure runs the trained SmolVLA checkpoint on current robot observations and logs predictions. It does not move the arm or actuate the gripper.

## Safety boundary

- The model server listens only on localhost.
- The model process contains no ROS imports.
- The ROS shadow process creates subscriptions but no publishers.
- `--live` is hard-locked and exits immediately.
- A prediction is never a controller command.
- Advancing to physical output requires a separate reviewed controller, workspace, External Control, manual-enable, and physical-deadman adapter.

## Validated artifacts

```text
Checkpoint:
/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/dump/downloaded_models/drive-download-20260823T063551Z-1-001/pretrained_model

Dataset:
/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/lerobot/ur5_orange_cube_v1

Repository ID:
local/ur5_orange_cube_v1

Task:
Pick up the orange cube and place it in the tray
```

The client refuses startup unless the checkpoint and dataset both contain two `480x640` cameras, a 7D state, a 7D action, canonical UR5 joint order, and absolute-joint/binary-gripper action semantics.

## Build after source changes

```bash
cd /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select ur5_vla_dataset_recorder --symlink-install
```

## Terminal 1 — model server

Run the model in the LeRobot 0.5.2 Conda environment:

```bash
cd /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/ros2_ws/src/ur5-vla-dataset-recorder

conda run --no-capture-output -n pi0 \
  python -m ur5_vla_dataset_recorder.policy_inference_server \
  --checkpoint /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/dump/downloaded_models/drive-download-20260823T063551Z-1-001/pretrained_model \
  --host 127.0.0.1 \
  --port 8765 \
  --device cuda
```

Wait for:

```text
SHADOW inference server loaded .../pretrained_model
No ROS imports or command publishers exist in this process
Listening on 127.0.0.1:8765
```

## Terminal 2 — ROS shadow client

Ensure the normal robot driver, both RealSense cameras, gripper state publisher, and Servo status publisher are already running. Then run:

First confirm that joint samples are actually arriving, rather than only checking that the topic name exists:

```bash
ros2 topic hz /joint_states
```

Stop the check with `Ctrl+C` after it prints a nonzero rate. If it prints no samples, do not substitute saved or fabricated joint state. Inspect/restart the normal UR driver session using the established operations procedure before starting shadow inference.

Then run:

```bash
source /opt/ros/humble/setup.bash
source /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/ros2_ws/install/setup.bash

ros2 run ur5_vla_dataset_recorder policy_executor \
  --checkpoint /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/dump/downloaded_models/drive-download-20260823T063551Z-1-001/pretrained_model \
  --dataset-root /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/lerobot/ur5_orange_cube_v1 \
  --repo-id local/ur5_orange_cube_v1 \
  --task "Pick up the orange cube and place it in the tray" \
  --server-host 127.0.0.1 \
  --server-port 8765 \
  --inference-period 1.0 \
  --max-target-delta 0.012566 \
  --max-prediction-jitter 0.005 \
  --stable-predictions 100
```

Expected output begins with:

```text
SHADOW INFERENCE ONLY: no motion or gripper publishers have been created
```

Each accepted observation later produces a line containing:

```text
SHADOW prediction only action=[...] raw_gripper=... max_abs_delta=... model_latency=... round_trip=...
```

The displayed gripper value is thresholded at `0.5` for inspection only. The unrounded model output remains visible as `raw_gripper`.

For the clean retraining run, do not advance unless 100 consecutive predictions become eligible with zero rejected target deltas. These limits match the demonstrated motion envelope; do not widen them to force a checkpoint to pass.

## Inputs and rejection rules

The client consumes:

- `/base/camera/color/image_raw`
- `/wrist/camera/color/image_raw`
- `/joint_states`
- `/gripper/state`
- `/servo_node/status`

Inference is skipped when an input is missing or stale, Servo status is nonzero, the gripper estimate is not binary, joint names cannot be reordered canonically, the model server is unavailable, or a prediction is malformed/non-finite.

## Stop

Press `Ctrl+C` in Terminal 2, then Terminal 1. Stopping either process cannot send a robot command because neither process owns a command publisher.

## Explicitly unsupported

Do not add `--live`. It exits by design. This shadow bridge does not establish controller readiness, External Control readiness, workspace safety, a physical deadman, a zero-motion stop contract, or permission to execute even one predicted action.
