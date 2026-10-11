# Guarded one-shot physical policy execution

This mode exists to inspect a validated checkpoint using a deliberately tiny physical movement. It is not continuous autonomous operation.

## Hard limits

- Arm only; learned gripper output is not executed.
- Exactly one `JointJog` message after each manual arm request.
- Pulse duration at most `0.10 s`.
- Joint speed at most `0.02 rad/s`, therefore the commanded pulse is at most approximately `0.002 rad` per joint.
- Predicted absolute target must be within `0.012566 rad` of current state.
- Prediction jitter must be at most `0.005 rad` for 100 consecutive predictions.
- MoveIt must report the predicted target collision-free.
- The Logitech teleoperation mapper must be stopped so it cannot send competing Servo commands. Keep `joy_node` running for the physical deadman.
- UR speed slider must be at or below 20%, External Control must be running, safety mode must be NORMAL or REDUCED, Servo status must be zero, and `forward_position_controller` must be active.
- Releasing joystick button 0 during the pulse immediately publishes a zero `JointJog` and latches a fault.

Never widen a limit to force a checkpoint to pass. An E-stop remains the final hardware protection, not the normal stop mechanism.

## Start the model server

```bash
cd /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/ros2_ws/src/ur5-vla-dataset-recorder

conda run --no-capture-output -n pi0 \
  python -m ur5_vla_dataset_recorder.policy_inference_server \
  --checkpoint CHECKPOINT/pretrained_model \
  --host 127.0.0.1 --port 8765 --device cuda
```

## Start the guarded executor

Use a system ROS terminal. Do not run it under Conda.

```bash
source /opt/ros/humble/setup.bash
source /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/ros2_ws/install/setup.bash

ros2 run ur5_vla_dataset_recorder policy_live_executor \
  --checkpoint CHECKPOINT/pretrained_model \
  --dataset-root DATASET_ROOT \
  --repo-id local/ur5_cube_clean_20ep \
  --task "Pick up the orange cube and place it in the bowl" \
  --risk-ack I_UNDERSTAND_ONE_ACTION_MOVES_UR5
```

The node begins disarmed. It may calculate predictions, but it cannot publish a motion until the arm service succeeds.

## Inspect readiness

```bash
ros2 service call /vla_live/status std_srvs/srv/Trigger "{}"
```

Continue only when the response says `READY TO ARM`. Clear the workspace, set the speed slider to 20% or less, keep the E-stop reachable, and have a second person observe the arm.

## Execute one tiny pulse

Hold joystick button 0 continuously, then from another terminal call:

```bash
ros2 service call /vla_live/arm_one_action std_srvs/srv/Trigger "{}"
```

The service performs one collision check and consumes the arm request whether the check succeeds or fails. A second movement always requires another explicit arm call. Release the deadman immediately if the direction is unexpected.

## Fault recovery

Inspect the fault first. Then, while remaining disarmed:

```bash
ros2 service call /vla_live/clear_fault std_srvs/srv/Trigger "{}"
```

Clearing a fault does not arm the executor and does not publish motion.
