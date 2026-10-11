# Dataset recorder operations

This is the short operational sequence. The repository `README.md` contains
installation, camera, service, schema, and troubleshooting details.

## 1. Prepare every terminal

```bash
conda deactivate
source /opt/ros/humble/setup.bash
source /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/ros2_ws/install/setup.bash
```

ROS 2 Humble uses the system Python 3.10. Running under a Conda Python 3.13
environment caused the `_rclpy_pybind11` import failure.

## 2. Start the robot stack

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5 \
  robot_ip:=10.0.1.38 \
  launch_rviz:=false
```

Start the External Control program on the teach pendant, then launch MoveIt:

```bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur5 \
  launch_rviz:=true
```

Switch to the Servo controller and start Servo:

```bash
ros2 control switch_controllers \
  --deactivate scaled_joint_trajectory_controller \
  --activate forward_position_controller

ros2 service call /servo_node/start_servo std_srvs/srv/Trigger "{}"
```

Require Servo status 0 before recording:

```bash
ros2 topic echo /servo_node/status --once
```

## 3. Start the cameras

Base D435i:

```bash
ros2 launch realsense2_camera rs_launch.py \
  serial_no:=_044322072365 \
  camera_namespace:=base \
  camera_name:=camera \
  initial_reset:=true \
  enable_color:=true \
  enable_depth:=true \
  align_depth.enable:=true \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30
```

Wrist D435:

```bash
ros2 launch realsense2_camera rs_launch.py \
  serial_no:=_827312073590 \
  camera_namespace:=wrist \
  camera_name:=camera \
  initial_reset:=true \
  enable_color:=true \
  enable_depth:=true \
  align_depth.enable:=true \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30
```

Verify all required image streams:

```bash
ros2 topic hz /base/camera/color/image_raw
ros2 topic hz /wrist/camera/color/image_raw
ros2 topic hz /base/camera/aligned_depth_to_color/image_raw
ros2 topic hz /wrist/camera/aligned_depth_to_color/image_raw
```

## 4. Start joystick input and the mapper

```bash
ros2 run joy joy_enumerate_devices

ros2 run joy joy_node --ros-args \
  -p device_id:=YOUR_LOGITECH_DEVICE_ID \
  -p deadzone:=0.1 \
  -p autorepeat_rate:=50.0
```

In another terminal, use the mapper installed by this repository:

```bash
ros2 run ur5_vla_dataset_recorder logitech_servo
```

Open or close the gripper once, then verify its normalized dataset streams:

```bash
ros2 topic echo /gripper/command --once
ros2 topic echo /gripper/state --once
```

The convention is `0.0 = closed`, `1.0 = open`. State is the last successful
command estimate because the pneumatic gripper has no position sensor.

## 5. Start the recorder

Before each new-dataset episode, use one command to return to the fixed,
guarded standard pose and then reactivate MoveIt Servo:

```bash
ros2 run ur5_vla_dataset_recorder prepare_dataset_start
```

This command physically moves the UR5. It requires External Control, normal or
reduced safety mode, no Servo command publishers, and a MoveIt-valid
interpolated path. If reported speed scaling is above 20%, it first requests a
20% slider setting and waits for fresh feedback at or below 20% before planning
or moving. An already-lower setting is preserved; a zero-speed stop is never
raised automatically. The slider stays reduced after reset. Service failure or
missing confirmation blocks movement. The standalone `move_to_standard_pose`
command still requires the speed to be set manually. After reaching the target, it
activates `forward_position_controller` and starts Servo. Keep the E-stop
reachable and confirm that the real swept path is clear; the current MoveIt
scene does not model the table, bowl, cameras, or people. Start the joystick
mapper only after this command reports `DATASET START READY`.

When the Logitech mapper is already running, the same workflow is available
directly from the joystick: release the deadman and hold physical button 7 for
two seconds. The mapper publishes zero motion, removes its Twist publisher,
runs the guarded reset, and restores teleoperation only after the reset command
finishes. Physical buttons 9, 10, and 11 remain reserved for episode recording.

```bash
ros2 launch ur5_vla_dataset_recorder recorder.launch.py
```

Set the task and check readiness:

```bash
ros2 param set /dataset_recorder task \
  "Pick up the red cube and place it in the tray"

ros2 service call /dataset_recorder/status std_srvs/srv/Trigger "{}"
```

Use physical button 9 to start. Use button 11 to save a successful episode or
button 10 to abort it.

## 6. Inspect recordings

```bash
ls -lah /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/staging/episodes/
du -sh /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset
find /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/staging/episodes -maxdepth 2 -type f | sort | head -100
```

Each finalized episode contains:

```text
metadata.yaml
frames.jsonl
images/base/*.jpg
images/wrist/*.jpg
depth/base/*.png
depth/wrist/*.png
```

## 7. Return to Plan and Execute

Stop the joystick mapper and Servo first:

```bash
ros2 service call /servo_node/stop_servo std_srvs/srv/Trigger "{}"

ros2 control switch_controllers \
  --deactivate forward_position_controller \
  --activate scaled_joint_trajectory_controller
```

Never replay recorded commands or a rosbag while a physical command controller
is active unless a dedicated, validated replay tool explicitly owns that motion.
