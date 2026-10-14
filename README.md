# Physical UR5 VLA implementation

Central local repository for physical UR5 vision-language-action recording, conversion, training, and evaluation at GRASP Lab, Ontario Tech University. **This is the authoritative README for current paths and operation.** Other Markdown files are package details, experiment reports, or historical records.

## Contents

- [Workspaces and hardware](#workspaces-and-hardware)
- [Location and environment](#location-and-environment)
- [Build and source](#build-and-source)
- [UR5 physical arm installation and teleoperation](#ur5-physical-arm-installation-and-teleoperation)
  - [Daily run: teleoperation and dataset recording, one step at a time](#daily-run-teleoperation-and-dataset-recording-one-step-at-a-time)
- [Directory map](#directory-map)
- [Current data and model](#current-data-and-model)
- [Dataset conversion and validation](#dataset-conversion-and-validation)
- [ROS package files](#ros-package-files)
- [Standalone scripts](#standalone-scripts)
- [Typical read-only checks](#typical-read-only-checks)
- [Historical material and known limits](#historical-material-and-known-limits)
- [Verification and Git](#verification-and-git)
  - [ROS executable smoke test](#ros-executable-smoke-test-2026-10-11)

## Workspaces and hardware

This repository contains two ROS 2 workspaces with different roles:

| Workspace | Purpose | When it is used |
| --- | --- | --- |
| `hardware_ws` | Robotiq gripper driver, controller plugins, serial transport, robot description, hardware tests, and UR5 Isaac Sim support | Source before `ros2_ws`; use its Robotiq launch files only when controlling compatible serial-connected Robotiq hardware or using the simulation assets |
| `ros2_ws` | Our UR5 VLA experiment package: dataset recorder, Logitech teleoperation, guarded standard pose, conversion, validation, and policy tools | Use for the current physical recording and evaluation workflow |

`hardware_ws/src/ros2_robotiq_gripper/` contains `robotiq_driver` (a
`ros2_control` hardware-interface plugin), `robotiq_controllers` (gripper
controller plugins), `robotiq_description` (URDF, xacro, controller YAML,
and launch files), and `robotiq_hardware_tests` (driver tests).
`hardware_ws/src/serial/` is the serial communication library used by the
Robotiq driver. `hardware_ws/src/ur5_isaac_simulation/` contains the
Isaac Sim bridge, action servers, descriptions, scenes, and launch assets.
These packages provide components and launch files; most do not have a
standalone `ros2 run` executable.

The current Logitech gripper buttons in `ros2_ws` call the UR controller's
`/io_and_status_controller/set_io` service. That is a separate path from the
serial Robotiq driver in `hardware_ws`; having the Robotiq packages installed
does not mean that driver is currently controlling the physical gripper.
Before starting a Robotiq controller, identify the connected gripper and
serial device and check for an existing controller. The relevant launch file
is `robotiq_description/launch/robotiq_control.launch.py` (its default port is
`/dev/ttyUSB0`); `hand-e_control.launch.xml` is for the Hand-E variant.

To verify which repository supplies these packages in the current terminal:

```bash
ros2 pkg prefix robotiq_driver
ros2 pkg prefix robotiq_controllers
ros2 pkg prefix robotiq_description
ros2 pkg prefix ur5_vla_dataset_recorder
```

The first three paths should be inside this repository's
`hardware_ws/install/`; the last should be inside `ros2_ws/install/`.

## Location and environment

The repository is mounted at:

```text
/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation
```

The user's `.bashrc` sets `UR5_VLA_ROOT` to this directory and sources ROS 2 Humble, `hardware_ws`, then `ros2_ws`. **Open a new terminal after this migration**; already-open terminals retain their previous ROS environment. Verify a fresh terminal before running a node:

```bash
echo "$UR5_VLA_ROOT"
ros2 pkg prefix ur5_vla_dataset_recorder
ros2 pkg prefix robotiq_driver
/usr/bin/python3 -c 'import ur5_vla_dataset_recorder as p; print(p.__file__)'
```

The first prefix and Python file must be under this repository's `ros2_ws`; `robotiq_driver` must be under its `hardware_ws`. To source manually:

```bash
export UR5_VLA_ROOT=/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation
source /opt/ros/humble/setup.bash
source "$UR5_VLA_ROOT/hardware_ws/install/setup.bash"
source "$UR5_VLA_ROOT/ros2_ws/install/setup.bash"
```

ROS 2 Humble, the UR robot driver, camera drivers, system Python 3.10, and the `pi0` Conda environment are system dependencies. The `pi0` environment's editable LeRobot 0.5.2 installation now points to `dump/vla_script_training/lerobot` **inside this repository**. Conda itself remains at `/home/grasplab/miniconda3/bin/conda`. The external drive must remain mounted while using this setup. Fresh terminals also set `HF_HOME` to `data/huggingface_cache` and `HF_LEROBOT_HOME` to `data/hf_lerobot_cache`; authentication tokens were not copied.

Use `/usr/bin/python3` for ROS-facing standalone scripts; the Conda base
`python3` may be a different Python version from ROS Humble's bindings.

## Build and source

`hardware_ws` contains the copied Robotiq/Isaac source; `ros2_ws` contains the dataset recorder and policy package. Build in this order:

```bash
source /opt/ros/humble/setup.bash
cd "$UR5_VLA_ROOT/hardware_ws"
colcon build --symlink-install
source install/setup.bash
cd "$UR5_VLA_ROOT/ros2_ws"
colcon build --symlink-install --packages-select ur5_vla_dataset_recorder
source install/setup.bash
```

Six hardware workspace packages and the recorder package built successfully from copied source on 2026-10-10. Build, install, and log directories are generated output excluded from Git.

## UR5 physical arm installation and teleoperation

### Installed components and workspace roles

The physical platform uses **UR5 (`ur_type:=ur5`), ROS 2 Humble on Ubuntu
22.04, and robot IP `10.0.1.38`**. Use system Python 3.10 for ROS commands.
The following installation was inspected on 2026-10-11:

| Component | Installed version or location | Role |
| --- | --- | --- |
| `ur_robot_driver`, `ur_moveit_config`, `ur_calibration` | ROS Debian packages 2.14.0 under `/opt/ros/humble` | Physical arm connection, trajectory controllers, MoveIt configuration, calibration extraction |
| `moveit_servo` | ROS Debian package 2.5.9 | Converts teleoperation Twist commands into joint targets |
| `joy` | ROS Debian package 3.3.0 | Logitech Extreme 3D Pro input |
| `realsense2_camera` | ROS Debian package 4.58.3 | Base D435i and wrist D435 camera streams |
| `hardware_ws` | Six source packages: `serial`, `robotiq_driver`, `robotiq_controllers`, `robotiq_description`, `robotiq_hardware_tests`, `ur5_isaac_simulation` | Serial Robotiq support and Isaac simulation assets |
| `ros2_ws` | `ur5_vla_dataset_recorder` | Logitech mapper, guarded reset, synchronized recording, and policy tools |

The UR arm driver is **not contained in `hardware_ws/src/`**. Its binary
installation provides the physical arm stack. The current pneumatic gripper
uses UR digital outputs through SetIO; the serial Robotiq launch files are for
the separate compatible gripper setup. Isaac launch files start simulation
components and are not the physical arm bring-up procedure.

### Install dependencies on a new workstation

First install Ubuntu 22.04 and ROS 2 Humble following the
[official ROS installation instructions](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html).
The UR Humble driver supports binary installation through `ros-humble-ur`;
see the [official Humble driver installation guide](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/blob/humble/ur_robot_driver/doc/installation/installation.rst).
With the ROS apt repository configured, install the host dependencies:

```bash
sudo apt update
sudo apt install \
  ros-humble-ur ros-humble-moveit ros-humble-moveit-servo \
  ros-humble-joy ros-humble-realsense2-camera \
  ros-humble-cv-bridge ros-humble-ur-msgs ros-humble-ur-dashboard-msgs \
  ros-humble-control-msgs ros-humble-controller-manager-msgs \
  ros-humble-moveit-msgs ros-humble-trajectory-msgs \
  python3-opencv python3-numpy python3-yaml \
  python3-colcon-common-extensions python3-rosdep
```

If rosdep has not been initialized on that machine, run `sudo rosdep init`
once, followed by `rosdep update`. Install workspace dependencies:

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths "$UR5_VLA_ROOT/hardware_ws/src" \
  "$UR5_VLA_ROOT/ros2_ws/src" --ignore-src --rosdistro humble -y
```

Set `UR5_VLA_ROOT` using [Location and environment](#location-and-environment)
first, then build and source both workspaces in the order shown in
[Build and source](#build-and-source). The existing `hardware_ws/src/serial`
supplies the serial dependency; no second checkout is needed. Isaac support
additionally needs `ros-humble-tf-transformations`, which is missing from the
current host and its package manifest; see the smoke-test limitations below.

### Robot network, External Control, and calibration

Connect the robot control box and PC over Ethernet. Configure a unique PC
address on the robot's subnet, matching the actual pendant network settings;
the PC address has not been recorded here. Check the chosen interface and
reachability with `ip -4 addr` and `ping -c 3 10.0.1.38`.

Install the External Control URCap appropriate for the robot's PolyScope
version, reboot as directed, configure its remote host as the PC's Ethernet
IP, and save a pendant program containing the External Control node. The
installed driver provides
`/opt/ros/humble/share/ur_robot_driver/resources/externalcontrol-1.0.5.urcap`.
Use the [official Humble robot setup guide](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/blob/humble/ur_robot_driver/doc/installation/robot_setup.rst)
for the pendant procedure. The documented lab workflow starts the driver
first and then plays this program on the pendant. The current pendant
configuration and PolyScope version were not inspected during this review.

Extract and retain this particular arm's factory calibration when setting up
a new workstation. For example, choose a persistent configuration location:

```bash
mkdir -p "$UR5_VLA_ROOT/hardware_ws/config"
ros2 launch ur_calibration calibration_correction.launch.py \
  robot_ip:=10.0.1.38 \
  target_filename:="$UR5_VLA_ROOT/hardware_ws/config/ur5_calibration.yaml"
```

Then supply `kinematics_params_file` with that file when launching the driver.
No arm-specific extracted calibration was found in the repository during
inspection; the Isaac `default_kinematics.yaml` is a simulation/default asset.
The installed MoveIt launch uses the description package's default kinematics,
so also configure a matching calibrated description for MoveIt if accurate
Cartesian geometry is required; passing the driver argument alone does not
update that separate launch.

### Bring up the physical arm and MoveIt Servo

Prepare every terminal using the environment/source commands above. Keep
long-running launch processes in separate terminals. The following controller
and teleoperation steps enable physical motion; use a clear swept path, low
pendant speed, and an accessible emergency stop.

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5 robot_ip:=10.0.1.38 launch_rviz:=false
```

If the extracted calibration exists, append
`kinematics_params_file:="$UR5_VLA_ROOT/hardware_ws/config/ur5_calibration.yaml"`.
Start the saved External Control program on the pendant. In a second terminal:

```bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur5 launch_rviz:=true launch_servo:=true
```

The installed launch explicitly supports `launch_servo`. Switch the arm from
trajectory execution to continuous Servo control, then start Servo:

```bash
ros2 control switch_controllers \
  --deactivate scaled_joint_trajectory_controller \
  --activate forward_position_controller
ros2 service call /servo_node/start_servo std_srvs/srv/Trigger "{}"
ros2 control list_controllers
ros2 topic echo /servo_node/status --once
ros2 topic echo /joint_states --once
```

Confirm `forward_position_controller` is active, the trajectory controller is
inactive, and Servo status is zero. The installed Servo configuration outputs
joint positions on `/forward_position_controller/commands` and uses a 0.1 s
incoming-command timeout. The command path is:

```text
Logitech → /joy → logitech_servo → /servo_node/delta_twist_cmds
  → MoveIt Servo → /forward_position_controller/commands
  → UR robot driver → physical UR5
```

### Cameras, joystick, and gripper

Connect both RealSense cameras over USB 3. Launch each camera in a separate
terminal using its serial number and the namespaces expected by the recorder:

```bash
# Base D435i
ros2 launch realsense2_camera rs_launch.py \
  serial_no:=_044322072365 camera_namespace:=base camera_name:=camera \
  initial_reset:=true enable_color:=true enable_depth:=true \
  align_depth.enable:=true rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30

# Wrist D435
ros2 launch realsense2_camera rs_launch.py \
  serial_no:=_827312073590 camera_namespace:=wrist camera_name:=camera \
  initial_reset:=true enable_color:=true enable_depth:=true \
  align_depth.enable:=true rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30
```

Verify RGB and aligned depth on both cameras:

```bash
ros2 topic hz /base/camera/color/image_raw
ros2 topic hz /wrist/camera/color/image_raw
ros2 topic hz /base/camera/aligned_depth_to_color/image_raw
ros2 topic hz /wrist/camera/aligned_depth_to_color/image_raw
```

Each `topic hz` command runs until interrupted. Enumerate the Logitech device,
replace the placeholder with its actual device ID, and start input:

```bash
ros2 run joy joy_enumerate_devices
ros2 run joy joy_node --ros-args \
  -p device_id:=YOUR_LOGITECH_DEVICE_ID \
  -p deadzone:=0.1 -p autorepeat_rate:=50.0
```

For a new-dataset episode, run the guarded reset **before starting the mapper**:

```bash
ros2 run ur5_vla_dataset_recorder prepare_dataset_start
```

This moves the arm to its standard pose, requests at most 20% speed scaling
when needed, checks an interpolated path with MoveIt, then restores Servo.
Wait for `DATASET START READY`, then start the mapper in another terminal:

```bash
ros2 run ur5_vla_dataset_recorder logitech_servo
```

| Logitech control | Behavior in the current mapper |
| --- | --- |
| Physical button 1 / trigger (`buttons[0]`) | Hold to enable arm Twist commands; release publishes zero Twist |
| Stick forward/back and left/right | Base-frame X/Y translation |
| Handle twist | Rotation about base-frame Z |
| Physical buttons 5 / 6 | Z down / up while trigger is held |
| Physical buttons 2 / 3 | Close / open pneumatic gripper through SetIO; these buttons do not require the arm trigger |
| Physical button 7, held for two seconds with trigger released | Guarded standard-pose reset, then resume teleoperation |
| Physical buttons 9 / 10 / 11 | Recorder start / discard / save-success when the recorder is running |

The mapper requests up to 0.30 m/s linear and 0.50 rad/s angular speed in
`base_link`; these are input command magnitudes, subject to Servo and robot
limits. Gripper digital outputs use pin 0 for close and pin 1 for open.
Operate the gripper once to initialize its estimated dataset state, then check
`/gripper/state` and `/gripper/command` with `ros2 topic echo ... --once`.
The convention is 0 = closed, 1 = open, with no measured finger/contact state.

### Record demonstrations and return to trajectory control

Start the 20 Hz recorder after the cameras, joint state, Servo status, arm
targets, and gripper streams are ready:

```bash
ros2 launch ur5_vla_dataset_recorder recorder.launch.py
ros2 param set /dataset_recorder task \
  "Pick up the orange cube and place it in the bowl"
ros2 service call /dataset_recorder/status std_srvs/srv/Trigger "{}"
```

Set the task to describe the actual demonstration. Use button 9 to start,
button 11 to save success, or button 10 to discard. Raw recordings go to
`data/dataset/staging/episodes/`. Recording observes motion and does not
command the arm. The current recorder also requires aligned depth, fresh
inputs, Servo status zero, and at least 20 GB free disk space.

To return to MoveIt Plan and Execute, release the trigger and stop the mapper,
then stop Servo and switch controllers:

```bash
ros2 service call /servo_node/stop_servo std_srvs/srv/Trigger "{}"
ros2 control switch_controllers \
  --deactivate forward_position_controller \
  --activate scaled_joint_trajectory_controller
ros2 control list_controllers
```

### Daily run: teleoperation and dataset recording, one step at a time

Use this sequence after installation and workspace builds are complete. Each
long-running process has its own terminal. Prepare every terminal with step 1.
The reset and teleoperation commands physically move the arm: clear the swept
path, use low pendant speed, and keep the emergency stop accessible. Keep
standalone policy, replay, and `control_engine` processes stopped throughout.

**1. Prepare every new terminal.**

Use the workspaces inside this repository:

```text
physical-ur5-vla-implementation/
├── hardware_ws/install/   # This repository's hardware packages
└── ros2_ws/install/       # This repository's teleop and recorder package
```

Start from a fresh terminal. Source the system ROS installation first, then
the two repository overlays explicitly with `local_setup.bash`:

```bash
conda deactivate
export UR5_VLA_ROOT=/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation
source /opt/ros/humble/setup.bash
source "$UR5_VLA_ROOT/hardware_ws/install/local_setup.bash"
source "$UR5_VLA_ROOT/ros2_ws/install/local_setup.bash"
```

If Conda remains active, repeat `conda deactivate`. If Conda was not active,
continue with the source commands. ROS commands use system Python.

Verify the selected packages before starting hardware:

```bash
ros2 pkg prefix ur5_vla_dataset_recorder
ros2 pkg prefix robotiq_driver
ros2 pkg prefix ur_robot_driver
ros2 pkg prefix ur_moveit_config
/usr/bin/python3 -c 'import ur5_vla_dataset_recorder as p; print(p.__file__)'
```

The recorder prefix must be
`$UR5_VLA_ROOT/ros2_ws/install/ur5_vla_dataset_recorder`, and the Robotiq
prefix must be `$UR5_VLA_ROOT/hardware_ws/install/robotiq_driver`.
The recorder Python module must also resolve inside this repository; a
`ros2_ws/build/` path is normal for its symlink installation. The UR arm driver
and MoveIt are installed under `/opt/ros/humble`. These resolutions were
verified in a clean shell on 2026-10-13. If a package resolves from an older
workspace, stop and correct the terminal environment before proceeding.

**2. Check robot connectivity.**

```bash
ping -c 3 10.0.1.38
```

**3. Terminal 1 — start the physical UR5 driver.**

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5 robot_ip:=10.0.1.38 launch_rviz:=false
```

Apply the calibration argument described above if the extracted calibration
file is available. Then start the saved External Control program on the teach
pendant. Leave the driver running.

**4. Terminal 2 — start MoveIt and Servo.**

```bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur5 launch_rviz:=true launch_servo:=true
```

**5. Terminal 3 — activate the Servo controller and check readiness.**

Run these commands one by one:

```bash
ros2 control switch_controllers \
  --deactivate scaled_joint_trajectory_controller \
  --activate forward_position_controller
ros2 service call /servo_node/start_servo std_srvs/srv/Trigger "{}"
ros2 control list_controllers
ros2 topic echo /servo_node/status --once
ros2 topic echo /joint_states --once
```

Require the forward position controller active, the scaled trajectory
controller inactive, fresh joint state, and Servo status **0**.

**6. Terminal 4 — start the base D435i camera.**

```bash
ros2 launch realsense2_camera rs_launch.py \
  serial_no:=_044322072365 \
  camera_namespace:=base camera_name:=camera \
  initial_reset:=true enable_color:=true enable_depth:=true \
  align_depth.enable:=true \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30
```

**7. Terminal 5 — start the wrist D435 camera.**

```bash
ros2 launch realsense2_camera rs_launch.py \
  serial_no:=_827312073590 \
  camera_namespace:=wrist camera_name:=camera \
  initial_reset:=true enable_color:=true enable_depth:=true \
  align_depth.enable:=true \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30
```

**8. Terminal 3 — verify all camera streams.**

Run each command separately and press Ctrl+C after observing its rate:

```bash
ros2 topic hz /base/camera/color/image_raw
ros2 topic hz /wrist/camera/color/image_raw
ros2 topic hz /base/camera/aligned_depth_to_color/image_raw
ros2 topic hz /wrist/camera/aligned_depth_to_color/image_raw
```

**9. Terminal 6 — identify and start the Logitech joystick.**

```bash
ros2 run joy joy_enumerate_devices
```

Replace `YOUR_LOGITECH_DEVICE_ID` with the reported numeric device ID:

```bash
ros2 run joy joy_node --ros-args \
  -p device_id:=YOUR_LOGITECH_DEVICE_ID \
  -p deadzone:=0.1 -p autorepeat_rate:=50.0
```

**10. Terminal 3 — move to the recording start pose.**

Leave the teleoperation mapper stopped while this guarded reset runs:

```bash
ros2 run ur5_vla_dataset_recorder prepare_dataset_start
```

Wait for **`DATASET START READY`**. The command moves the arm and restores
Servo after the reset; a failed readiness check must be resolved first.

**11. Terminal 7 — start teleoperation.**

```bash
ros2 run ur5_vla_dataset_recorder logitech_servo
```

Hold physical button 1 (trigger) to move using the stick and handle twist.
Buttons 5/6 move down/up while the trigger is held. Release the trigger to
publish zero arm Twist. Buttons 2/3 close/open the gripper independently of the
trigger. Operate the gripper once to initialize its recording state. The
full mapping is in the control table above.

**12. Terminal 3 — check gripper streams.**

```bash
ros2 topic echo /gripper/state --once
ros2 topic echo /gripper/command --once
```

The values are 0 = closed and 1 = open; state is the acknowledged command
estimate rather than a measured finger position.

**13. Terminal 8 — start the dataset recorder.**

```bash
ros2 launch ur5_vla_dataset_recorder recorder.launch.py
```

**14. Terminal 3 — set the task to match the actual demonstration.**

```bash
ros2 param set /dataset_recorder task \
  "Pick up the orange cube and place it in the bowl"
```

**15. Check recorder readiness before starting an episode.**

```bash
ros2 service call /dataset_recorder/status std_srvs/srv/Trigger "{}"
```

Resolve any missing/stale input or disk-space issue reported by the service.

**16. Record each demonstration with the joystick.**

1. Place the object and prepare the start pose.
2. Press physical **button 9** to start recording.
3. Hold the trigger and perform the task, using the gripper buttons as needed.
4. Release the trigger after the demonstration.
5. Press **button 11** to save a successful episode, or **button 10** to discard it.

Before the next episode, release the trigger and hold **button 7 for two
seconds** to run the guarded standard-pose reset through the mapper. Wait for
completion before starting another recording. Finish or discard the current
episode before resetting.

**17. Terminal 3 — inspect saved episodes.**

```bash
ls -lah "$UR5_VLA_ROOT/data/dataset/staging/episodes/"
```

Review the saved demonstrations before converting an explicit selection;
follow [Dataset conversion and validation](#dataset-conversion-and-validation).

**18. Finish teleoperation and return to trajectory control if needed.**

Release the trigger, finish or discard any active episode, and stop the mapper
in terminal 7 with Ctrl+C. In terminal 3:

```bash
ros2 service call /servo_node/stop_servo std_srvs/srv/Trigger "{}"
ros2 control switch_controllers \
  --deactivate forward_position_controller \
  --activate scaled_joint_trajectory_controller
ros2 control list_controllers
```

The controller switch returns the arm to MoveIt Plan and Execute. When ending
the session, stop the recorder, joystick input, cameras, MoveIt, and driver
with Ctrl+C in their respective terminals; stop the driver last. Follow the
lab procedure for ending the pendant program and powering down the robot.

### Setup limits and troubleshooting

If the driver cannot connect, check robot/PC Ethernet addressing, the URCap's
PC host address, and that the External Control program is playing. If the
arm does not respond to teleoperation, check `/joy`, controller activation,
the Servo start-service response, and `/servo_node/status`. If recording is
blocked, its status service reports readiness problems; check all four camera
streams and initialize the gripper estimate. An `_rclpy_pybind11` error usually
means a Conda Python environment was used instead of ROS's system Python.

The current MoveIt scene does not model the table, bowl, cameras, or people;
a successful collision query is not a check of the real swept path. The
installation, local launch arguments, and mapper configuration were inspected
for this chapter, but no physical bring-up or teleoperation was executed.
Supervised hardware validation remains required. Keep standalone
`scripts/control_engine*.py` and policy/replay publishers stopped during this
Servo workflow: they form a separate control path and can compete with it.

## Directory map

| Path from repository root | Contents |
| --- | --- |
| `hardware_ws/src/` | Robotiq controller, driver, description, serial library, and UR5 Isaac simulation source |
| `ros2_ws/src/ur5-vla-dataset-recorder/` | Current ROS package, launch/config files, tests, tools, and package notebooks |
| `scripts/` | Standalone control, replay, and offline/shadow test entrypoints |
| `notebooks/` | Four later SmolVLA and π0.5 Colab notebooks and troubleshooting copies |
| `docs/episode_audit_20261002/` | Reviewed episode lists, audit code, contact sheets, and reports |
| `docs/thesis_chapter/` | Canonical methodology chapter, builder, figures, PDF, DOCX, and evidence register |
| `data/dataset/` | Raw episodes (`staging/`), converted datasets (`lerobot/`), and quarantine |
| `data/models/` | Trained checkpoints and evaluation results |
| `data/huggingface_cache/` | Copied model/dataset Hub cache used by fresh terminals |
| `data/hf_lerobot_cache/` | Copied LeRobot local dataset cache used by fresh terminals |
| `dump/desktop_vla/` | Earlier experiments, source code, simulation assets, and outputs |
| `dump/vla_script_training/` | Earlier LeRobot checkout, conversion/training scripts, and outputs |
| `dump/downloads_physical_ur5/` | Earlier downloaded training/deployment materials |
| `dump/downloaded_models/` | Downloaded checkpoints, including the 20 GB `Checkpoints/` archive |

See [Workspaces and hardware](#workspaces-and-hardware) for the role of each
hardware workspace package.

To regenerate the canonical chapter, run `/usr/bin/python3
"$UR5_VLA_ROOT/docs/thesis_chapter/build_chapter.py"`. Graphviz and system
matplotlib are required. Markdown and python-docx are installed locally under
`.python_deps/` (excluded from Git); on a new machine, install them with
`/usr/bin/python3 -m pip install --target "$UR5_VLA_ROOT/.python_deps"
markdown python-docx`.

The original project directories were copied, not moved. They are no longer sourced by the user's shell startup. Historical logs, report JSON, notebook output cells, and the dated handoff document may still quote original paths as evidence of where an experiment ran; they are not current runtime configuration.

## Current data and model

The `data/` layout below was verified on 2026-10-11. Paths are relative to
the repository root; episode media and checkpoint internals are omitted from
the tree for readability.

```text
data/
├── dataset/
│   ├── staging/
│   │   ├── episodes/                         # 112 finalized raw episodes
│   │   └── clean_20_cube_raw_nonconverted/   # Separate raw selection
│   ├── quarantine/
│   │   ├── invalid_episodes_2026-08-25/
│   │   └── partial_conversions/
│   └── lerobot/
│       ├── ur5_cube_110ep_v3/
│       ├── ur5_cube_30ep/
│       ├── ur5_cube_clean_20ep/
│       ├── ur5_cube_clean_20ep_delta/
│       ├── ur5_cube_clean_53ep/
│       ├── ur5_cube_clean_73ep_v2/
│       └── ur5_orange_cube_v1/
├── models/
│   ├── checkpoints/
│   │   └── ur5_cube_110ep_v3_smolvla_a100_b128_18000_steps_20261003_180950/
│   └── evaluation/                         # Empty after cleanup
├── hf_lerobot_cache/
│   ├── lerobot/
│   └── local/
└── huggingface_cache/
    ├── datasets/
    └── hub/
```

| Folder | Purpose and contents |
| --- | --- |
| `data/dataset/staging/` | Original demonstrations before conversion. Episodes contain `frames.jsonl`, `metadata.yaml`, base/wrist RGB images, and depth when recorded. Active or unfinished recorder episodes use hidden `.incomplete` directories. The separate `clean_20_cube_raw_nonconverted/` selection is retained. |
| `data/dataset/quarantine/` | Preserved invalid recordings and interrupted conversion outputs, kept separate from the selected training datasets. |
| `data/dataset/lerobot/` | Converted, versioned training datasets. Each dataset keeps its metadata, frame data, and camera media together. Absolute and delta action datasets are distinct versions. |
| `data/models/checkpoints/` | Trained policy weights, model/training configuration, preprocessing and postprocessing assets, intermediate checkpoints, training logs, metrics, environment records, and checksum records. |
| `data/models/evaluation/` | Destination for future prediction CSVs, evaluation reports, and test logs. All previous contents were deleted on 2026-10-11; the directory remains empty. |
| `data/hf_lerobot_cache/` | LeRobot local dataset cache selected by `HF_LEROBOT_HOME`. |
| `data/huggingface_cache/` | Hugging Face model and dataset download caches selected by `HF_HOME`. |

`data/dataset/` contains only `staging/`, `quarantine/`, and `lerobot/`.
`data/models/` contains only `checkpoints/` and `evaluation/`. The previous
`data/ur5_vla_dataset/` name has been replaced, and its `exports/` folder was
deleted. Dataset and checkpoint contents were preserved during the directory
reorganization; evaluation contents were intentionally removed. The recorder
and model-check script use the new paths.

`data/dataset/staging/episodes/` contains 112 finalized raw episode directories. `data/dataset/lerobot/ur5_cube_110ep_v3/` is the reviewed 110-episode dataset: **285,204 frames at 20 Hz**. Its `meta/info.json`, `meta/stats.json`, `meta/tasks.parquet`, `data/`, and `videos/` form the LeRobot dataset; keep the folder intact. Other `lerobot/` children are earlier dataset versions and must be identified by exact folder name when reporting results.

The final 18,000-step SmolVLA checkpoint is:

```text
data/models/checkpoints/ur5_cube_110ep_v3_smolvla_a100_b128_18000_steps_20261003_180950/pretrained_model/
```

It contains `model.safetensors`, `config.json`, the training config, and policy pre/postprocessor files. `data/models/evaluation/` is currently empty after the requested cleanup; future model checks write their reports there. `data/dataset/quarantine/` contains rejected or interrupted material; `data/dataset/staging/` contains raw recording material. The former `exports/` folder was deleted. Do not silently mix those categories when training.

Converted dataset folders currently include `ur5_cube_30ep`,
`ur5_cube_clean_20ep`, `ur5_cube_clean_20ep_delta`,
`ur5_cube_clean_53ep`, `ur5_cube_clean_73ep_v2`,
`ur5_orange_cube_v1`, and `ur5_cube_110ep_v3`. The folder name identifies
the selection and conversion; check its `meta/info.json` and
`meta/ur5_vla_conversion.json` before comparing experiments. Raw episode
directories contain `frames.jsonl`, metadata, and referenced camera media;
the README describes them by schema and dataset version rather than listing
all 1.4 million files individually.

At the original migration, the main raw/converted data copy completed with **1,427,363 files** in both source and destination trees and an rsync dry run showing zero differences. The copied hardware source, later checkpoint archive, LeRobot cache, and Hugging Face Hub/datasets caches also showed zero differences. Migration-time size estimates were about 139 GB of project data, roughly another 32 GB of copied caches, and more than 110 GB of older experiments/checkpoints; these are historical estimates before the exports/evaluation cleanup. These large folders are ignored by Git: **a normal Git push does not upload datasets or model weights**.

## Dataset conversion and validation

### Scripts, environment, and data flow

Conversion runs locally after recording and reviewing demonstrations:

```text
data/dataset/staging/episodes/<selected episodes>
  → raw validation and full media decoding
  → convert_lerobot.py (LeRobot dataset creation)
  → data/dataset/lerobot/<new dataset name>
  → validate_lerobot.py (converted dataset checks)
  → training notebook for the chosen policy
```

| File relative to the repository root | Role |
| --- | --- |
| `ros2_ws/src/ur5-vla-dataset-recorder/ur5_vla_dataset_recorder/convert_lerobot.py` | Main raw-to-LeRobot converter; invoked as a Python module |
| `ros2_ws/src/ur5-vla-dataset-recorder/ur5_vla_dataset_recorder/raw_dataset.py` | Raw metadata, timing, state/action, and media validation helpers |
| `ros2_ws/src/ur5-vla-dataset-recorder/ur5_vla_dataset_recorder/validate_lerobot.py` | Loads the converted dataset using LeRobot and writes a validation report |
| `docs/episode_audit_20261002/convert_reviewed_episodes.sh` | Wrapper for the dated 37-episode and combined 110-episode selections; runs conversion and validation |
| `docs/episode_audit_20261002/accepted_new_episodes.txt`, `combined_episodes.txt` | Exact episode selections used by that wrapper |

**`pi0` is the name of the local Conda environment**, not the model chosen by
the converter. It has LeRobot 0.5.2 and the Python dependencies needed for
conversion, video encoding, and validation. Its editable LeRobot installation
points to `dump/vla_script_training/lerobot`. The resulting dataset can be used
with SmolVLA or another compatible policy. ROS nodes use system Python;
conversion uses this separate environment and does not command hardware.

Prepare the source path and verify the installed environment:

```bash
export UR5_PACKAGE="$UR5_VLA_ROOT/ros2_ws/src/ur5-vla-dataset-recorder"
export PYTHONPATH="$UR5_PACKAGE${PYTHONPATH:+:$PYTHONPATH}"
cd "$UR5_PACKAGE"

/home/grasplab/miniconda3/bin/conda run --no-capture-output -n pi0 \
  python -c 'import sys, lerobot; import importlib.metadata as m; print(sys.executable); print(lerobot.__file__); print(m.version("lerobot"))'
```

Set `UR5_VLA_ROOT` as described in [Location and environment](#location-and-environment)
before these commands. The absolute Conda executable avoids depending on shell
activation. The converter source still writes the historical compatibility
label `0.4.3 / dataset codebase v3.0` into conversion metadata; this label is
distinct from the installed LeRobot version checked above.

### Convert a new reviewed selection

Choose a **new dataset name**, set the task to match the demonstration, and
replace both example episode names with existing reviewed directory names.
Use the same terminal prepared above:

```bash
UR5_DATASET_NAME=MY_NEW_DATASET
UR5_TASK="Pick up the orange cube and place it in the bowl"
UR5_EPISODES=(episode_YYYYMMDD_HHMMSS episode_YYYYMMDD_HHMMSS)

/home/grasplab/miniconda3/bin/conda run --no-capture-output -n pi0 \
  python -m ur5_vla_dataset_recorder.convert_lerobot \
  --input-root "$UR5_VLA_ROOT/data/dataset/staging/episodes" \
  --output-root "$UR5_VLA_ROOT/data/dataset/lerobot/$UR5_DATASET_NAME" \
  --repo-id "local/$UR5_DATASET_NAME" \
  --task "$UR5_TASK" \
  --episodes "${UR5_EPISODES[@]}" \
  --depth none --storage video --action-mode absolute
```

For a longer reviewed selection, set the same array from your own text file
containing one episode directory name per line:

```bash
mapfile -t UR5_EPISODES < /absolute/path/to/reviewed_episode_names.txt
```

Load that list before running the conversion command. Explicit selection
keeps rejected recordings out of the dataset. Omitting `--episodes` selects
all finalized episode directories under the input root; the current raw
folder includes recordings excluded from the reviewed 110-episode dataset,
so omission is not equivalent to the reviewed selection.

The converter validates **all selected episodes before creating the output**.
It requires successful finalized recordings and checks metadata, canonical
joint ordering, frame/timestamp consistency, finite state/action values,
gripper conventions, and referenced media. It decodes all referenced RGB and
depth media during preflight, even when depth is excluded from the output.
Any raw validation failure stops conversion. Existing output directories are
refused by default. `--overwrite` deletes an existing output directory, so use
a new version name for routine conversions.

### Conversion options and resulting schema

| Option | Meaning |
| --- | --- |
| `--input-root` | Directory containing finalized raw episode folders |
| `--output-root` | New converted dataset directory |
| `--repo-id` | LeRobot dataset identifier, such as `local/MY_NEW_DATASET`; this command does not upload it to the Hub |
| `--episodes` | Explicit episode directory names; omission selects all finalized episodes |
| `--task` | Override the task for every selected episode; omission uses each episode's metadata task |
| `--depth none` | Current RGB-only training workflow; source depth files remain intact |
| `--depth meters` | Add float32 depth in metres and validity masks; requires suitable raw depth and metadata |
| `--storage video` | Store base/wrist RGB as videos; `image` is the alternative |
| `--action-mode absolute` | Six absolute joint-position targets in radians and an absolute binary gripper command |
| `--action-mode delta` | Six same-frame differences `q_command - q_measured`; gripper remains absolute |
| `--overwrite` | Replace an existing output by deleting it; not used in the commands above |

The current schema expects 640×480 camera images and creates a 20 Hz dataset.
`observation.state` contains six measured joints in canonical UR5 order plus
the estimated gripper state. `action` contains the six arm targets or deltas
plus the gripper command. The gripper convention is **0 = closed, 1 = open**.
Exact raw episode timing is retained as `source_episode_time_s` alongside
LeRobot's regular frame timestamps. The task is stored with each frame.
Raw recordings are read without being changed.

The output contains `meta/`, frame data under `data/`, and camera media under
`videos/` for video storage. `meta/ur5_vla_conversion.json` records the source
episode names, joint order, units, action semantics, gripper convention, depth
mode, and storage choice. Keep the whole dataset directory together.

### Validate the converted dataset

After conversion succeeds, run the validator in the same prepared terminal:

```bash
/home/grasplab/miniconda3/bin/conda run --no-capture-output -n pi0 \
  python -m ur5_vla_dataset_recorder.validate_lerobot \
  --root "$UR5_VLA_ROOT/data/dataset/lerobot/$UR5_DATASET_NAME" \
  --repo-id "local/$UR5_DATASET_NAME" \
  --expected-episodes "${#UR5_EPISODES[@]}" \
  --task "$UR5_TASK" \
  --video-backend pyav \
  --report "$UR5_VLA_ROOT/data/dataset/lerobot/$UR5_DATASET_NAME/validation_report.md"
```

Add `--expected-frames NUMBER` when the reviewed frame total is known. The
validator scans every state/action row, checks seven-dimensional finite values
and binary gripper actions, checks conversion semantics and optional expected
counts/task, and requires close/open transitions in each episode. Delta
datasets are also checked against a 0.012566 rad arm-delta envelope. It decodes
both camera views at the beginning, middle, and end of each episode; converted
video validation is sampled, rather than a full-frame decode. A passing run
prints `PASS`; validation errors produce a nonzero exit and a report.

### Existing reviewed 37- and 110-episode workflows

The dated wrapper uses fixed selections and destinations:

| Mode | Selection | Output dataset | Expected frames |
| --- | --- | --- | --- |
| `new` | 37 accepted October 1 episodes | `ur5_cube_oct01_37ep` | 88,497 |
| `combined` | Previous 73 plus the accepted 37 | `ur5_cube_110ep_v3` | 285,204 |

```bash
# Show the wrapper's available modes.
bash "$UR5_VLA_ROOT/docs/episode_audit_20261002/convert_reviewed_episodes.sh" --help

# Convert and validate the reviewed 110-episode selection.
bash "$UR5_VLA_ROOT/docs/episode_audit_20261002/convert_reviewed_episodes.sh" combined
```

Use `new` instead of `combined` for the fixed 37-episode selection. The
110-episode output **already exists**, so `combined` currently refuses to
overwrite it. These modes are historical selections, not automatic selection
of newly recorded episodes. For another selection, use the explicit new-name
command above. The combined selection retains a historical episode with
multiple grasp/release transitions, as documented in the audit report.

### Progress and troubleshooting

`[preflight]` messages show raw validation and media decoding; `[convert]`
messages show frame staging and video encoding; `[finalize]` marks metadata
and statistics completion. Encoding can take time. A failed or interrupted
conversion may leave a partial output directory: inspect it and keep it
separate from valid datasets before retrying with a new output name.

If imports fail, verify the `pi0` environment and `PYTHONPATH` above. If the
output already exists, choose a new dataset name. If preflight fails, inspect
the reported episode/media/timing error and revise the reviewed selection.
If the converted validator fails, retain its report for diagnosis and do not
use that output as a validated training dataset. Local conversion does not
need the robot running. Package conversion notebooks provide a separate Colab
route using cloud paths; archived `dump/` CSV converters use an older raw
format and do not replace this `frames.jsonl` workflow.

## ROS package files

The installed console commands are defined in `ros2_ws/src/ur5-vla-dataset-recorder/setup.py`. These modules live in its `ur5_vla_dataset_recorder/` directory:

| Module | Purpose |
| --- | --- |
| `recorder_node.py`, `episode_writer.py` | 20 Hz synchronized recording and file writing; default output is this repository's data folder |
| `logitech_servo_node.py`, `standard_pose_node.py` | Joystick teleoperation, gripper IO, and guarded standard-pose preparation |
| `core.py`, `raw_dataset.py` | Joint ordering, schema checks, and raw episode validation |
| `convert_lerobot.py`, `validate_lerobot.py` | Raw-to-LeRobot conversion and converted-dataset validation |
| `action_semantics.py`, `episode_control.py`, `episode_execution.py` | Absolute/delta action conventions and episode state/control helpers |
| `safety.py`, `live_control.py` | Readiness gates and live-control checks |
| `policy_executor.py`, `shadow_protocol.py` | Read-only live shadow client and localhost prediction protocol |
| `policy_inference_server.py`, `offline_inference.py` | Model server and saved-dataset inference; offline inference does not move the robot |
| `policy_live_executor.py`, `policy_episode_executor.py` | Live policy execution and bounded episode execution; these can command hardware |
| `package_dataset.py`, `split_archive.py`, `install_checkpoint.py` | Dataset archive and checkpoint transport utilities |

`launch/recorder.launch.py` loads `config/recorder.yaml`. The recorder config uses `data/dataset` relative to this repository, and the recorder resolves relative `output_root` values from the repository root. `test/` contains package checks. `tools/` contains notebook generators for the 30-, 20-, 53-, 73-, and 110-episode experiments, generic conversion-notebook generators, and a historical training progress monitor. That monitor uses the copied `dump/downloaded_models/Checkpoints/` path and an old process ID; it is not a live monitor for a new run.

The package's `notebooks/` contains conversion notebooks and SmolVLA/π0/π0.5 training notebooks for named dataset versions. Colab notebooks intentionally use `/content` and Google Drive paths because they execute in Colab. `notebooks/` at repository root contains four later 110-episode training and π0.5 troubleshooting notebooks. All 17 package/root notebook files parsed as JSON; cloud training was not rerun during this migration.

Package conversion notebooks: `convert_raw_ur5_to_lerobot_colab.ipynb`,
`convert_ur5_cube_30ep_colab.ipynb`, and
`convert_ur5_cube_clean_20ep_colab.ipynb`. Package SmolVLA notebooks:
`train_ur5_smolvla_colab.ipynb`, `train_ur5_cube_30ep_smolvla_40000_colab.ipynb`,
the 20- and 53-episode clean variants, the 73-episode A100 variant, and
`train_ur5_cube_110ep_v3_smolvla_a100_colab.ipynb`. The package also has
`train_ur5_cube_clean_73ep_v2_pi0_a100_colab.ipynb` and
`train_ur5_cube_110ep_v3_pi05_a100_colab.ipynb`. The root notebook files are
`train_ur5_cube_110ep_v3_smolvla_a100_colab.ipynb`,
`train_ur5_cube_110ep_v3_pi05_a100_colab.ipynb`,
`pi05_error_20261004.ipynb`, and `pi05_fixed_auth_20261004.ipynb`.

Notebook generators in `tools/` are `generate_colab_notebook.py`,
`generate_conversion_colab_notebook.py`, `generate_generic_conversion_colab.py`,
`generate_30ep_training_colab.py`, `generate_clean_20ep_colab.py`,
`generate_clean_20ep_b8_60k_guarded_colab.py`,
`generate_clean_53ep_80k_colab.py`, `generate_clean_53ep_b8_60k_colab.py`,
`generate_pi0_73ep_a100_colab.py`, `generate_110ep_training_colab.py`, and
`generate_pi05_110ep_colab.py`. They regenerate specific notebook variants;
review the named input/output notebook before rerunning a generator.

`docs/episode_audit_20261002/` contains `audit_episodes.py` (raw validation
and contact sheets), `finalize_report.py` (selection and report),
`convert_reviewed_episodes.sh` (new/combined conversion),
`accepted_new_episodes.txt` and `combined_episodes.txt` (exact selections),
CSV/JSON audit summaries, contact sheets, and `REPORT.md`. The canonical
`docs/thesis_chapter/` contains `build_chapter.py`, the Markdown chapter,
generated HTML/DOCX/PDF, figures, checkpoint comparison CSV, and a SHA-256
evidence register. `docs/SMOLVLA_110EP_COLAB_README.md` is the training
notebook handoff note; this root README remains the current path guide.

## Standalone scripts

Some filenames resemble ROS package modules, but they are distinct programs,
not duplicate copies. The files under `ros2_ws/src/ur5-vla-dataset-recorder/`
are installed by `colcon` and invoked with `ros2 run`; they share tested
package helpers and accept explicit artifact paths. In particular,
`ros2 run ur5_vla_dataset_recorder policy_executor` is a **shadow-only**
inference client and rejects `--live`. The package's
`policy_live_executor` and `policy_episode_executor` are separate guarded
programs; the latter defaults to shadow mode.

The root `scripts/` folder preserves standalone deployment and experiment
entrypoints from the earlier `bypass` workflow. `scripts/policy_executor.py`
selects a specific downloaded checkpoint, starts the inference server, and
publishes seven abstract actions on `/control_abstraction_layer` for the
separate `control_engine.py` to consume. It imports observation/protocol
helpers from the ROS package; it is not the same code as the package's
shadow-only `policy_executor.py`. `scripts/policy_executor_phased.py` is an
earlier phase-aware variant. `scripts/episode_executor.py` replays saved
actions to the abstraction topic, whereas the package's
`policy_episode_executor.py` executes new policy predictions with explicit
shadow/physical modes. Keep these roles distinct when choosing a command;
the standalone scripts can cause physical motion when paired with the live
control engine.

| File in `scripts/` | Purpose and status |
| --- | --- |
| `project_paths.py` | Derives the repository, package, data, downloaded-model, and cache paths; gives model-server subprocesses a clean project environment |
| `control_engine.py` | Smooths abstract commands and publishes arm/gripper commands; **physical motion** unless in dry-run mode |
| `control_engine_phased.py` | Phase-aware control-engine variant; **physical motion** unless in dry-run mode |
| `policy_executor.py` | Starts the copied 110-episode SmolVLA server and publishes 7D actions; **physical motion** |
| `trajectory_offset.py` | Imported helper for continuous Cartesian translation of policy arm targets; uses the MoveIt robot description, local kinematics, and MoveIt path checks |
| `grasp_offset.py` | Shared offset validation/planning and the earlier one-grasp intervention; imported by the executor, not run separately |
| `policy_executor_phased.py` | Phase-gated executor using the 110-episode checkpoint; **physical motion** outside shadow mode |
| `episode_executor.py` | Replays saved episode actions to the abstraction topic; **physical motion** |
| `standard_pose.py` | Standalone standard-pose helper that publishes controller targets; **physical motion** |
| `test_110ep_model.sh` | `offline`, `server`, and read-only `shadow` model checks using this repository's paths |
| `pi05_110ep_dataset_validation.json`, `smolvla_110ep_notebook_validation.json` | Saved validation records, not executables |
| `thesis_chapter/` | Duplicate snapshot from initial copy; use `docs/thesis_chapter/` as canonical |

The copied live scripts import the ROS package from this repository's source tree before another installation can be chosen. Their model and data constants resolve under this repository. The original standalone scripts outside this repository were not edited.

### Continuous offset experiment

The measured forward direction is approximately +X in the robot's `base`
frame. The experimental calibration file currently specifies **0.5 mm** in
that direction. To translate the entire policy trajectory, use this executor
command in a prepared ROS terminal, with the control engine in its separate
terminal:

```bash
cd "$UR5_VLA_ROOT/scripts"
/usr/bin/python3 policy_executor.py \
  --grasp-offset-file "$UR5_VLA_ROOT/data/models/evaluation/codex test analysis/offset_calibration.json" \
  --offset-mode continuous
```

This mode translates every arm target during approach, grasp, carry, and
placement, preserving its orientation and gripper value. It blends chunks
before translation so the offset is applied once per target. It uses the
MoveIt robot model to solve each translated pose and checks the interpolated
joint path against the current planning scene before queuing the chunk.
Planning failure latches a fault and clears queued actions. The scene's
unmodelled table/objects and the standalone engine's existing limitations
still apply; this is an experimental action correction, not camera calibration.

`--offset-mode grasp` selects the earlier correction immediately before the
first close. Omitting `--grasp-offset-file` disables both modes. The file is
loaded at startup: stop and safely reset before restarting to change modes or
offset size. Do not restart an executor while its startup commands could
release a held object. The continuous mode passed a read-only 50-target
planning preview and geometry/queue regression checks; physical task success
with this mode has not yet been measured.

## Typical read-only checks

```bash
ros2 pkg prefix ur5_vla_dataset_recorder
bash "$UR5_VLA_ROOT/scripts/test_110ep_model.sh" --help
bash "$UR5_VLA_ROOT/scripts/test_110ep_model.sh" offline
cat "$UR5_VLA_ROOT/data/dataset/lerobot/ur5_cube_110ep_v3/meta/info.json"
```

Recording is launched with `ros2 launch ur5_vla_dataset_recorder recorder.launch.py` after the robot and cameras are ready. Its output goes to the repository data folder. The live control and policy scripts above can move the robot; they were **imported and syntax-checked**, not executed on hardware during this migration.

## Historical material and known limits

`dump/` preserves earlier simulation, bottle, lemon, and cube experiments. The archived `dump/vla_script_training/convert_data*.py` and `conversion_data_cube_twoCam.py` scripts now look under `dump/raw/dataset_raw_cube` or `dump/raw/dataset_raw_bottle` and write under `data/hf_lerobot_cache`; they refuse to overwrite an existing converted dataset. Those earlier CSV raw directories were not present on this workstation, so the scripts cannot complete until the matching inputs are supplied. The current `frames.jsonl` raw dataset has a different schema. Archived training logs and notebook outputs may include old machine paths; they are historical observations, not current configuration. The dated package handoff document has the same status.

ROS hardware drivers, Conda, CUDA, Google Drive, and the physical arm remain external dependencies. A one-frame GPU inference completed from the copied checkpoint and dataset. That validates local model loading and an offline prediction; it does **not** establish physical task success or certify live motion safety. No live control process was started by this migration.

## Verification and Git

### ROS executable smoke test (2026-10-11)

All 16 installed console executables from `ur5_vla_dataset_recorder` and
`ur5_isaac_simulation` were discovered from the new workspace overlays.
The `serial` and Robotiq packages provide libraries, plugins, descriptions,
and launch files; they do not install additional `ros2 run` executables here.
ROS nodes were started only in isolated domain 210 with
`ROS_LOCALHOST_ONLY=1`; no physical robot command was sent. A timeout exit
code of 124 means the node remained running until deliberately stopped.

| Executable | Result | Limit |
| --- | --- | --- |
| `dataset_recorder` | Started; `/dataset_recorder` and start/stop services visible | Recording with live cameras and arm untested |
| `logitech_servo` | Started; warned that SetIO was absent in isolation | Joystick, gripper IO, and arm response untested |
| `move_to_standard_pose`, `prepare_dataset_start` | Started and waited for robot state/services; interrupted after five seconds | Physical movement and safety checks untested |
| `policy_executor`, `policy_live_executor`, `policy_episode_executor` | `--help` passed | Live action execution untested |
| `convert_lerobot`, `validate_lerobot`, `package_lerobot_dataset`, `install_lerobot_checkpoint`, `split_lerobot_archive` | `--help` passed | Dataset operations were not repeated in this smoke test |
| `offline_policy_check` | Fails under system Python because `torch` is missing; module `--help` passes with `pi0` Python | Run as `/home/grasplab/miniconda3/envs/pi0/bin/python -m ur5_vla_dataset_recorder.offline_inference ...` after sourcing overlays |
| Isaac `ur5_controller`, `gripper_controller` | Started and remained active until timeout | Both print an `rcl_shutdown already called` traceback on interrupt; action execution untested |
| Isaac `ur5_isaac_ros2` | Fails at import: `tf_transformations` missing | Install `ros-humble-tf-transformations` in the host ROS environment, then retest with Isaac Sim/display available |

The Isaac package currently omits its `tf_transformations` runtime dependency
from `package.xml`; the host ROS package is not installed. The Isaac node
also uses a GUI, so a headless smoke test cannot establish complete startup
even after the dependency is installed. A supervised live test is still needed
for physical nodes when the robot, controller state, speed slider, cameras,
and swept path are ready.

On 2026-10-10: both workspaces built; the recorder package resolved from this repository in a fresh shell; `pi0` imported LeRobot and the UR5 package from here; 67 system-Python package tests passed with one skipped, and six model-protocol tests passed in `pi0`; Python compilation, shell syntax, and notebook JSON checks passed; one-frame GPU inference completed with only the new cache locations for the canonical 110-episode checkpoint and for both downloaded checkpoints selected by the copied policy scripts; both copies of the thesis chapter builder completed and refreshed 57 evidence entries. The 95 standalone files copied from the earlier working folder matched their originals byte-for-byte at the time of copying. Source data, checkpoint, and cache copies were checked with rsync dry runs.

`.gitignore` excludes data, model weights, older generated outputs, upstream LeRobot checkouts, Python caches, and both ROS build trees. Source code, notebooks, reports, and this README remain available to commit. No commit or push was made as part of this migration.
