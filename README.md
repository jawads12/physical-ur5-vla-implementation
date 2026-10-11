# Physical UR5 VLA implementation

Central local repository for physical UR5 vision-language-action recording, conversion, training, and evaluation at GRASP Lab, Ontario Tech University. **This is the authoritative README for current paths and operation.** Other Markdown files are package details, experiment reports, or historical records.

## Contents

- [Workspaces and hardware](#workspaces-and-hardware)
- [Location and environment](#location-and-environment)
- [Build and source](#build-and-source)
- [Directory map](#directory-map)
- [Current data and model](#current-data-and-model)
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

ROS 2 Humble, the UR robot driver, camera drivers, system Python 3.10, and the `pi0` Conda environment are system dependencies. The `pi0` environment's editable LeRobot 0.5.2 installation now points to `legacy/vla_script_training/lerobot` **inside this repository**. Conda itself remains at `/home/grasplab/miniconda3/bin/conda`. The external drive must remain mounted while using this setup. Fresh terminals also set `HF_HOME` to `data/huggingface_cache` and `HF_LEROBOT_HOME` to `data/hf_lerobot_cache`; authentication tokens were not copied.

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

## Directory map

| Path from repository root | Contents |
| --- | --- |
| `hardware_ws/src/` | Robotiq controller, driver, description, serial library, and UR5 Isaac simulation source |
| `ros2_ws/src/ur5-vla-dataset-recorder/` | Current ROS package, launch/config files, tests, tools, and package notebooks |
| `scripts/` | Standalone control, replay, and offline/shadow test entrypoints |
| `notebooks/` | Four later SmolVLA and π0.5 Colab notebooks and troubleshooting copies |
| `docs/episode_audit_20261002/` | Reviewed episode lists, audit code, contact sheets, and reports |
| `docs/thesis_chapter/` | Canonical methodology chapter, builder, figures, PDF, DOCX, and evidence register |
| `data/ur5_vla_dataset/` | Raw episodes, converted datasets, checkpoints, exports, evaluation, quarantine |
| `data/huggingface_cache/` | Copied model/dataset Hub cache used by fresh terminals |
| `data/hf_lerobot_cache/` | Copied LeRobot local dataset cache used by fresh terminals |
| `legacy/desktop_vla/` | Earlier experiments, source code, simulation assets, and outputs |
| `legacy/vla_script_training/` | Earlier LeRobot checkout, conversion/training scripts, and outputs |
| `legacy/downloads_physical_ur5/` | Earlier downloaded training/deployment materials |
| `legacy/downloaded_models/` | Downloaded checkpoints, including the 20 GB `Checkpoints/` archive |

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

`data/ur5_vla_dataset/staging/episodes/` contains 112 finalized raw episode directories. `data/ur5_vla_dataset/lerobot/ur5_cube_110ep_v3/` is the reviewed 110-episode dataset: **285,204 frames at 20 Hz**. Its `meta/info.json`, `meta/stats.json`, `meta/tasks.parquet`, `data/`, and `videos/` form the LeRobot dataset; keep the folder intact. Other `lerobot/` children are earlier dataset versions and must be identified by exact folder name when reporting results.

The final 18,000-step SmolVLA checkpoint is:

```text
data/ur5_vla_dataset/checkpoints/ur5_cube_110ep_v3_smolvla_a100_b128_18000_steps_20261003_180950/pretrained_model/
```

It contains `model.safetensors`, `config.json`, the training config, and policy pre/postprocessor files. `data/ur5_vla_dataset/evaluation/` holds saved offline reports, including `smolvla_110ep_18000/new_repo_gpu_smoke.csv` and its JSON report from a one-frame GPU check. `exports/` contains packaged datasets; `quarantine/` contains rejected or interrupted material; `staging/` contains raw recording material. Do not silently mix those categories when training.

Converted dataset folders currently include `ur5_cube_30ep`,
`ur5_cube_clean_20ep`, `ur5_cube_clean_20ep_delta`,
`ur5_cube_clean_53ep`, `ur5_cube_clean_73ep_v2`,
`ur5_orange_cube_v1`, and `ur5_cube_110ep_v3`. The folder name identifies
the selection and conversion; check its `meta/info.json` and
`meta/ur5_vla_conversion.json` before comparing experiments. Raw episode
directories contain `frames.jsonl`, metadata, and referenced camera media;
the README describes them by schema and dataset version rather than listing
all 1.4 million files individually.

The main raw/converted data copy completed with **1,427,363 files** in both source and destination trees and an rsync dry run showing zero differences. The copied hardware source, later checkpoint archive, LeRobot cache, and Hugging Face Hub/datasets caches also showed zero differences. The project data is about 139 GB, with roughly another 32 GB of copied caches and more than 110 GB of older experiments/checkpoints. These large folders are ignored by Git: **a normal Git push does not upload datasets or model weights**.

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

`launch/recorder.launch.py` loads `config/recorder.yaml`. The recorder config uses `data/ur5_vla_dataset` relative to this repository, and the recorder resolves relative `output_root` values from the repository root. `test/` contains package checks. `tools/` contains notebook generators for the 30-, 20-, 53-, 73-, and 110-episode experiments, generic conversion-notebook generators, and a historical training progress monitor. That monitor uses the copied `legacy/downloaded_models/Checkpoints/` path and an old process ID; it is not a live monitor for a new run.

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

| File in `scripts/` | Purpose and status |
| --- | --- |
| `project_paths.py` | Derives the repository, package, data, downloaded-model, and cache paths; gives model-server subprocesses a clean project environment |
| `control_engine.py` | Smooths abstract commands and publishes arm/gripper commands; **physical motion** unless in dry-run mode |
| `control_engine_phased.py` | Phase-aware control-engine variant; **physical motion** unless in dry-run mode |
| `policy_executor.py` | Starts the copied 110-episode SmolVLA server and publishes 7D actions; **physical motion** |
| `policy_executor_phased.py` | Earlier 73-episode phase-gated executor; **physical motion** outside shadow mode |
| `episode_executor.py` | Replays saved episode actions to the abstraction topic; **physical motion** |
| `standard_pose.py` | Standalone standard-pose helper that publishes controller targets; **physical motion** |
| `test_110ep_model.sh` | `offline`, `server`, and read-only `shadow` model checks using this repository's paths |
| `pi05_110ep_dataset_validation.json`, `smolvla_110ep_notebook_validation.json` | Saved validation records, not executables |
| `thesis_chapter/` | Duplicate snapshot from initial copy; use `docs/thesis_chapter/` as canonical |

The copied live scripts import the ROS package from this repository's source tree before another installation can be chosen. Their model and data constants resolve under this repository. The original standalone scripts outside this repository were not edited.

## Typical read-only checks

```bash
ros2 pkg prefix ur5_vla_dataset_recorder
bash "$UR5_VLA_ROOT/scripts/test_110ep_model.sh" --help
bash "$UR5_VLA_ROOT/scripts/test_110ep_model.sh" offline
cat "$UR5_VLA_ROOT/data/ur5_vla_dataset/lerobot/ur5_cube_110ep_v3/meta/info.json"
```

Recording is launched with `ros2 launch ur5_vla_dataset_recorder recorder.launch.py` after the robot and cameras are ready. Its output goes to the repository data folder. The live control and policy scripts above can move the robot; they were **imported and syntax-checked**, not executed on hardware during this migration.

## Historical material and known limits

`legacy/` preserves earlier simulation, bottle, lemon, and cube experiments. The archived `legacy/vla_script_training/convert_data*.py` and `conversion_data_cube_twoCam.py` scripts now look under `legacy/raw/dataset_raw_cube` or `legacy/raw/dataset_raw_bottle` and write under `data/hf_lerobot_cache`; they refuse to overwrite an existing converted dataset. Those earlier CSV raw directories were not present on this workstation, so the scripts cannot complete until the matching inputs are supplied. The current `frames.jsonl` raw dataset has a different schema. Archived training logs and notebook outputs may include old machine paths; they are historical observations, not current configuration. The dated package handoff document has the same status.

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
