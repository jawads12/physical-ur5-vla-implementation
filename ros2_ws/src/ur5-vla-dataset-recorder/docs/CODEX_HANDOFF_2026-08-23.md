# Codex Handoff — Physical UR5 VLA Project

Date: 2026-08-23 (America/Toronto)

## Update — 2026-08-24: reusable raw-to-LeRobot notebook

A generic self-contained Colab converter was added:

```text
notebooks/convert_raw_ur5_to_lerobot_colab.ipynb
tools/generate_generic_conversion_colab.py
```

It expects the user's current Drive layout:

```text
/content/drive/MyDrive/pHYSICAL UR5/datasets/raw
/content/drive/MyDrive/pHYSICAL UR5/datasets/lerobot
```

The notebook discovers folders whose direct children are `episode_*`, presents an input dropdown, proposes an editable output name, requires an explicit task instruction, and accepts optional explicitly named episode exclusions. It copies raw data to `/content`, fully validates and converts with the embedded checked-in converter, validates the LeRobot result, copies it to `<output>.copying`, SHA-256 verifies every file, and only then renames it under `datasets/lerobot/<output>`. It refuses existing final or staging outputs and contains no overwrite/delete mode.

The current raw Drive link contains an `episodes` folder. The destination Drive link is the existing LeRobot library containing `ur5_cube_30ep` and the older split archive. The generic notebook was uploaded to the same Drive notebook folder as the training notebooks.

## Update — 2026-08-24: new 28-episode training notebook

The converted `ur5_cube_30ep` dataset is now visible in the user's Google Drive folder. Connector readback confirmed its `meta/info.json` reports:

```text
episodes: 28
frames:   47,204
tasks:    1
fps:      20
cameras:  observation.images.base and observation.images.wrist, AV1 480x640
state:    7D
action:   7D
```

`meta/ur5_vla_conversion.json` lists the 28 accepted source episodes and does not include rejected `episode_20260822_035620`.

A separate 40,000-step notebook was generated instead of modifying the previous 10-episode/5,000-step notebook:

```text
notebooks/train_ur5_cube_30ep_smolvla_40000_colab.ipynb
tools/generate_30ep_training_colab.py
```

The notebook expects the Drive dataset at:

```text
/content/drive/MyDrive/pHYSICAL UR5/datasets/lerobot/ur5_cube_30ep
```

It copies the unpacked dataset to `/content/ur5_cube_30ep`, verifies file inventory, validates 28 episodes/47,204 frames and the exact bowl task, performs the SmolVLA compatibility pass, then trains for 40,000 steps. Resumable checkpoints are made every 5,000 steps and copied to Drive through verified `.copying` directories. The notebook was subsequently changed at the user's request to retain every Drive checkpoint (`005000` through `040000`). Only older local copies are removed after verified Drive copies exist. The single training cell now also creates the final `pretrained_model`, reports, downloadable tar, and checksum automatically; no later copy/package cell is required.

The first 40,000-step run completed under the older two-checkpoint-retention version. Drive contains `035000` and `040000`; the usable final weights are already under `checkpoints/040000/pretrained_model`. Earlier `005000` through `030000` folders from that run were pruned and cannot be recreated without retraining.

Training for 40,000 steps is a user-selected experiment, not a guarantee of better physical behavior. Compare checkpoints offline and in shadow mode before guarded execution.

## Current objective

The user is uploading the raw physical-UR5 dataset to Google Drive and will run a separate Colab notebook to convert it to LeRobot v3. The new dataset is named `ur5_cube_30ep`, but it contains 28 accepted episodes because one of the 29 finalized recordings is invalid and the user explicitly authorized excluding it.

The canonical task text for every converted frame is:

```text
Pick up the orange cube and place it in the bowl
```

Do not carry forward the old `place it in the tray` instruction into the new dataset.

## Repository

```text
/home/grasplab/ur5_vla_ws/src/ur5-vla-dataset-recorder
```

Branch at handoff:

```text
agent/lerobot-conversion-safety
```

Latest existing commit:

```text
f6766d1 Sync complete Colab checkpoints to Drive during training
```

The working tree contains intentional uncommitted changes. Preserve them:

```text
M  docs/PHASED_DEPLOYMENT.md
M  ur5_vla_dataset_recorder/convert_lerobot.py
M  ur5_vla_dataset_recorder/policy_executor.py
?? docs/LIVE_SHADOW_INFERENCE.md
?? notebooks/convert_ur5_cube_30ep_colab.ipynb
?? test/test_shadow_inference.py
?? tools/generate_conversion_colab_notebook.py
?? ur5_vla_dataset_recorder/policy_inference_server.py
?? ur5_vla_dataset_recorder/shadow_protocol.py
```

## Raw dataset status

Raw root:

```text
/home/grasplab/ur5_vla_dataset/staging/episodes
```

Size at handoff: approximately 19 GB.

There are 29 finalized episode directories and 48,470 total raw frames. Full validation with `decode_all_media=True` produced:

```text
28 episodes PASS
1 episode FAIL
```

The rejected episode is:

```text
episode_20260822_035620
```

Failure evidence:

```text
frames: 16 -> 17
times: 0.828752576 -> 0.878377634 seconds
maximum discontinuity: elbow_joint 1.471315211 rad

absolute changes:
shoulder_pan_joint   0.294840634 rad
shoulder_lift_joint  1.101030654 rad
elbow_joint          1.471315211 rad
wrist_1_joint        0.058437655 rad
wrist_2_joint        0.631923494 rad
wrist_3_joint        0.020703548 rad
```

The episode also reports two dropped frames. Do not edit or silently repair the finalized raw episode. The user explicitly chose `exclude it and go forward`.

Expected selected result:

```text
episodes: 28
frames:   47,204
fps:      20
repo ID:  local/ur5_cube_30ep
depth:    none in converted dataset; raw lossless depth remains archived
storage:  AV1 RGB video
```

## New conversion notebook

Generated notebook:

```text
notebooks/convert_ur5_cube_30ep_colab.ipynb
```

Generator:

```text
tools/generate_conversion_colab_notebook.py
```

The notebook embeds the checked-in `core.py`, `raw_dataset.py`, `convert_lerobot.py`, and `validate_lerobot.py`, so the user only needs to upload the notebook and raw dataset. If any embedded source changes, rerun:

```bash
cd /home/grasplab/ur5_vla_ws/src/ur5-vla-dataset-recorder
python tools/generate_conversion_colab_notebook.py
```

Expected Drive layout:

```text
/content/drive/MyDrive/pHYSICAL UR5/
├── datasets/raw/ur5_cube_30ep/episodes/episode_*/
└── datasets/lerobot/ur5_cube_30ep/       # created after validation passes
```

Notebook behavior:

1. Mounts Drive.
2. Pins Torch 2.7.1, Torchvision 0.22.1, and LeRobot 0.4.3.
3. Checks for 29 finalized Drive episodes and the known rejected episode.
4. Requires at least 45 GiB free under `/content`.
5. Uses verbose `rsync` to copy raw data from Drive to `/content`.
6. Selects exactly 28 episodes by excluding only `episode_20260822_035620`.
7. Performs full raw validation inside the converter with visible per-episode progress.
8. Converts locally to `/content/ur5_cube_30ep`.
9. Validates 28 episodes, 47,204 frames, two cameras, 7D state/action, binary gripper transitions, task text, and representative video decoding.
10. Writes `validation_report.md` and `conversion_run_report.json`.
11. Copies to Drive through `ur5_cube_30ep.copying`.
12. SHA-256 verifies every copied file before renaming to the final Drive directory.
13. Loads the final Drive copy through the official LeRobot API.

The notebook refuses an existing Drive output unless the user deliberately changes:

```python
REPLACE_EXISTING_DRIVE_OUTPUT = True
```

The converter is intentionally CPU/disk intensive. Raw preflight validation and serial AV1 encoding can take a long time. Do not mistake silence for a hang; the local converter was updated to print:

```text
[preflight i/N] ...
[convert i/N] 500/M frames staged
[convert i/N] encoding base/wrist videos ...
[convert i/N] episode saved
[finalize] ...
```

No conversion process was running on the Ubuntu PC at the time of handoff. A previous partial local attempt was interrupted; `/home/grasplab/ur5_vla_dataset/lerobot/ur5_cube_30ep` did not contain completed output and is not the authoritative destination.

## Existing 10-episode converted dataset

```text
/home/grasplab/ur5_vla_dataset/lerobot/ur5_orange_cube_v1
repo ID: local/ur5_orange_cube_v1
episodes: 10
frames: 20,583
task: Pick up the orange cube and place it in the tray
```

This old dataset remains valid but uses the tray task. Do not mix its task label with the new bowl dataset except where the user explicitly overrides all 28 selected episodes to the bowl instruction during conversion.

## Existing trained checkpoint

Downloaded archive directory:

```text
/home/grasplab/Downloads/drive-download-20260823T063551Z-1-001
```

Archive:

```text
ur5_orange_cube_smolvla_5000_steps_complete_checkpoint.tar
```

The archive SHA-256 passed. Extracted complete checkpoint:

```text
/home/grasplab/Downloads/drive-download-20260823T063551Z-1-001/pretrained_model
```

It contains weights, config, train config, preprocessor/postprocessor JSON, and normalization tensors. Its schema is correct for the physical dataset:

```text
observation.images.base   3x480x640
observation.images.wrist  3x480x640
observation.state         7D
action                    7D
chunk size                50
```

Offline inference on 100 saved frames passed structurally:

```text
finite outputs: yes
gripper threshold accuracy: 100%
joint MAE: approximately 0.014–0.029 rad
```

Reports:

```text
/home/grasplab/Downloads/drive-download-20260823T063551Z-1-001/offline_validation/report_100_frames.json
/home/grasplab/Downloads/drive-download-20260823T063551Z-1-001/offline_validation/actions_100_frames.csv
```

## Live shadow inference work

Implemented a two-process, localhost-only shadow bridge:

```text
ur5_vla_dataset_recorder/policy_inference_server.py
ur5_vla_dataset_recorder/shadow_protocol.py
ur5_vla_dataset_recorder/policy_executor.py
```

The model server runs in the LeRobot environment and contains no ROS imports. The ROS process subscribes to base/wrist RGB, named joints, gripper state, and Servo status. It creates no ROS publishers. `--live` is hard-locked.

Documentation:

```text
docs/LIVE_SHADOW_INFERENCE.md
docs/PHASED_DEPLOYMENT.md
```

Live shadow inference succeeded after the robot was placed inside the broad training envelope. Inputs were fresh, gripper was open, and Servo status was zero. The model repeatedly produced finite 7D predictions, but first-action deltas were:

```text
0.052–0.125 rad
```

The configured single-target limit is 0.05 rad, so execution was rejected. No model action was ever sent to the robot. The user later asked to make deployment unrestricted; that request was refused. Do not remove the deadman, freshness, joint, velocity, workspace, controller, External Control, manual-enable, or fault-latching requirements merely to make the robot move.

The old 10-episode/5,000-step model is a pipeline experiment, not a deployment-ready controller. The user plans to train a larger dataset next.

## Python environments and compatibility

System ROS environment:

```text
ROS 2 Humble
system Python 3.10
```

Do not run ROS nodes inside Conda.

Base Conda environment at handoff:

```text
Python 3.13
Torch 2.7.1+cu126
LeRobot reports 0.4.3 from editable source:
/home/grasplab/Desktop/VLA/berk/lerobot/lerobot
```

That editable source rejects the checkpoint `use_peft` field because its local `SmolVLAConfig` differs from the checkpoint-producing version. It is adequate for dataset conversion, but do not use it to load this checkpoint.

Checkpoint-compatible environment:

```text
conda environment: pi0
Python 3.12
LeRobot 0.5.2 editable source:
/home/grasplab/Desktop/VLA_script_training/lerobot
```

The checkpoint loaded and inferred successfully in this environment.

Colab previously encountered a partially initialized Torchvision module. The working installation order is:

```text
torch==2.7.1
torchvision==0.22.1
torchaudio==2.7.1
then lerobot==0.4.3
```

Restart/delete the Colab runtime if Torch/Torchvision was imported before installing those pins.

## Tests

The shadow implementation added:

```text
test/test_shadow_inference.py
```

It covers protocol framing, finite values, BGR-to-RGB conversion, freshness/Servo gating, artifact schema matching, no publishers, and the live lock.

At the last full run before notebook creation:

```text
25 tests passed
```

The converter-specific suite passed after verbose progress was added:

```text
6 tests passed
```

Run before committing or handing off further:

```bash
cd /home/grasplab/ur5_vla_ws/src/ur5-vla-dataset-recorder
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m compileall -q ur5_vla_dataset_recorder test tools
git diff --check
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids an unrelated ROS `launch_testing` plugin importing Python 3.10 packages into the Conda Python 3.13 process.

## Recommended next-session sequence

1. Inspect and test `notebooks/convert_ur5_cube_30ep_colab.ipynb`.
2. Have the user upload raw episodes to the exact Drive input path or edit `DRIVE_RAW_ROOT` in Section A.
3. Run the notebook top-to-bottom in Colab and retain every PASS/report.
4. Confirm the final Drive dataset contains exactly 28 episodes and 47,204 frames with the bowl task.
5. Create a separate training notebook/config for `local/ur5_cube_30ep`; do not reuse hard-coded counts/task from the old 10-episode tray notebook.
6. Train locally under `/content`, saving complete resumable checkpoints to Drive.
7. Repeat offline and live-shadow evaluation. Do not unlock physical output based only on a lower training loss or larger episode count.

## Clean 20-episode retraining revision (2026-08-24)

Behavior/timing analysis found one action discontinuity, four episodes with multi-second acquisition gaps, and five episodes containing multiple task attempts (one overlaps the gap group). The exact nine exclusions and reasons are stored in `config/ur5_cube_clean_20ep.json`. Original raw files were not modified or deleted.

The resulting source selection is exactly 20 episodes / 32,022 frames, and all 20 pass raw validation. `raw_dataset.py` now rejects any adjacent raw-frame interval greater than 0.20 seconds, preventing the converter from compressing a recording pause into adjacent 20 Hz training samples.

Dedicated notebooks:

```text
notebooks/convert_ur5_cube_clean_20ep_colab.ipynb
notebooks/train_ur5_cube_clean_20ep_smolvla_40000_colab.ipynb
```

The training notebook saves and verifies every 5,000-step checkpoint directly to Drive through 40,000 steps. Physical deployment remains blocked until a checkpoint passes the artifact, offline, 100-prediction shadow, and guarded single-action gates in `docs/PHASED_DEPLOYMENT.md`.

## User communication preference observed

The user wants direct, cell-by-cell workflows with visible progress and minimal ambiguity. Clearly distinguish validation, conversion, Drive copying, training, and physical deployment. Do not use terms such as “smoke” without immediately explaining that they mean a short pipeline test rather than a separate dataset or model.
