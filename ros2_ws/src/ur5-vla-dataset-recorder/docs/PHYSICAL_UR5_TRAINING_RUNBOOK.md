# Physical UR5 SmolVLA Training Instructions

## What this notebook does

The notebook is a single cell-by-cell workflow:

1. Mount Google Drive.
2. Reassemble and verify the converted LeRobot dataset.
3. Load and validate the 10 episodes and 20,583 frames.
4. Display synchronized base-camera and wrist-camera examples.
5. Load `lerobot/smolvla_base` and verify that one batch is compatible.
6. Fine-tune SmolVLA once for the selected number of steps, syncing each complete checkpoint to Drive.
7. Create a convenient final trained-weights directory in Drive.
8. Create a downloadable checkpoint archive and SHA-256 checksum.

The notebook performs one training run using the value in `TRAIN_STEPS`.

## Drive locations

Project folder: `https://drive.google.com/drive/folders/1MorO9fN1JCQ115NrBQgZCIeldwEJUOE3`

- Notebook: `notebooks/train_ur5_smolvla_colab.ipynb`
- Converted dataset parts: `datasets/lerobot/ur5_orange_cube_v1.tar.part-000` through `.part-006`
- Dataset checksums: `datasets/lerobot/ur5_orange_cube_v1.tar.sha256` and `.parts.sha256`
- Trained policy: `checkpoints/ur5_smolvla_trained/<RUN_NAME>/pretrained_model/`
- Resumable numbered checkpoints: `checkpoints/ur5_smolvla_trained/<RUN_NAME>/checkpoints/<STEP>/`
- Training reports: `reports/training/`
- Downloadable checkpoint: `deployment/downloaded_checkpoints/<RUN_NAME>_complete_checkpoint.tar`

The seven uploaded dataset parts total 634,357,760 bytes. The reconstructed archive must have SHA-256:

`d9edefcf4fe9feb16d828bd31b691f2c3f04165bd6f54e9d6518e9880e7b1ec0`

## Run it in Colab

1. In Drive, open `notebooks/train_ur5_smolvla_colab.ipynb` with Google Colaboratory.
2. Select **Runtime → Change runtime type → GPU**.
3. Run every cell in order from Section A to Section I.
4. In Section E, leave `TRAIN_STEPS = 5000` for the prepared run or change it before continuing.
5. Keep the Colab tab connected while Section G trains.
6. During training, confirm `SYNCED COMPLETE CHECKPOINT TO DRIVE` appears after completed checkpoint intervals, then wait for `TRAINING COMPLETE`.
7. Run Sections H and I. Wait for `TRAINED WEIGHTS SAVED TO DRIVE` and `READY TO DOWNLOAD`.

### Torchvision partially initialized error

If Colab reports `torchvision ... has no attribute 'extension'`, the runtime imported Torch before pip replaced Torch/Torchvision. This is an environment-state error, not a dataset error.

1. Close any older copy of the notebook and reopen the current Drive notebook.
2. Select **Runtime → Disconnect and delete runtime**.
3. Reconnect with a GPU.
4. Run again from Section A without running cells out of order.

The current notebook installs the official matching pair `torch==2.7.1` and `torchvision==0.22.1` before either package is imported. Section C stops immediately if Torch, Torchvision native operators, or CUDA cannot load.

## Training configuration

- Base model: `lerobot/smolvla_base`
- LeRobot: 0.4.3
- Dataset: validated LeRobot v3 physical-UR5 dataset
- Task: “Pick up the orange cube and place it in the tray”
- Inputs: base D435i RGB, wrist D435 RGB, six measured UR5 joints, estimated pneumatic command state
- Outputs: six absolute joint targets and one binary gripper command
- Cameras: the policy configuration is derived from the dataset and retains `observation.images.base` and `observation.images.wrist`
- Batch size: 1
- Training steps: 5,000 by default
- Checkpoint interval: 500 steps
- Mixed precision: enabled
- Hugging Face upload: disabled
- WandB: disabled
- Training storage: local Colab `/content`; each complete numbered checkpoint and its training state are copied to Drive during training

## Hybrid checkpoint protection

Training computations and active files remain under `/content` for speed. At each 500-step interval, Section G waits for LeRobot to finish the model and optimizer state, copies the entire numbered checkpoint into a hidden `.copying` folder on Drive, verifies the copied file-size manifest, and renames it into place. An interrupted partial copy is never presented as a complete checkpoint.

Each checkpoint is approximately 1.5 GB, so a 5,000-step run with ten saved checkpoints needs roughly 15 GB of Drive capacity. Section H still runs after training to create the final `pretrained_model` path used by Section I.

## What counts as the trained weights

Use the complete `pretrained_model` directory, not only `model.safetensors`. It contains the model weights plus configuration, preprocessing/postprocessing, normalization information, feature definitions, and processor/tokenizer assets required to reload SmolVLA.

The `.tar` archive created in Section I contains this complete directory. Verify it with the adjacent `.sha256` file after downloading it to the Ubuntu robot PC.

### Camera feature mismatch

The converted dataset intentionally retains meaningful camera names (`base` and `wrist`). The compatibility cell constructs a fresh SmolVLA training configuration with `pretrained_path=lerobot/smolvla_base`, allowing LeRobot to derive those two input features from the dataset before it loads the pretrained weights. It also resolves the policy delta timestamps so the check uses a 50-action training chunk rather than an invalid single action. Do not replace this with `PreTrainedConfig.from_pretrained(...)`, which imports the base checkpoint's generic camera names too early.

## Physical robot boundary

Training does not command the robot. First load the checkpoint for offline inference, compare predictions with recorded demonstrations, and use ROS shadow mode. The pneumatic gripper state is estimated command history rather than measured finger position, force, contact, or grasp success.
