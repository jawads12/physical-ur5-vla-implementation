# Google Drive and Colab Pro Workflow

## Architecture

```text
Ubuntu robot PC                           Google Drive                         Colab Pro
record -> validate -> LeRobot v3 -> tar -> datasets/lerobot/...tar -> copy to /content
                                                                      |
                             reports/ <- validation/train reports <- validate
                             checkpoints/ <- complete checkpoints <- SmolVLA
                                                                      |
Ubuntu PC <- manual download <- deployment/downloaded_checkpoints/ <- complete tar
    |
offline inference -> ROS shadow logs -> mock testing -> guarded testing much later
```

No Hugging Face dataset/checkpoint upload is used for the first experiment. The pretrained base model may be downloaded from `lerobot/smolvla_base`; project data and trained checkpoints remain local/Drive.

## Required Drive security change

The folder `pHYSICAL UR5` was reported as **Anyone with the link: Editor**. Change it to **Restricted before uploading anything**. Explicitly add only the intended Colab Google account as Editor. In that account, add a shortcut for the shared folder inside My Drive. The notebook defaults to `/content/drive/MyDrive/pHYSICAL UR5` but requires `DRIVE_ROOT` to be reviewed and verified.

Folder: `https://drive.google.com/drive/folders/1MorO9fN1JCQ115NrBQgZCIeldwEJUOE3`

```text
pHYSICAL UR5/
├── datasets/
│   ├── raw_archives/
│   └── lerobot/
│       ├── ur5_orange_cube_v1.tar
│       ├── ur5_orange_cube_v1.tar.sha256
│       ├── ur5_orange_cube_v1.tar.metadata.sha256
│       ├── ur5_orange_cube_v1.tar.manifest.json
│       └── ur5_orange_cube_v1.tar.validation.md
├── notebooks/train_ur5_smolvla_colab.ipynb
├── checkpoints/ur5_smolvla_test/
├── reports/dataset_validation/
├── reports/training/
└── deployment/downloaded_checkpoints/
```

## Package on the Ubuntu PC

This creates an uncompressed tar so existing AV1 video is not recompressed. It refuses existing outputs unless `--overwrite` is explicitly supplied, lists and checks archive paths, extracts to a uniquely named temporary directory, loads the extracted copy through LeRobot 0.4.3, runs the official converted validator, then deletes only that temporary directory.

```bash
cd /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/ros2_ws/src/ur5-vla-dataset-recorder
conda activate ur5-lerobot
python -m ur5_vla_dataset_recorder.package_dataset \
  --dataset-root /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/lerobot/ur5_orange_cube_v1 \
  --archive /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/exports/ur5_orange_cube_v1.tar \
  --repo-id local/ur5_orange_cube_v1
```

Verify again manually:

```bash
cd /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/exports
sha256sum -c ur5_orange_cube_v1.tar.sha256
tar -tf ur5_orange_cube_v1.tar | less
```

If a connector has a 512 MiB per-file limit, preserve the tar byte-for-byte as 400 MiB parts:

```bash
python -m ur5_vla_dataset_recorder.split_archive \
  --archive /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/exports/ur5_orange_cube_v1.tar
```

Upload both `.part-NNN` files plus `.parts.sha256` and `.parts.json`. The notebook detects them, validates each part, concatenates them in lexical order under `/content`, and verifies the original tar SHA-256 before extraction.

## Manual upload — do not automate

After restricting permissions, upload these five files into `pHYSICAL UR5/datasets/lerobot/`:

```text
/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/exports/ur5_orange_cube_v1.tar
/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/exports/ur5_orange_cube_v1.tar.sha256
/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/exports/ur5_orange_cube_v1.tar.metadata.sha256
/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/exports/ur5_orange_cube_v1.tar.manifest.json
/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/exports/ur5_orange_cube_v1.tar.validation.md
```

Upload [train_ur5_smolvla_colab.ipynb](../notebooks/train_ur5_smolvla_colab.ipynb) into `pHYSICAL UR5/notebooks/`, open it with Colab Pro, choose a GPU runtime, and run top to bottom through the ten-step smoke test. Review every assertion and output. The 5,000-step and resume cells intentionally stop until their booleans are manually enabled.

## Expected notebook outputs

- Runtime and dependency report, including GPU/VRAM and PyAV/TorchCodec status.
- Dataset SHA-256 success, 10 episodes, 20,583 frames, correct features/task, finite binary gripper values, and 20 decoded representative views.
- Loader, one-batch, model-load, forward, one-step, and ten-step results with loss/time/RAM/VRAM/error information.
- Complete smoke checkpoint under `checkpoints/ur5_smolvla_test/smoke_10_steps/`.
- Resolved configuration, preflight report, inference ranges/jumps/gripper values, and demonstrated-versus-predicted plot under `reports/training/`.
- Full checkpoints under `checkpoints/ur5_smolvla_test/full_5000/` only after explicit opt-in.
- Complete deployable tar and SHA-256 under `deployment/downloaded_checkpoints/`.

Ten episodes are only a pipeline smoke test. Roughly 50 high-quality, varied but in-distribution episodes are recommended before serious evaluation. The pneumatic gripper state is an estimate of the last acknowledged SetIO command, not measured motion/contact. Raw depth remains archived but is excluded from the first RGB-only SmolVLA model.

## Resume and disconnection handling

The notebook saves numbered complete checkpoints locally and copies them to Drive. After a disconnect, remount Drive, rerun environment/dataset setup, inspect `checkpoints/ur5_smolvla_test/full_5000/`, set `RESUME_TRAINING=True`, and run the resume cell. It verifies weights/config, copies the complete checkpoint locally, extracts the recorded step, passes `--resume=true`, and refuses to silently select no checkpoint.

## Download and install a complete checkpoint on the PC

Manually download both the complete checkpoint tar and `.sha256` from `pHYSICAL UR5/deployment/downloaded_checkpoints/` to `~/Downloads/`. Then:

```bash
conda activate ur5-lerobot
cd /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/ros2_ws/src/ur5-vla-dataset-recorder
python -m ur5_vla_dataset_recorder.install_checkpoint \
  --archive ~/Downloads/step_5000_complete_checkpoint.tar \
  --checksum ~/Downloads/step_5000_complete_checkpoint.tar.sha256 \
  --output /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/checkpoints/ur5_smolvla_test
```

The installer verifies SHA-256, rejects traversal/links and accidental overwrite, requires complete model/config/processor assets, and loads the checkpoint with the installed LeRobot version before installation. `model.safetensors` alone is insufficient because model configuration, preprocessing/postprocessing, normalization statistics, feature definitions, and tokenizer/processor assets must match training.

## Offline validation and shadow mode

```bash
conda activate ur5-lerobot
cd /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/ros2_ws/src/ur5-vla-dataset-recorder
python -m ur5_vla_dataset_recorder.offline_inference \
  --dataset-root /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/lerobot/ur5_orange_cube_v1 \
  --repo-id local/ur5_orange_cube_v1 \
  --checkpoint /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/checkpoints/ur5_smolvla_test \
  --output-csv /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/checkpoints/ur5_smolvla_test/offline_actions.csv \
  --report /media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/checkpoints/ur5_smolvla_test/offline_validation.json \
  --max-frames 100 --device cuda --video-backend pyav
```

Record inference latency, peak GPU memory, predicted ranges and jumps, and compare the CSV against demonstrations before shadow mode. For shadow mode, deactivate Conda and use ROS Python 3.10 exactly as documented in `LEROBOT_PIPELINE.md`. The executor creates no publishers and `--live` remains hard-locked. The future progression remains offline -> real-sensor shadow -> comparison -> mock robot -> one bounded physical action -> a few closed-loop actions -> one guarded full attempt.
