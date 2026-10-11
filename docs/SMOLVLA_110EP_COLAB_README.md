# SmolVLA training: 110 episodes

Notebook: `train_ur5_cube_110ep_v3_smolvla_a100_colab.ipynb`

1. Upload the notebook to Google Colab and select an A100 GPU in a fresh runtime.
2. Put the complete converted dataset folder (data, meta, videos, and other included files) at `MyDrive/pHYSICAL UR5/datasets/lerobot/ur5_cube_110ep_v3/`. The local source is `/media/grasplab/ed18b0ca-2841-4109-a97f-532234a9abb5/physical-ur5-vla-implementation/data/ur5_vla_dataset/lerobot/ur5_cube_110ep_v3`.
3. Run cells in order. Confirm the Drive mount and use the W&B login prompt, or set `USE_WANDB = False` in the configuration cell.
4. The notebook verifies the data, benchmarks batch sizes, fine-tunes `lerobot/smolvla_base`, and copies completed checkpoints and the final model to Drive.

The dataset contains 110 episodes and 285,204 frames at 20 Hz. The budget targets eight dataset passes; batch 96 gives 24,000 steps with checkpoints every 3,000 steps. Batch size is selected by the benchmark and may differ.

Completed checkpoints and the final model go under `MyDrive/pHYSICAL UR5/checkpoints/ur5_smolvla_trained/<run-name>/`. A model archive and checksum go under `MyDrive/pHYSICAL UR5/deployment/downloaded_checkpoints/`.

Local validation read every state/action row and decoded 330 paired video samples using LeRobot 0.4.3. All five offline notebook tests passed, and code cells parsed including IPython installation magics. Colab installation, GPU benchmarks, and training have not been run here. The optional plot section was not executed locally because Matplotlib is absent from the local training environment; the notebook installs it in Colab.

This notebook starts a new fine-tune, retains all completed checkpoints, and does not automatically resume an interrupted run. All 110 episodes are training data; there is no held-out evaluation split.
