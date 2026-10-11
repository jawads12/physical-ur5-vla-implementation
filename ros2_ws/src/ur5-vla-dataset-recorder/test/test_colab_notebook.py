import ast
import json
from pathlib import Path


def test_colab_notebook_json_code_and_sections():
    path = Path(__file__).parents[1] / "notebooks/train_ur5_smolvla_colab.ipynb"
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    combined = "\n".join("".join(cell["source"]) for cell in notebook["cells"]).lower()
    required = [
        "runtime verification", "google drive", "safely extract", "20583", "smolvla_base",
        "train_steps = 5000", "train smolvla", "training complete", "trained weights saved to drive",
        "complete trained weights", "pretrained_model", "model.safetensors", "ready to download",
        "torch==2.7.1", "torchvision==0.22.1", "disconnect and delete runtime",
        "resolve_delta_timestamps", "smolvlaconfig", "training_dataset", "cfg.chunk_size",
        "sync_complete_checkpoints", "synced complete checkpoint to drive", "optimizer_state.safetensors",
        ".copying", "file_sizes(staging) == file_sizes(source)",
    ]
    assert all(value in combined for value in required)
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
            continue
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise AssertionError(f"invalid Python in notebook cell {index}: {exc}") from exc


def test_30ep_40000_step_colab_notebook_contract_and_code():
    path = Path(__file__).parents[1] / "notebooks/train_ur5_cube_30ep_smolvla_40000_colab.ipynb"
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["name"] == path.name

    combined = "\n".join("".join(cell["source"]) for cell in notebook["cells"]).lower()
    required = [
        "datasets/lerobot/ur5_cube_30ep",
        'repo_id = "local/ur5_cube_30ep"',
        "pick up the orange cube and place it in the bowl",
        "total_episodes == 28",
        "total_frames == 47204",
        "train_steps = 40000",
        "save_freq = 5000",
        '"drive_checkpoint_retention": "all"',
        "keep_local_checkpoints = 2",
        "ur5_cube_30ep_smolvla_40000_steps",
        "rsync",
        "synced complete checkpoint to drive",
        "drive checkpoint retained",
        "automatic final model copy and packaging",
        "ready to download",
        "model.safetensors",
        "optimizer_state.safetensors",
        "training_step.json",
        "push_to_hub=false",
    ]
    assert all(value in combined for value in required)
    assert "ur5_orange_cube_v1" not in combined
    assert "place it in the tray" not in combined
    assert "import rclpy" not in combined
    assert "pruned old" not in combined
    assert "keep_checkpoints" not in combined
    assert "## h — copy" not in combined
    assert "## i — package" not in combined

    training_cells = [cell for cell in notebook["cells"] if "subprocess.Popen" in "".join(cell["source"])]
    assert len(training_cells) == 1
    training_source = "".join(training_cells[0]["source"])
    assert "export_archive =" in training_source
    assert 'final_target = DRIVE_RUN_DIR / "pretrained_model"' in training_source

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
            continue
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise AssertionError(f"invalid Python in notebook cell {index}: {exc}") from exc


def test_clean_20ep_40000_step_colab_notebook_contract_and_code():
    path = Path(__file__).parents[1] / "notebooks/train_ur5_cube_clean_20ep_smolvla_40000_colab.ipynb"
    notebook = json.loads(path.read_text())
    combined = "\n".join("".join(cell["source"]) for cell in notebook["cells"]).lower()

    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["name"] == path.name
    required = [
        "datasets/lerobot/ur5_cube_clean_20ep", 'repo_id = "local/ur5_cube_clean_20ep"',
        "total_episodes == 20", "total_frames == 32022", "train_steps = 40000",
        "save_freq = 5000", '"drive_checkpoint_retention": "all"',
        "ur5_cube_clean_20ep_smolvla_40000_steps", "automatic final model copy and packaging",
        "optimizer_state.safetensors", "training_step.json", "push_to_hub=false",
    ]
    assert all(value in combined for value in required)
    assert "47204" not in combined
    assert "range(28)" not in combined
    assert "import rclpy" not in combined

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
            continue
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise AssertionError(f"invalid Python in notebook cell {index}: {exc}") from exc
