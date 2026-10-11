import ast
import json
from pathlib import Path


def test_conversion_colab_notebook_json_code_and_contract():
    path = Path(__file__).parents[1] / "notebooks/convert_ur5_cube_30ep_colab.ipynb"
    notebook = json.loads(path.read_text())

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["colab"]["name"] == path.name

    combined = "\n".join("".join(cell["source"]) for cell in notebook["cells"]).lower()
    required = [
        "physical ur5/datasets/raw/ur5_cube_30ep/episodes",
        'drive_output_root = drive_root / "datasets/lerobot/ur5_cube_30ep"',
        "episode_20260822_035620",
        "expected_episodes = 28",
        "expected_frames = 47204",
        "pick up the orange cube and place it in the bowl",
        "lerobot==0.4.3",
        "rsync",
        "validation_report.md",
        "conversion_run_report.json",
        "drive_copy_manifest.json",
        "sha256",
        ".copying",
        "replace_existing_drive_output = false",
    ]
    assert all(value in combined for value in required)

    # Conversion is deliberately offline: this notebook must not control ROS or
    # publish a dataset/model to a remote Hugging Face repository.
    assert "import rclpy" not in combined
    assert "push_to_hub" not in combined

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


def test_generic_conversion_colab_notebook_contract_and_code():
    path = Path(__file__).parents[1] / "notebooks/convert_raw_ur5_to_lerobot_colab.ipynb"
    notebook = json.loads(path.read_text())

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["colab"]["name"] == path.name
    combined = "\n".join("".join(cell["source"]) for cell in notebook["cells"]).lower()
    required = [
        "datasets/raw",
        "datasets/lerobot",
        "discover_episode_roots",
        "input_widget = widgets.dropdown",
        "output_widget = widgets.text",
        "task_widget = widgets.text",
        "excluded_episodes",
        "finalized_episode_paths",
        "lerobot==0.4.3",
        "--info=progress2",
        "converted_episodes, converted_frames = convert",
        "validate_dataset",
        "conversion_run_report.json",
        "drive_copy_manifest.json",
        "sha256",
        ".copying",
        "assert not drive_output_root.exists()",
        "existing outputs are never deleted or overwritten",
    ]
    assert all(value in combined for value in required)
    assert "import rclpy" not in combined
    assert "push_to_hub" not in combined

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


def test_clean_20ep_conversion_notebook_contract_and_code():
    path = Path(__file__).parents[1] / "notebooks/convert_ur5_cube_clean_20ep_colab.ipynb"
    notebook = json.loads(path.read_text())
    combined = "\n".join("".join(cell["source"]) for cell in notebook["cells"]).lower()

    assert notebook["metadata"]["colab"]["name"] == path.name
    required = [
        'datasets/raw/episodes', 'datasets/lerobot/ur5_cube_clean_20ep',
        'repo_id = "local/ur5_cube_clean_20ep"', 'expected_source_episodes = 29',
        'expected_episodes = 20', 'expected_frames = 32022', 'excluded_episodes',
        'episode_20260821_020621', 'episode_20260822_035620',
        'pick up the orange cube and place it in the bowl', 'validation_report.md',
        'conversion_run_report.json', 'drive_copy_manifest.json', '.copying', 'sha256',
    ]
    assert all(value in combined for value in required)
    assert "import rclpy" not in combined
    assert "push_to_hub" not in combined

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
