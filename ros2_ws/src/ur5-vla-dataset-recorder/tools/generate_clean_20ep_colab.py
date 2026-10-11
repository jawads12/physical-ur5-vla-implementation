"""Generate dedicated clean-dataset conversion and 40k SmolVLA notebooks."""

from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "config/ur5_cube_clean_20ep.json").read_text())


def text(cell: dict) -> str:
    return "".join(cell["source"])


def set_text(cell: dict, value: str) -> None:
    cell["source"] = value.strip().splitlines(keepends=True)


def find_cell(notebook: dict, marker: str) -> dict:
    cells = [cell for cell in notebook["cells"] if marker in text(cell)]
    if len(cells) != 1:
        raise RuntimeError(f"Expected one cell containing {marker!r}, found {len(cells)}")
    return cells[0]


# Refresh the source notebooks first so their embedded validator and training logic
# exactly match the checked-in project sources.
runpy.run_path(str(ROOT / "tools/generate_conversion_colab_notebook.py"))
runpy.run_path(str(ROOT / "tools/generate_30ep_training_colab.py"))

excluded = MANIFEST["excluded_episodes"]
excluded_literal = repr(excluded)

conversion_path = ROOT / "notebooks/convert_ur5_cube_30ep_colab.ipynb"
conversion = json.loads(conversion_path.read_text())
set_text(
    conversion["cells"][0],
    """
# Build `ur5_cube_clean_20ep` from the physical UR5 raw archive

This notebook preserves the original 29 raw episodes and creates a separate clean LeRobot v3 dataset. It excludes the nine episodes recorded with an unsafe action discontinuity, multi-second acquisition gaps, or multiple pick/place attempts. The exact exclusions and reasons are recorded in the output provenance report.

Run every cell from top to bottom. The selected 20 episodes are fully decoded and validated before conversion. The result is hash-verified before it becomes visible in the Drive LeRobot folder. This notebook never contacts ROS or controls the robot.
""",
)

config = find_cell(conversion, 'DRIVE_ROOT = Path("/content/drive/MyDrive/pHYSICAL UR5")')
set_text(
    config,
    f'''
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/pHYSICAL UR5")
DRIVE_RAW_ROOT = DRIVE_ROOT / "datasets/raw/episodes"
DRIVE_OUTPUT_ROOT = DRIVE_ROOT / "datasets/lerobot/ur5_cube_clean_20ep"
DRIVE_COPYING_ROOT = DRIVE_OUTPUT_ROOT.with_name(f"{{DRIVE_OUTPUT_ROOT.name}}.copying")

LOCAL_RAW_ROOT = Path("/content/ur5_cube_clean_20ep_raw/episodes")
LOCAL_OUTPUT_ROOT = Path("/content/ur5_cube_clean_20ep")

REPO_ID = "{MANIFEST['repo_id']}"
TASK = "{MANIFEST['task']}"
EXCLUDED_EPISODES = {excluded_literal}
EXPECTED_SOURCE_EPISODES = {MANIFEST['expected_source_episodes']}
EXPECTED_EPISODES = {MANIFEST['expected_selected_episodes']}
EXPECTED_FRAMES = {MANIFEST['expected_selected_frames']}
REPLACE_EXISTING_DRIVE_OUTPUT = False

assert DRIVE_ROOT.is_dir(), f"Drive root not found: {{DRIVE_ROOT}}"
assert DRIVE_RAW_ROOT.is_dir(), f"Raw episodes folder not found: {{DRIVE_RAW_ROOT}}"
print("Raw input:", DRIVE_RAW_ROOT)
print("Final output:", DRIVE_OUTPUT_ROOT)
print("Task:", TASK)
''',
)

inspect = find_cell(conversion, "drive_episodes = finalized_episode_paths(DRIVE_RAW_ROOT)")
set_text(
    inspect,
    """
import shutil

drive_episodes = finalized_episode_paths(DRIVE_RAW_ROOT)
drive_names = [path.name for path in drive_episodes]
assert len(drive_episodes) == EXPECTED_SOURCE_EPISODES, (
    f"Expected {EXPECTED_SOURCE_EPISODES} finalized Drive episodes, found {len(drive_episodes)}"
)
missing_exclusions = set(EXCLUDED_EPISODES) - set(drive_names)
assert not missing_exclusions, f"Excluded episodes missing from source: {sorted(missing_exclusions)}"

selected_names = [name for name in drive_names if name not in EXCLUDED_EPISODES]
assert len(selected_names) == EXPECTED_EPISODES, (len(selected_names), EXPECTED_EPISODES)

disk = shutil.disk_usage("/content")
print(f"Drive finalized episodes: {len(drive_episodes)}")
print(f"Selected episodes: {len(selected_names)}")
print("Excluded episodes and reasons:")
for name, reason in EXCLUDED_EPISODES.items():
    print(f"  - {name}: {reason}")
print(f"Local free space: {disk.free / 2**30:.1f} GiB")
assert disk.free >= 35 * 2**30, "At least 35 GiB of free /content storage is required"
""",
)

for cell in conversion["cells"]:
    value = text(cell)
    value = value.replace("ur5_cube_30ep", "ur5_cube_clean_20ep")
    value = value.replace("all 28 selected raw episodes", "all 20 selected raw episodes")
    value = value.replace("After all 28 pass", "After all 20 pass")
    value = value.replace("exactly 29 finalized episodes before exclusion", "exactly 29 finalized episodes before exclusion")
    set_text(cell, value)

# Restore the raw input path changed by the broad dataset-name replacement.
config = find_cell(conversion, 'DRIVE_RAW_ROOT = DRIVE_ROOT / "datasets/raw/episodes"')

provenance = find_cell(conversion, "conversion_report = {")
provenance_text = text(provenance)
provenance_text = provenance_text.replace(
    '"excluded_episode": EXCLUDED_EPISODE,\n    "excluded_reason": "arm action jump 1.471315 rad between raw frames 16 and 17",',
    '"excluded_episodes": EXCLUDED_EPISODES,\n    "selection_manifest": "config/ur5_cube_clean_20ep.json",',
)
provenance_text = provenance_text.replace(
    'assert resolved.name in {"ur5_cube_clean_20ep", "ur5_cube_clean_20ep.copying"}',
    'assert resolved.name in {"ur5_cube_clean_20ep", "ur5_cube_clean_20ep.copying"}',
)
set_text(provenance, provenance_text)

conversion["metadata"]["colab"]["name"] = "convert_ur5_cube_clean_20ep_colab.ipynb"
conversion_output = ROOT / "notebooks/convert_ur5_cube_clean_20ep_colab.ipynb"
conversion_output.write_text(json.dumps(conversion, indent=1) + "\n")

training_path = ROOT / "notebooks/train_ur5_cube_30ep_smolvla_40000_colab.ipynb"
training = json.loads(training_path.read_text())
for cell in training["cells"]:
    value = text(cell)
    value = value.replace("ur5_cube_30ep", "ur5_cube_clean_20ep")
    value = value.replace("28 episodes", "20 clean episodes")
    value = value.replace("28-accepted-episode", "20-clean-episode")
    value = value.replace("total_episodes == 28 and dataset.meta.total_frames == 47204", "total_episodes == 20 and dataset.meta.total_frames == 32022")
    value = value.replace("plt.subplots(28, 2", "plt.subplots(20, 2")
    value = value.replace("range(28)", "range(20)")
    value = value.replace('"dataset_episodes": 28, "dataset_frames": 47204', '"dataset_episodes": 20, "dataset_frames": 32022')
    set_text(cell, value)

training["metadata"]["colab"]["name"] = "train_ur5_cube_clean_20ep_smolvla_40000_colab.ipynb"
training_output = ROOT / "notebooks/train_ur5_cube_clean_20ep_smolvla_40000_colab.ipynb"
training_output.write_text(json.dumps(training, indent=1) + "\n")

print(conversion_output)
print(training_output)
