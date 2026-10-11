"""Generate a reusable Drive raw-UR5-to-LeRobot Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


embedded_sources = {
    name: (ROOT / "ur5_vla_dataset_recorder" / name).read_text()
    for name in ("__init__.py", "core.py", "raw_dataset.py", "convert_lerobot.py", "validate_lerobot.py")
}


cells = [
    markdown(
        """
# Reusable UR5 raw episodes → LeRobot v3 converter

Use this notebook for each newly recorded physical-UR5 dataset. It discovers raw folders containing `episode_*` directories, lets you select the input, proposes a matching output name, validates every selected episode, converts locally under `/content`, validates the LeRobot result, and saves it under the Drive LeRobot folder.

It never controls ROS or the robot, never modifies the raw source, never uploads to Hugging Face, and never overwrites an existing converted dataset. Conversion is CPU/disk intensive; a GPU is not required.

Run every cell from top to bottom. Before starting, ensure `pHYSICAL UR5` is available under **My Drive** for the Colab account.
"""
    ),
    markdown(
        """
## A — Mount Drive and locate the raw/output libraries

The supplied Drive folders correspond to these mounted paths. Edit the two path lines only if your My Drive shortcut layout differs.
"""
    ),
    code(
        """
from google.colab import drive
drive.mount("/content/drive")
"""
    ),
    code(
        """
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/pHYSICAL UR5")
DRIVE_RAW_LIBRARY = DRIVE_ROOT / "datasets/raw"
DRIVE_LEROBOT_LIBRARY = DRIVE_ROOT / "datasets/lerobot"

assert DRIVE_RAW_LIBRARY.is_dir(), f"Raw library not found: {DRIVE_RAW_LIBRARY}"
assert DRIVE_LEROBOT_LIBRARY.is_dir(), f"LeRobot output library not found: {DRIVE_LEROBOT_LIBRARY}"
print("Raw library:", DRIVE_RAW_LIBRARY)
print("LeRobot library:", DRIVE_LEROBOT_LIBRARY)
"""
    ),
    markdown(
        """
## B — Select the input and describe the dataset

Choose a folder whose direct children are raw `episode_*` folders. Enter the exact task instruction demonstrated in every selected episode. The suggested output name is editable; it becomes a new folder under `datasets/lerobot`.

Optional exclusions are comma-separated exact episode folder names. Leave exclusions blank for a normal dataset. Never exclude an episode merely to hide an unexplained validation failure—inspect and document the cause first.
"""
    ),
    code(
        """
import os, re
import ipywidgets as widgets
from IPython.display import display

def discover_episode_roots(root: Path, max_depth=4):
    candidates = []
    root_depth = len(root.parts)
    for current, directories, _ in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.parts) - root_depth
        if depth > max_depth:
            directories[:] = []
            continue
        if any(name.startswith("episode_") for name in directories):
            candidates.append(current_path)
            directories[:] = [name for name in directories if not name.startswith("episode_")]
    return sorted(set(candidates), key=lambda path: str(path).lower())

def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return value or "ur5_dataset"

def proposed_output(path: Path):
    episode_count = sum(child.is_dir() and child.name.startswith("episode_") for child in path.iterdir())
    if path.name.lower() == "episodes" and path.parent != DRIVE_RAW_LIBRARY:
        base = path.parent.name
    elif path.name.lower() == "episodes":
        base = "ur5_dataset"
    else:
        base = path.name
    return f"{safe_name(base)}_{episode_count}ep"

INPUT_CANDIDATES = discover_episode_roots(DRIVE_RAW_LIBRARY)
assert INPUT_CANDIDATES, f"No folder containing episode_* directories found under {DRIVE_RAW_LIBRARY}"
choices = [(str(path.relative_to(DRIVE_RAW_LIBRARY)), str(path)) for path in INPUT_CANDIDATES]

INPUT_WIDGET = widgets.Dropdown(options=choices, description="Raw input:", layout=widgets.Layout(width="95%"))
OUTPUT_WIDGET = widgets.Text(value=proposed_output(Path(INPUT_WIDGET.value)), description="Output:", layout=widgets.Layout(width="95%"))
TASK_WIDGET = widgets.Text(value="", placeholder="Example: Pick up the orange cube and place it in the bowl", description="Task:", layout=widgets.Layout(width="95%"))
EXCLUDE_WIDGET = widgets.Text(value="", placeholder="episode_name_1, episode_name_2 (normally blank)", description="Exclude:", layout=widgets.Layout(width="95%"))

def input_changed(change):
    if change.get("name") == "value":
        OUTPUT_WIDGET.value = proposed_output(Path(change["new"]))

INPUT_WIDGET.observe(input_changed, names="value")
display(INPUT_WIDGET, OUTPUT_WIDGET, TASK_WIDGET, EXCLUDE_WIDGET)
print("Select the input, review/edit the output name, enter the task, then run the next cell.")
"""
    ),
    markdown(
        """
### Freeze and verify your selection

After this cell passes, do not change the widgets. Restart from Section B if you selected the wrong dataset or instruction.
"""
    ),
    code(
        """
DRIVE_RAW_ROOT = Path(INPUT_WIDGET.value)
OUTPUT_NAME = OUTPUT_WIDGET.value.strip()
TASK = TASK_WIDGET.value.strip()
EXCLUDED_EPISODES = sorted({item.strip() for item in EXCLUDE_WIDGET.value.split(",") if item.strip()})

assert DRIVE_RAW_ROOT in INPUT_CANDIDATES, "Input is not one of the discovered raw episode folders"
assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", OUTPUT_NAME), (
    "Output must use only letters, numbers, dot, underscore, and hyphen; no slash or spaces"
)
assert TASK, "Task instruction is required"

DRIVE_OUTPUT_ROOT = DRIVE_LEROBOT_LIBRARY / OUTPUT_NAME
DRIVE_COPYING_ROOT = DRIVE_LEROBOT_LIBRARY / f"{OUTPUT_NAME}.copying"
LOCAL_RAW_ROOT = Path("/content/raw_ur5_work/episodes")
LOCAL_OUTPUT_ROOT = Path("/content/lerobot_work") / OUTPUT_NAME
REPO_ID = f"local/{OUTPUT_NAME}"

assert not DRIVE_OUTPUT_ROOT.exists(), f"Output already exists; choose another name: {DRIVE_OUTPUT_ROOT}"
assert not DRIVE_COPYING_ROOT.exists(), f"Incomplete .copying folder exists; inspect it before retrying: {DRIVE_COPYING_ROOT}"

print("SELECTED CONFIGURATION")
print("Raw input:", DRIVE_RAW_ROOT)
print("Output:", DRIVE_OUTPUT_ROOT)
print("Repository ID:", REPO_ID)
print("Task:", TASK)
print("Excluded episodes:", EXCLUDED_EPISODES or "none")
"""
    ),
    markdown(
        """
## C — Install the pinned converter environment

If Torch or Torchvision was already imported in this runtime, use **Runtime → Disconnect and delete runtime**, reconnect, and start again. The pinned matching versions prevent the partially initialized Torchvision error previously seen in Colab.
"""
    ),
    code(
        """
import sys
assert "torch" not in sys.modules and "torchvision" not in sys.modules, (
    "Torch/Torchvision already imported. Disconnect and delete the runtime, reconnect, and restart."
)
"""
    ),
    code(
        """
%pip install -q --upgrade pip
%pip install -q --index-url https://download.pytorch.org/whl/cu126 "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1"
%pip install -q "lerobot==0.4.3" "av>=12" "PyYAML>=6" "Pillow>=10" ipywidgets
!ffmpeg -version | head -2
"""
    ),
    code(
        """
import importlib.metadata as md
import torch, torchvision, av

print("lerobot", md.version("lerobot"))
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("av", av.__version__)
assert md.version("lerobot") == "0.4.3"
assert torch.__version__.startswith("2.7.1")
assert torchvision.__version__.startswith("0.22.1")
assert torchvision.extension._has_ops(), "Torchvision native operators did not load"
print("Pinned environment PASS")
"""
    ),
    markdown(
        """
## D — Install the project validator/converter

The checked-in converter source is embedded in this notebook; no repository clone is needed.
"""
    ),
    code(
        """
import sys
from pathlib import Path

PACKAGE_PARENT = Path("/content/ur5_converter")
PACKAGE_DIR = PACKAGE_PARENT / "ur5_vla_dataset_recorder"
PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_FILES = __EMBEDDED_SOURCES__
for filename, source_text in SOURCE_FILES.items():
    (PACKAGE_DIR / filename).write_text(source_text)
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from ur5_vla_dataset_recorder.convert_lerobot import convert
from ur5_vla_dataset_recorder.raw_dataset import finalized_episode_paths
from ur5_vla_dataset_recorder.validate_lerobot import validate_dataset
print("Embedded converter installed:", PACKAGE_DIR)
""".replace("__EMBEDDED_SOURCES__", repr(embedded_sources))
    ),
    markdown(
        """
## E — Inspect and copy the raw dataset locally

Only finalized episodes are eligible. The original Drive source remains untouched. `rsync` shows progress while copying to `/content`.
"""
    ),
    code(
        """
import shutil, subprocess

drive_episodes = finalized_episode_paths(DRIVE_RAW_ROOT)
drive_names = [path.name for path in drive_episodes]
assert drive_names, f"No finalized episodes found in {DRIVE_RAW_ROOT}"
unknown_exclusions = sorted(set(EXCLUDED_EPISODES) - set(drive_names))
assert not unknown_exclusions, f"Excluded names are not finalized episodes: {unknown_exclusions}"
selected_names = [name for name in drive_names if name not in EXCLUDED_EPISODES]
assert selected_names, "All finalized episodes were excluded"

raw_bytes = sum(path.stat().st_size for path in DRIVE_RAW_ROOT.rglob("*") if path.is_file())
disk = shutil.disk_usage("/content")
minimum_free = max(45 * 2**30, raw_bytes + 25 * 2**30)
print(f"Finalized episodes: {len(drive_names)}")
print(f"Selected episodes: {len(selected_names)}")
print(f"Excluded episodes: {EXCLUDED_EPISODES or 'none'}")
print(f"Raw size: {raw_bytes / 2**30:.1f} GiB; local free: {disk.free / 2**30:.1f} GiB")
assert disk.free >= minimum_free, f"Need at least {minimum_free / 2**30:.1f} GiB free under /content"

assert not LOCAL_RAW_ROOT.exists(), f"Local raw work folder exists; restart runtime: {LOCAL_RAW_ROOT}"
LOCAL_RAW_ROOT.mkdir(parents=True)
subprocess.run(["rsync", "-a", "--info=progress2", f"{DRIVE_RAW_ROOT}/", f"{LOCAL_RAW_ROOT}/"], check=True)
local_names = [path.name for path in finalized_episode_paths(LOCAL_RAW_ROOT)]
assert local_names == drive_names, "Local raw episode list differs from Drive"
print("Raw local copy PASS:", LOCAL_RAW_ROOT)
"""
    ),
    markdown(
        """
## F — Fully validate and convert

This cell decodes all selected RGB/depth media during preflight, then converts RGB, 7D state/action, timestamps, and the chosen task to LeRobot v3. It prints per-episode and per-500-frame progress. Raw lossless depth remains archived in the source; the converted policy dataset is RGB-only.
"""
    ),
    code(
        """
assert not LOCAL_OUTPUT_ROOT.exists(), f"Local output already exists: {LOCAL_OUTPUT_ROOT}"
LOCAL_OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)

converted_episodes, converted_frames = convert(
    input_root=LOCAL_RAW_ROOT,
    output_root=LOCAL_OUTPUT_ROOT,
    repo_id=REPO_ID,
    task=TASK,
    episodes=selected_names,
    overwrite=False,
    depth="none",
    storage="video",
)
assert converted_episodes == len(selected_names)
assert converted_frames > 0
print(f"CONVERSION PASS: {converted_episodes} episodes / {converted_frames} frames")
"""
    ),
    markdown(
        """
## G — Validate the converted LeRobot dataset

This scans every state/action row and decodes representative synchronized video frames from every episode through the official LeRobot API.
"""
    ),
    code(
        """
LOCAL_VALIDATION_REPORT = LOCAL_OUTPUT_ROOT / "validation_report.md"
errors = validate_dataset(
    root=LOCAL_OUTPUT_ROOT,
    repo_id=REPO_ID,
    report=LOCAL_VALIDATION_REPORT,
    expected_episodes=converted_episodes,
    expected_frames=converted_frames,
    task=TASK,
    video_backend="pyav",
)
assert not errors, "Converted validation failed:\\n- " + "\\n- ".join(errors)
print(LOCAL_VALIDATION_REPORT.read_text())
"""
    ),
    markdown(
        """
## H — Record provenance and publish to the LeRobot Drive folder

The completed local dataset is copied first to `<name>.copying`. Every file is SHA-256 verified before that staging directory is renamed to the final output name. Existing outputs are never deleted or overwritten.
"""
    ),
    code(
        """
import datetime, hashlib, json, subprocess

conversion_report = {
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "repo_id": REPO_ID,
    "task": TASK,
    "episodes": converted_episodes,
    "frames": converted_frames,
    "fps": 20,
    "source_drive_root": str(DRIVE_RAW_ROOT),
    "excluded_episodes": EXCLUDED_EPISODES,
    "selected_source_episodes": selected_names,
    "depth_mode": "none; raw depth remains in source upload",
    "storage": "video",
    "lerobot": md.version("lerobot"),
}
(LOCAL_OUTPUT_ROOT / "conversion_run_report.json").write_text(json.dumps(conversion_report, indent=2) + "\\n")

assert not DRIVE_OUTPUT_ROOT.exists(), f"Final output appeared during conversion: {DRIVE_OUTPUT_ROOT}"
assert not DRIVE_COPYING_ROOT.exists(), f"Staging output appeared during conversion: {DRIVE_COPYING_ROOT}"
subprocess.run(["rsync", "-a", "--info=progress2", f"{LOCAL_OUTPUT_ROOT}/", f"{DRIVE_COPYING_ROOT}/"], check=True)

def file_hashes(root: Path):
    result = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        result[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return result

local_hashes = file_hashes(LOCAL_OUTPUT_ROOT)
drive_hashes = file_hashes(DRIVE_COPYING_ROOT)
assert drive_hashes == local_hashes, "Drive copy hash verification failed; final output was not published"
DRIVE_COPYING_ROOT.rename(DRIVE_OUTPUT_ROOT)
(DRIVE_OUTPUT_ROOT / "drive_copy_manifest.json").write_text(json.dumps(local_hashes, indent=2) + "\\n")
print("VERIFIED DRIVE COPY PASS:", DRIVE_OUTPUT_ROOT)
print("Files verified:", len(local_hashes))
"""
    ),
    markdown(
        """
## I — Final Drive readback

Do not train on the new dataset unless this final cell passes.
"""
    ),
    code(
        """
from lerobot.datasets.lerobot_dataset import LeRobotDataset

drive_dataset = LeRobotDataset(REPO_ID, root=DRIVE_OUTPUT_ROOT, video_backend="pyav")
assert drive_dataset.meta.total_episodes == converted_episodes
assert drive_dataset.meta.total_frames == converted_frames
assert TASK in list(drive_dataset.meta.tasks.index)
for report_name in ("validation_report.md", "conversion_run_report.json", "drive_copy_manifest.json"):
    assert (DRIVE_OUTPUT_ROOT / report_name).is_file(), report_name

print("FINAL PASS")
print("Drive dataset:", DRIVE_OUTPUT_ROOT)
print("Repository ID:", REPO_ID)
print("Episodes / frames:", converted_episodes, converted_frames)
print("Task:", TASK)
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "convert_raw_ur5_to_lerobot_colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = ROOT / "notebooks/convert_raw_ur5_to_lerobot_colab.ipynb"
output.write_text(json.dumps(notebook, indent=1) + "\n")
print(output)
