"""Generate the self-contained Drive raw-to-LeRobot conversion notebook."""

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
# Convert UR5 raw episodes to LeRobot v3 on Google Colab

This notebook performs one linear, restartable workflow:

1. Mount Google Drive.
2. Copy the uploaded raw episode folder from Drive to `/content`.
3. Exclude the one known-invalid episode.
4. Fully validate all 28 selected raw episodes with visible progress.
5. Convert RGB, 7D state, 7D absolute action, timestamps, and task text to LeRobot v3.
6. Validate the converted dataset through the official LeRobot API.
7. Hash-copy the completed dataset into a separate Drive output folder.

It never contacts ROS, never controls the robot, never uploads to Hugging Face, and never converts directly on mounted Drive. Run every cell from top to bottom. A GPU is not required; conversion is CPU/disk intensive and may take a long time.

Expected raw upload layout:

```text
MyDrive/pHYSICAL UR5/datasets/raw/ur5_cube_30ep/episodes/
├── episode_20260821_012008/
├── ...
└── episode_20260822_035749/
```
"""
    ),
    markdown(
        """
## A — Mount Drive and configure paths

Edit only `DRIVE_ROOT` or `DRIVE_RAW_ROOT` if your Drive shortcut/upload layout differs. The final Drive output is refused if it already exists unless you deliberately set `REPLACE_EXISTING_DRIVE_OUTPUT = True`.
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
DRIVE_RAW_ROOT = DRIVE_ROOT / "datasets/raw/ur5_cube_30ep/episodes"
DRIVE_OUTPUT_ROOT = DRIVE_ROOT / "datasets/lerobot/ur5_cube_30ep"
DRIVE_COPYING_ROOT = DRIVE_OUTPUT_ROOT.with_name(f"{DRIVE_OUTPUT_ROOT.name}.copying")

LOCAL_RAW_ROOT = Path("/content/ur5_cube_30ep_raw/episodes")
LOCAL_OUTPUT_ROOT = Path("/content/ur5_cube_30ep")

REPO_ID = "local/ur5_cube_30ep"
TASK = "Pick up the orange cube and place it in the bowl"
EXCLUDED_EPISODE = "episode_20260822_035620"
EXPECTED_EPISODES = 28
EXPECTED_FRAMES = 47204
REPLACE_EXISTING_DRIVE_OUTPUT = False

assert DRIVE_ROOT.is_dir(), f"Drive root not found: {DRIVE_ROOT}"
assert DRIVE_RAW_ROOT.is_dir(), f"Raw episodes folder not found: {DRIVE_RAW_ROOT}"
print("Raw input:", DRIVE_RAW_ROOT)
print("Final output:", DRIVE_OUTPUT_ROOT)
print("Task:", TASK)
"""
    ),
    markdown(
        """
## B — Install the pinned conversion environment

Use the same LeRobot generation as the physical dataset pipeline. The explicit PyTorch/Torchvision pair prevents the partially initialized Torchvision error previously seen in Colab. If Torch or Torchvision was already imported in this runtime, use **Runtime → Disconnect and delete runtime**, reconnect, and start again.
"""
    ),
    code(
        """
import sys
assert "torch" not in sys.modules and "torchvision" not in sys.modules, (
    "Torch/Torchvision was already imported. Disconnect and delete the runtime, reconnect, and restart."
)
"""
    ),
    code(
        """
%pip install -q --upgrade pip
%pip install -q --index-url https://download.pytorch.org/whl/cu126 "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1"
%pip install -q "lerobot==0.4.3" "av>=12" "PyYAML>=6" "Pillow>=10"
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
## C — Install the project converter inside the runtime

The notebook contains the checked-in validator/converter source, so no repository clone or extra Drive upload is required.
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
for filename, source in SOURCE_FILES.items():
    (PACKAGE_DIR / filename).write_text(source)

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
## D — Inspect Drive input and local disk

The Drive folder must contain exactly 29 finalized episodes before exclusion. The known-invalid episode is retained in Drive but is never copied into the selected conversion set.
"""
    ),
    code(
        """
import shutil

drive_episodes = finalized_episode_paths(DRIVE_RAW_ROOT)
drive_names = [path.name for path in drive_episodes]
assert len(drive_episodes) == 29, f"Expected 29 finalized Drive episodes, found {len(drive_episodes)}"
assert EXCLUDED_EPISODE in drive_names, f"Known excluded episode missing: {EXCLUDED_EPISODE}"

selected_names = [name for name in drive_names if name != EXCLUDED_EPISODE]
assert len(selected_names) == EXPECTED_EPISODES

disk = shutil.disk_usage("/content")
print(f"Drive finalized episodes: {len(drive_episodes)}")
print(f"Selected episodes: {len(selected_names)}")
print(f"Excluded: {EXCLUDED_EPISODE}")
print(f"Local free space: {disk.free / 2**30:.1f} GiB")
assert disk.free >= 45 * 2**30, "At least 45 GiB of free /content storage is required"
"""
    ),
    markdown(
        """
## E — Copy raw episodes from Drive to `/content`

This is a local working copy. The original Drive upload is never modified. `rsync` displays transfer progress.
"""
    ),
    code(
        """
import shutil, subprocess

def reset_local_directory(path: Path):
    resolved = path.resolve()
    assert resolved != Path("/content") and Path("/content") in resolved.parents, resolved
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)

reset_local_directory(LOCAL_RAW_ROOT)
subprocess.run(
    ["rsync", "-a", "--info=progress2", f"{DRIVE_RAW_ROOT}/", f"{LOCAL_RAW_ROOT}/"],
    check=True,
)

local_episodes = finalized_episode_paths(LOCAL_RAW_ROOT)
assert [path.name for path in local_episodes] == drive_names, "Local raw copy episode list differs from Drive"
print("Raw copy complete:", LOCAL_RAW_ROOT)
"""
    ),
    markdown(
        """
## F — Full raw validation and conversion

This cell is intentionally verbose. First it prints one `preflight` result per episode while decoding every RGB/depth file. After all 28 pass, it prints every 500 staged frames and each base/wrist AV1 encoding stage. Do not close the browser or interrupt the runtime.

The output is RGB-only for SmolVLA/LeRobot policies. Lossless raw depth remains preserved in the Drive raw upload.
"""
    ),
    code(
        """
if LOCAL_OUTPUT_ROOT.exists():
    resolved = LOCAL_OUTPUT_ROOT.resolve()
    assert resolved != Path("/content") and Path("/content") in resolved.parents
    shutil.rmtree(resolved)

count, frames = convert(
    input_root=LOCAL_RAW_ROOT,
    output_root=LOCAL_OUTPUT_ROOT,
    repo_id=REPO_ID,
    task=TASK,
    episodes=selected_names,
    overwrite=False,
    depth="none",
    storage="video",
)

assert count == EXPECTED_EPISODES, (count, EXPECTED_EPISODES)
assert frames == EXPECTED_FRAMES, (frames, EXPECTED_FRAMES)
print(f"CONVERSION PASS: {count} episodes / {frames} frames")
"""
    ),
    markdown(
        """
## G — Validate the converted LeRobot dataset

This scans every state/action row and decodes the beginning, middle, and end views from every episode through the official LeRobot API.
"""
    ),
    code(
        """
LOCAL_VALIDATION_REPORT = LOCAL_OUTPUT_ROOT / "validation_report.md"
errors = validate_dataset(
    root=LOCAL_OUTPUT_ROOT,
    repo_id=REPO_ID,
    report=LOCAL_VALIDATION_REPORT,
    expected_episodes=EXPECTED_EPISODES,
    expected_frames=EXPECTED_FRAMES,
    task=TASK,
    video_backend="pyav",
)
assert not errors, "Converted validation failed:\\n- " + "\\n- ".join(errors)
print(LOCAL_VALIDATION_REPORT.read_text())
"""
    ),
    markdown(
        """
## H — Record provenance and copy the completed dataset to Drive

The notebook copies into `ur5_cube_30ep.copying`, hashes every file locally and on Drive, and renames the directory only after exact verification. A pre-existing final output is refused by default.
"""
    ),
    code(
        """
import datetime, hashlib, json, os, shutil, subprocess

conversion_report = {
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "repo_id": REPO_ID,
    "task": TASK,
    "episodes": EXPECTED_EPISODES,
    "frames": EXPECTED_FRAMES,
    "fps": 20,
    "source_drive_root": str(DRIVE_RAW_ROOT),
    "excluded_episode": EXCLUDED_EPISODE,
    "excluded_reason": "arm action jump 1.471315 rad between raw frames 16 and 17",
    "selected_source_episodes": selected_names,
    "depth_mode": "none; raw depth remains in source upload",
    "storage": "video",
    "lerobot": md.version("lerobot"),
}
(LOCAL_OUTPUT_ROOT / "conversion_run_report.json").write_text(json.dumps(conversion_report, indent=2) + "\\n")

def safe_drive_remove(path: Path):
    resolved = path.resolve()
    assert resolved.name in {"ur5_cube_30ep", "ur5_cube_30ep.copying"}
    assert "datasets/lerobot" in str(resolved)
    if resolved.exists():
        shutil.rmtree(resolved)

if DRIVE_OUTPUT_ROOT.exists() or DRIVE_COPYING_ROOT.exists():
    assert REPLACE_EXISTING_DRIVE_OUTPUT, (
        "Drive output or .copying folder exists. Inspect it first; set "
        "REPLACE_EXISTING_DRIVE_OUTPUT=True only to replace this exact dataset."
    )
    safe_drive_remove(DRIVE_OUTPUT_ROOT)
    safe_drive_remove(DRIVE_COPYING_ROOT)

DRIVE_COPYING_ROOT.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(
    ["rsync", "-a", "--info=progress2", f"{LOCAL_OUTPUT_ROOT}/", f"{DRIVE_COPYING_ROOT}/"],
    check=True,
)

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
assert drive_hashes == local_hashes, "Drive copy hash verification failed"

DRIVE_COPYING_ROOT.rename(DRIVE_OUTPUT_ROOT)
(DRIVE_OUTPUT_ROOT / "drive_copy_manifest.json").write_text(json.dumps(local_hashes, indent=2) + "\\n")
print("DRIVE COPY PASS:", DRIVE_OUTPUT_ROOT)
print("Files verified:", len(local_hashes))
"""
    ),
    markdown(
        """
## I — Final Drive verification

Do not train until this cell passes. The Drive output is loaded directly only for final verification; training should later copy it back to `/content` first.
"""
    ),
    code(
        """
from lerobot.datasets.lerobot_dataset import LeRobotDataset

drive_dataset = LeRobotDataset(REPO_ID, root=DRIVE_OUTPUT_ROOT, video_backend="pyav")
assert drive_dataset.meta.total_episodes == EXPECTED_EPISODES
assert drive_dataset.meta.total_frames == EXPECTED_FRAMES
assert TASK in list(drive_dataset.meta.tasks.index)
assert (DRIVE_OUTPUT_ROOT / "validation_report.md").is_file()
assert (DRIVE_OUTPUT_ROOT / "conversion_run_report.json").is_file()
assert (DRIVE_OUTPUT_ROOT / "drive_copy_manifest.json").is_file()

print("FINAL PASS")
print("Drive dataset:", DRIVE_OUTPUT_ROOT)
print("Repository ID:", REPO_ID)
print("Episodes / frames:", EXPECTED_EPISODES, EXPECTED_FRAMES)
print("Task:", TASK)
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "convert_ur5_cube_30ep_colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = ROOT / "notebooks/convert_ur5_cube_30ep_colab.ipynb"
output.write_text(json.dumps(notebook, indent=1) + "\n")
print(output)
