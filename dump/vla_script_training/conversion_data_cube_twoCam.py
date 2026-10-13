#!/usr/bin/env python3
"""
convert_data_cube_2cam.py — Convert dataset_raw_cube into LeRobot format with TWO cameras.

Reads:
    dump/raw/dataset_raw_cube/episode_*/data.csv
    dump/raw/dataset_raw_cube/episode_*/rgb/
    dump/raw/dataset_raw_cube/episode_*/rgb_up/

Writes:
    local/ur5_cube_2cam
"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATASET_PATH = PROJECT_ROOT / "dump/raw/dataset_raw_cube"

REPO_ID = "local/ur5_cube_2cam"
ROBOT_TYPE = "ur5_isaac"
FPS = 20
TASK_STRING = "pick up the blue cube"

STATE_PREFIX = "q_"
ACTION_PREFIX = "a_"

CAMERA_SOURCES = [
    {
        "feature_key": "observation.images.camera1",
        "folder": "rgb",
        "csv_column": "rgb_file",
    },
    {
        "feature_key": "observation.images.camera2",
        "folder": "rgb_up",
        "csv_column": "rgb_up_file",
    },
]


def discover_columns(header):
    state_cols = [c for c in header if c.startswith(STATE_PREFIX)]
    action_cols = [c for c in header if c.startswith(ACTION_PREFIX)]

    assert state_cols, f"No columns starting with '{STATE_PREFIX}' in header"
    assert action_cols, f"No columns starting with '{ACTION_PREFIX}' in header"

    return state_cols, action_cols


def load_episode(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames

    assert rows and header, f"Empty CSV: {csv_path}"

    state_cols, action_cols = discover_columns(header)
    return rows, state_cols, action_cols, header


def resolve_image_path(camera_dir, csv_name):
    csv_name = str(csv_name).strip()
    direct_path = camera_dir / csv_name

    if direct_path.exists():
        return direct_path

    target_stem = Path(csv_name).stem

    for ext in [
        ".png", ".jpg", ".jpeg", ".bmp", ".webp",
        ".PNG", ".JPG", ".JPEG", ".BMP", ".WEBP"
    ]:
        candidate = camera_dir / f"{target_stem}{ext}"
        if candidate.exists():
            return candidate

    matches = [p for p in camera_dir.iterdir() if p.is_file() and p.stem == target_stem]

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"\nImage listed in CSV but not found.\n"
        f"CSV value: {csv_name}\n"
        f"Camera folder: {camera_dir}\n"
        f"Tried direct path: {direct_path}\n"
    )


def add_frame_safe(dataset, frame, task):
    try:
        dataset.add_frame(frame, task=task)
    except TypeError:
        frame["task"] = task
        dataset.add_frame(frame)


def validate_camera_sources(header):
    for cam in CAMERA_SOURCES:
        csv_column = cam["csv_column"]
        assert csv_column in header, f"CSV column not found: {csv_column}"


def main():
    print("\n" + "=" * 70)
    print("  dataset_raw_cube → LeRobot dataset with TWO cameras")
    print("=" * 70)

    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATASET_PATH}")

    episodes = sorted(RAW_DATASET_PATH.glob("episode_*"))
    assert episodes, f"No episode_* folders found in {RAW_DATASET_PATH}"

    first_rows, state_cols, action_cols, header = load_episode(episodes[0] / "data.csv")
    validate_camera_sources(header)

    print(f"Raw dataset: {RAW_DATASET_PATH}")
    print(f"Episodes: {len(episodes)}")
    print(f"State dims: {len(state_cols)} {state_cols}")
    print(f"Action dims: {len(action_cols)} {action_cols}")

    features = {}

    for cam in CAMERA_SOURCES:
        feature_key = cam["feature_key"]
        folder = cam["folder"]
        csv_column = cam["csv_column"]

        first_img_path = resolve_image_path(
            episodes[0] / folder,
            first_rows[0][csv_column],
        )

        first_img = np.array(Image.open(first_img_path).convert("RGB"))
        H, W, C = first_img.shape

        features[feature_key] = {
            "dtype": "video",
            "shape": (H, W, 3),
            "names": ["height", "width", "channel"],
        }

        print(f"\nCamera feature: {feature_key}")
        print(f"  folder: {folder}")
        print(f"  csv column: {csv_column}")
        print(f"  shape: ({H}, {W}, 3)")

    features["observation.state"] = {
        "dtype": "float32",
        "shape": (len(state_cols),),
        "names": [c.removeprefix(STATE_PREFIX) for c in state_cols],
    }

    features["action"] = {
        "dtype": "float32",
        "shape": (len(action_cols),),
        "names": [c.removeprefix(ACTION_PREFIX) for c in action_cols],
    }

    cache_dir = PROJECT_ROOT / "data/hf_lerobot_cache" / REPO_ID

    if cache_dir.exists():
        raise FileExistsError(f"Converted dataset already exists: {cache_dir}")

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=cache_dir,
        robot_type=ROBOT_TYPE,
        fps=FPS,
        features=features,
        image_writer_threads=4,
        image_writer_processes=2,
    )

    total_frames = 0

    for idx, ep_dir in enumerate(episodes):
        rows, sc, ac, ep_header = load_episode(ep_dir / "data.csv")

        assert sc == state_cols, f"State schema mismatch in {ep_dir.name}"
        assert ac == action_cols, f"Action schema mismatch in {ep_dir.name}"

        for row in rows:
            frame = {}

            for cam in CAMERA_SOURCES:
                feature_key = cam["feature_key"]
                folder = cam["folder"]
                csv_column = cam["csv_column"]

                img_path = resolve_image_path(
                    ep_dir / folder,
                    row[csv_column],
                )

                frame[feature_key] = np.array(Image.open(img_path).convert("RGB"))

            frame["observation.state"] = np.array(
                [float(row[c]) for c in state_cols],
                dtype=np.float32,
            )

            frame["action"] = np.array(
                [float(row[c]) for c in action_cols],
                dtype=np.float32,
            )

            add_frame_safe(dataset, frame, TASK_STRING)
            total_frames += 1

        dataset.save_episode()

        print(
            f"✔ Episode {idx + 1:3d}/{len(episodes)} | "
            f"{ep_dir.name} | {len(rows)} frames"
        )

    dataset.finalize()

    print("\n✅ Two-camera conversion complete")
    print(f"Episodes: {len(episodes)}")
    print(f"Frames: {total_frames}")
    print(f"Repo ID: {REPO_ID}")
    print(f"Output cache: {cache_dir}")


if __name__ == "__main__":
    main()
