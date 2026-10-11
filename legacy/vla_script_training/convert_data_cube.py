#!/usr/bin/env python3
"""
convert_data_cube.py — Convert dataset_raw_cube into LeRobot v2.1 HuggingFace format.

Reads:  legacy/raw/dataset_raw_cube/episode_*/data.csv + rgb/
Writes: data/hf_lerobot_cache/local/ur5_cube in this repository.

Usage:
    python convert_data_cube.py
"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ═══════════════════════════════════════════════════════════════
# Configuration — edit these if your paths or task change
# ═══════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATASET_PATH = PROJECT_ROOT / "legacy/raw/dataset_raw_cube"
REPO_ID          = "local/ur5_cube"
ROBOT_TYPE       = "ur5_isaac"
FPS              = 20
TASK_STRING      = "pick up the blue cube"

# Column prefixes in data.csv
STATE_PREFIX  = "q_"   # joint-state columns
ACTION_PREFIX = "a_"   # action columns


def discover_columns(header: list[str]):
    """Return ordered lists of state and action column names."""
    state_cols  = [c for c in header if c.startswith(STATE_PREFIX)]
    action_cols = [c for c in header if c.startswith(ACTION_PREFIX)]
    assert state_cols,  f"No columns starting with '{STATE_PREFIX}' in header"
    assert action_cols, f"No columns starting with '{ACTION_PREFIX}' in header"
    return state_cols, action_cols


def load_episode(csv_path: Path):
    """Load a single episode CSV and return rows + column lists."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)
        header = reader.fieldnames
    assert rows and header, f"Empty CSV: {csv_path}"
    state_cols, action_cols = discover_columns(header)
    return rows, state_cols, action_cols


def add_frame_safe(dataset, frame: dict, task: str):
    """Add a frame with task string (handles API variants)."""
    try:
        dataset.add_frame(frame, task=task)
    except TypeError:
        frame["task"] = task
        dataset.add_frame(frame)


def main():
    print("\n" + "=" * 60)
    print("  dataset_raw_cube  →  LeRobot HuggingFace dataset")
    print("=" * 60)

    # ── Validate raw dataset ──────────────────────────────────
    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATASET_PATH}")

    episodes = sorted(RAW_DATASET_PATH.glob("episode_*"))
    assert episodes, f"No episode_* folders in {RAW_DATASET_PATH}"

    # ── Infer schema from first episode ───────────────────────
    first_rows, state_cols, action_cols = load_episode(episodes[0] / "data.csv")
    first_img = np.array(
        Image.open(episodes[0] / "rgb" / first_rows[0]["rgb_file"]).convert("RGB")
    )
    H, W, C = first_img.shape

    print(f"  Episodes:     {len(episodes)}")
    print(f"  Image shape:  ({H}, {W}, {C})")
    print(f"  State dims:   {len(state_cols)}  {state_cols}")
    print(f"  Action dims:  {len(action_cols)} {action_cols}")
    print(f"  Output:       {REPO_ID}\n")

    # ── Delete existing dataset if present ────────────────────
    cache_dir = PROJECT_ROOT / "data/hf_lerobot_cache" / REPO_ID
    if cache_dir.exists():
        raise FileExistsError(f"Converted dataset already exists: {cache_dir}")

    # ── Create LeRobot dataset ────────────────────────────────
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=cache_dir,
        robot_type=ROBOT_TYPE,
        fps=FPS,
        features={
            "observation.images.rgb": {
                "dtype": "video",
                "shape": (H, W, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_cols),),
                "names": [c.removeprefix(STATE_PREFIX) for c in state_cols],
            },
            "action": {
                "dtype": "float32",
                "shape": (len(action_cols),),
                "names": [c.removeprefix(ACTION_PREFIX) for c in action_cols],
            },
        },
        image_writer_threads=4,
        image_writer_processes=2,
    )

    # ── Populate episodes ─────────────────────────────────────
    total_frames = 0
    for idx, ep_dir in enumerate(episodes):
        rows, sc, ac = load_episode(ep_dir / "data.csv")
        assert sc == state_cols and ac == action_cols, (
            f"Schema mismatch in {ep_dir.name}"
        )

        for row in rows:
            img_path = ep_dir / "rgb" / row["rgb_file"]
            assert img_path.exists(), f"Missing image: {img_path}"

            frame = {
                "observation.images.rgb": np.array(
                    Image.open(img_path).convert("RGB")
                ),
                "observation.state": np.array(
                    [float(row[c]) for c in state_cols], dtype=np.float32
                ),
                "action": np.array(
                    [float(row[c]) for c in action_cols], dtype=np.float32
                ),
            }
            add_frame_safe(dataset, frame, TASK_STRING)
            total_frames += 1

        dataset.save_episode()
        print(f"  ✔ Episode {idx + 1:3d}/{len(episodes)}  "
              f"{ep_dir.name}  ({len(rows)} frames)")

    # ── Finalize ──────────────────────────────────────────────
    dataset.finalize()
    print(f"\n✅ Done — {len(episodes)} episodes, {total_frames} frames → {REPO_ID}")


if __name__ == "__main__":
    main()
