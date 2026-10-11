#!/usr/bin/env python3
import shutil
from pathlib import Path
import csv

import numpy as np
from PIL import Image

# LeRobot v2.1/v3 API entrypoint (current docs)
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_columns(header):
    """
    Your CSV has columns like:
      q_<joint>, a_<joint> (or a_gripper_cmd), plus other fields like timestamps.
    We'll build:
      state_cols  = all columns starting with 'q_'
      action_cols = all columns starting with 'a_'
    """
    state_cols = [c for c in header if c.startswith("q_")]
    action_cols = [c for c in header if c.startswith("a_")]
    if not state_cols:
        raise ValueError("No state columns found. Expected columns starting with 'q_'.")
    if not action_cols:
        raise ValueError("No action columns found. Expected columns starting with 'a_'.")
    return state_cols, action_cols


def load_episode_csv(csv_path: Path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames
    if not rows or not header:
        raise ValueError(f"Empty CSV: {csv_path}")
    state_cols, action_cols = parse_columns(header)
    return rows, state_cols, action_cols


def try_add_frame(dataset, frame_dict, task_str: str):
    """
    LeRobot has supported both patterns in the wild:
      - dataset.add_frame(frame_dict, task=task_str)
      - dataset.add_frame({**frame_dict, "task": task_str})

    We'll try the explicit arg first, then fallback.
    """
    try:
        dataset.add_frame(frame_dict, task=task_str)
    except TypeError:
        # fallback: task inside frame dict
        frame_dict2 = dict(frame_dict)
        frame_dict2["task"] = task_str
        dataset.add_frame(frame_dict2)


def main():
    print("\n=== dataset_raw -> LeRobot (v2.1-style) ===\n")

    raw_root = Path(input("Enter raw dataset root (dataset_raw): ").strip()).expanduser().resolve()
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)

    out_repo_id = input(
        "Enter repo_id (HF style 'username/name' OR local name like 'local/ur5_bluecube'): "
    ).strip()

    robot_type = input("Enter robot_type string (e.g., 'ur5_isaac'): ").strip() or "ur5_isaac"
    fps = int(input("Enter fps (your logging rate, e.g., 20): ").strip() or "20")
    task_str = input("Enter task string (e.g., 'pick a blue cube'): ").strip() or "pick a blue cube"

    # detect episodes
    episodes = sorted([p for p in raw_root.glob("episode_*") if p.is_dir()])
    if not episodes:
        raise ValueError(f"No episode_* folders found under {raw_root}")

    # Read first episode to infer schema
    first_csv = episodes[0] / "data.csv"
    rows0, state_cols, action_cols = load_episode_csv(first_csv)

    # Infer image shape from first frame
    first_rgb_rel = rows0[0].get("rgb_file", None)
    if first_rgb_rel is None:
        raise ValueError("CSV must contain 'rgb_file' column.")
    first_img_path = (episodes[0] / "rgb" / first_rgb_rel).resolve()
    if not first_img_path.exists():
        raise FileNotFoundError(first_img_path)
    img0 = np.array(Image.open(first_img_path).convert("RGB"))
    H, W, C = img0.shape

    print("\nDetected:")
    print(f"  episodes      = {len(episodes)}")
    print(f"  image shape   = {(H, W, C)}")
    print(f"  state dim     = {len(state_cols)}  cols={state_cols}")
    print(f"  action dim    = {len(action_cols)} cols={action_cols}\n")

    # If output exists, delete (same behavior as the Libero example)
    # NOTE: LeRobotDataset stores under its own root cache; repo_id identifies the dataset folder.
    # If you used the same repo_id before, remove it by creating a fresh name or deleting the folder.
    # We'll attempt to delete via dataset root if it exists.
    # (LeRobotDataset.create will error if it already exists.)
    # So just tell the user clearly.
    print("IMPORTANT: If this repo_id already exists, LeRobotDataset.create will fail.\n"
          "Use a new repo_id OR delete the existing dataset folder.\n")

    # Create dataset (v2.1-style episodes) with canonical v3 key names
    # Keys align with v3 docs: observation.state, action, observation.images.<camera> :contentReference[oaicite:2]{index=2}
    dataset = LeRobotDataset.create(
        repo_id=out_repo_id,
        robot_type=robot_type,
        fps=fps,
        features={
            "observation.images.rgb": {
                "dtype": "video",
                "shape": (H, W, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_cols),),
                "names": [c.replace("q_", "", 1) for c in state_cols],
            },
            "action": {
                "dtype": "float32",
                "shape": (len(action_cols),),
                "names": [c.replace("a_", "", 1) for c in action_cols],
            },
        },
        # keep these moderate; you can increase later
        image_writer_threads=4,
        image_writer_processes=2,
    )

    total_frames = 0

    for ep_i, ep_dir in enumerate(episodes):
        csv_path = ep_dir / "data.csv"
        rgb_dir = ep_dir / "rgb"
        rows, state_cols_ep, action_cols_ep = load_episode_csv(csv_path)

        # Ensure schema consistent
        if state_cols_ep != state_cols or action_cols_ep != action_cols:
            raise ValueError(
                f"Schema mismatch in {ep_dir.name}.\n"
                f"Expected state_cols={state_cols}\nGot={state_cols_ep}\n"
                f"Expected action_cols={action_cols}\nGot={action_cols_ep}\n"
            )

        for r in rows:
            rgb_file = r["rgb_file"]
            img_path = rgb_dir / rgb_file
            if not img_path.exists():
                raise FileNotFoundError(f"Missing image: {img_path}")

            img = np.array(Image.open(img_path).convert("RGB"))

            state = np.array([float(r[c]) for c in state_cols], dtype=np.float32)
            act = np.array([float(r[c]) for c in action_cols], dtype=np.float32)

            frame = {
                "observation.images.rgb": img,
                "observation.state": state,
                "action": act,
            }

            # Make sure task is present (this prevents your “Missing features: {'task'}” issue)
            try_add_frame(dataset, frame, task_str)

            total_frames += 1

        dataset.save_episode()
        print(f"[OK] saved episode {ep_i+1}/{len(episodes)}: {ep_dir.name}  frames={len(rows)}")

    # Must finalize before loading/pushing, otherwise parquet can be invalid :contentReference[oaicite:3]{index=3}
    dataset.finalize()
    print(f"\nDONE. Wrote {len(episodes)} episodes, {total_frames} total frames.")
    print("Next: convert v2.1 -> v3.0 using the official converter command (below).")


if __name__ == "__main__":
    main()

