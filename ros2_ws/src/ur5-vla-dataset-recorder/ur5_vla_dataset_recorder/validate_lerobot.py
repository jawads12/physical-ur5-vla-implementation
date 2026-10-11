"""Validate a converted dataset by loading it through the official LeRobot API."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .action_semantics import ABSOLUTE_ACTION_SEMANTICS, DELTA_ACTION_SEMANTICS


def validate_dataset(root: Path, repo_id: str, report: Path, expected_episodes: int | None = None,
                     expected_frames: int | None = None, task: str | None = None,
                     video_backend: str = "pyav") -> list[str]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    errors, warnings = [], []
    dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend=video_backend)
    features = dataset.meta.features
    required = {"observation.images.base", "observation.images.wrist", "observation.state", "action"}
    if not required.issubset(features):
        errors.append(f"missing features: {sorted(required - set(features))}")
    if tuple(features.get("observation.state", {}).get("shape", ())) != (7,):
        errors.append("observation.state feature is not shape (7,)")
    if tuple(features.get("action", {}).get("shape", ())) != (7,):
        errors.append("action feature is not shape (7,)")
    if expected_episodes is not None and dataset.meta.total_episodes != expected_episodes:
        errors.append(f"expected {expected_episodes} episodes, found {dataset.meta.total_episodes}")
    if expected_frames is not None and dataset.meta.total_frames != expected_frames:
        errors.append(f"expected {expected_frames} frames, found {dataset.meta.total_frames}")
    tasks = list(dataset.meta.tasks.index)
    if task is not None and task not in tasks:
        errors.append(f"task text missing: {task!r}")
    metadata_path = root / "meta" / "ur5_vla_conversion.json"
    conversion_meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    action_semantics = conversion_meta.get("action_semantics")
    if action_semantics not in {ABSOLUTE_ACTION_SEMANTICS, DELTA_ACTION_SEMANTICS}:
        errors.append("UR5 action semantics metadata missing or incorrect")

    gripper_by_episode: dict[int, list[float]] = {}
    samples = set()
    episodes = dataset.meta.episodes
    for ep in range(dataset.meta.total_episodes):
        start, stop = int(episodes[ep]["dataset_from_index"]), int(episodes[ep]["dataset_to_index"])
        samples.update({start, (start + stop - 1) // 2, stop - 1})
    # Scan scalar/vector parquet data for every row without needlessly decoding
    # both videos 20k times. Visual samples below still use official __getitem__.
    for index in range(len(dataset)):
        sample = dataset.hf_dataset[index]
        state = np.asarray(sample["observation.state"])
        action = np.asarray(sample["action"])
        if state.shape != (7,) or action.shape != (7,) or not np.isfinite(state).all() or not np.isfinite(action).all():
            errors.append(f"frame {index}: invalid state/action shape or values")
            continue
        if min(abs(float(action[6])), abs(float(action[6]) - 1.0)) > 1e-5:
            errors.append(f"frame {index}: non-binary gripper action {action[6]}")
        if action_semantics == DELTA_ACTION_SEMANTICS and np.max(np.abs(action[:6])) > 0.012566:
            errors.append(f"frame {index}: demonstrated delta exceeds hard safety envelope")
        ep = int(sample["episode_index"])
        gripper_by_episode.setdefault(ep, []).append(float(action[6]))
    for index in sorted(samples):
        sample = dataset[index]
        for key in ("observation.images.base", "observation.images.wrist"):
            image = np.asarray(sample[key])
            if image.shape not in ((3, 480, 640), (480, 640, 3)):
                errors.append(f"sample {index}: {key} has shape {image.shape}")
    for ep, values in gripper_by_episode.items():
        changes = list(zip(values, values[1:]))
        if not any(a > .5 and b < .5 for a, b in changes) or not any(a < .5 and b > .5 for a, b in changes):
            errors.append(f"episode {ep}: missing close/open transition")

    lines = ["# Converted LeRobot Dataset Validation", "", f"**Verdict: {'FAIL' if errors else 'PASS'}**", "",
             f"- Root: `{root}`", f"- Repo ID: `{repo_id}`", f"- Episodes: {dataset.meta.total_episodes}",
             f"- Frames: {dataset.meta.total_frames}", f"- FPS: {dataset.meta.fps}",
             f"- Features: `{', '.join(features)}`", f"- Tasks: `{tasks}`", f"- Sampled frames: {len(samples)}", "",
             "## Critical errors", ""]
    lines.extend([f"- {value}" for value in errors] or ["None."])
    lines += ["", "## Warnings", ""]
    lines.extend([f"- {value}" for value in warnings] or ["None."])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--task")
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    args = parser.parse_args()
    errors = validate_dataset(**vars(args))
    if errors:
        print("Validation failed:\n- " + "\n- ".join(errors))
        raise SystemExit(1)
    print(f"PASS: {args.report}")


if __name__ == "__main__":
    main()
