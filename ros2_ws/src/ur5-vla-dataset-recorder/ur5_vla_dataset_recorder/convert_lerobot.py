"""Convert validated staging episodes to a local LeRobotDataset v3 dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .core import UR5_JOINT_ORDER
from .action_semantics import ABSOLUTE_ACTION_SEMANTICS, DELTA_ACTION_SEMANTICS, absolute_to_delta
from .raw_dataset import canonical_state_action, finalized_episode_paths, load_depth_m, load_rgb, validate_raw_episode


def feature_schema(depth: str = "none", storage: str = "video", action_mode: str = "absolute") -> dict:
    names = [*UR5_JOINT_ORDER, "estimated_gripper_state"]
    if action_mode not in {"absolute", "delta"}:
        raise ValueError(f"unsupported action mode: {action_mode}")
    action_names = [
        *([*UR5_JOINT_ORDER] if action_mode == "absolute" else [f"delta_{name}" for name in UR5_JOINT_ORDER]),
        "gripper_command",
    ]
    features = {
        "observation.images.base": {"dtype": storage, "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
        "observation.images.wrist": {"dtype": storage, "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": names},
        "action": {"dtype": "float32", "shape": (7,), "names": action_names},
        "source_episode_time_s": {"dtype": "float64", "shape": (1,), "names": ["seconds"]},
    }
    if depth == "meters":
        for camera in ("base", "wrist"):
            features[f"observation.depth.{camera}"] = {"dtype": "float32", "shape": (480, 640), "names": ["height", "width"]}
            features[f"observation.depth_valid.{camera}"] = {"dtype": "uint8", "shape": (480, 640), "names": ["height", "width"]}
    return features


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output exists; pass --overwrite to replace it: {path}")
        if path.resolve() in (Path.home().resolve(), Path("/").resolve()):
            raise ValueError("refusing unsafe output path")
        shutil.rmtree(path)


def convert(input_root: Path, output_root: Path, repo_id: str, task: str | None = None,
            episodes: list[str] | None = None, overwrite: bool = False, depth: str = "none",
            storage: str = "video", action_mode: str = "absolute") -> tuple[int, int]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    paths = finalized_episode_paths(input_root, episodes)
    if not paths:
        raise ValueError("no finalized episodes selected")
    validations = []
    print(f"[preflight] validating {len(paths)} raw episodes with full media decoding", flush=True)
    for index, path in enumerate(paths, 1):
        print(f"[preflight {index}/{len(paths)}] validating {path.name} ...", flush=True)
        validation = validate_raw_episode(path, decode_all_media=True)
        validations.append(validation)
        verdict = "PASS" if validation.ok else "FAIL"
        print(
            f"[preflight {index}/{len(paths)}] {verdict} {path.name}: "
            f"{len(validation.frames)} frames",
            flush=True,
        )
    failures = [item for item in validations if not item.ok]
    if failures:
        detail = "\n".join(f"{item.path.name}: {'; '.join(item.errors[:10])}" for item in failures)
        raise ValueError(f"raw validation failed; nothing converted:\n{detail}")
    print("[preflight] PASS; creating LeRobot output", flush=True)
    prepare_output(output_root, overwrite)
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=20, features=feature_schema(depth, storage, action_mode),
                                    root=output_root, robot_type="ur5", use_videos=storage == "video",
                                    tolerance_s=0.01)
    total = 0
    for episode_index, validation in enumerate(validations, 1):
        metadata = validation.metadata
        episode_task = task or metadata.get("task")
        frame_count = len(validation.frames)
        print(
            f"[convert {episode_index}/{len(validations)}] writing {validation.path.name}: "
            f"{frame_count} frames",
            flush=True,
        )
        for frame_index, raw in enumerate(validation.frames, 1):
            state, action = canonical_state_action(raw, metadata)
            if action_mode == "delta":
                action = absolute_to_delta(state, action)
            frame = {
                "observation.images.base": load_rgb(validation.path / raw["observation.images.base"]),
                "observation.images.wrist": load_rgb(validation.path / raw["observation.images.wrist"]),
                "observation.state": state,
                "action": action,
                # LeRobot 0.4.3 generates its built-in regular 20 Hz timestamp. The
                # exact original episode clock is retained as a separate feature.
                "source_episode_time_s": np.asarray([raw["episode_time_s"]], dtype=np.float64),
                "task": episode_task,
            }
            if depth == "meters":
                for camera in ("base", "wrist"):
                    values, valid = load_depth_m(validation.path / raw[f"observation.depth.{camera}"], float(metadata["depth_unit_m"]))
                    frame[f"observation.depth.{camera}"] = values
                    frame[f"observation.depth_valid.{camera}"] = valid
            dataset.add_frame(frame)
            total += 1
            if frame_index % 500 == 0 or frame_index == frame_count:
                print(
                    f"[convert {episode_index}/{len(validations)}] "
                    f"{frame_index}/{frame_count} frames staged",
                    flush=True,
                )
        print(
            f"[convert {episode_index}/{len(validations)}] encoding base/wrist videos ...",
            flush=True,
        )
        dataset.save_episode(parallel_encoding=False)
        print(f"[convert {episode_index}/{len(validations)}] episode saved", flush=True)
    print("[finalize] computing metadata/statistics and cleaning temporary assets ...", flush=True)
    dataset.finalize()
    conversion_meta = {
        "source_episode_names": [item.path.name for item in validations],
        "joint_order": list(UR5_JOINT_ORDER), "joint_unit": "radian",
        "state_semantics": "six measured joint positions plus estimated last-successful pneumatic command",
        "action_mode": action_mode,
        "action_semantics": ABSOLUTE_ACTION_SEMANTICS if action_mode == "absolute" else DELTA_ACTION_SEMANTICS,
        "gripper_closed_value": 0.0, "gripper_open_value": 1.0,
        "gripper_is_measured": False, "depth_mode": depth,
        "depth_note": "Raw depth remains in source episodes. meters mode masks raw 0 and 65535 to zero and stores validity masks.",
        "image_storage": storage, "lerobot_version_target": "0.4.3 / dataset codebase v3.0",
    }
    (output_root / "meta" / "ur5_vla_conversion.json").write_text(json.dumps(conversion_meta, indent=2) + "\n")
    print(f"[finalize] complete: {len(validations)} episodes / {total} frames", flush=True)
    return len(validations), total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task")
    parser.add_argument("--episodes", nargs="*", help="episode directory names; default: all finalized")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--depth", choices=("none", "meters"), default="none")
    parser.add_argument("--storage", choices=("video", "image"), default="video")
    parser.add_argument("--action-mode", choices=("absolute", "delta"), default="absolute")
    args = parser.parse_args()
    count, frames = convert(**vars(args))
    print(f"Converted {count} episodes / {frames} frames to {args.output_root}")


if __name__ == "__main__":
    main()
