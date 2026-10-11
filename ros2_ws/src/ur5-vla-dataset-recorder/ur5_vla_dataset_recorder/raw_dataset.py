"""ROS-independent validation and loading of staged UR5 demonstrations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image

from .core import UR5_JOINT_ORDER, reorder_command

CAMERA_AGE_LIMIT_S = 0.10
GRIPPER_AGE_LIMIT_S = 0.20
MAX_ACTION_JUMP_RAD = 0.20
MAX_FRAME_INTERVAL_S = 0.20


@dataclass
class EpisodeValidation:
    path: Path
    frames: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def finalized_episode_paths(root: Path, selected: Iterable[str] | None = None) -> list[Path]:
    wanted = set(selected or [])
    paths = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith(".") or path.name.endswith(".incomplete"):
            continue
        if wanted and path.name not in wanted:
            continue
        if (path / "metadata.yaml").exists() or (path / "frames.jsonl").exists():
            paths.append(path)
    missing = wanted - {p.name for p in paths}
    if missing:
        raise ValueError(f"selected episodes not found/finalized: {sorted(missing)}")
    return paths


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_depth_m(path: Path, unit_m: float) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        mode = image.mode
        raw = np.asarray(image)
    # Pillow commonly exposes lossless 16-bit PNG mode I as int32. Verify the
    # original mode and range before restoring its unsigned 16-bit semantics.
    if raw.dtype == np.int32 and mode == "I" and raw.size and raw.min() >= 0 and raw.max() <= 65535:
        raw = raw.astype(np.uint16)
    if raw.dtype != np.uint16:
        raise ValueError(f"depth must be uint16, got {raw.dtype}: {path}")
    valid = (raw != 0) & (raw != np.iinfo(np.uint16).max)
    depth = np.zeros(raw.shape, dtype=np.float32)
    depth[valid] = raw[valid].astype(np.float32) * np.float32(unit_m)
    return depth, valid.astype(np.uint8)


def _finite_vector(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(
        isinstance(x, (int, float)) and math.isfinite(float(x)) for x in value
    )


def _binary(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and min(
        abs(float(value)), abs(float(value) - 1.0)
    ) <= 1e-6


def validate_raw_episode(path: Path, decode_all_media: bool = True) -> EpisodeValidation:
    result = EpisodeValidation(path=path)
    if path.name.startswith(".") or path.name.endswith(".incomplete"):
        result.errors.append("episode is not finalized")
    try:
        result.metadata = yaml.safe_load((path / "metadata.yaml").read_text())
    except Exception as exc:
        result.errors.append(f"metadata parse failed: {exc}")
        return result
    try:
        for line_number, line in enumerate((path / "frames.jsonl").read_text().splitlines(), 1):
            try:
                result.frames.append(json.loads(line))
            except Exception as exc:
                result.errors.append(f"frames.jsonl line {line_number}: {exc}")
    except Exception as exc:
        result.errors.append(f"manifest read failed: {exc}")
        return result

    meta, frames = result.metadata, result.frames
    if meta.get("outcome") != "success":
        result.errors.append(f"outcome is {meta.get('outcome')!r}, not success")
    if tuple(meta.get("joint_order", ())) != UR5_JOINT_ORDER:
        result.errors.append("joint_order does not match canonical UR5 order")
    if tuple(meta.get("command_joint_order", ())) != UR5_JOINT_ORDER:
        result.errors.append("command_joint_order does not match canonical UR5 order")
    if meta.get("frames") != len(frames):
        result.errors.append(f"metadata frames={meta.get('frames')} but manifest has {len(frames)}")
    indexes = [frame.get("frame_index") for frame in frames]
    if indexes != list(range(len(frames))):
        result.errors.append("frame indexes are not sequential from zero")
    timestamps = np.asarray([frame.get("episode_time_s", np.nan) for frame in frames], dtype=float)
    if not np.isfinite(timestamps).all() or (len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0)):
        result.errors.append("timestamps are non-finite or not strictly increasing")
    elif len(timestamps) > 1:
        intervals = np.diff(timestamps)
        gap_indexes = np.flatnonzero(intervals > MAX_FRAME_INTERVAL_S)
        if gap_indexes.size:
            largest_index = int(gap_indexes[np.argmax(intervals[gap_indexes])])
            result.errors.append(
                f"{gap_indexes.size} frame interval(s) exceed {MAX_FRAME_INTERVAL_S:.2f}s; "
                f"largest is {intervals[largest_index]:.6f}s between frames "
                f"{largest_index} and {largest_index + 1}"
            )

    action_rows = []
    gripper_commands = []
    referenced: dict[str, list[Path]] = {name: [] for name in ("base_rgb", "wrist_rgb", "base_depth", "wrist_depth")}
    path_keys = {
        "base_rgb": "observation.images.base",
        "wrist_rgb": "observation.images.wrist",
        "base_depth": "observation.depth.base",
        "wrist_depth": "observation.depth.wrist",
    }
    stale_count = 0
    for index, frame in enumerate(frames):
        state, action = frame.get("observation.state"), frame.get("action")
        if not _finite_vector(state, 7):
            result.errors.append(f"frame {index}: invalid 7D state")
        if not _finite_vector(action, 7):
            result.errors.append(f"frame {index}: invalid 7D action")
        if _finite_vector(action, 7):
            action_rows.append(action[:6])
        values = [
            state[6] if _finite_vector(state, 7) else None,
            action[6] if _finite_vector(action, 7) else None,
            frame.get("gripper.state"), frame.get("gripper.command"),
        ]
        if not all(_binary(value) for value in values):
            result.errors.append(f"frame {index}: invalid binary gripper value")
        elif values[0] != values[2] or values[1] != values[3]:
            result.errors.append(f"frame {index}: gripper projection mismatch")
        else:
            gripper_commands.append(float(values[3]))
        if frame.get("servo_status") != 0 or frame.get("valid") is not True:
            result.errors.append(f"frame {index}: invalid frame or nonzero Servo status")
        ages = frame.get("receive_age_s", {})
        limits = {
            "joint_state": CAMERA_AGE_LIMIT_S, "arm_command": CAMERA_AGE_LIMIT_S,
            "base_image": CAMERA_AGE_LIMIT_S, "wrist_image": CAMERA_AGE_LIMIT_S,
            "base_depth": CAMERA_AGE_LIMIT_S, "wrist_depth": CAMERA_AGE_LIMIT_S,
            "gripper_state": GRIPPER_AGE_LIMIT_S, "gripper_command": GRIPPER_AGE_LIMIT_S,
        }
        if any(not math.isfinite(float(ages.get(key, math.inf))) or float(ages.get(key, math.inf)) > limit for key, limit in limits.items()):
            stale_count += 1
        for name, key in path_keys.items():
            relative = frame.get(key)
            media_path = path / relative if isinstance(relative, str) else path / "__missing__"
            referenced[name].append(media_path)
            if not media_path.is_file():
                result.errors.append(f"frame {index}: missing {key}: {relative}")
    if stale_count:
        result.errors.append(f"{stale_count} frames contain stale observations/actions")
    if action_rows and np.max(np.abs(np.diff(np.asarray(action_rows), axis=0)), initial=0) > MAX_ACTION_JUMP_RAD:
        result.errors.append(f"arm action jump exceeds {MAX_ACTION_JUMP_RAD} rad/frame")
    transitions = list(zip(gripper_commands, gripper_commands[1:]))
    if not any(a == 1 and b == 0 for a, b in transitions):
        result.errors.append("no gripper close transition")
    if not any(a == 0 and b == 1 for a, b in transitions):
        result.errors.append("no gripper open transition")

    for name, paths in referenced.items():
        directory = path / ("images" if "rgb" in name else "depth") / name.split("_")[0]
        suffix = ".jpg" if "rgb" in name else ".png"
        actual = set(directory.glob(f"*{suffix}")) if directory.is_dir() else set()
        if len(actual) != len(frames) or set(paths) != actual:
            result.errors.append(f"{name} count/reference mismatch ({len(actual)} files, {len(paths)} refs)")
        if decode_all_media:
            for media_path in paths:
                if not media_path.is_file():
                    continue
                try:
                    if "rgb" in name:
                        image = load_rgb(media_path)
                        if image.shape != (480, 640, 3):
                            raise ValueError(f"unexpected shape {image.shape}")
                    else:
                        load_depth_m(media_path, float(meta.get("depth_unit_m", 0.001)))
                except Exception as exc:
                    result.errors.append(f"unreadable {name} {media_path.name}: {exc}")
    if frames:
        for index in sorted({0, len(frames) // 2, len(frames) - 1}):
            try:
                base = load_rgb(referenced["base_rgb"][index])
                wrist = load_rgb(referenced["wrist_rgb"][index])
                if hashlib.sha256(base.tobytes()).digest() == hashlib.sha256(wrist.tobytes()).digest():
                    result.errors.append(f"frame {index}: base and wrist RGB are identical")
            except Exception:
                pass
    return result


def canonical_state_action(frame: dict[str, Any], metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical float32 7D vectors, reordering arm components if required."""
    state = frame["observation.state"]
    action = frame["action"]
    state_arm = reorder_command(state[:6], metadata["joint_order"])
    action_arm = reorder_command(action[:6], metadata["command_joint_order"])
    return (
        np.asarray(state_arm + [float(state[6])], dtype=np.float32),
        np.asarray(action_arm + [float(action[6])], dtype=np.float32),
    )
