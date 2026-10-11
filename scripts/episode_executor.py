#!/usr/bin/python3
"""Replay six absolute joint actions plus gripper on /control_abstraction_layer."""

import argparse
import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from project_paths import DATA


# Change these placeholders, or override them with command-line arguments.
EPISODE_ROOT = Path(
    DATA / "staging" / "clean_20_cube_raw_nonconverted"
)
EPISODE_NAME = "episode_20260828_014312"


def load_episode(root: Path, name: str) -> list[dict]:
    frames_file = root / name / "frames.jsonl"
    with frames_file.open() as stream:
        frames = [json.loads(line) for line in stream if line.strip()]
    if not frames:
        raise RuntimeError(f"No frames found in {frames_file}")
    return frames


class EpisodeExecutor(Node):
    def __init__(self, frames: list[dict], episode_name: str):
        super().__init__("episode_executor")
        self.frames = frames
        self.episode_name = episode_name
        self.index = 0
        self.start_time = time.monotonic()
        self.publisher = self.create_publisher(
            Float64MultiArray, "/control_abstraction_layer", 10
        )
        self.timer = self.create_timer(0.001, self.publish_due_frames)
        self.get_logger().info(
            f"Replaying {episode_name}: {len(frames)} frames on /control_abstraction_layer"
        )

    def publish_due_frames(self):
        elapsed = time.monotonic() - self.start_time
        while self.index < len(self.frames):
            frame = self.frames[self.index]
            if float(frame["episode_time_s"]) > elapsed:
                return
            action = frame["action"]
            message = Float64MultiArray()
            message.data = [float(value) for value in action[:7]]
            self.publisher.publish(message)
            self.index += 1

        self.get_logger().info(f"Finished replaying {self.episode_name}")
        self.timer.cancel()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-root", type=Path, default=EPISODE_ROOT)
    parser.add_argument("--episode-name", default=EPISODE_NAME)
    args = parser.parse_args()

    frames = load_episode(args.episode_root, args.episode_name)
    rclpy.init()
    node = EpisodeExecutor(frames, args.episode_name)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
