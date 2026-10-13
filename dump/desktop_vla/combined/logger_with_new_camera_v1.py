#!/usr/bin/env python3
import os
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Joy, JointState, Image as RosImage


class UR5DatasetLogger(Node):
    """
    Records episodes ONLY when axes[3] is fully pressed (~1.0).

    Controls (only when axes[3] is gated):
      - button 9  -> START episode
      - button 8  -> ABORT episode (delete; do not save)
      - button 10 -> SUCCESS episode (finalize/save; count success)

    Logs at 20 Hz:
      - /rgb     (sensor_msgs/Image)  -> observation.images.rgb
      - /rgbUp   (sensor_msgs/Image)  -> observation.images.rgb_up
      - /joint_states                 -> observation.state (arm joints + finger_joint)
      - /dataset_joint_commands       -> action (arm joints + gripper_cmd)

    SmolVLA / LeRobot notes:
      * Two camera views are logged; keep these names consistent through conversion.
      * observation.state now includes the real gripper joint (finger_joint) so the
        policy can OBSERVE gripper opening, not just command it. This matches the
        gripper present in the action (gripper_cmd) and is a common cause of failed
        VLA grasps (action has gripper, state does not).
      * action is logged from the same tick as state, keeping them time-aligned.
      * A natural-language task string is required by SmolVLA; it is added during
        conversion (convert_data.py TASK_STRING) and also recorded here in meta.json
        for traceability.
    """

    def __init__(self):
        super().__init__("ur5_dataset_logger")

        # -----------------------
        # User settings
        # -----------------------
        self.dataset_root = Path.home() / "Desktop" / "dataset_raw_bottle"
        self.dataset_root.mkdir(parents=True, exist_ok=True)

        self.rate_hz = 20.0
        self.dt = 1.0 / self.rate_hz

        # Natural-language task — recorded in meta.json for SmolVLA traceability.
        # Keep this identical to TASK_STRING in convert_data.py.
        self.task_string = "pick up the bottle"

        # Gate: axes[3] must be ~1.0 to log + accept buttons
        self.GATE_AXIS = 3
        self.gate_threshold = 0.95

        # Buttons (your mapping)
        self.BTN_START = 9
        self.BTN_ABORT = 8
        self.BTN_SUCCESS = 10

        # Topics
        self.topic_rgb = "/rgb"
        self.topic_rgb_up = "/rgbUp"  # Second camera topic
        self.topic_js = "/joint_states"
        self.topic_ds = "/dataset_joint_commands"
        self.topic_joy = "/joy"

        # -----------------------
        # Gripper state joint
        # -----------------------
        # The Robotiq 2F-140 drive joint reported in /joint_states. Its mimic joints
        # (inner/outer knuckle, inner finger) follow it, so this one joint fully
        # describes gripper opening. Set to None to disable gripper-state logging.
        self.gripper_state_joint = "finger_joint"

        # -----------------------
        # Internal state
        # -----------------------
        self.axes = []
        self.buttons = []
        self.prev_buttons = []

        self.latest_img = None
        self.latest_img_stamp = None

        self.latest_img_up = None
        self.latest_img_up_stamp = None

        self.latest_js = {}      # name -> position
        self.latest_js_stamp = None

        self.latest_act = {}     # name -> position (includes gripper_cmd)
        self.latest_act_stamp = None
        self.action_names = None

        self.episode_active = False
        self.episode_dir = None
        self.rgb_dir = None
        self.rgb_up_dir = None
        self.csv_file = None
        self.csv_writer = None
        self.frame_idx = 0

        self.success_count = 0
        self.episode_index = self._next_episode_index()

        # -----------------------
        # QoS (camera often requires reliable)
        # -----------------------
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )
        default_qos = QoSProfile(depth=10)

        # Subscribers
        self.create_subscription(Joy, self.topic_joy, self.on_joy, default_qos)
        self.create_subscription(RosImage, self.topic_rgb, self.on_rgb, cam_qos)
        self.create_subscription(RosImage, self.topic_rgb_up, self.on_rgb_up, cam_qos)
        self.create_subscription(JointState, self.topic_js, self.on_joint_states, default_qos)
        self.create_subscription(JointState, self.topic_ds, self.on_dataset_actions, default_qos)

        # Timer for logging at fixed rate
        self.timer = self.create_timer(self.dt, self.on_tick)

        self.get_logger().info(
            "Dataset Logger ready.\n"
            f"Gate: axes[{self.GATE_AXIS}] >= {self.gate_threshold} required.\n"
            f"Buttons (only when gated): START={self.BTN_START}, ABORT={self.BTN_ABORT}, SUCCESS={self.BTN_SUCCESS}\n"
            f"Cameras: {self.topic_rgb} -> rgb, {self.topic_rgb_up} -> rgb_up\n"
            f"Gripper state joint logged: {self.gripper_state_joint}\n"
            f"Task: '{self.task_string}'\n"
            f"Saving to: {str(self.dataset_root)}\n"
        )

    # -----------------------
    # Helpers
    # -----------------------
    def _next_episode_index(self) -> int:
        existing = sorted([p for p in self.dataset_root.glob("episode_*") if p.is_dir()])
        if not existing:
            return 1
        nums = []
        for p in existing:
            try:
                nums.append(int(p.name.split("_")[1]))
            except Exception:
                pass
        return (max(nums) + 1) if nums else 1

    def _state_joint_names(self):
        """
        observation.state joints = all action joints except the gripper command
        pseudo-joint, plus the real gripper drive joint (finger_joint) appended.
        Single source of truth so the CSV header and rows always match.
        """
        names = [n for n in self.action_names if n != "gripper_cmd"]
        if self.gripper_state_joint:
            names.append(self.gripper_state_joint)
        return names

    def gated(self) -> bool:
        if not self.axes or self.GATE_AXIS >= len(self.axes):
            return False
        return float(self.axes[self.GATE_AXIS]) >= self.gate_threshold

    def rising_edge(self, idx: int) -> bool:
        if idx >= len(self.buttons) or idx >= len(self.prev_buttons):
            return False
        return self.buttons[idx] == 1 and self.prev_buttons[idx] == 0

    # -----------------------
    # Callbacks
    # -----------------------
    def on_joy(self, msg: Joy):
        self.axes = list(msg.axes)
        self.buttons = list(msg.buttons)
        if not self.prev_buttons:
            self.prev_buttons = [0] * len(self.buttons)

        if self.gated():
            if self.rising_edge(self.BTN_START):
                self.start_episode()
            if self.rising_edge(self.BTN_ABORT):
                self.abort_episode()
            if self.rising_edge(self.BTN_SUCCESS):
                self.success_episode()

        self.prev_buttons = list(self.buttons)

    def on_rgb(self, msg: RosImage):
        self.latest_img = msg
        self.latest_img_stamp = msg.header.stamp

    def on_rgb_up(self, msg: RosImage):
        self.latest_img_up = msg
        self.latest_img_up_stamp = msg.header.stamp

    def on_joint_states(self, msg: JointState):
        for n, p in zip(msg.name, msg.position):
            self.latest_js[n] = float(p)
        self.latest_js_stamp = msg.header.stamp

    def on_dataset_actions(self, msg: JointState):
        self.action_names = list(msg.name)
        for n, p in zip(msg.name, msg.position):
            self.latest_act[n] = float(p)
        self.latest_act_stamp = msg.header.stamp

    # -----------------------
    # Episode control
    # -----------------------
    def start_episode(self):
        if self.episode_active:
            self.get_logger().warn("Episode already active. Press SUCCESS or ABORT first.")
            return

        if not self.action_names:
            self.get_logger().warn("No /dataset_joint_commands received yet. Move joystick once so teleop publishes.")
            return

        # Create folders
        ep_name = f"episode_{self.episode_index:04d}"
        self.episode_dir = self.dataset_root / ep_name

        self.rgb_dir = self.episode_dir / "rgb"
        self.rgb_dir.mkdir(parents=True, exist_ok=True)

        self.rgb_up_dir = self.episode_dir / "rgb_up"
        self.rgb_up_dir.mkdir(parents=True, exist_ok=True)

        # Open CSV
        self.csv_file = open(self.episode_dir / "data.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        state_joint_names = self._state_joint_names()
        action_joint_names = list(self.action_names)

        # Header includes both cameras + gripper state (q_finger_joint) + gripper action.
        header = (
            ["t_idx", "time_wall",
             "img_sec", "img_nsec", "img_up_sec", "img_up_nsec",
             "js_sec", "js_nsec", "act_sec", "act_nsec",
             "rgb_file", "rgb_up_file"]
            + [f"q_{n}" for n in state_joint_names]
            + [f"a_{n}" for n in action_joint_names]
        )
        self.csv_writer.writerow(header)
        self.csv_file.flush()

        # Meta
        meta = {
            "episode_index": self.episode_index,
            "created_time": time.time(),
            "rate_hz": self.rate_hz,
            "task": self.task_string,
            "gate_axis": self.GATE_AXIS,
            "gate_threshold": self.gate_threshold,
            "topics": {
                "rgb": self.topic_rgb,
                "rgb_up": self.topic_rgb_up,
                "joint_states": self.topic_js,
                "dataset_actions": self.topic_ds,
                "joy": self.topic_joy
            },
            "state_joints": state_joint_names,
            "action_names": action_joint_names,
            "gripper_state_joint": self.gripper_state_joint
        }
        (self.episode_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        self.frame_idx = 0
        self.episode_active = True
        self.get_logger().info(f"START episode {ep_name} (logging only while axes[{self.GATE_AXIS}] is fully pressed).")

    def abort_episode(self):
        if not self.episode_active:
            self.get_logger().warn("No active episode to abort.")
            return

        self._close_episode_files()

        # Delete folder
        try:
            for p in sorted(self.episode_dir.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                else:
                    p.rmdir()
            self.episode_dir.rmdir()
        except Exception as e:
            self.get_logger().warn(f"Abort cleanup warning: {e}")

        self.get_logger().info("ABORT: episode deleted (not saved).")
        self._reset_episode_state()

    def success_episode(self):
        if not self.episode_active:
            self.get_logger().warn("No active episode to mark success.")
            return

        self._close_episode_files()

        # Mark success in meta.json
        meta_path = self.episode_dir / "meta.json"
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        meta["success"] = True
        meta["num_frames_logged"] = int(self.frame_idx)
        meta["saved_time"] = time.time()
        meta_path.write_text(json.dumps(meta, indent=2))

        self.success_count += 1
        self.get_logger().info(
            f"SUCCESS: saved episode_{self.episode_index:04d} with {self.frame_idx} frames. "
            f"Total successful episodes: {self.success_count}"
        )

        self.episode_index += 1
        self._reset_episode_state()

    def _close_episode_files(self):
        try:
            if self.csv_file:
                self.csv_file.flush()
                self.csv_file.close()
        except Exception:
            pass

    def _reset_episode_state(self):
        self.episode_active = False
        self.episode_dir = None
        self.rgb_dir = None
        self.rgb_up_dir = None
        self.csv_file = None
        self.csv_writer = None
        self.frame_idx = 0

    # -----------------------
    # Logging tick
    # -----------------------
    def on_tick(self):
        if (not self.episode_active) or (not self.gated()):
            return

        # Need all streams (both cameras + state + action).
        if (self.latest_img is None or self.latest_img_up is None or
                self.latest_js_stamp is None or self.latest_act_stamp is None):
            return

        state_joint_names = self._state_joint_names()
        action_joint_names = list(self.action_names)

        # All required joints must be present in their streams; otherwise skip this
        # tick rather than write a malformed row (keeps state/action time-aligned).
        for j in state_joint_names:
            if j not in self.latest_js:
                return
        for j in action_joint_names:
            if j not in self.latest_act:
                return

        # Save primary image
        rgb_file = f"{self.frame_idx:06d}.png"
        rgb_path = self.rgb_dir / rgb_file
        if not rgb_path.exists():
            self._save_ros_image_as_png(self.latest_img, rgb_path)

        # Save secondary image (rgbUp)
        rgb_up_file = f"{self.frame_idx:06d}.png"
        rgb_up_path = self.rgb_up_dir / rgb_up_file
        if not rgb_up_path.exists():
            self._save_ros_image_as_png(self.latest_img_up, rgb_up_path)

        # Write CSV row
        now_wall = time.time()

        img_sec = int(self.latest_img_stamp.sec)
        img_nsec = int(self.latest_img_stamp.nanosec)
        img_up_sec = int(self.latest_img_up_stamp.sec)
        img_up_nsec = int(self.latest_img_up_stamp.nanosec)
        js_sec = int(self.latest_js_stamp.sec)
        js_nsec = int(self.latest_js_stamp.nanosec)
        act_sec = int(self.latest_act_stamp.sec)
        act_nsec = int(self.latest_act_stamp.nanosec)

        q_vals = [self.latest_js[j] for j in state_joint_names]
        a_vals = [self.latest_act[j] for j in action_joint_names]

        row = (
            [self.frame_idx, now_wall,
             img_sec, img_nsec, img_up_sec, img_up_nsec,
             js_sec, js_nsec, act_sec, act_nsec,
             rgb_file, rgb_up_file]
            + q_vals
            + a_vals
        )
        self.csv_writer.writerow(row)

        # Flush periodically
        if (self.frame_idx % 20) == 0:
            self.csv_file.flush()

        self.frame_idx += 1

    def _save_ros_image_as_png(self, msg: RosImage, out_path: Path):
        h, w = int(msg.height), int(msg.width)
        enc = (msg.encoding or "").lower()

        data = np.frombuffer(msg.data, dtype=np.uint8)

        if enc == "rgb8":
            img = data.reshape((h, w, 3))
        elif enc == "bgr8":
            img = data.reshape((h, w, 3))[:, :, ::-1]  # BGR -> RGB
        elif enc == "rgba8":
            img = data.reshape((h, w, 4))[:, :, :3]
        elif enc == "bgra8":
            img = data.reshape((h, w, 4))[:, :, :3][:, :, ::-1]
        else:
            img = data.reshape((h, w, 3))

        Image.fromarray(img).save(str(out_path))


def main():
    rclpy.init()
    node = UR5DatasetLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
