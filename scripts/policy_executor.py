#!/usr/bin/python3
"""Run an absolute-action SmolVLA policy and publish seven abstract values."""

import argparse
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from project_paths import MODELS, PACKAGE, inference_environment

sys.path.insert(0, str(PACKAGE))

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64, Float64MultiArray, Int8

from ur5_vla_dataset_recorder.core import UR5_JOINT_ORDER, reorder_joint_positions
from ur5_vla_dataset_recorder.policy_executor import ShadowObservationBuffer
from ur5_vla_dataset_recorder.shadow_protocol import request_prediction


# Final 110-episode, 18,000-step SmolVLA checkpoint.
MODEL_PATH = Path(
    MODELS / "checkpoints" /
    "ur5_cube_110ep_v3_smolvla_a100_b128_18000_steps_20261003_180950" /
    "pretrained_model"
)

TASK = "Pick up the orange cube and place it in the bowl"
INITIAL_GRIPPER = 1.0
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765
ACTION_PERIOD = 0.05  # 20 Hz, matching the recorded demonstrations.
PREFETCH_ACTIONS = 12
BLEND_ACTIONS = 10
REPOSITORY = PACKAGE


class PolicyExecutor(Node):
    def __init__(self, grasp_offset_file=None, offset_mode="continuous"):
        super().__init__("bypass_policy_executor")
        self.buffer = ShadowObservationBuffer()
        self.gripper = INITIAL_GRIPPER
        self.inference_running = False
        self.action_queue = deque()
        self.raw_action_queue = deque()
        self.last_action = None
        self.last_raw_action = None
        self.queue_lock = threading.Lock()
        self.inference_generation = 0
        self.measured_joints = None
        self.measured_joints_at = 0.0
        self.grasp_assist = None
        self.trajectory_offset = None
        self.offset_fault = False
        self.confirmed_gripper = None
        self.confirmed_gripper_at = 0.0
        if grasp_offset_file is not None:
            if offset_mode == "continuous":
                from grasp_offset import load_offset
                from trajectory_offset import TrajectoryOffsetPlanner
                from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
                frame, vector = load_offset(grasp_offset_file)
                self.trajectory_offset = TrajectoryOffsetPlanner(self, UR5_JOINT_ORDER, frame, vector)
                self.create_subscription(
                    Float64, "/control_engine/gripper_state", self.gripper_confirmation,
                    QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL),
                )
                self.get_logger().warning(
                    f"CONTINUOUS OFFSET ENABLED: {vector} metres in {frame}; every arm target, including carry and placement"
                )
            else:
                from grasp_offset import GraspOffsetAssist
                self.grasp_assist = GraspOffsetAssist(self, UR5_JOINT_ORDER, grasp_offset_file)
        self.publisher = self.create_publisher(
            Float64MultiArray, "/control_abstraction_layer", 10
        )
        self.create_subscription(
            Image,
            "/base/camera/color/image_raw",
            lambda message: self.image("observation.images.base", message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            "/wrist/camera/color/image_raw",
            lambda message: self.image("observation.images.wrist", message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointState, "/joint_states", self.joints, qos_profile_sensor_data
        )
        self.create_subscription(Float64, "/gripper/state", self.gripper_state, 10)
        self.create_subscription(
            Int8,
            "/servo_node/status",
            lambda message: self.buffer.update_servo(
                message.data, time.monotonic()
            ),
            10,
        )
        self.inference_timer = self.create_timer(0.05, self.schedule_inference)
        self.action_timer = self.create_timer(ACTION_PERIOD, self.publish_action)

    def image(self, key, message):
        self.buffer.update_image(
            key,
            message.height,
            message.width,
            message.encoding,
            message.step,
            bytes(message.data),
            time.monotonic(),
        )

    def joints(self, message):
        ordered = reorder_joint_positions(
            message.name, message.position, UR5_JOINT_ORDER
        )
        self.buffer.update_joints(ordered, time.monotonic())
        self.measured_joints = list(ordered)
        self.measured_joints_at = time.monotonic()

    def gripper_state(self, message):
        self.gripper = 1.0 if float(message.data) >= 0.5 else 0.0

    def gripper_confirmation(self, message):
        if float(message.data) in (0.0, 1.0):
            self.confirmed_gripper = float(message.data)
            self.confirmed_gripper_at = time.monotonic()

    def schedule_inference(self):
        if self.offset_fault:
            return
        if self.grasp_assist is not None and self.grasp_assist.active:
            return
        with self.queue_lock:
            if self.inference_running or len(self.action_queue) > PREFETCH_ACTIONS:
                return
            self.inference_running = True
            generation = self.inference_generation
        now = time.monotonic()
        if (self.trajectory_offset is not None and self.confirmed_gripper is not None
                and now - self.confirmed_gripper_at <= 0.2):
            self.gripper = self.confirmed_gripper
        if (self.grasp_assist is not None and
                self.grasp_assist.confirmed_gripper is not None and
                now - self.grasp_assist.confirmed_gripper_at <= 0.2):
            self.gripper = self.grasp_assist.confirmed_gripper
        # INITIAL_GRIPPER supplies the first observation; later predictions or
        # /gripper/state messages update this estimate.
        self.buffer.update_gripper(self.gripper, now)
        try:
            state, images, _ = self.buffer.snapshot(now, 0.5, 1.0, 0.5)
        except Exception as error:
            self.get_logger().warning(f"Waiting for observations: {error}")
            with self.queue_lock:
                self.inference_running = False
            return
        threading.Thread(
            target=self.infer, args=(state, images, generation), daemon=True
        ).start()

    def infer(self, state, images, generation):
        try:
            chunk, _ = request_prediction(
                SERVER_HOST, SERVER_PORT, 5.0, state, TASK, images
            )
            actions = [list(action) for action in chunk]
            if len(actions) != 50 or any(len(action) != 7 for action in actions):
                raise ValueError(f"Expected action chunk [50, 7], got {len(actions)} actions")
            with self.queue_lock:
                if generation != self.inference_generation:
                    return
                if self.grasp_assist is not None and self.grasp_assist.active:
                    return
                if self.trajectory_offset is not None:
                    boundary = list(self.raw_action_queue[-1]) if self.raw_action_queue else self.last_raw_action
                else:
                    boundary = list(self.action_queue[-1]) if self.action_queue else self.last_action
            if boundary is not None:
                count = min(BLEND_ACTIONS, len(actions))
                for index in range(count):
                    phase = (index + 1) / count
                    weight = phase * phase * (3.0 - 2.0 * phase)
                    actions[index][:6] = [
                        old + weight * (new - old)
                        for old, new in zip(boundary[:6], actions[index][:6])
                    ]
            raw_actions = [list(action) for action in actions]
            if self.trajectory_offset is not None:
                if self.measured_joints is None or time.monotonic() - self.measured_joints_at > 0.1:
                    raise ValueError("fresh measured joints required for continuous correction")
                # If actions are already queued, check the path from their endpoint.
                with self.queue_lock:
                    reference = list(self.action_queue[-1][:6]) if self.action_queue else list(self.measured_joints)
                actions, report = self.trajectory_offset.plan_actions(actions, reference)
                self.get_logger().info(f"CONTINUOUS OFFSET CHECKED: {report}")
            with self.queue_lock:
                if generation != self.inference_generation or self.offset_fault:
                    return
                if self.grasp_assist is not None and self.grasp_assist.active:
                    return
                self.action_queue.extend(actions)
                if self.trajectory_offset is not None:
                    self.raw_action_queue.extend(raw_actions)
            self.get_logger().info(
                "Queued 50 policy actions with joint-space boundary blending"
            )
        except Exception as error:
            if self.trajectory_offset is not None:
                with self.queue_lock:
                    self.offset_fault = True
                    self.action_queue.clear()
                    self.raw_action_queue.clear()
                self.get_logger().error(f"CONTINUOUS OFFSET BLOCKED; fault latched: {error}")
            self.get_logger().error(f"Inference failed: {error}")
        finally:
            with self.queue_lock:
                self.inference_running = False

    def publish_action(self):
        if self.offset_fault:
            return
        if self.trajectory_offset is not None:
            with self.queue_lock:
                if not self.action_queue:
                    return
        if (self.trajectory_offset is not None and
                (self.measured_joints is None or time.monotonic() - self.measured_joints_at > 0.1)):
            with self.queue_lock:
                self.offset_fault = True
                self.action_queue.clear()
                self.raw_action_queue.clear()
            self.get_logger().error("CONTINUOUS OFFSET BLOCKED: stale measured joints")
            return
        if self.grasp_assist is not None and self.grasp_assist.active:
            action = self.grasp_assist.tick(
                self.measured_joints, self.measured_joints_at,
                self.grasp_assist.open_confirmed,
            )
            if action is not None:
                self.last_action = list(action)
                message = Float64MultiArray()
                message.data = action
                self.publisher.publish(message)
            elif self.grasp_assist.stage == "complete":
                self.gripper = 0.0
                with self.queue_lock:
                    self.action_queue.clear()
            return
        with self.queue_lock:
            if not self.action_queue:
                return
            action = self.action_queue.popleft()
            if self.trajectory_offset is not None:
                self.last_raw_action = self.raw_action_queue.popleft()
            self.last_action = list(action)
            remaining = len(self.action_queue)
        if (self.grasp_assist is not None and
                self.grasp_assist.stage == "idle" and float(action[6]) < 0.5):
            if not self.grasp_assist.open_confirmed:
                self.grasp_assist.stage = "fault"
                self.get_logger().error("OFFSET BLOCKED: fresh gripper-open IO required before correction")
                return
            if self.measured_joints is None or time.monotonic() - self.measured_joints_at > 0.1:
                self.grasp_assist.stage = "fault"
                self.get_logger().error("OFFSET BLOCKED: fresh measured joints required before correction")
                return
            with self.queue_lock:
                self.action_queue.clear()
                self.inference_generation += 1
                self.grasp_assist.start(action[:6], self.measured_joints)
            return
        self.gripper = 1.0 if float(action[6]) >= 0.5 else 0.0
        message = Float64MultiArray()
        message.data = [float(value) for value in action[:6]] + [self.gripper]
        self.publisher.publish(message)
        if remaining == 0:
            self.get_logger().info("Action chunk completed; requesting next observation")


def wait_for_server(process, timeout=120.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Inference server exited during startup")
        try:
            with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=0.2):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError("Timed out waiting for inference server")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grasp-offset-file", type=Path,
        help="measured Cartesian translation; applied throughout the trajectory by default",
    )
    parser.add_argument(
        "--offset-mode", choices=("continuous", "grasp"), default="continuous",
        help="continuous: offset all arm targets; grasp: earlier one-close intervention",
    )
    args = parser.parse_args()
    if args.grasp_offset_file is not None:
        from grasp_offset import load_offset
        load_offset(args.grasp_offset_file)
    if not MODEL_PATH.is_dir():
        raise SystemExit(f"MODEL_PATH does not exist: {MODEL_PATH}")
    server = subprocess.Popen(
        [
            "/home/grasplab/miniconda3/bin/conda",
            "run",
            "--no-capture-output",
            "-n",
            "pi0",
            "python",
            "-m",
            "ur5_vla_dataset_recorder.policy_inference_server",
            "--checkpoint",
            str(MODEL_PATH),
            "--host",
            SERVER_HOST,
            "--port",
            str(SERVER_PORT),
            "--device",
            "cuda",
        ],
        cwd=REPOSITORY,
        env=inference_environment(),
    )
    try:
        wait_for_server(server)
        rclpy.init()
        node = PolicyExecutor(args.grasp_offset_file, args.offset_mode)
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
