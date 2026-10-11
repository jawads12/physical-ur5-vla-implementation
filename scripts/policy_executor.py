#!/usr/bin/python3
"""Run an absolute-action SmolVLA policy and publish seven abstract values."""

import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from project_paths import DOWNLOADED_MODELS, PACKAGE, inference_environment

sys.path.insert(0, str(PACKAGE))

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64, Float64MultiArray, Int8

from ur5_vla_dataset_recorder.core import UR5_JOINT_ORDER, reorder_joint_positions
from ur5_vla_dataset_recorder.policy_executor import ShadowObservationBuffer
from ur5_vla_dataset_recorder.shadow_protocol import request_prediction


# Selected checkpoint copied into the central repository.
MODEL_PATH = Path(
    DOWNLOADED_MODELS / "pretrained_model-20261007T190844Z-1-001" / "pretrained_model"
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
    def __init__(self):
        super().__init__("bypass_policy_executor")
        self.buffer = ShadowObservationBuffer()
        self.gripper = INITIAL_GRIPPER
        self.inference_running = False
        self.action_queue = deque()
        self.last_action = None
        self.queue_lock = threading.Lock()
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

    def gripper_state(self, message):
        self.gripper = 1.0 if float(message.data) >= 0.5 else 0.0

    def schedule_inference(self):
        with self.queue_lock:
            if self.inference_running or len(self.action_queue) > PREFETCH_ACTIONS:
                return
            self.inference_running = True
        now = time.monotonic()
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
            target=self.infer, args=(state, images), daemon=True
        ).start()

    def infer(self, state, images):
        try:
            chunk, _ = request_prediction(
                SERVER_HOST, SERVER_PORT, 5.0, state, TASK, images
            )
            actions = [list(action) for action in chunk]
            if len(actions) != 50 or any(len(action) != 7 for action in actions):
                raise ValueError(f"Expected action chunk [50, 7], got {len(actions)} actions")
            with self.queue_lock:
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
                self.action_queue.extend(actions)
            self.get_logger().info(
                "Queued 50 policy actions with joint-space boundary blending"
            )
        except Exception as error:
            self.get_logger().error(f"Inference failed: {error}")
        finally:
            with self.queue_lock:
                self.inference_running = False

    def publish_action(self):
        with self.queue_lock:
            if not self.action_queue:
                return
            action = self.action_queue.popleft()
            self.last_action = list(action)
            remaining = len(self.action_queue)
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
        node = PolicyExecutor()
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
