#!/usr/bin/python3
"""Phase-gated, five-action receding-horizon SmolVLA executor.

The existing policy_executor.py is intentionally left unchanged.  Like that
executor, this node publishes seven-value commands to
/control_abstraction_layer when run normally.  --shadow is available only for
explicit offline diagnostics.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from enum import Enum
from pathlib import Path

from project_paths import DATA, MODELS, PACKAGE, inference_environment

sys.path.insert(0, str(PACKAGE))

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, Int8
from ur_dashboard_msgs.msg import SafetyMode

from ur5_vla_dataset_recorder.core import UR5_JOINT_ORDER, reorder_joint_positions
from ur5_vla_dataset_recorder.policy_executor import ShadowObservationBuffer
from ur5_vla_dataset_recorder.shadow_protocol import request_prediction


MODEL_PATH = Path(
    MODELS / "checkpoints" /
    "ur5_cube_110ep_v3_smolvla_a100_b128_18000_steps_20261003_180950" /
    "pretrained_model"
)
RAW_EPISODES = DATA / "staging" / "episodes"
REPOSITORY = PACKAGE
TASK = "Pick up the orange cube and place it in the bowl"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765
ACTION_HZ = 20.0
EXECUTED_ACTIONS_PER_PREDICTION = 5
REPLAN_REMAINING_ACTIONS = 4
TRANSITION_CONFIRMATIONS = 3
GRIPPER_SETTLE_SECONDS = 1.0
RELEASE_SETTLE_SECONDS = 2.0
MAX_RUNTIME_SECONDS = 180.0

# A valid absolute-action prediction should begin near the measured pose.  A
# larger error means the policy skipped ahead instead of producing the next
# local part of the demonstration.
MAX_FIRST_TARGET_ERROR_RAD = 0.08
MAX_CHUNK_STEP_RAD = 0.05
MAX_PHASE_DISTANCE_RAD = 0.12
MAX_SPEED_SCALING_PERCENT = 20.0


class Phase(str, Enum):
    APPROACH = "approach"
    CARRY = "carry"
    RELEASED = "released"
    COMPLETE = "complete"
    FAULT = "fault"


class DemonstrationPhaseIndex:
    """Joint-space references split by demonstrated gripper transitions."""

    def __init__(self, root: Path):
        references = {"approach": [], "carry": [], "released": []}
        self.close = []
        self.open = []
        episode_paths = sorted(glob.glob(str(root / "episode_*" / "frames.jsonl")))
        if not episode_paths:
            raise RuntimeError(f"no raw episodes found under {root}")

        for path in episode_paths:
            phase = "approach"
            previous_gripper = 1
            with open(path, encoding="utf-8") as stream:
                for line_number, line in enumerate(stream):
                    frame = json.loads(line)
                    action = [float(value) for value in frame["action"]]
                    if len(action) != 7 or not all(math.isfinite(value) for value in action):
                        raise RuntimeError(f"invalid action in {path}:{line_number + 1}")
                    gripper = int(action[6] >= 0.5)
                    if phase == "approach" and previous_gripper == 1 and gripper == 0:
                        self.close.append(action[:6])
                        phase = "carry"
                    elif phase == "carry" and previous_gripper == 0 and gripper == 1:
                        self.open.append(action[:6])
                        phase = "released"
                    # Subsampling keeps nearest-phase checks inexpensive while
                    # retaining the 20 Hz trajectory geometry closely enough.
                    if line_number % 5 == 0:
                        references[phase].append(action[:6])
                    previous_gripper = gripper

        if not self.close or not self.open or any(not values for values in references.values()):
            raise RuntimeError("dataset does not contain complete open-close-open episodes")
        self.references = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in references.items()
        }
        self.close = np.asarray(self.close, dtype=np.float64)
        self.open = np.asarray(self.open, dtype=np.float64)

    @staticmethod
    def _distance(values: np.ndarray, target) -> float:
        point = np.asarray(target, dtype=np.float64)
        return float(np.min(np.max(np.abs(values - point), axis=1)))

    def distances(self, target) -> dict[str, float]:
        return {
            name: self._distance(values, target)
            for name, values in self.references.items()
        }

    def transition_distance(self, target, opened: bool) -> float:
        return self._distance(self.open if opened else self.close, target)


class PhasedPolicyExecutor(Node):
    def __init__(self, live: bool, phases: DemonstrationPhaseIndex):
        super().__init__("bypass_policy_executor_phased")
        self.live = live
        self.phases = phases
        self.buffer = ShadowObservationBuffer()
        self.lock = threading.RLock()
        self.queue = deque()
        self.inference_running = False
        self.latest_joints = None
        self.actual_gripper = None if live else 1.0
        self.servo_status = None
        self.speed_scaling = None
        self.safety_mode = None
        self.program_running = None
        self.phase = Phase.APPROACH
        self.transition_candidate = None
        self.transition_votes = 0
        self.pending_gripper = None
        self.pending_since = 0.0
        self.released_at = 0.0
        self.started_at = time.monotonic()
        self.fault_reason = None
        if not live:
            self.buffer.update_gripper(1.0, time.monotonic())

        self.publisher = None
        if live:
            self.publisher = self.create_publisher(
                Float64MultiArray, "/control_abstraction_layer", 10
            )

        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Image,
            "/base/camera/color/image_raw",
            lambda message: self._image("observation.images.base", message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            "/wrist/camera/color/image_raw",
            lambda message: self._image("observation.images.wrist", message),
            qos_profile_sensor_data,
        )
        self.create_subscription(JointState, "/joint_states", self._joints, qos_profile_sensor_data)
        self.create_subscription(Int8, "/servo_node/status", self._servo, 10)
        self.create_subscription(
            Float64,
            "/control_engine/gripper_state",
            self._gripper_ack,
            state_qos,
        )
        self.create_subscription(
            Float64,
            "/speed_scaling_state_broadcaster/speed_scaling",
            lambda message: setattr(self, "speed_scaling", float(message.data)),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            SafetyMode,
            "/io_and_status_controller/safety_mode",
            lambda message: setattr(self, "safety_mode", int(message.mode)),
            state_qos,
        )
        self.create_subscription(
            Bool,
            "/io_and_status_controller/robot_program_running",
            lambda message: setattr(self, "program_running", bool(message.data)),
            state_qos,
        )
        self.create_timer(0.05, self._schedule_inference)
        self.create_timer(1.0 / ACTION_HZ, self._action_tick)
        mode = "LIVE" if live else "SHADOW"
        self.get_logger().warning(
            f"{mode} phased executor: five actions per observation; phase={self.phase.value}"
        )

    def _image(self, key, message):
        self.buffer.update_image(
            key,
            message.height,
            message.width,
            message.encoding,
            message.step,
            bytes(message.data),
            time.monotonic(),
        )

    def _joints(self, message):
        try:
            joints = reorder_joint_positions(message.name, message.position, UR5_JOINT_ORDER)
            self.latest_joints = [float(value) for value in joints]
            self.buffer.update_joints(joints, time.monotonic())
        except Exception as error:
            self._fault(f"joint state rejected: {error}")

    def _servo(self, message):
        self.servo_status = int(message.data)
        self.buffer.update_servo(self.servo_status, time.monotonic())

    def _gripper_ack(self, message):
        gripper = 1.0 if float(message.data) >= 0.5 else 0.0
        self.actual_gripper = gripper
        self.buffer.update_gripper(gripper, time.monotonic())
        self._confirm_pending_if_ready()

    def _confirm_pending_if_ready(self):
        if self.pending_gripper is None or self.actual_gripper != self.pending_gripper:
            return
        if time.monotonic() - self.pending_since < GRIPPER_SETTLE_SECONDS:
            return
        completed = self.pending_gripper
        self.pending_gripper = None
        self.transition_votes = 0
        self.transition_candidate = None
        if completed == 0.0:
            self.phase = Phase.CARRY
        else:
            self.phase = Phase.RELEASED
            self.released_at = time.monotonic()
        self.get_logger().warning(
            f"confirmed physical gripper transition; phase={self.phase.value}"
        )

    def _runtime_failure(self):
        if time.monotonic() - self.started_at > MAX_RUNTIME_SECONDS:
            return "runtime limit exceeded"
        if self.latest_joints is None or self.actual_gripper is None:
            return "waiting for measured joints/gripper IO"
        if self.servo_status != 0:
            return f"Servo status is {self.servo_status}, expected 0"
        if self.live:
            if any(value is None for value in (
                self.speed_scaling, self.safety_mode, self.program_running
            )):
                return "waiting for live robot status"
            if self.speed_scaling is None or not 0.0 < self.speed_scaling <= MAX_SPEED_SCALING_PERCENT:
                return f"speed scaling must be in (0, 20]%, got {self.speed_scaling}"
            if self.safety_mode not in (SafetyMode.NORMAL, SafetyMode.REDUCED):
                return f"unsafe/unknown safety mode {self.safety_mode}"
            if self.program_running is not True:
                return "External Control is not running"
        return None

    def _fault(self, reason):
        if self.phase == Phase.FAULT:
            return
        self.phase = Phase.FAULT
        self.fault_reason = str(reason)
        with self.lock:
            self.queue.clear()
        self.get_logger().error(f"PHASED EXECUTOR FAULT: {reason}")

    def _schedule_inference(self):
        if self.phase in (Phase.RELEASED, Phase.COMPLETE, Phase.FAULT):
            return
        failure = self._runtime_failure()
        if failure:
            # Missing startup inputs are expected; an abnormal live condition
            # after input initialization is not.
            if (
                self.live
                and not failure.startswith("waiting for")
                and self.latest_joints is not None
                and self.actual_gripper is not None
            ):
                self._fault(failure)
            return
        with self.lock:
            if (
                self.inference_running
                or self.pending_gripper is not None
                or len(self.queue) > REPLAN_REMAINING_ACTIONS
            ):
                return
            self.inference_running = True
        now = time.monotonic()
        if not self.live:
            self.buffer.update_gripper(self.actual_gripper, now)
        try:
            state, images, _ = self.buffer.snapshot(now, 0.5, 0.5, 0.5)
        except Exception as error:
            with self.lock:
                self.inference_running = False
            self.get_logger().warning(f"waiting for fresh observations: {error}")
            return
        threading.Thread(target=self._infer, args=(state, images), daemon=True).start()

    def _infer(self, state, images):
        try:
            chunk, latency = request_prediction(
                SERVER_HOST, SERVER_PORT, 5.0, state, TASK, images
            )
            if len(chunk) != 50 or any(len(action) != 7 for action in chunk):
                raise ValueError("policy output is not [50, 7]")
            selected = [[float(value) for value in action] for action in chunk[:EXECUTED_ACTIONS_PER_PREDICTION]]
            self._accept_prediction(state, selected)
            self.get_logger().info(
                f"accepted {len(selected)} fresh actions phase={self.phase.value} latency={latency:.3f}s"
            )
        except Exception as error:
            self._fault(error)
        finally:
            with self.lock:
                self.inference_running = False

    def _accept_prediction(self, state, actions):
        for action in actions:
            if not all(math.isfinite(value) for value in action):
                raise ValueError("policy produced a non-finite action")
        if max(abs(a - b) for a, b in zip(actions[0][:6], state[:6])) > MAX_FIRST_TARGET_ERROR_RAD:
            raise ValueError("first target skipped too far from the measured pose")
        previous = actions[0]
        for action in actions[1:]:
            if max(abs(a - b) for a, b in zip(action[:6], previous[:6])) > MAX_CHUNK_STEP_RAD:
                raise ValueError("prediction contains an excessive intra-chunk step")
            previous = action

        allowed = "approach" if self.phase == Phase.APPROACH else "carry"
        for action in actions:
            distances = self.phases.distances(action[:6])
            allowed_distance = distances[allowed]
            if allowed_distance > MAX_PHASE_DISTANCE_RAD:
                raise ValueError(
                    f"target is outside demonstrated {allowed} corridor: {allowed_distance:.4f} rad"
                )

        expected_gripper = 1 if self.phase == Phase.APPROACH else 0
        transition_index = next(
            (index for index, action in enumerate(actions) if int(action[6] >= 0.5) != expected_gripper),
            None,
        )
        executable = actions if transition_index is None else actions[:transition_index]
        desired_transition = 1 - expected_gripper
        old_phase = self.phase
        self._vote_transition(
            desired_transition if transition_index is not None else None,
            state[:6],
        )
        if self.pending_gripper is not None or self.phase != old_phase:
            executable = []
        with self.lock:
            # Replace rather than append: no stale remainder survives a new
            # observation/prediction boundary.
            self.queue.clear()
            self.queue.extend(executable)

    def _vote_transition(self, candidate, joints):
        if candidate is None:
            self.transition_candidate = None
            self.transition_votes = 0
            return
        if candidate == self.transition_candidate:
            self.transition_votes += 1
        else:
            self.transition_candidate = candidate
            self.transition_votes = 1
        if self.transition_votes < TRANSITION_CONFIRMATIONS:
            return
        distance = self.phases.transition_distance(joints, opened=bool(candidate))
        if distance > MAX_PHASE_DISTANCE_RAD:
            raise ValueError(
                f"gripper transition rejected outside demonstrated transition region: {distance:.4f} rad"
            )
        self.pending_gripper = float(candidate)
        self.pending_since = time.monotonic()
        with self.lock:
            self.queue.clear()
        if not self.live:
            # Shadow mode has no physical IO transaction. Simulate only the
            # phase bookkeeping so complete sequences can be validated.
            self.actual_gripper = float(candidate)
            self.buffer.update_gripper(float(candidate), time.monotonic())
            if candidate == 0:
                self.phase = Phase.CARRY
            else:
                self.phase = Phase.RELEASED
                self.released_at = time.monotonic()
            self.pending_gripper = None
        self.get_logger().warning(
            f"transition accepted: {'OPEN' if candidate else 'CLOSE'}"
        )

    def _action_tick(self):
        if self.phase == Phase.FAULT:
            return
        if self.phase == Phase.RELEASED:
            if time.monotonic() - self.released_at >= RELEASE_SETTLE_SECONDS:
                self.phase = Phase.COMPLETE
                self.get_logger().warning("TASK COMPLETE; command output stopped")
            return
        failure = self._runtime_failure()
        if failure:
            return
        if self.pending_gripper is not None:
            self._confirm_pending_if_ready()
            if self.pending_gripper is None:
                return
            self._publish([*self.latest_joints, self.pending_gripper])
            return
        with self.lock:
            if not self.queue:
                return
            action = self.queue.popleft()
        gripper = 1.0 if self.phase == Phase.APPROACH else 0.0
        self._publish([*action[:6], gripper])

    def _publish(self, values):
        if self.publisher is None:
            self.get_logger().info(
                "SHADOW command " + ", ".join(f"{value:.5f}" for value in values)
            )
            return
        message = Float64MultiArray()
        message.data = [float(value) for value in values]
        self.publisher.publish(message)


def wait_for_server(process, timeout=120.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("inference server exited during startup")
        try:
            with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=0.2):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError("timed out waiting for inference server")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shadow",
        action="store_true",
        help="disable /control_abstraction_layer output for offline diagnostics",
    )
    args = parser.parse_args()
    if not MODEL_PATH.is_dir():
        raise SystemExit(f"MODEL_PATH does not exist: {MODEL_PATH}")
    phases = DemonstrationPhaseIndex(RAW_EPISODES)
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
        node = PhasedPolicyExecutor(not args.shadow, phases)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
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
