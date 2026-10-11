"""ROS 2 live-observation, shadow-only policy inference client.

The node creates subscriptions and a localhost inference client only. It never
creates a controller or gripper publisher. Live output remains hard-locked.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .core import UR5_JOINT_ORDER, ValidationError, reorder_joint_positions
from .safety import PredictionStabilityGate, StabilityConfig
from .shadow_protocol import ImagePayload, ProtocolError, request_prediction
from .action_semantics import ABSOLUTE_ACTION_SEMANTICS, DELTA_ACTION_SEMANTICS, delta_to_absolute


EXPECTED_INPUTS = {
    "observation.images.base": [3, 480, 640],
    "observation.images.wrist": [3, 480, 640],
    "observation.state": [7],
}


@dataclass(frozen=True)
class TimedImage:
    payload: ImagePayload
    received_at: float


class ShadowObservationBuffer:
    """Thread-safe latest-value buffer with receive-time freshness checks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._images: dict[str, TimedImage] = {}
        self._joints: tuple[list[float], float] | None = None
        self._gripper: tuple[float, float] | None = None
        self._servo: tuple[int, float] | None = None

    def update_image(
        self, key: str, height: int, width: int, encoding: str, step: int, data: bytes, now: float
    ) -> None:
        payload = ImagePayload(key, int(height), int(width), str(encoding), int(step), bytes(data))
        with self._lock:
            self._images[key] = TimedImage(payload, float(now))

    def update_joints(self, values: Sequence[float], now: float) -> None:
        joints = [float(value) for value in values]
        if len(joints) != 6 or not all(math.isfinite(value) for value in joints):
            raise ValidationError("joint state must contain six finite ordered values")
        with self._lock:
            self._joints = (joints, float(now))

    def update_gripper(self, value: float, now: float) -> None:
        gripper = float(value)
        if not math.isfinite(gripper):
            raise ValidationError("gripper state is not finite")
        with self._lock:
            self._gripper = (gripper, float(now))

    def update_servo(self, value: int, now: float) -> None:
        with self._lock:
            self._servo = (int(value), float(now))

    def snapshot(
        self,
        now: float,
        max_sensor_age_s: float,
        max_gripper_age_s: float,
        max_camera_age_s: float | None = None,
    ) -> tuple[list[float], list[ImagePayload], dict[str, float]]:
        with self._lock:
            images = dict(self._images)
            joints = None if self._joints is None else (list(self._joints[0]), self._joints[1])
            gripper = self._gripper
            servo = self._servo
        required = ("observation.images.base", "observation.images.wrist")
        missing = [key for key in required if key not in images]
        if missing or joints is None or gripper is None or servo is None:
            labels = [*missing]
            if joints is None:
                labels.append("joint_state")
            if gripper is None:
                labels.append("gripper_state")
            if servo is None:
                labels.append("servo_status")
            raise ValidationError(f"waiting for live inputs: {labels}")
        ages = {
            "base_image": max(0.0, now - images[required[0]].received_at),
            "wrist_image": max(0.0, now - images[required[1]].received_at),
            "joint_state": max(0.0, now - joints[1]),
            "gripper_state": max(0.0, now - gripper[1]),
            "servo_status": max(0.0, now - servo[1]),
        }
        camera_limit = max_sensor_age_s if max_camera_age_s is None else max_camera_age_s
        stale = [key for key in ("base_image", "wrist_image") if ages[key] > camera_limit]
        stale.extend(key for key in ("joint_state", "servo_status") if ages[key] > max_sensor_age_s)
        if ages["gripper_state"] > max_gripper_age_s:
            stale.append("gripper_state")
        if stale:
            raise ValidationError(f"stale live inputs: {stale}; ages={ages}")
        if servo[0] != 0:
            raise ValidationError(f"Servo status is nonzero: {servo[0]}")
        gripper_value = float(gripper[0])
        if min(abs(gripper_value), abs(gripper_value - 1.0)) > 1e-3:
            raise ValidationError(f"estimated gripper state is not binary: {gripper_value}")
        state = [*joints[0], float(round(gripper_value))]
        return state, [images[key].payload for key in required], ages


def validate_local_artifacts(checkpoint: Path, dataset_root: Path, repo_id: str) -> str:
    config_path = checkpoint / "config.json"
    info_path = dataset_root / "meta/info.json"
    conversion_path = dataset_root / "meta/ur5_vla_conversion.json"
    for path in (config_path, info_path, conversion_path):
        if not path.is_file():
            raise ValidationError(f"required local artifact is missing: {path}")
    config = json.loads(config_path.read_text())
    inputs = config.get("input_features", {})
    if set(inputs) != set(EXPECTED_INPUTS):
        raise ValidationError(f"checkpoint input feature mismatch: {sorted(inputs)}")
    for key, shape in EXPECTED_INPUTS.items():
        if inputs[key].get("shape") != shape:
            raise ValidationError(f"checkpoint {key} shape mismatch: {inputs[key].get('shape')}")
    if config.get("output_features", {}).get("action", {}).get("shape") != [7]:
        raise ValidationError("checkpoint action feature must be 7D")
    train_config_path = checkpoint / "train_config.json"
    if train_config_path.is_file():
        trained_repo = json.loads(train_config_path.read_text()).get("dataset", {}).get("repo_id")
        if trained_repo and trained_repo != repo_id:
            raise ValidationError(f"checkpoint trained on {trained_repo}, not {repo_id}")
    info = json.loads(info_path.read_text())
    if info.get("robot_type") != "ur5" or info.get("features", {}).get("action", {}).get("shape") != [7]:
        raise ValidationError("dataset is not the expected 7D UR5 dataset")
    conversion = json.loads(conversion_path.read_text())
    if conversion.get("joint_order") != list(UR5_JOINT_ORDER):
        raise ValidationError("dataset joint order does not match the live canonical order")
    action_semantics = conversion.get("action_semantics")
    if action_semantics not in {ABSOLUTE_ACTION_SEMANTICS, DELTA_ACTION_SEMANTICS}:
        raise ValidationError("dataset action semantics do not match the shadow adapter")
    return action_semantics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task", default="Pick up the orange cube and place it in the bowl")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8765)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--inference-period", type=float, default=1.0)
    parser.add_argument("--max-sensor-age", type=float, default=0.25)
    parser.add_argument("--max-gripper-age", type=float, default=0.50)
    parser.add_argument("--max-target-delta", type=float, default=0.012566)
    parser.add_argument("--max-prediction-jitter", type=float, default=0.005)
    parser.add_argument("--stable-predictions", type=int, default=100)
    parser.add_argument("--live", action="store_true", help="request physical output; deliberately unsupported")
    args = parser.parse_args()
    if args.live:
        raise SystemExit(
            "LIVE MODE IS LOCKED: shadow inference must be reviewed before a separate site-specific "
            "controller/External-Control/workspace/deadman adapter is implemented."
        )
    if args.inference_period <= 0 or args.request_timeout <= 0:
        raise SystemExit("inference period and request timeout must be positive")
    action_semantics = validate_local_artifacts(args.checkpoint, args.dataset_root, args.repo_id)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float64, Int8

    class ShadowExecutor(Node):
        def __init__(self):
            super().__init__("ur5_vla_policy_shadow")
            self.buffer = ShadowObservationBuffer()
            self._inference_lock = threading.Lock()
            self._inference_running = False
            self._stopping = threading.Event()
            self.stability_gate = PredictionStabilityGate(StabilityConfig(
                max_target_delta_rad=args.max_target_delta,
                max_prediction_jitter_rad=args.max_prediction_jitter,
                required_consecutive=args.stable_predictions,
            ))
            self.create_subscription(
                Image,
                "/base/camera/color/image_raw",
                lambda msg: self._image("observation.images.base", msg),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                Image,
                "/wrist/camera/color/image_raw",
                lambda msg: self._image("observation.images.wrist", msg),
                qos_profile_sensor_data,
            )
            self.create_subscription(JointState, "/joint_states", self._joints, qos_profile_sensor_data)
            self.create_subscription(Float64, "/gripper/state", self._gripper, 10)
            self.create_subscription(Int8, "/servo_node/status", self._servo, 10)
            self.create_timer(args.inference_period, self._schedule_inference)
            self.get_logger().warning(
                "SHADOW INFERENCE ONLY: no motion or gripper publishers have been created"
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

        def _gripper(self, message):
            try:
                self.buffer.update_gripper(message.data, time.monotonic())
            except ValidationError as exc:
                self.get_logger().error(str(exc))

        def _servo(self, message):
            self.buffer.update_servo(message.data, time.monotonic())

        def _joints(self, message):
            try:
                ordered = reorder_joint_positions(message.name, message.position, UR5_JOINT_ORDER)
                self.buffer.update_joints(ordered, time.monotonic())
            except Exception as exc:
                self.get_logger().error(f"rejected joint state: {exc}")

        def _schedule_inference(self):
            with self._inference_lock:
                if self._inference_running or self._stopping.is_set():
                    return
                try:
                    state, images, ages = self.buffer.snapshot(
                        time.monotonic(), args.max_sensor_age, args.max_gripper_age
                    )
                except ValidationError as exc:
                    self.get_logger().warning(f"shadow inference skipped: {exc}")
                    return
                self._inference_running = True
            threading.Thread(
                target=self._infer,
                args=(state, images, ages),
                daemon=True,
                name="shadow-inference-request",
            ).start()

        def _infer(self, state, images, ages):
            started = time.monotonic()
            try:
                chunk, model_latency = request_prediction(
                    args.server_host,
                    args.server_port,
                    args.request_timeout,
                    state,
                    args.task,
                    images,
                )
                round_trip = time.monotonic() - started
                first = chunk[0]
                gripper_binary = int(first[6] >= 0.5)
                raw_action = [*first[:6], float(gripper_binary)]
                if action_semantics == DELTA_ACTION_SEMANTICS:
                    action = delta_to_absolute(state, raw_action, args.max_target_delta).tolist()
                else:
                    action = raw_action
                deltas = [action[index] - state[index] for index in range(6)]
                displayed_action = [round(value, 5) for value in action[:6]] + [gripper_binary]
                eligible, gate_reason = self.stability_gate.observe(
                    action, state[:6]
                )
                self.get_logger().info(
                    "SHADOW prediction only "
                    f"action={displayed_action} raw_gripper={first[6]:.5f} "
                    f"max_abs_delta={max(map(abs, deltas)):.5f}rad "
                    f"eligible={eligible} gate='{gate_reason}' "
                    f"model_latency={model_latency:.3f}s round_trip={round_trip:.3f}s ages={ages}"
                )
            except (OSError, ProtocolError, ValueError) as exc:
                self.stability_gate.reset()
                self.get_logger().error(f"shadow inference failed: {exc}")
            finally:
                with self._inference_lock:
                    self._inference_running = False

        def destroy_node(self):
            self._stopping.set()
            return super().destroy_node()

    rclpy.init()
    node = ShadowExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
