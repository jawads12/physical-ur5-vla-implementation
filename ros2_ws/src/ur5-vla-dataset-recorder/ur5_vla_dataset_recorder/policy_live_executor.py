"""One-shot, deadman-gated physical UR5 policy executor through MoveIt Servo.

This node never switches controllers and never executes a policy action chunk.
It publishes one short JointJog pulse only after an explicit arm service call,
100 stable predictions, collision checking, and all live readiness checks.
The pneumatic gripper is deliberately disabled in this first physical phase.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

from .core import UR5_JOINT_ORDER, ValidationError, reorder_joint_positions
from .live_control import LiveReadiness, OneShotArmGate
from .policy_executor import ShadowObservationBuffer, validate_local_artifacts
from .safety import PredictionStabilityGate, SafetyConfig, SafetyFault, SafetySupervisor, StabilityConfig
from .shadow_protocol import ProtocolError, request_prediction
from .action_semantics import DELTA_ACTION_SEMANTICS, delta_to_absolute


RISK_ACK = "I_UNDERSTAND_ONE_ACTION_MOVES_UR5"


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
    parser.add_argument("--max-target-delta", type=float, default=0.012566)
    parser.add_argument("--max-prediction-jitter", type=float, default=0.005)
    parser.add_argument("--stable-predictions", type=int, default=100)
    parser.add_argument("--max-joint-speed", type=float, default=0.02)
    parser.add_argument("--pulse-duration", type=float, default=0.10)
    parser.add_argument("--max-speed-scaling", type=float, default=20.0)
    parser.add_argument("--deadman-button", type=int, default=0)
    parser.add_argument("--risk-ack", required=True)
    args = parser.parse_args()
    if args.risk_ack != RISK_ACK:
        raise SystemExit(f"Refusing physical mode: --risk-ack must equal {RISK_ACK!r}")
    if not (0 < args.pulse_duration <= 0.10):
        raise SystemExit("pulse duration must be in (0, 0.10] seconds")
    if not (0 < args.max_joint_speed <= 0.02):
        raise SystemExit("max joint speed must be in (0, 0.02] rad/s")
    action_semantics = validate_local_artifacts(args.checkpoint, args.dataset_root, args.repo_id)

    import rclpy
    from control_msgs.msg import JointJog
    from controller_manager_msgs.srv import ListControllers
    from moveit_msgs.srv import GetStateValidity
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import Image, JointState, Joy
    from std_msgs.msg import Bool, Float64, Int8
    from std_srvs.srv import Trigger
    from ur_dashboard_msgs.msg import SafetyMode

    class LiveExecutor(Node):
        def __init__(self):
            super().__init__("ur5_vla_policy_live_one_shot")
            self.buffer = ShadowObservationBuffer()
            self.lock = threading.Lock()
            self.inference_running = False
            self.stopping = False
            self.latest_action = None
            self.latest_action_at = 0.0
            self.latest_model_latency = math.inf
            self.deadman = (False, 0.0)
            self.program = (False, 0.0)
            self.safety_mode = (0, 0.0)
            self.speed_scaling = (math.inf, 0.0)
            self.controller = ("unknown", 0.0)
            self.pending_collision = False
            self.motion_until = 0.0
            self.stability = PredictionStabilityGate(StabilityConfig(
                args.max_target_delta, args.max_prediction_jitter, args.stable_predictions
            ))
            safety_config = SafetyConfig(
                max_target_delta_rad=args.max_target_delta,
                max_velocity_rad_s=args.max_joint_speed,
                max_sensor_age_s=0.25,
                max_gripper_age_s=0.50,
                max_inference_age_s=max(1.0, args.request_timeout),
                expected_controller="forward_position_controller",
            )
            self.gate = OneShotArmGate(SafetySupervisor(safety_config))
            self.jog_pub = self.create_publisher(JointJog, "/servo_node/delta_joint_cmds", 10)
            self.controller_client = self.create_client(ListControllers, "/controller_manager/list_controllers")
            self.validity_client = self.create_client(GetStateValidity, "/check_state_validity")
            state_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(Image, "/base/camera/color/image_raw", lambda m: self._image("observation.images.base", m), qos_profile_sensor_data)
            self.create_subscription(Image, "/wrist/camera/color/image_raw", lambda m: self._image("observation.images.wrist", m), qos_profile_sensor_data)
            self.create_subscription(JointState, "/joint_states", self._joints, qos_profile_sensor_data)
            self.create_subscription(Float64, "/gripper/state", lambda m: self.buffer.update_gripper(m.data, time.monotonic()), 10)
            self.create_subscription(Int8, "/servo_node/status", lambda m: self.buffer.update_servo(m.data, time.monotonic()), 10)
            self.create_subscription(Joy, "/joy", self._joy, qos_profile_sensor_data)
            self.create_subscription(Bool, "/io_and_status_controller/robot_program_running", self._program, state_qos)
            self.create_subscription(SafetyMode, "/io_and_status_controller/safety_mode", self._safety, state_qos)
            self.create_subscription(Float64, "/speed_scaling_state_broadcaster/speed_scaling", self._speed, state_qos)
            self.create_service(Trigger, "/vla_live/arm_one_action", self._arm)
            self.create_service(Trigger, "/vla_live/clear_fault", self._clear)
            self.create_service(Trigger, "/vla_live/status", self._status)
            self.create_timer(args.inference_period, self._schedule_inference)
            self.create_timer(0.5, self._poll_controller)
            self.create_timer(0.02, self._deadman_monitor)
            self.get_logger().warning(
                "PHYSICAL ONE-SHOT MODE: arm publisher exists; gripper output disabled; "
                "no command is sent until /vla_live/arm_one_action passes every gate"
            )

        def _image(self, key, msg):
            self.buffer.update_image(key, msg.height, msg.width, msg.encoding, msg.step, bytes(msg.data), time.monotonic())

        def _joints(self, msg):
            try:
                self.buffer.update_joints(reorder_joint_positions(msg.name, msg.position, UR5_JOINT_ORDER), time.monotonic())
            except Exception as exc:
                self.get_logger().error(f"joint state rejected: {exc}")

        def _joy(self, msg):
            pressed = 0 <= args.deadman_button < len(msg.buttons) and int(msg.buttons[args.deadman_button]) == 1
            with self.lock:
                self.deadman = (pressed, time.monotonic())

        def _program(self, msg):
            with self.lock:
                self.program = (bool(msg.data), time.monotonic())

        def _safety(self, msg):
            with self.lock:
                self.safety_mode = (int(msg.mode), time.monotonic())

        def _speed(self, msg):
            with self.lock:
                self.speed_scaling = (float(msg.data), time.monotonic())

        def _poll_controller(self):
            if not self.controller_client.service_is_ready():
                return
            future = self.controller_client.call_async(ListControllers.Request())
            future.add_done_callback(self._controller_result)

        def _controller_result(self, future):
            try:
                active = [item.name for item in future.result().controller if item.state == "active"]
                value = "forward_position_controller" if "forward_position_controller" in active else ",".join(active)
                with self.lock:
                    self.controller = (value, time.monotonic())
            except Exception as exc:
                self.get_logger().error(f"controller query failed: {exc}")

        def _readiness_failures(self, require_prediction=True):
            now = time.monotonic()
            with self.lock:
                deadman, deadman_at = self.deadman
                program, program_at = self.program
                mode, mode_at = self.safety_mode
                speed, speed_at = self.speed_scaling
                controller, controller_at = self.controller
                action_at = self.latest_action_at
            failures = []
            if "logitech_servo_mapper" in self.get_node_names(): failures.append("stop logitech_servo_mapper to remove competing teleoperation")
            if not deadman or now - deadman_at > 0.25: failures.append("deadman button 0 is not held with fresh /joy data")
            if not program: failures.append("UR External Control program is not running")
            if mode not in (SafetyMode.NORMAL, SafetyMode.REDUCED): failures.append(f"unsafe UR safety mode {mode}")
            if not math.isfinite(speed) or speed > args.max_speed_scaling: failures.append(f"speed scaling must be <= {args.max_speed_scaling}%, got {speed}")
            if controller != "forward_position_controller" or now - controller_at > 1.0: failures.append(f"forward_position_controller not freshly active: {controller}")
            if require_prediction and (self.latest_action is None or now - action_at > max(1.5, args.inference_period * 2)): failures.append("no fresh stable policy prediction")
            if self.stability.consecutive < args.stable_predictions: failures.append(f"stable predictions {self.stability.consecutive}/{args.stable_predictions}")
            if self.pending_collision: failures.append("collision check already pending")
            if not self.validity_client.service_is_ready(): failures.append("MoveIt /check_state_validity service unavailable")
            return failures

        def _status(self, request, response):
            failures = self._readiness_failures()
            response.success = not failures
            response.message = "READY TO ARM" if not failures else "BLOCKED: " + "; ".join(failures)
            return response

        def _clear(self, request, response):
            self.gate.supervisor.clear_fault()
            self.gate.armed = False
            response.success = True
            response.message = "fault cleared; executor remains disarmed"
            return response

        def _arm(self, request, response):
            failures = self._readiness_failures()
            if failures:
                response.success = False
                response.message = "BLOCKED: " + "; ".join(failures)
                return response
            try:
                self.gate.arm()
            except Exception as exc:
                response.success = False
                response.message = str(exc)
                return response
            action = list(self.latest_action)
            collision_request = GetStateValidity.Request()
            collision_request.group_name = "ur_manipulator"
            collision_request.robot_state.is_diff = True
            collision_request.robot_state.joint_state.name = list(UR5_JOINT_ORDER)
            collision_request.robot_state.joint_state.position = action[:6]
            self.pending_collision = True
            future = self.validity_client.call_async(collision_request)
            future.add_done_callback(lambda completed: self._collision_result(completed, action))
            response.success = True
            response.message = "armed once; collision check pending; keep deadman held"
            return response

        def _collision_result(self, future, action):
            self.pending_collision = False
            try:
                failures = self._readiness_failures()
                if failures:
                    raise SafetyFault("; ".join(failures))
                collision_free = bool(future.result().valid)
                now = time.monotonic()
                state, _, ages = self.buffer.snapshot(now, 0.25, 0.50)
                with self.lock:
                    deadman, deadman_at = self.deadman
                    program, program_at = self.program
                    controller, controller_at = self.controller
                ages["inference"] = now - self.latest_action_at
                readiness = LiveReadiness(
                    joints=state[:6], ages_s=ages, servo_status=0,
                    controller=controller if now - controller_at <= 1.0 else "stale",
                    program_running=program,
                    deadman_held=deadman and now - deadman_at <= 0.25,
                    collision_free=collision_free,
                )
                period = max(args.pulse_duration, max(abs(action[i] - state[i]) for i in range(6)) / args.max_joint_speed)
                approved = self.gate.approve(action, readiness, period)
                errors = [approved[i] - state[i] for i in range(6)]
                msg = JointJog()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.joint_names = list(UR5_JOINT_ORDER)
                msg.velocities = [max(-args.max_joint_speed, min(args.max_joint_speed, error / args.pulse_duration)) for error in errors]
                msg.duration = args.pulse_duration
                self.jog_pub.publish(msg)
                self.motion_until = time.monotonic() + args.pulse_duration
                self.get_logger().warning(f"ONE PHYSICAL PULSE PUBLISHED velocities={msg.velocities}; executor disarmed")
            except Exception as exc:
                self.gate.cancel(str(exc))
                self.get_logger().error(f"PHYSICAL COMMAND BLOCKED: {exc}")

        def _deadman_monitor(self):
            if time.monotonic() >= self.motion_until:
                return
            with self.lock:
                held, stamp = self.deadman
            if held and time.monotonic() - stamp <= 0.25:
                return
            stop = JointJog()
            stop.header.stamp = self.get_clock().now().to_msg()
            stop.joint_names = list(UR5_JOINT_ORDER)
            stop.velocities = [0.0] * 6
            stop.duration = 0.0
            self.jog_pub.publish(stop)
            self.motion_until = 0.0
            self.gate.cancel("deadman released during pulse")
            self.get_logger().error("DEADMAN RELEASED: zero JointJog published and fault latched")

        def _schedule_inference(self):
            with self.lock:
                if self.inference_running or self.stopping:
                    return
                self.inference_running = True
            try:
                state, images, _ = self.buffer.snapshot(time.monotonic(), 0.25, 0.50)
            except ValidationError as exc:
                with self.lock: self.inference_running = False
                self.stability.reset()
                self.get_logger().warning(f"prediction skipped: {exc}")
                return
            threading.Thread(target=self._infer, args=(state, images), daemon=True).start()

        def _infer(self, state, images):
            started = time.monotonic()
            try:
                chunk, latency = request_prediction(args.server_host, args.server_port, args.request_timeout, state, args.task, images)
                first = list(chunk[0])
                gripper = float(int(first[6] >= 0.5))
                raw_action = [*first[:6], gripper]
                if action_semantics == DELTA_ACTION_SEMANTICS:
                    # Decode only after checking the raw learned delta. No clipping.
                    action = delta_to_absolute(state, raw_action, args.max_target_delta).tolist()
                else:
                    action = raw_action
                eligible, reason = self.stability.observe(action, state[:6])
                if eligible:
                    with self.lock:
                        self.latest_action = action
                        self.latest_action_at = time.monotonic()
                        self.latest_model_latency = latency
                self.get_logger().info(f"LIVE-GATED prediction eligible={eligible} gate='{reason}' max_delta={max(abs(action[i]-state[i]) for i in range(6)):.6f}")
            except (OSError, ProtocolError, ValueError) as exc:
                self.stability.reset()
                self.get_logger().error(f"inference failed: {exc}")
            finally:
                with self.lock: self.inference_running = False

        def destroy_node(self):
            with self.lock: self.stopping = True
            self.gate.cancel("node shutdown")
            return super().destroy_node()

    rclpy.init()
    node = LiveExecutor()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        executor.shutdown()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == "__main__":
    main()
