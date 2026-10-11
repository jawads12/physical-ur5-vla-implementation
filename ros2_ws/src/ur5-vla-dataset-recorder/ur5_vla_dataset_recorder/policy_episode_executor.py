"""Guarded receding-horizon UR5 pick-and-place policy executor.

Shadow mode is the default and creates no command publishers or SetIO client.
Physical mode requires an explicit acknowledgement and a separate arm service call.
Every short arm pulse is checked against fresh inputs and MoveIt state validity.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from pathlib import Path

from .action_semantics import DELTA_ACTION_SEMANTICS, delta_to_absolute
from .core import UR5_JOINT_ORDER, ValidationError, reorder_joint_positions
from .episode_control import EpisodeFault, EpisodeLimits, EpisodePhase, GuardedEpisodeState
from .episode_execution import PulseConfig, gripper_pins, plan_joint_pulse
from .policy_executor import ShadowObservationBuffer, validate_local_artifacts
from .safety import PredictionStabilityGate, StabilityConfig
from .shadow_protocol import ProtocolError, request_prediction


RISK_ACK = "I_UNDERSTAND_FULL_EPISODE_MOVES_UR5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task", default="Pick up the orange cube and place it in the bowl")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8765)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--inference-period", type=float, default=0.20)
    parser.add_argument("--max-camera-age", type=float, default=0.50)
    parser.add_argument("--mode", choices=("shadow", "physical"), default="shadow")
    parser.add_argument("--risk-ack", default="")
    parser.add_argument("--stable-predictions", type=int, default=100)
    parser.add_argument("--max-target-delta", type=float, default=0.0065)
    parser.add_argument("--max-prediction-jitter", type=float, default=0.005)
    parser.add_argument("--max-joint-speed", type=float, default=0.02)
    parser.add_argument("--pulse-duration", type=float, default=0.10)
    parser.add_argument("--max-speed-scaling", type=float, default=20.0)
    parser.add_argument("--deadman-button", type=int, default=0)
    parser.add_argument("--max-duration", type=float, default=210.0)
    parser.add_argument("--max-pulses", type=int, default=1000)
    args = parser.parse_args()
    if args.mode == "physical" and args.risk_ack != RISK_ACK:
        raise SystemExit(f"physical mode requires --risk-ack {RISK_ACK!r}")
    if not 0.0 < args.max_camera_age <= 0.50:
        raise SystemExit("--max-camera-age must be in (0, 0.50] seconds")
    pulse_config = PulseConfig(args.pulse_duration, args.max_joint_speed)
    semantics = validate_local_artifacts(args.checkpoint, args.dataset_root, args.repo_id)

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
    from ur_msgs.srv import SetIO

    class EpisodeExecutor(Node):
        def __init__(self):
            super().__init__("ur5_vla_policy_episode_executor")
            self.buffer = ShadowObservationBuffer()
            self.lock = threading.RLock()
            self.episode = GuardedEpisodeState(EpisodeLimits(
                max_duration_s=args.max_duration, max_pulses=args.max_pulses
            ))
            self.stability = PredictionStabilityGate(StabilityConfig(
                args.max_target_delta, args.max_prediction_jitter, args.stable_predictions
            ))
            self.latest_action = None
            self.latest_action_at = 0.0
            self.inference_running = False
            self.pending_collision = False
            self.gripper_transaction = False
            self.stopping = False
            self.deadman = (False, 0.0)
            self.program = (False, 0.0)
            self.safety_mode = (0, 0.0)
            self.speed = (math.inf, 0.0)
            self.controller = ("unknown", 0.0)
            self.last_gripper = None
            self.last_gripper_prediction_at = 0.0
            self.motion_until = 0.0
            self.controller_client = self.create_client(ListControllers, "/controller_manager/list_controllers")
            self.validity_client = self.create_client(GetStateValidity, "/check_state_validity")
            self.jog_pub = self.io_client = self.gripper_pub = self.command_pub = None
            if args.mode == "physical":
                self.jog_pub = self.create_publisher(JointJog, "/servo_node/delta_joint_cmds", 10)
                self.io_client = self.create_client(SetIO, "/io_and_status_controller/set_io")
                gripper_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
                self.gripper_pub = self.create_publisher(Float64, "/gripper/state", gripper_qos)
                self.command_pub = self.create_publisher(Float64, "/gripper/command", gripper_qos)
            state_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                   durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(Image, "/base/camera/color/image_raw", lambda m: self._image("observation.images.base", m), qos_profile_sensor_data)
            self.create_subscription(Image, "/wrist/camera/color/image_raw", lambda m: self._image("observation.images.wrist", m), qos_profile_sensor_data)
            self.create_subscription(JointState, "/joint_states", self._joints, qos_profile_sensor_data)
            self.create_subscription(Float64, "/gripper/state", self._gripper, 10)
            self.create_subscription(Int8, "/servo_node/status", lambda m: self.buffer.update_servo(m.data, time.monotonic()), 10)
            self.create_subscription(Joy, "/joy", self._joy, qos_profile_sensor_data)
            self.create_subscription(Bool, "/io_and_status_controller/robot_program_running", lambda m: self._set("program", bool(m.data)), state_qos)
            self.create_subscription(SafetyMode, "/io_and_status_controller/safety_mode", lambda m: self._set("safety_mode", int(m.mode)), state_qos)
            self.create_subscription(Float64, "/speed_scaling_state_broadcaster/speed_scaling", lambda m: self._set("speed", float(m.data)), state_qos)
            self.create_service(Trigger, "/vla_episode/arm", self._arm)
            self.create_service(Trigger, "/vla_episode/stop", self._stop_service)
            self.create_service(Trigger, "/vla_episode/clear_fault", self._clear)
            self.create_service(Trigger, "/vla_episode/status", self._status)
            self.create_timer(args.inference_period, self._schedule_inference)
            self.create_timer(0.05, self._control_tick)
            self.create_timer(0.5, self._poll_controller)
            self.create_timer(0.05, self._gripper_heartbeat)
            self.get_logger().warning(f"EPISODE EXECUTOR mode={args.mode}; starts DISARMED")

        def _set(self, name, value):
            with self.lock: setattr(self, name, (value, time.monotonic()))

        def _image(self, key, msg):
            self.buffer.update_image(key, msg.height, msg.width, msg.encoding, msg.step, bytes(msg.data), time.monotonic())

        def _joints(self, msg):
            try: self.buffer.update_joints(reorder_joint_positions(msg.name, msg.position, UR5_JOINT_ORDER), time.monotonic())
            except Exception as exc: self._fault(f"joint state rejected: {exc}")

        def _gripper(self, msg):
            value = float(round(msg.data))
            self.last_gripper = value
            self.buffer.update_gripper(value, time.monotonic())

        def _joy(self, msg):
            held = 0 <= args.deadman_button < len(msg.buttons) and int(msg.buttons[args.deadman_button]) == 1
            self._set("deadman", held)

        def _poll_controller(self):
            if not self.controller_client.service_is_ready(): return
            future = self.controller_client.call_async(ListControllers.Request())
            future.add_done_callback(self._controller_result)

        def _controller_result(self, future):
            try:
                active = [c.name for c in future.result().controller if c.state == "active"]
                self._set("controller", "forward_position_controller" if "forward_position_controller" in active else ",".join(active))
            except Exception as exc: self.get_logger().error(f"controller query failed: {exc}")

        def _failures(self, prediction=True):
            now = time.monotonic()
            with self.lock:
                deadman, deadman_at = self.deadman; program, program_at = self.program
                mode, mode_at = self.safety_mode; speed, speed_at = self.speed
                controller, controller_at = self.controller
            failures = []
            if "logitech_servo_mapper" in self.get_node_names(): failures.append("competing logitech_servo_mapper is running")
            if not deadman or now - deadman_at > .25: failures.append("deadman is not freshly held")
            # These UR status topics use transient-local QoS and may publish
            # only when the value changes. Receipt proves that the current
            # latched value was obtained; unlike sensor streams, age alone is
            # not a reason to reject it.
            if program_at == 0.0 or not program: failures.append("External Control is not running")
            if mode_at == 0.0 or mode not in (SafetyMode.NORMAL, SafetyMode.REDUCED): failures.append(f"unsafe/unknown safety mode {mode}")
            # This UR driver publishes the slider value in percent (observed
            # 20.0 at a 20% pendant setting), matching the CLI unit.
            if not math.isfinite(speed) or speed > args.max_speed_scaling or now - speed_at > 1.0:
                failures.append(f"speed scaling unsafe/stale: {speed:.1f}% (limit {args.max_speed_scaling:.1f}%)")
            if controller != "forward_position_controller" or now - controller_at > 1.0: failures.append(f"controller unsafe/stale: {controller}")
            if prediction and (self.latest_action is None or now - self.latest_action_at > max(.75, 3 * args.inference_period)): failures.append("policy prediction is stale")
            if not self.validity_client.service_is_ready(): failures.append("MoveIt state-validity service unavailable")
            if args.mode == "physical" and (self.io_client is None or not self.io_client.service_is_ready()): failures.append("UR SetIO unavailable")
            return failures

        def _status(self, request, response):
            failures = self._failures()
            response.success = not failures and self.stability.consecutive >= args.stable_predictions
            response.message = (f"phase={self.episode.phase.value} stable={self.stability.consecutive}/{args.stable_predictions} "
                                f"pulses={self.episode.pulses} failures={failures}")
            return response

        def _arm(self, request, response):
            failures = self._failures()
            if self.stability.consecutive < args.stable_predictions: failures.append("initial prediction qualification incomplete")
            if failures:
                response.success = False; response.message = "BLOCKED: " + "; ".join(failures); return response
            try:
                state, _, _ = self.buffer.snapshot(time.monotonic(), .25, .50, args.max_camera_age)
                self.episode.arm(time.monotonic(), state[:6], bool(state[6]))
            except Exception as exc:
                response.success = False; response.message = str(exc); return response
            response.success = True; response.message = f"{args.mode} episode armed; hold deadman continuously"
            return response

        def _stop_service(self, request, response):
            self._fault("operator stop requested")
            response.success = True; response.message = "zero command sent; fault latched"
            return response

        def _clear(self, request, response):
            if self.motion_until > time.monotonic() or self.pending_collision or self.gripper_transaction:
                response.success = False; response.message = "cannot clear while work is pending"; return response
            self.episode = GuardedEpisodeState(EpisodeLimits(max_duration_s=args.max_duration, max_pulses=args.max_pulses))
            self.stability.reset(); response.success = True; response.message = "cleared; remains disarmed and must requalify"
            return response

        def _zero(self):
            if self.jog_pub is None: return
            msg = JointJog(); msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = list(UR5_JOINT_ORDER); msg.velocities = [0.0] * 6; msg.duration = 0.0
            self.jog_pub.publish(msg); self.motion_until = 0.0

        def _fault(self, reason):
            if self.episode.phase != EpisodePhase.FAULT: self.episode.fault(str(reason))
            self._zero(); self.get_logger().error(f"EPISODE FAULT: {reason}")

        def _control_tick(self):
            if self.episode.phase in (EpisodePhase.DISARMED, EpisodePhase.COMPLETE, EpisodePhase.FAULT): return
            failures = self._failures()
            if failures: self._fault("; ".join(failures)); return
            try:
                now = time.monotonic(); self.episode.check_time(now)
                if self.episode.update_completion(now):
                    self._zero(); self.get_logger().warning("EPISODE COMPLETE after confirmed release and settle"); return
                if self.episode.phase == EpisodePhase.RELEASED:
                    self._zero()
                    return
                if self.pending_collision or now < self.motion_until: return
                state, _, _ = self.buffer.snapshot(now, .25, .50, args.max_camera_age)
                action = list(self.latest_action)
                if self.latest_action_at > self.last_gripper_prediction_at:
                    self.last_gripper_prediction_at = self.latest_action_at
                    command = self.episode.observe_gripper(now, bool(action[6]))
                    if command is not None:
                        if args.mode == "physical": self._start_gripper(bool(command))
                        else: self.last_gripper = float(command); self.buffer.update_gripper(command, now)
                request = GetStateValidity.Request(); request.group_name = "ur_manipulator"
                request.robot_state.is_diff = True; request.robot_state.joint_state.name = list(UR5_JOINT_ORDER)
                request.robot_state.joint_state.position = action[:6]
                self.pending_collision = True
                self.validity_client.call_async(request).add_done_callback(lambda f: self._collision_result(f, action))
            except Exception as exc: self._fault(exc)

        def _collision_result(self, future, action):
            self.pending_collision = False
            try:
                if not bool(future.result().valid): raise EpisodeFault("MoveIt rejected predicted target")
                failures = self._failures()
                if failures: raise EpisodeFault("; ".join(failures))
                now = time.monotonic(); state, _, _ = self.buffer.snapshot(now, .25, .50, args.max_camera_age)
                if max(abs(action[i] - state[i]) for i in range(6)) > args.max_target_delta:
                    raise EpisodeFault("target changed beyond per-pulse envelope before dispatch")
                velocities = plan_joint_pulse(action[:6], state[:6], pulse_config)
                self.episode.record_pulse(now, state[:6])
                if args.mode == "physical":
                    msg = JointJog(); msg.header.stamp = self.get_clock().now().to_msg()
                    msg.joint_names = list(UR5_JOINT_ORDER); msg.velocities = velocities; msg.duration = args.pulse_duration
                    self.jog_pub.publish(msg)
                self.motion_until = now + args.pulse_duration
            except Exception as exc: self._fault(exc)

        def _start_gripper(self, opened):
            if self.gripper_transaction: return
            self.gripper_transaction = True; writes = gripper_pins(opened)
            self._io_write(writes, 0, opened)

        def _io_write(self, writes, index, opened):
            pin, value = writes[index]; request = SetIO.Request()
            request.fun = 1; request.pin = pin; request.state = value
            self.io_client.call_async(request).add_done_callback(lambda f: self._io_result(f, writes, index, opened))

        def _io_result(self, future, writes, index, opened):
            try:
                if not future.result().success: raise EpisodeFault(f"SetIO pin {writes[index][0]} rejected")
                if index + 1 < len(writes): self._io_write(writes, index + 1, opened); return
                value = float(int(opened)); self.last_gripper = value
                message = Float64(); message.data = value
                self.gripper_pub.publish(message); self.command_pub.publish(message)
                self.buffer.update_gripper(value, time.monotonic()); self.gripper_transaction = False
            except Exception as exc:
                self.gripper_transaction = False; self._fault(exc)

        def _gripper_heartbeat(self):
            if args.mode == "physical" and self.last_gripper is not None:
                message = Float64(); message.data = self.last_gripper; self.gripper_pub.publish(message)

        def _schedule_inference(self):
            with self.lock:
                if self.inference_running or self.stopping: return
                self.inference_running = True
            try: state, images, _ = self.buffer.snapshot(time.monotonic(), .25, .50, args.max_camera_age)
            except ValidationError as exc:
                self.inference_running = False; self.stability.reset(); return
            threading.Thread(target=self._infer, args=(state, images), daemon=True).start()

        def _infer(self, state, images):
            try:
                chunk, _ = request_prediction(args.server_host, args.server_port, args.request_timeout, state, args.task, images)
                first = list(chunk[0]); raw = [*first[:6], float(int(first[6] >= .5))]
                action = delta_to_absolute(state, raw, args.max_target_delta).tolist() if semantics == DELTA_ACTION_SEMANTICS else raw
                eligible, reason = self.stability.observe(action, state[:6])
                # Before arming require the full stability qualification. During an
                # episode retain only individually bounded, finite predictions.
                if eligible or self.episode.phase not in (EpisodePhase.DISARMED, EpisodePhase.FAULT, EpisodePhase.COMPLETE):
                    if max(abs(action[i] - state[i]) for i in range(6)) > args.max_target_delta: raise ValueError("prediction exceeds envelope")
                    with self.lock: self.latest_action = action; self.latest_action_at = time.monotonic()
                self.get_logger().info(f"prediction eligible={eligible} {reason} phase={self.episode.phase.value}")
            except (OSError, ProtocolError, ValueError) as exc:
                self.stability.reset()
                if self.episode.phase not in (EpisodePhase.DISARMED, EpisodePhase.COMPLETE, EpisodePhase.FAULT): self._fault(f"inference failed: {exc}")
            finally:
                with self.lock: self.inference_running = False

        def destroy_node(self):
            with self.lock: self.stopping = True
            if rclpy.ok():
                try: self._zero()
                except Exception as exc: self.get_logger().warning(f"shutdown zero publish failed: {exc}")
            return super().destroy_node()

    rclpy.init(); node = EpisodeExecutor(); executor = MultiThreadedExecutor(num_threads=4); executor.add_node(node)
    try: executor.spin()
    except KeyboardInterrupt: pass
    finally:
        executor.remove_node(node); node.destroy_node(); executor.shutdown()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == "__main__":
    main()
