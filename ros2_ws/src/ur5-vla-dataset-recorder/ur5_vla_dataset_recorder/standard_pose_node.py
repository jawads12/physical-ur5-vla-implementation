"""Move the physical UR5 to the project's fixed demonstration start pose.

This command is intentionally conservative: it requires a healthy UR program,
normal/reduced safety mode, a speed slider at or below 20%, no Servo command
publishers, and a collision-free interpolated joint path before executing one
slow scaled joint trajectory.

The dataset-start entry point can lower an excessive speed slider before
planning, but never raises a stopped or already-low speed setting.
"""

from __future__ import annotations

import math
import time

from .core import UR5_JOINT_ORDER, reorder_joint_positions


# Previously observed, stationary real-robot pose with a comfortably bent
# elbow. This is the canonical start pose for the next dataset generation.
STANDARD_POSE_RAD = (
    3.2679903507,
    -1.6692879836,
    1.7310266495,
    -1.9356344382,
    -1.6694858710,
    -0.0394843260,
)

MAX_SPEED_SCALING_PERCENT = 20.0
NOMINAL_JOINT_SPEED_RAD_S = 0.10
MIN_TRAJECTORY_SECONDS = 5.0
COLLISION_SAMPLE_STEP_RAD = 0.04
FINAL_TOLERANCE_RAD = 0.02
FAST_STATE_MAX_AGE_SECONDS = 0.5
STATE_VALIDITY_ATTEMPTS = 3


def interpolated_joint_path(start, target, max_step=COLLISION_SAMPLE_STEP_RAD):
    if len(start) != 6 or len(target) != 6:
        raise ValueError("start and target must each contain six joints")
    largest = max(abs(float(b) - float(a)) for a, b in zip(start, target))
    segments = max(1, math.ceil(largest / max_step))
    return [
        [float(a) + (float(b) - float(a)) * index / segments for a, b in zip(start, target)]
        for index in range(segments + 1)
    ]


def trajectory_duration(start, target):
    largest = max(abs(float(b) - float(a)) for a, b in zip(start, target))
    return max(MIN_TRAJECTORY_SECONDS, largest / NOMINAL_JOINT_SPEED_RAD_S)


def retry_state_validity_query(query, before_retry):
    """Retry only missing responses to a read-only MoveIt validity query.

    Returned collision rejections and non-timeout failures are never retried.
    before_retry rechecks robot readiness before another request is sent.
    """
    for attempt in range(1, STATE_VALIDITY_ATTEMPTS + 1):
        try:
            return query()
        except TimeoutError:
            if attempt == STATE_VALIDITY_ATTEMPTS:
                raise
            before_retry(attempt + 1)


def ensure_reset_speed_limit(read_safe_speed, request_fraction, wait_for_feedback):
    """Lower an excessive speed, requiring fresh feedback before proceeding.

    read_safe_speed must check freshness, robot readiness and command ownership.
    A zero slider is an operator stop; it must never be raised automatically.
    """
    speed = read_safe_speed()
    if not math.isfinite(speed) or not 0.0 < speed <= 100.0:
        raise RuntimeError(f"speed scaling must be finite and in (0, 100]%, got {speed}%")
    if speed <= MAX_SPEED_SCALING_PERCENT:
        return False
    if not request_fraction(MAX_SPEED_SCALING_PERCENT / 100.0):
        raise RuntimeError("UR controller rejected speed-slider reduction")
    wait_for_feedback()
    speed = read_safe_speed()
    if not 0.0 < speed <= MAX_SPEED_SCALING_PERCENT:
        raise RuntimeError(f"speed-slider reduction not confirmed: got {speed}%")
    return True


def main(args=None, *, resume_servo=False, auto_reduce_speed=False):
    import rclpy
    from action_msgs.msg import GoalStatus
    from control_msgs.action import FollowJointTrajectory
    from controller_manager_msgs.srv import ListControllers, SwitchController
    from moveit_msgs.srv import GetStateValidity
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool, Float64
    from std_srvs.srv import Trigger
    from trajectory_msgs.msg import JointTrajectoryPoint
    from ur_dashboard_msgs.msg import SafetyMode
    from ur_msgs.srv import SetSpeedSliderFraction

    class StandardPoseMover(Node):
        def __init__(self):
            super().__init__("ur5_standard_pose_mover")
            self.joints = None
            self.joints_at = 0.0
            self.program_running = None
            self.program_at = 0.0
            self.safety_mode = None
            self.safety_at = 0.0
            self.speed_scaling = None
            self.speed_at = 0.0

            state_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(JointState, "/joint_states", self._joints, qos_profile_sensor_data)
            self.create_subscription(
                Bool,
                "/io_and_status_controller/robot_program_running",
                self._program,
                state_qos,
            )
            self.create_subscription(
                SafetyMode,
                "/io_and_status_controller/safety_mode",
                self._safety,
                state_qos,
            )
            self.create_subscription(
                Float64,
                "/speed_scaling_state_broadcaster/speed_scaling",
                self._speed,
                state_qos,
            )
            self.list_client = self.create_client(ListControllers, "/controller_manager/list_controllers")
            self.switch_client = self.create_client(SwitchController, "/controller_manager/switch_controller")
            self.validity_client = self.create_client(GetStateValidity, "/check_state_validity")
            self.stop_servo_client = self.create_client(Trigger, "/servo_node/stop_servo")
            self.start_servo_client = self.create_client(Trigger, "/servo_node/start_servo")
            self.speed_client = self.create_client(
                SetSpeedSliderFraction, "/io_and_status_controller/set_speed_slider"
            )
            self.trajectory_client = ActionClient(
                self,
                FollowJointTrajectory,
                "/scaled_joint_trajectory_controller/follow_joint_trajectory",
            )

        def _joints(self, message):
            self.joints = reorder_joint_positions(message.name, message.position, UR5_JOINT_ORDER)
            self.joints_at = time.monotonic()

        def _program(self, message):
            self.program_running = bool(message.data)
            self.program_at = time.monotonic()

        def _safety(self, message):
            self.safety_mode = int(message.mode)
            self.safety_at = time.monotonic()

        def _speed(self, message):
            self.speed_scaling = float(message.data)
            self.speed_at = time.monotonic()

        def wait_for_inputs(self, timeout=5.0):
            deadline = time.monotonic() + timeout
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
                if all(
                    value is not None
                    for value in (self.joints, self.program_running, self.safety_mode, self.speed_scaling)
                ):
                    return
            raise RuntimeError("timed out waiting for fresh robot safety state")

        def assert_live_safety(self, *, allow_speed_reduction=False):
            now = time.monotonic()
            ages = {
                "joints": now - self.joints_at,
                "program": now - self.program_at,
                "safety": now - self.safety_at,
                "speed": now - self.speed_at,
            }
            if ages["joints"] > FAST_STATE_MAX_AGE_SECONDS or ages["speed"] > FAST_STATE_MAX_AGE_SECONDS:
                raise RuntimeError(f"fast robot state became stale: ages={ages}")
            # Program and safety mode are transient-local state-change topics,
            # not periodic heartbeats. Their last values remain authoritative;
            # a state change is delivered immediately and checked below.
            if not self.program_running:
                raise RuntimeError("UR External Control program is not running")
            if self.safety_mode not in (SafetyMode.NORMAL, SafetyMode.REDUCED):
                raise RuntimeError(f"unsafe UR safety mode: {self.safety_mode}")
            speed_limit = 100.0 if allow_speed_reduction else MAX_SPEED_SCALING_PERCENT
            if not 0.0 < self.speed_scaling <= speed_limit:
                raise RuntimeError(
                    f"speed slider must be in (0, {speed_limit}]%, "
                    f"got {self.speed_scaling}%"
                )
            competing = []
            for topic in ("/servo_node/delta_joint_cmds", "/servo_node/delta_twist_cmds"):
                if self.get_publishers_info_by_topic(topic):
                    competing.append(topic)
            if competing:
                raise RuntimeError(f"competing Servo command publisher exists: {competing}")

        def prepare_reset_speed(self):
            def read_safe_speed():
                # Only the upper speed bound is relaxed while REDUCING it.
                # No controller switching or trajectory dispatch happens here.
                self.assert_live_safety(allow_speed_reduction=True)
                return self.speed_scaling

            def request_fraction(fraction):
                self.get_logger().warning(
                    f"Reducing speed slider to {100.0 * fraction:.1f}% before guarded reset"
                )
                request = SetSpeedSliderFraction.Request()
                request.speed_slider_fraction = fraction
                return self.call(self.speed_client, request, "speed-slider reduction").success

            def wait_for_feedback():
                # Do not accept a sample received before the service completed.
                after = time.monotonic()
                deadline = after + 3.0
                while rclpy.ok() and time.monotonic() < deadline:
                    rclpy.spin_once(self, timeout_sec=0.05)
                    speed = read_safe_speed()
                    if self.speed_at > after and speed <= MAX_SPEED_SCALING_PERCENT:
                        return
                raise TimeoutError("speed-slider reduction did not produce fresh feedback at or below 20%")

            if ensure_reset_speed_limit(read_safe_speed, request_fraction, wait_for_feedback):
                self.get_logger().info(
                    f"Reset speed confirmed: {self.speed_scaling:.1f}%; slider remains reduced after reset"
                )

        def call(self, client, request, description, timeout=5.0):
            if not client.wait_for_service(timeout_sec=timeout):
                raise RuntimeError(f"{description} service unavailable")
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
            if not future.done():
                client.remove_pending_request(future)
                future.cancel()
                raise TimeoutError(f"{description} service timed out")
            if future.result() is None:
                raise RuntimeError(f"{description} service returned no response")
            return future.result()

        def active_controllers(self):
            response = self.call(self.list_client, ListControllers.Request(), "controller list")
            return {item.name for item in response.controller if item.state == "active"}

        def validate_path(self, start):
            path = interpolated_joint_path(start, STANDARD_POSE_RAD)
            self.get_logger().info(f"Checking {len(path)} interpolated states for self-collision")
            for index, positions in enumerate(path):
                request = GetStateValidity.Request()
                request.group_name = "ur_manipulator"
                request.robot_state.is_diff = True
                request.robot_state.joint_state.name = list(UR5_JOINT_ORDER)
                request.robot_state.joint_state.position = positions
                description = f"MoveIt state-validity sample {index + 1}/{len(path)}"

                def before_retry(attempt):
                    self.assert_live_safety()
                    self.get_logger().warning(
                        f"{description}: no service response; retrying read-only "
                        f"check ({attempt}/{STATE_VALIDITY_ATTEMPTS})"
                    )

                response = retry_state_validity_query(
                    lambda: self.call(self.validity_client, request, description),
                    before_retry,
                )
                if not response.valid:
                    contacts = [f"{item.contact_body_1}<->{item.contact_body_2}" for item in response.contacts]
                    raise RuntimeError(
                        f"MoveIt rejected path sample {index + 1}/{len(path)}: {contacts or 'collision'}"
                    )

        def switch_to_scaled_controller(self):
            if self.stop_servo_client.wait_for_service(timeout_sec=2.0):
                self.call(self.stop_servo_client, Trigger.Request(), "Servo stop")
            active = self.active_controllers()
            if "scaled_joint_trajectory_controller" in active:
                return
            request = SwitchController.Request()
            request.activate_controllers = ["scaled_joint_trajectory_controller"]
            request.deactivate_controllers = [
                name
                for name in (
                    "forward_position_controller",
                    "joint_trajectory_controller",
                    "forward_velocity_controller",
                    "forward_effort_controller",
                )
                if name in active
            ]
            request.strictness = SwitchController.Request.STRICT
            request.activate_asap = True
            request.timeout.sec = 5
            response = self.call(self.switch_client, request, "controller switch", timeout=8.0)
            if not response.ok:
                raise RuntimeError("controller manager rejected switch to scaled trajectory controller")

        def switch_to_servo_controller(self):
            active = self.active_controllers()
            if "forward_position_controller" not in active:
                request = SwitchController.Request()
                request.activate_controllers = ["forward_position_controller"]
                request.deactivate_controllers = [
                    name
                    for name in (
                        "scaled_joint_trajectory_controller",
                        "joint_trajectory_controller",
                        "forward_velocity_controller",
                        "forward_effort_controller",
                    )
                    if name in active
                ]
                request.strictness = SwitchController.Request.STRICT
                request.activate_asap = True
                request.timeout.sec = 5
                response = self.call(self.switch_client, request, "Servo controller switch", timeout=8.0)
                if not response.ok:
                    raise RuntimeError("controller manager rejected switch to forward position controller")
            response = self.call(self.start_servo_client, Trigger.Request(), "Servo start")
            if not response.success:
                raise RuntimeError(f"MoveIt Servo refused to start: {response.message}")
            active = self.active_controllers()
            if "forward_position_controller" not in active:
                raise RuntimeError("forward_position_controller did not become active")
            self.get_logger().info(
                "DATASET START READY: standard pose reached, forward_position_controller active, Servo started"
            )

        def execute(self):
            self.wait_for_inputs()
            if auto_reduce_speed:
                self.prepare_reset_speed()
            self.assert_live_safety()
            start = list(self.joints)
            target = list(STANDARD_POSE_RAD)
            duration = trajectory_duration(start, target)
            self.get_logger().warning(
                "PHYSICAL RESET: moving to fixed standard pose; keep E-stop ready and swept path clear"
            )
            self.get_logger().info(f"Current joints: {[round(value, 6) for value in start]}")
            self.get_logger().info(f"Target joints:  {[round(value, 6) for value in target]}")
            self.validate_path(start)
            self.assert_live_safety()
            self.switch_to_scaled_controller()
            if not self.trajectory_client.wait_for_server(timeout_sec=5.0):
                raise RuntimeError("scaled trajectory action server unavailable")

            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = list(UR5_JOINT_ORDER)
            point = JointTrajectoryPoint()
            point.positions = target
            point.time_from_start.sec = int(duration)
            point.time_from_start.nanosec = int((duration - int(duration)) * 1_000_000_000)
            goal.trajectory.points = [point]
            goal.goal_time_tolerance.sec = 10

            self.assert_live_safety()
            goal_future = self.trajectory_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, goal_future, timeout_sec=5.0)
            if not goal_future.done() or goal_future.result() is None:
                raise RuntimeError("trajectory goal request timed out")
            handle = goal_future.result()
            if not handle.accepted:
                raise RuntimeError("scaled trajectory controller rejected standard pose goal")

            self.get_logger().warning(f"Trajectory accepted; nominal duration={duration:.1f}s")
            result_future = handle.get_result_async()
            while rclpy.ok() and not result_future.done():
                rclpy.spin_once(self, timeout_sec=0.05)
                try:
                    self.assert_live_safety()
                except Exception:
                    cancel = handle.cancel_goal_async()
                    rclpy.spin_until_future_complete(self, cancel, timeout_sec=2.0)
                    raise
            wrapped = result_future.result()
            if wrapped is None or wrapped.status != GoalStatus.STATUS_SUCCEEDED:
                code = wrapped.result.error_code if wrapped is not None else None
                text = wrapped.result.error_string if wrapped is not None else "no result"
                raise RuntimeError(f"standard pose trajectory failed: status={getattr(wrapped, 'status', None)}, code={code}, {text}")

            final_error = max(abs(a - b) for a, b in zip(self.joints, target))
            if final_error > FINAL_TOLERANCE_RAD:
                raise RuntimeError(f"trajectory reported success but final error is {final_error:.6f} rad")
            self.get_logger().info(
                f"STANDARD POSE REACHED; maximum final joint error={final_error:.6f} rad"
            )

    rclpy.init(args=args)
    node = StandardPoseMover()
    try:
        node.execute()
        if resume_servo:
            node.switch_to_servo_controller()
    except KeyboardInterrupt:
        node.get_logger().error("standard pose movement interrupted")
        raise SystemExit(130)
    except Exception as error:
        node.get_logger().error(f"STANDARD POSE BLOCKED/FAILED: {error}")
        raise SystemExit(1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def prepare_dataset_start(args=None):
    """Reset to the standard pose and leave the arm ready for Servo teleoperation."""
    return main(args=args, resume_servo=True, auto_reduce_speed=True)


if __name__ == "__main__":
    main()
