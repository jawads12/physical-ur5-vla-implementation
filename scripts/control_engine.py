#!/usr/bin/python3
"""Smooth seven-value abstract commands for the UR5 arm and gripper."""

import argparse
import math
import sys
import time

from project_paths import PACKAGE

sys.path.insert(0, str(PACKAGE))

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import Trigger
from ur_msgs.srv import SetIO

from ur5_vla_dataset_recorder.core import UR5_JOINT_ORDER, reorder_joint_positions


OUTPUT_HZ = 125.0
FILTER_TIME_CONSTANT = 0.28
MAX_VELOCITY = 0.10       # rad/s per joint
MAX_ACCELERATION = 0.30   # rad/s^2 per joint
MAX_TARGET_STEP = 0.15    # rad between consecutive 20 Hz policy targets
MAX_INITIAL_ERROR = 0.50  # rad from measured pose to first target


class ControlEngine(Node):
    def __init__(self, dry_run=False, validation_count=0):
        super().__init__("control_engine")
        self.dry_run = dry_run
        self.validation_count = validation_count
        self.output_count = 0
        self.validation_done = False
        self.last_source_warning = 0.0
        self.target = None
        self.filtered_target = None
        self.command = None
        self.velocity = [0.0] * 6
        self.last_raw_target = None
        self.last_tick = time.monotonic()
        controller_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.arm_publisher = self.create_publisher(
            Float64MultiArray,
            "/forward_position_controller/commands",
            controller_qos,
        )
        self.create_subscription(
            Float64MultiArray,
            "/control_abstraction_layer",
            self.command_callback,
            10,
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )
        self.io_client = self.create_client(
            SetIO, "/io_and_status_controller/set_io"
        )
        self.servo_stop_client = self.create_client(
            Trigger, "/servo_node/stop_servo"
        )
        self.last_gripper = None
        self.gripper_busy = False
        self.confirmed_gripper = None
        self.gripper_state_publisher = self.create_publisher(
            Float64, "/control_engine/gripper_state",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.create_timer(0.05, self.publish_gripper_confirmation)
        self.create_timer(1.0 / OUTPUT_HZ, self.publish_smoothed_arm)
        self.create_timer(1.0, self.ensure_servo_stopped)
        mode = "DRY RUN" if dry_run else "LIVE"
        self.get_logger().info(f"{mode}: smoothed arm output at {OUTPUT_HZ:.0f} Hz")

    def ensure_servo_stopped(self):
        if self.servo_stop_client.service_is_ready():
            self.servo_stop_client.call_async(Trigger.Request())

    def joint_state_callback(self, message):
        if self.command is not None:
            return
        try:
            measured = reorder_joint_positions(
                message.name, message.position, UR5_JOINT_ORDER
            )
        except Exception as error:
            self.get_logger().warning(f"Waiting for ordered joint state: {error}")
            return
        self.command = [float(value) for value in measured]
        self.filtered_target = list(self.command)

    def command_callback(self, message):
        if len(message.data) != 7:
            self.get_logger().error(
                f"Expected 7 values, received {len(message.data)}"
            )
            return

        # Gripper transitions are discrete and deliberately independent of
        # arm filtering/rejection.
        gripper = 1.0 if float(message.data[6]) >= 0.5 else 0.0
        if gripper != self.last_gripper and not self.gripper_busy:
            self.last_gripper = gripper
            if self.dry_run:
                self.get_logger().info(
                    f"DRY RUN gripper {'OPEN' if gripper else 'CLOSED'}"
                )
            else:
                self.start_gripper(gripper)

        candidate = [float(value) for value in message.data[:6]]
        if not all(math.isfinite(value) for value in candidate):
            self.get_logger().error("Rejected non-finite arm target")
            return
        reference = self.last_raw_target if self.last_raw_target is not None else self.command
        limit = MAX_TARGET_STEP if self.last_raw_target is not None else MAX_INITIAL_ERROR
        if reference is None:
            self.get_logger().warning("Rejected target while waiting for /joint_states")
            return
        largest_step = max(abs(a - b) for a, b in zip(candidate, reference))
        if largest_step > limit:
            self.get_logger().error(
                f"Rejected arm target jump {largest_step:.4f} rad (limit {limit:.4f})"
            )
            return
        self.last_raw_target = candidate
        self.target = candidate

    def publish_smoothed_arm(self):
        if self.target is None or self.command is None:
            return
        source_count = self.count_publishers("/control_abstraction_layer")
        if source_count != 1:
            now = time.monotonic()
            if now - self.last_source_warning > 1.0:
                self.get_logger().error(
                    f"Arm output blocked: expected 1 abstraction publisher, found {source_count}"
                )
                self.last_source_warning = now
            self.velocity = [0.0] * 6
            return
        now = time.monotonic()
        dt = min(max(now - self.last_tick, 0.001), 0.02)
        self.last_tick = now
        alpha = 1.0 - math.exp(-dt / FILTER_TIME_CONSTANT)
        output = []
        new_velocity = []
        for index in range(6):
            filtered = self.filtered_target[index] + alpha * (
                self.target[index] - self.filtered_target[index]
            )
            self.filtered_target[index] = filtered
            desired_velocity = max(
                -MAX_VELOCITY,
                min(MAX_VELOCITY, (filtered - self.command[index]) / dt),
            )
            dv = MAX_ACCELERATION * dt
            velocity = max(
                self.velocity[index] - dv,
                min(self.velocity[index] + dv, desired_velocity),
            )
            new_velocity.append(velocity)
            output.append(self.command[index] + velocity * dt)
        self.velocity = new_velocity
        self.command = output
        self.output_count += 1
        if self.dry_run:
            self.get_logger().info(
                f"DRY {self.output_count:03d}: "
                + ", ".join(f"{value:.6f}" for value in output)
            )
            if self.validation_count and self.output_count >= self.validation_count:
                self.get_logger().info(
                    f"DRY RUN COMPLETE: {self.output_count} filtered commands; none published"
                )
                self.validation_done = True
            return
        arm = Float64MultiArray()
        arm.data = output
        self.arm_publisher.publish(arm)

    def start_gripper(self, value):
        if not self.io_client.service_is_ready():
            self.get_logger().error("SetIO service is unavailable")
            return
        self.gripper_busy = True
        writes = (
            ((0, 0.0), (1, 1.0))
            if value == 1.0
            else ((1, 0.0), (0, 1.0))
        )
        self.send_io(writes, 0, value)

    def publish_gripper_confirmation(self):
        # This is confirmed command history, never measured finger position.
        # Do not publish a guessed initial state before a successful IO pair.
        if self.confirmed_gripper is None:
            return
        message = Float64()
        message.data = self.confirmed_gripper
        self.gripper_state_publisher.publish(message)

    def send_io(self, writes, index, gripper_value):
        pin, state = writes[index]
        request = SetIO.Request()
        request.fun = 1
        request.pin = pin
        request.state = state
        future = self.io_client.call_async(request)
        future.add_done_callback(
            lambda result: self.io_finished(
                result, writes, index, gripper_value
            )
        )

    def io_finished(self, future, writes, index, gripper_value):
        try:
            if not future.result().success:
                raise RuntimeError(f"SetIO rejected pin {writes[index][0]}")
            if index == 0:
                self.send_io(writes, 1, gripper_value)
                return
            self.confirmed_gripper = float(gripper_value)
            self.publish_gripper_confirmation()
            state = "OPEN" if gripper_value == 1.0 else "CLOSED"
            self.get_logger().info(f"Gripper changed to {state}")
        except Exception as error:
            self.get_logger().error(f"Gripper operation failed: {error}")
        finally:
            if index == 1:
                self.gripper_busy = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validation-count", type=int, default=0)
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = ControlEngine(args.dry_run, args.validation_count)
    try:
        while rclpy.ok() and not node.validation_done:
            rclpy.spin_once(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
