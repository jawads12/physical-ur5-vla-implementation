#!/usr/bin/python3
"""Control-engine variant for policy_executor_phased.py.

It retains the existing 125 Hz smoothing behavior, resynchronizes its filter
with measured joints whenever abstract commands stop, and publishes a
transient-local gripper acknowledgment only after both SetIO writes succeed.
"""

from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64

from control_engine import ControlEngine
from ur5_vla_dataset_recorder.core import UR5_JOINT_ORDER, reorder_joint_positions


SOURCE_TIMEOUT_SECONDS = 0.20


class PhasedControlEngine(ControlEngine):
    def __init__(self, dry_run=False, validation_count=0, initial_gripper=1.0):
        self.last_abstract_command_at = 0.0
        self.initial_gripper = float(initial_gripper)
        self.confirmed_gripper = float(initial_gripper)
        super().__init__(dry_run=dry_run, validation_count=validation_count)
        gripper_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.gripper_state_publisher = self.create_publisher(
            Float64, "/control_engine/gripper_state", gripper_qos
        )
        self.last_gripper = self.initial_gripper
        self._publish_gripper_ack(self.confirmed_gripper)
        self.create_timer(0.10, self._publish_gripper_heartbeat)
        self.get_logger().info(
            "phased extension enabled: measured-state resync and gripper acknowledgments"
        )

    def command_callback(self, message):
        self.last_abstract_command_at = time.monotonic()
        super().command_callback(message)

    def joint_state_callback(self, message):
        try:
            measured = reorder_joint_positions(
                message.name, message.position, UR5_JOINT_ORDER
            )
        except Exception as error:
            self.get_logger().warning(f"Waiting for ordered joint state: {error}")
            return
        inactive = (
            self.last_abstract_command_at == 0.0
            or time.monotonic() - self.last_abstract_command_at > SOURCE_TIMEOUT_SECONDS
        )
        if self.command is None or inactive:
            values = [float(value) for value in measured]
            self.command = list(values)
            self.filtered_target = list(values)
            self.target = None
            self.velocity = [0.0] * 6
            self.last_raw_target = None

    def _publish_gripper_ack(self, value):
        message = Float64()
        message.data = float(value)
        self.gripper_state_publisher.publish(message)

    def _publish_gripper_heartbeat(self):
        # Publish only the last IO-confirmed value.  ControlEngine updates
        # last_gripper before starting an IO transaction, so using that field
        # here could falsely acknowledge a failed transition.
        self._publish_gripper_ack(self.confirmed_gripper)

    def io_finished(self, future, writes, index, gripper_value):
        try:
            if not future.result().success:
                raise RuntimeError(f"SetIO rejected pin {writes[index][0]}")
            if index == 0:
                self.send_io(writes, 1, gripper_value)
                return
            state = "OPEN" if gripper_value == 1.0 else "CLOSED"
            self.confirmed_gripper = float(gripper_value)
            self._publish_gripper_ack(gripper_value)
            self.get_logger().info(f"Gripper changed and acknowledged as {state}")
        except Exception as error:
            self.get_logger().error(f"Gripper operation failed: {error}")
        finally:
            if index == 1:
                self.gripper_busy = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validation-count", type=int, default=0)
    parser.add_argument(
        "--initial-gripper",
        choices=("open", "closed"),
        default="open",
        help="initial physical gripper state; this does not actuate the gripper",
    )
    args, ros_args = parser.parse_known_args()
    initial = 1.0 if args.initial_gripper == "open" else 0.0
    rclpy.init(args=ros_args)
    node = PhasedControlEngine(
        dry_run=args.dry_run,
        validation_count=args.validation_count,
        initial_gripper=initial,
    )
    try:
        while rclpy.ok() and not node.validation_done:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
