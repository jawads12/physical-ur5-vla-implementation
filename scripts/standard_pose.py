#!/usr/bin/python3
"""Move the UR5 from its current joints to the saved standard pose."""

import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


JOINT_ORDER = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

STANDARD_POSE = (
    3.2679903507,
    -1.6692879836,
    1.7310266495,
    -1.9356344382,
    -1.6694858710,
    -0.0394843260,
)


class StandardPose(Node):
    def __init__(self, duration):
        super().__init__("bypass_standard_pose")
        self.duration = duration
        self.start_joints = None
        self.started_at = None
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(
            Float64MultiArray, "/forward_position_controller/commands", qos
        )
        self.create_subscription(JointState, "/joint_states", self.joints, 10)
        self.timer = self.create_timer(0.01, self.update)

    def joints(self, message):
        if self.start_joints is not None:
            return
        positions = dict(zip(message.name, message.position))
        if not all(name in positions for name in JOINT_ORDER):
            return
        self.start_joints = [float(positions[name]) for name in JOINT_ORDER]
        self.started_at = time.monotonic()
        self.get_logger().info(
            f"Moving to standard pose over {self.duration:.1f} seconds"
        )

    def update(self):
        if self.start_joints is None:
            return
        fraction = min(1.0, (time.monotonic() - self.started_at) / self.duration)
        # Smoothstep interpolation starts and ends with zero commanded velocity.
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        command = [
            start + blend * (target - start)
            for start, target in zip(self.start_joints, STANDARD_POSE)
        ]
        message = Float64MultiArray()
        message.data = command
        self.publisher.publish(message)
        if fraction >= 1.0:
            self.get_logger().info("Standard pose command completed")
            self.timer.cancel()
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()
    if args.duration <= 0.0:
        raise SystemExit("--duration must be positive")

    rclpy.init()
    node = StandardPose(args.duration)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
