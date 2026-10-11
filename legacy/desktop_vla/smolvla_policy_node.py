#!/usr/bin/env python3
"""
SmolVLA Policy Inference Node for UR5 (ROS2 + Isaac Sim)
=========================================================

This ROS2 node bridges a fine-tuned SmolVLA policy with a simulated UR5
arm in Isaac Sim. It:
  1. Subscribes to camera images + joint states from Isaac Sim
  2. Runs SmolVLA inference to predict action chunks
  3. Publishes joint commands back to the simulator

Usage:
  source /opt/ros/humble/setup.bash
  conda activate lerobot
  python smolvla_policy_node.py --ros-args \
      -p model_path:=smolvla_ur5_finetuned \
      -p control_freq:=5.0 \
      -p language_instruction:="Pick up the blue cup and put it into the brown cup."

Prerequisites:
  - Isaac Sim running with UR5 + ROS2 bridge
  - Topics publishing: /joint_states, /camera/color/image_raw, /wrist_camera/color/image_raw
  - pip install lerobot[smolvla] cv_bridge
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import torch
import numpy as np
from collections import deque
from pathlib import Path

try:
    from cv_bridge import CvBridge
except ImportError:
    print("ERROR: cv_bridge not found. Install with: sudo apt install ros-humble-cv-bridge")
    raise

from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy


class SmolVLAPolicyNode(Node):
    """
    ROS2 node that runs SmolVLA inference in a control loop.

    Observation flow:
        Isaac Sim → /joint_states      → self.current_joint_state
        Isaac Sim → /camera/...        → self.current_third_person_image
        Isaac Sim → /wrist_camera/...  → self.current_wrist_image

    Action flow:
        SmolVLA policy → action_chunk → action_queue → JointTrajectory
        → /joint_trajectory_controller/joint_trajectory → Isaac Sim
    """

    def __init__(self):
        super().__init__('smolvla_policy_node')

        # ──────────────────────── Parameters ────────────────────────
        self.declare_parameter('model_path', 'smolvla_ur5_finetuned')
        self.declare_parameter('control_freq', 5.0)
        self.declare_parameter('language_instruction',
                               'Pick up the blue cup and put it into the brown cup.')

        # Topic names (override via ros-args if your Isaac Sim uses different names)
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('third_person_camera_topic', '/camera/color/image_raw')
        self.declare_parameter('wrist_camera_topic', '/wrist_camera/color/image_raw')
        self.declare_parameter('joint_command_topic',
                               '/joint_trajectory_controller/joint_trajectory')

        model_path = self.get_parameter('model_path').value
        self.control_freq = self.get_parameter('control_freq').value
        self.language_instruction = self.get_parameter('language_instruction').value

        joint_states_topic = self.get_parameter('joint_states_topic').value
        third_person_topic = self.get_parameter('third_person_camera_topic').value
        wrist_topic = self.get_parameter('wrist_camera_topic').value
        joint_cmd_topic = self.get_parameter('joint_command_topic').value

        # ──────────────────────── Load Policy ────────────────────────
        self.get_logger().info(f'Loading SmolVLA policy from: {model_path}')
        self.policy = SmolVLAPolicy.from_pretrained(model_path)
        self.policy.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = self.policy.to(self.device)
        self.get_logger().info(f'Policy loaded on {self.device} '
                               f'({sum(p.numel() for p in self.policy.parameters())/1e6:.0f}M params)')

        # ──────────────────────── State ────────────────────────
        self.bridge = CvBridge()
        self.current_joint_state = None
        self.current_third_person_image = None
        self.current_wrist_image = None
        self.action_queue = deque()
        self.step_count = 0

        # UR5 joint names (standard)
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
        ]

        # ──────────────────────── Subscribers ────────────────────────
        self.joint_state_sub = self.create_subscription(
            JointState, joint_states_topic,
            self.joint_state_cb, 10)

        self.third_person_sub = self.create_subscription(
            Image, third_person_topic,
            self.third_person_image_cb, 10)

        self.wrist_sub = self.create_subscription(
            Image, wrist_topic,
            self.wrist_image_cb, 10)

        # ──────────────────────── Publisher ────────────────────────
        self.joint_cmd_pub = self.create_publisher(
            JointTrajectory, joint_cmd_topic, 10)

        # ──────────────────────── Control Timer ────────────────────────
        self.timer = self.create_timer(1.0 / self.control_freq, self.control_loop)

        self.get_logger().info(
            f'Node ready | freq={self.control_freq}Hz | '
            f'instruction="{self.language_instruction}"')
        self.get_logger().info(
            f'Subscribing: {joint_states_topic}, {third_person_topic}, {wrist_topic}')
        self.get_logger().info(
            f'Publishing:  {joint_cmd_topic}')

    # ──────────────────────── Subscriber Callbacks ────────────────────────

    def joint_state_cb(self, msg: JointState):
        """Extract UR5 joint positions in canonical order."""
        positions = []
        for name in self.joint_names:
            if name in msg.name:
                idx = msg.name.index(name)
                positions.append(msg.position[idx])

        if len(positions) == 6:
            # Dataset uses 8-dim state: 6 joints + gripper_closed + extra
            # Pad with 0s for gripper state (adjust if you have gripper feedback)
            self.current_joint_state = np.array(
                positions + [0.0, 0.0], dtype=np.float32)

    def third_person_image_cb(self, msg: Image):
        """Convert ROS Image to numpy array (480×640×3)."""
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        if cv_img.shape[:2] != (480, 640):
            import cv2
            cv_img = cv2.resize(cv_img, (640, 480))
        self.current_third_person_image = cv_img

    def wrist_image_cb(self, msg: Image):
        """Convert wrist camera ROS Image to numpy array (480×640×3)."""
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        if cv_img.shape[:2] != (480, 640):
            import cv2
            cv_img = cv2.resize(cv_img, (640, 480))
        self.current_wrist_image = cv_img

    # ──────────────────────── Main Control Loop ────────────────────────

    def control_loop(self):
        """
        Runs at control_freq Hz.
        If actions remain in the queue, execute the next one.
        Otherwise, run a new inference pass to generate a fresh action chunk.
        """
        # Execute queued action if available
        if self.action_queue:
            action = self.action_queue.popleft()
            self._send_joint_command(action)
            self.step_count += 1
            return

        # Wait for all observations to be received
        if (self.current_joint_state is None or
            self.current_third_person_image is None or
            self.current_wrist_image is None):
            self.get_logger().warn(
                'Waiting for observations... '
                f'joints={self.current_joint_state is not None} '
                f'cam={self.current_third_person_image is not None} '
                f'wrist={self.current_wrist_image is not None}',
                throttle_duration_sec=3.0)
            return

        # ──── Build observation dict ────
        # Images: (1, C, H, W) float32 in [0, 1]
        third_person = torch.from_numpy(
            self.current_third_person_image.transpose(2, 0, 1).copy()
        ).float().unsqueeze(0) / 255.0

        wrist = torch.from_numpy(
            self.current_wrist_image.transpose(2, 0, 1).copy()
        ).float().unsqueeze(0) / 255.0

        state = torch.from_numpy(
            self.current_joint_state.copy()
        ).float().unsqueeze(0)

        observation = {
            'observation.images.image': third_person.to(self.device),
            'observation.images.hand_image': wrist.to(self.device),
            'observation.state': state.to(self.device),
            'language_instruction': [self.language_instruction],
        }

        # ──── Inference ────
        with torch.no_grad():
            action_chunk = self.policy.select_action(observation)

        actions = action_chunk.cpu().numpy()

        # Queue all actions from the chunk
        for i in range(actions.shape[0]):
            self.action_queue.append(actions[i])

        self.get_logger().info(
            f'[step {self.step_count}] Generated action chunk: '
            f'{actions.shape[0]} steps, '
            f'first action: [{", ".join(f"{a:.4f}" for a in actions[0][:4])}...]')

    def _send_joint_command(self, action: np.ndarray):
        """
        Convert policy action → JointTrajectory and publish.

        ⚠️  IMPORTANT: The Berkeley UR5 dataset uses END-EFFECTOR delta actions
        (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper), NOT joint-space deltas.

        For proper deployment, you should either:
          A) Use an IK solver (MoveIt2) to convert EE deltas → joint positions
          B) Use Isaac Sim's ArticulationController with IK mode

        The code below provides a BASIC joint-delta interpretation as a starting
        point. You MUST adjust this for your specific setup.
        """
        if self.current_joint_state is None:
            return

        # --- OPTION 1: Interpret as joint-space deltas (simplified) ---
        # Take the first 6 values as joint deltas
        joint_deltas = action[:6] if len(action) >= 6 else np.zeros(6)
        new_positions = self.current_joint_state[:6] + joint_deltas

        # Build JointTrajectory message
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = new_positions.tolist()
        point.time_from_start = Duration(
            sec=0, nanosec=int((1.0 / self.control_freq) * 1e9))
        trajectory.points = [point]

        self.joint_cmd_pub.publish(trajectory)


def main(args=None):
    rclpy.init(args=args)
    node = SmolVLAPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
