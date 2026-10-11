#!/usr/bin/env python3
"""
ROS2 node that sends observations to Colab PI0 server and publishes actions.

Usage:
    python3 inference_client.py --ros-args \
        -p server_url:="https://YOUR-NGROK-URL.ngrok.io" \
        -p task:="pick a blue cube" \
        -p rate_hz:=10.0
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState, Image as RosImage
from std_msgs.msg import Header
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

import numpy as np
import requests
import base64
from PIL import Image
import io
import time


class PI0InferenceClient(Node):
    """
    ROS2 node that:
    1. Subscribes to camera images and joint states
    2. Sends observations to remote PI0 inference server
    3. Publishes actions to control the robot
    """

    def __init__(self):
        super().__init__('pi0_inference_client')

        # ===============================
        # Parameters
        # ===============================
        self.declare_parameter('server_url', 'https://YOUR-NGROK-URL.ngrok.io')
        self.declare_parameter('task', 'pick a blue cube')
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('action_horizon', 1)  # How many steps from chunk to execute
        self.declare_parameter('timeout_s', 60.0)  # PI0 can take 10-30s, especially first call
        self.declare_parameter('max_retries', 3)

        self.server_url = self.get_parameter('server_url').value
        self.task = self.get_parameter('task').value
        self.rate_hz = self.get_parameter('rate_hz').value
        self.action_horizon = self.get_parameter('action_horizon').value
        self.timeout_s = self.get_parameter('timeout_s').value
        self.max_retries = self.get_parameter('max_retries').value

        # Track consecutive failures for backoff
        self.consecutive_failures = 0
        self.last_error_msg = ""

        # ===============================
        # Joint Configuration (matching teleop.py)
        # ===============================
        self.CTRL_JOINTS = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_3_joint',
        ]
        self.LOCK_WRIST_2 = 1.5708
        self.PUBLISH_JOINTS = self.CTRL_JOINTS + ['wrist_2_joint']

        # Gripper joints (for action client)
        self.GRIPPER_JOINTS = ['left_inner_finger_joint', 'right_inner_finger_joint']
        self.GRIP_OPEN_POS = 0.0
        self.GRIP_CLOSE_POS = 0.5

        # ===============================
        # State Storage
        # ===============================
        self.latest_image = None
        self.latest_image_stamp = None
        self.latest_joints = {}
        self.latest_joints_stamp = None

        self.action_chunk = None
        self.action_idx = 0

        self.is_running = False
        self.current_gripper_state = 0  # 0=open, 1=closed

        # ===============================
        # ROS2 Setup
        # ===============================
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )
        default_qos = QoSProfile(depth=10)

        # Subscribers
        self.create_subscription(
            RosImage, '/rgb', self.on_image, cam_qos
        )
        self.create_subscription(
            JointState, '/joint_states', self.on_joint_states, default_qos
        )

        # Publishers
        self.cmd_pub = self.create_publisher(JointState, '/joint_command', 10)

        # Action clients (for gripper)
        self.grip_client = ActionClient(
            self, FollowJointTrajectory, '/ur5_gripper/follow_joint_trajectory'
        )

        # Control loop timer
        self.dt = 1.0 / self.rate_hz
        self.timer = self.create_timer(self.dt, self.control_loop)

        # Status check timer
        self.create_timer(10.0, self.print_status)

        # Initial health check
        self.check_server_health()

        self.get_logger().info("=" * 60)
        self.get_logger().info("PI0 Inference Client Initialized")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Server URL: {self.server_url}")
        self.get_logger().info(f"Task: {self.task}")
        self.get_logger().info(f"Rate: {self.rate_hz} Hz")
        self.get_logger().info(f"Timeout: {self.timeout_s}s (increase if PI0 is slow)")
        self.get_logger().info(f"Controlled joints: {self.CTRL_JOINTS}")
        self.get_logger().info("=" * 60)
        self.get_logger().info("TIP: First inference may take 30-60s for model warm-up!")

    # ===============================
    # Server Communication
    # ===============================
    def check_server_health(self):
        """Verify connection to inference server"""
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self.get_logger().info(f"✅ Server healthy: {data}")
                self.is_running = True
            else:
                self.get_logger().error(f"❌ Server returned status {resp.status_code}")
                self.is_running = False
        except requests.ConnectionError:
            self.get_logger().error(f"❌ Cannot connect to server at {self.server_url}")
            self.is_running = False
        except Exception as e:
            self.get_logger().error(f"❌ Health check failed: {e}")
            self.is_running = False

    def reset_policy(self):
        """Reset the policy state on the server"""
        try:
            resp = requests.post(f"{self.server_url}/reset", timeout=5)
            if resp.status_code == 200:
                self.get_logger().info("Policy reset successful")
                self.action_chunk = None
                self.action_idx = 0
            else:
                self.get_logger().warn(f"Policy reset failed: {resp.status_code}")
        except Exception as e:
            self.get_logger().error(f"Reset error: {e}")

    # ===============================
    # Callbacks
    # ===============================
    def on_image(self, msg: RosImage):
        """Store latest camera image"""
        try:
            h, w = msg.height, msg.width
            enc = (msg.encoding or "").lower()
            data = np.frombuffer(msg.data, dtype=np.uint8)

            if enc == 'rgb8':
                self.latest_image = data.reshape((h, w, 3))
            elif enc == 'bgr8':
                self.latest_image = data.reshape((h, w, 3))[:, :, ::-1].copy()
            elif enc == 'rgba8':
                self.latest_image = data.reshape((h, w, 4))[:, :, :3].copy()
            elif enc == 'bgra8':
                self.latest_image = data.reshape((h, w, 4))[:, :, :3][:, :, ::-1].copy()
            else:
                self.latest_image = data.reshape((h, w, 3))

            self.latest_image_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"Image decode error: {e}")

    def on_joint_states(self, msg: JointState):
        """Store latest joint positions"""
        for name, pos in zip(msg.name, msg.position):
            self.latest_joints[name] = float(pos)
        self.latest_joints_stamp = msg.header.stamp

    # ===============================
    # Data Conversion
    # ===============================
    def image_to_base64(self, image_np: np.ndarray) -> str:
        """Convert numpy image to base64 string (JPEG for faster transfer)"""
        pil_image = Image.fromarray(image_np)
        # Resize to 224x224 (PI0 expects this) to reduce payload size
        pil_image = pil_image.resize((224, 224), Image.BILINEAR)
        buffer = io.BytesIO()
        # Use JPEG for much smaller payload (PNG can be 10x larger)
        pil_image.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def get_state_vector(self) -> np.ndarray:
        """Get current joint positions as state vector (matching training order)"""
        state = []
        for joint in self.CTRL_JOINTS:
            if joint in self.latest_joints:
                state.append(self.latest_joints[joint])
            else:
                state.append(0.0)
        return np.array(state, dtype=np.float32)

    # ===============================
    # Inference
    # ===============================
    def request_inference(self) -> np.ndarray:
        """Send observation to server and get action chunk with retry logic"""
        if self.latest_image is None:
            self.get_logger().warn("No image available")
            return None

        # Prepare payload
        payload = {
            'image': self.image_to_base64(self.latest_image),
            'state': self.get_state_vector().tolist(),
            'task': self.task,
        }

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                resp = requests.post(
                    f"{self.server_url}/infer",
                    json=payload,
                    timeout=self.timeout_s
                )
                latency = (time.time() - start_time) * 1000

                if resp.status_code == 200:
                    result = resp.json()
                    if result.get('success'):
                        self.consecutive_failures = 0
                        self.get_logger().info(f"✅ Inference OK (latency: {latency:.1f}ms, action_dim: {result.get('action_dim', '?')})")
                        return np.array(result['action'])
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        traceback = result.get('traceback', '')
                        if error_msg != self.last_error_msg:
                            self.get_logger().error(f"❌ Server error: {error_msg}")
                            if traceback:
                                self.get_logger().error(f"Traceback:\n{traceback[:500]}")
                            self.last_error_msg = error_msg
                elif resp.status_code == 500:
                    # Try to get error details from response
                    try:
                        err_data = resp.json()
                        error_msg = err_data.get('error', 'Unknown')
                        traceback = err_data.get('traceback', '')
                        if error_msg != self.last_error_msg:
                            self.get_logger().error(f"❌ HTTP 500 - Server error: {error_msg}")
                            if traceback:
                                self.get_logger().error(f"Traceback:\n{traceback[:1000]}")
                            self.last_error_msg = error_msg
                    except:
                        self.get_logger().error(f"❌ HTTP 500 - Response: {resp.text[:200]}")
                else:
                    self.get_logger().error(f"❌ HTTP {resp.status_code}: {resp.text[:200]}")

            except requests.Timeout:
                self.get_logger().warn(f"⏱️ Timeout (>{self.timeout_s}s) - attempt {attempt+1}/{self.max_retries}")
            except requests.ConnectionError as e:
                self.get_logger().error(f"🔌 Connection error: {e}")
                # Don't retry immediately on connection errors
                break
            except Exception as e:
                self.get_logger().error(f"❌ Request error: {type(e).__name__}: {e}")

            # Exponential backoff between retries
            if attempt < self.max_retries - 1:
                wait_time = min(2 ** attempt, 10)  # Max 10 seconds
                self.get_logger().info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)

        self.consecutive_failures += 1
        if self.consecutive_failures >= 5:
            self.get_logger().warn(f"⚠️ {self.consecutive_failures} consecutive failures - server may be down")

        return None

    # ===============================
    # Robot Control
    # ===============================
    def send_gripper_command(self, close: bool):
        """Send gripper open/close command via action client"""
        target_state = 1 if close else 0
        if target_state == self.current_gripper_state:
            return  # Already in desired state

        if not self.grip_client.server_is_ready():
            self.get_logger().debug("Gripper action server not ready")
            return

        pos = self.GRIP_CLOSE_POS if close else self.GRIP_OPEN_POS

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.GRIPPER_JOINTS
        goal.trajectory.header.stamp = self.get_clock().now().to_msg()

        pt = JointTrajectoryPoint()
        pt.positions = [pos, pos]
        pt.velocities = [0.0, 0.0]
        pt.time_from_start.sec = 1
        pt.time_from_start.nanosec = 0

        goal.trajectory.points = [pt]
        goal.goal_time_tolerance.sec = 1

        self.grip_client.send_goal_async(goal)
        self.current_gripper_state = target_state
        self.get_logger().info(f"Gripper: {'CLOSE' if close else 'OPEN'}")

    def control_loop(self):
        """Main control loop - request actions and send to robot"""
        if not self.is_running:
            return

        # Check if we have observations
        if self.latest_image is None:
            return
        if len(self.latest_joints) < len(self.CTRL_JOINTS):
            return

        # Check if we need new actions
        if self.action_chunk is None or self.action_idx >= len(self.action_chunk):
            self.action_chunk = self.request_inference()
            self.action_idx = 0

            if self.action_chunk is None:
                return

        # Handle action chunk shape
        # Expected: (horizon, action_dim) or (action_dim,)
        if len(self.action_chunk.shape) == 2:
            action = self.action_chunk[self.action_idx]
        elif len(self.action_chunk.shape) == 1:
            action = self.action_chunk
        else:
            self.get_logger().error(f"Unexpected action shape: {self.action_chunk.shape}")
            return

        self.action_idx += 1

        # Parse action: [joint1, joint2, ..., joint5, gripper_cmd]
        # Your dataset has 6 action dims: 5 joints + 1 gripper
        if len(action) >= 6:
            joint_positions = action[:5]
            gripper_cmd = action[5]
        else:
            joint_positions = action[:-1] if len(action) > 1 else action
            gripper_cmd = action[-1] if len(action) > 1 else 0.0

        # Publish joint command
        cmd = JointState()
        cmd.header = Header()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = self.PUBLISH_JOINTS
        cmd.position = list(joint_positions) + [self.LOCK_WRIST_2]

        self.cmd_pub.publish(cmd)

        # Handle gripper
        if gripper_cmd > 0.5:
            self.send_gripper_command(close=True)
        else:
            self.send_gripper_command(close=False)

    def print_status(self):
        """Periodic status update"""
        has_img = self.latest_image is not None
        has_js = len(self.latest_joints) >= len(self.CTRL_JOINTS)
        self.get_logger().info(
            f"Status: running={self.is_running}, "
            f"has_image={has_img}, has_joints={has_js}, "
            f"action_idx={self.action_idx}"
        )


def main():
    rclpy.init()
    node = PI0InferenceClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

