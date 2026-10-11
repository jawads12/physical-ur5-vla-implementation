#!/usr/bin/env python3
"""
Zero-Shot SmolVLA inference node for UR5.

Downloads and runs `lerobot/smolvla_base` (or any other SmolVLA variant)
without any local fine-tuning.

Usage:
    python3 smolvla_inference.py --ros-args \
        -p task:="pick up the lemon" \
        -p rate_hz:=20.0 \
        -p action_horizon:=15
"""

import gc
import os
import sys
import traceback
import time

import numpy as np
import torch
from PIL import Image

# Disable torch.compile to avoid runtime compilation overhead
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_CUDAGRAPH_DISABLE"] = "1"

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState, Image as RosImage
from std_msgs.msg import Header
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

# LeRobot SmolVLA Imports
try:
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.common.policies.factory import make_pre_post_processors
except ImportError:
    # Fallback for different LeRobot versions
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors


def pick_best_gpu() -> int:
    """Return the GPU index with the most free VRAM."""
    best_gpu = 0
    best_free = 0
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_free = free
                best_gpu = i
    return best_gpu

def load_smolvla_model(repo_id: str = "lerobot/smolvla_base"):
    """Load SmolVLA directly from HuggingFace."""
    if not torch.cuda.is_available():
        print("❌ No CUDA GPUs detected. Exiting.")
        sys.exit(1)

    best_gpu = pick_best_gpu()
    target_device = f"cuda:{best_gpu}"
    
    print(f"Downloading/Loading {repo_id} to {target_device}...")
    
    # Load the policy
    policy = SmolVLAPolicy.from_pretrained(repo_id)
    
    # SmolVLA is designed to be highly efficient, but we strictly use bfloat16
    # because the SmolLM2/SigLIP backbone overflows easily in standard fp16
    policy = policy.to(dtype=torch.bfloat16, device=target_device)
    policy.eval()

    print("✅ Model loaded successfully in bfloat16.")

    # Setup Preprocessors
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        repo_id,
        preprocessor_overrides={"device_processor": {"device": target_device}},
    )

    return policy, preprocess, postprocess


class SmolVLAInferenceNode(Node):
    def __init__(self, policy, preprocess, postprocess):
        super().__init__("smolvla_inference_node")

        self.policy = policy
        self.preprocess = preprocess
        self.postprocess = postprocess

        # Default task for the base model
        self.declare_parameter("task", "pick up the lemon")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("action_horizon", 15)  # SmolVLA chunks might differ, keep horizon moderate

        self.task = self.get_parameter("task").value
        self.rate_hz = self.get_parameter("rate_hz").value
        self.action_horizon = self.get_parameter("action_horizon").value

        # --- UR5 Joint Config ---
        self.CTRL_JOINTS = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_3_joint",
        ]
        self.LOCK_WRIST_2 = 1.5708
        self.PUBLISH_JOINTS = self.CTRL_JOINTS + ["wrist_2_joint"]
        self.GRIPPER_JOINTS = ["left_inner_finger_joint", "right_inner_finger_joint"]
        
        self.GRIP_OPEN_POS = 0.0
        self.GRIP_CLOSE_POS = 0.5

        # --- State ---
        self.latest_image = None
        self.latest_joints = {}
        self.action_chunk = None
        self.action_idx = 0
        self.current_gripper_state = 0
        self.is_running = True

        # --- ROS2 Setup ---
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )
        default_qos = QoSProfile(depth=10)

        self.create_subscription(RosImage, "/rgb", self.on_image, cam_qos)
        self.create_subscription(JointState, "/joint_states", self.on_joint_states, default_qos)

        self.cmd_pub = self.create_publisher(JointState, "/joint_command", 10)
        self.grip_client = ActionClient(self, FollowJointTrajectory, "/ur5_gripper/follow_joint_trajectory")

        self.dt = 1.0 / self.rate_hz
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("=" * 60)
        self.get_logger().info("⚠️ WARNING: RUNNING ZERO-SHOT BASE MODEL ⚠️")
        self.get_logger().info("KEEP YOUR HAND ON THE E-STOP! Output actions are not aligned to your UR5!")
        self.get_logger().info("=" * 60)

    def on_image(self, msg: RosImage):
        try:
            h, w = msg.height, msg.width
            enc = (msg.encoding or "").lower()
            data = np.frombuffer(msg.data, dtype=np.uint8)

            if enc == "rgb8":
                self.latest_image = data.reshape((h, w, 3))
            elif enc == "bgr8":
                self.latest_image = data.reshape((h, w, 3))[:, :, ::-1].copy()
            else:
                self.latest_image = data.reshape((h, w, 3))
        except Exception as e:
            self.get_logger().error(f"Image decode error: {e}")

    def on_joint_states(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.latest_joints[name] = float(pos)

    def get_state_vector(self) -> np.ndarray:
        state = []
        for joint in self.CTRL_JOINTS:
            state.append(self.latest_joints.get(joint, 0.0))
        return np.array(state, dtype=np.float32)

    def run_inference(self) -> np.ndarray:
        if self.latest_image is None:
            return None

        try:
            t0 = time.time()

            # 1. DYNAMICALLY QUERY BASE MODEL CONFIGURATION
            # Find the exact keys and shapes the base model is expecting
            img_keys = [k for k, v in self.policy.config.input_features.items() if v.type.name == 'VISUAL']
            target_img_key = img_keys[0] if img_keys else 'observation.images.rgb'
            
            expected_img_shape = self.policy.config.input_features[target_img_key].shape
            target_h, target_w = expected_img_shape[1], expected_img_shape[2] # e.g., 256x256

            state_keys = [k for k, v in self.policy.config.input_features.items() if v.type.name == 'STATE']
            target_state_key = state_keys[0] if state_keys else 'observation.state'
            expected_state_dim = self.policy.config.input_features[target_state_key].shape[0]

            # 2. ADAPT THE IMAGE
            image_pil = Image.fromarray(self.latest_image)
            image_resized = image_pil.resize((target_w, target_h), Image.BILINEAR)
            image_array = np.array(image_resized, dtype=np.uint8)
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0

            # 3. ADAPT THE STATE (Pad with zeros if the base model expects more dimensions)
            state_np = self.get_state_vector()
            padded_state = np.zeros(expected_state_dim, dtype=np.float32)
            copy_len = min(len(state_np), expected_state_dim)
            padded_state[:copy_len] = state_np[:copy_len]
            state_tensor = torch.from_numpy(padded_state).float()

            # 4. BUILD THE OBSERVATION DICT USING THE BASE MODEL'S KEYS
            observation = {
                target_img_key: image_tensor,
                target_state_key: state_tensor,
                "task": self.task,
            }

            # 5. RUN INFERENCE
            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    batch = self.preprocess(observation)
                    action = self.policy.select_action(batch)
                action = self.postprocess(action)

            action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else np.array(action)
            action_np = np.squeeze(action_np)
            
            latency_ms = (time.time() - t0) * 1000
            self.get_logger().info(f"SmolVLA Inference: {latency_ms:.1f}ms | shape: {action_np.shape}")
            
            return action_np
            
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            self.get_logger().error(traceback.format_exc())
            return None

    def control_loop(self):
        if not self.is_running or self.latest_image is None or len(self.latest_joints) < len(self.CTRL_JOINTS):
            return

        chunk_len = self.action_chunk.shape[0] if self.action_chunk is not None and len(self.action_chunk.shape) == 2 else 1
        horizon_limit = min(self.action_horizon, chunk_len)

        if self.action_chunk is None or self.action_idx >= horizon_limit:
            self.action_chunk = self.run_inference()
            self.action_idx = 0
            if self.action_chunk is None:
                return

        if len(self.action_chunk.shape) == 2:
            action = self.action_chunk[self.action_idx]
        else:
            action = self.action_chunk

        self.action_idx += 1

        # NOTE: SmolVLA base model actions might not perfectly match your 6D layout
        # We try to extract what we can safely.
        if len(action) >= 6:
            joint_positions = action[:5]
            gripper_cmd = action[5]
        else:
            joint_positions = action[:-1] if len(action) > 1 else action
            gripper_cmd = action[-1] if len(action) > 1 else 0.0

        # Publish Joint Command
        cmd = JointState()
        cmd.header = Header()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = self.PUBLISH_JOINTS
        cmd.position = [float(p) for p in joint_positions] + [self.LOCK_WRIST_2]
        self.cmd_pub.publish(cmd)

        # Handle Gripper
        target_close = True if gripper_cmd > 0.5 else False
        if target_close != (self.current_gripper_state == 1) and self.grip_client is not None:
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = self.GRIPPER_JOINTS
            goal.trajectory.header.stamp = self.get_clock().now().to_msg()

            pt = JointTrajectoryPoint()
            pos = self.GRIP_CLOSE_POS if target_close else self.GRIP_OPEN_POS
            pt.positions = [pos, pos]
            pt.velocities = [0.0, 0.0]
            pt.time_from_start.sec = 1
            
            goal.trajectory.points = [pt]
            self.grip_client.send_goal_async(goal)
            self.current_gripper_state = 1 if target_close else 0

def main():
    print("\n" + "=" * 60)
    print("  Zero-Shot SmolVLA Inference")
    print("=" * 60)
    
    # We load the base un-finetuned model from Hugging Face
    repo_id = "lerobot/smolvla_base"
    
    try:
        policy, preprocess, postprocess = load_smolvla_model(repo_id)
    except Exception as e:
        print(f"❌ Failed to load {repo_id}: {e}")
        traceback.print_exc()
        sys.exit(1)

    rclpy.init()
    node = SmolVLAInferenceNode(policy, preprocess, postprocess)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
