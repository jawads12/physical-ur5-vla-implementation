#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray
import torch
import numpy as np
import cv2
from transformers import AutoTokenizer
from pathlib import Path

# ====================== CONFIG ======================
POLICY_PATH = str(Path.home() / "Desktop/VLA/berk/smolvla_ur5_finetuned")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TASK = "Pick up the blue cup and put it into the brown cup."   # ← change this to a task you trained on!
IMAGE_SIZE = (480, 640)
STATE_DIM = 8
PUB_TOPIC = "/joint_commands"
# ===================================================

class SmolVLAPolicyNode(Node):
    def __init__(self):
        super().__init__("smolvla_policy_node")
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        self.policy = SmolVLAPolicy.from_pretrained(POLICY_PATH).to(DEVICE)
        self.policy.eval()
        self.policy.reset()

        # Debug: show what the model thinks its action dim is
        action_dim = getattr(self.policy.config, 'action_dim', 'UNKNOWN')
        self.get_logger().info(f"✅ SmolVLA loaded (action_dim = {action_dim}) on {DEVICE}")

        self.create_subscription(Image, "/rgb", self.rgb_cb, 10)
        self.create_subscription(Image, "/rgbUp", self.rgbup_cb, 10)
        self.create_subscription(Image, "/depthUp", self.depth_cb, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_cb, 10)

        self.latest_images = {}
        self.latest_state = None

        self.timer = self.create_timer(0.1, self.inference_timer)
        self.cmd_pub = self.create_publisher(Float64MultiArray, PUB_TOPIC, 10)

        self.get_logger().info(f"✅ Node ready — publishing to {PUB_TOPIC}")

    def _rosimg_to_tensor(self, msg, is_depth=False):
        if is_depth:
            arr = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            arr = np.nan_to_num(arr, nan=0.0)
            if arr.max() > 0:
                arr = (arr / arr.max() * 255).astype(np.uint8)
            else:
                arr = np.zeros((msg.height, msg.width), dtype=np.uint8)
            arr = np.stack([arr] * 3, axis=-1)
        else:
            if msg.encoding == "rgb8":
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif msg.encoding == "bgr8":
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            else:
                arr = np.zeros((msg.height, msg.width, 3), dtype=np.uint8)

        if arr.shape[:2] != IMAGE_SIZE[::-1]:
            arr = cv2.resize(arr, IMAGE_SIZE)

        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    def rgb_cb(self, msg):
        self.latest_images["hand_image"] = self._rosimg_to_tensor(msg)

    def rgbup_cb(self, msg):
        self.latest_images["image"] = self._rosimg_to_tensor(msg)

    def depth_cb(self, msg):
        self.latest_images["image_with_depth"] = self._rosimg_to_tensor(msg, is_depth=True)

    def joint_cb(self, msg):
        positions = list(msg.position)
        self.latest_state = torch.tensor(
            positions + [0.0] * (STATE_DIM - len(positions)),
            dtype=torch.float32
        )[:STATE_DIM]

    def inference_timer(self):
        if len(self.latest_images) < 3 or self.latest_state is None:
            self.get_logger().warn("Waiting for all cameras + joint state...")
            return

        batch = {}
        for k, v in self.latest_images.items():
            batch[f"observation.images.{k}"] = v.unsqueeze(0).to(DEVICE)

        batch["observation.state"] = self.latest_state.unsqueeze(0).to(DEVICE)

        tokenized = self.tokenizer(
            [TASK], max_length=48, truncation=True, padding="max_length", return_tensors="pt"
        )
        batch["observation.language.tokens"] = tokenized["input_ids"].to(DEVICE)
        batch["observation.language.attention_mask"] = tokenized["attention_mask"].bool().to(DEVICE)

        with torch.no_grad():
            action_chunk = self.policy.select_action(batch)

        # === CRITICAL DEBUG ===
        self.get_logger().info(f"DEBUG → action_chunk shape: {action_chunk.shape}")

        action = action_chunk[0, 0].cpu().numpy().flatten()   # force 1D array

        if len(action) != 7:
            self.get_logger().warn(f"Action dimension is {len(action)} (expected 7) → padding with zeros")
            action = np.pad(action, (0, 7 - len(action)), mode='constant')

        action = action[:7]
        action = np.clip(action, -3.14, 3.14)   # safe for UR5 joint positions

        msg = Float64MultiArray()
        msg.data = action.tolist()
        self.cmd_pub.publish(msg)

        self.get_logger().info(f"✅ Action sent → {action.round(3)}")

def main():
    rclpy.init()
    node = SmolVLAPolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
