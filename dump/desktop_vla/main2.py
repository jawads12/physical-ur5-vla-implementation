#!/usr/bin/env python3
"""
ROS2 inference node for a SmolVLA checkpoint.

Subscribes to:
  • /rgb
  • /rgbUp
  • /depthUp
  • /joint_states

Publishes:
  • /joint_command
  • gripper action (if control_msgs is available)

What this script does:
  - Loads SmolVLA using the current LeRobot API
  - Maps your ROS topics to the checkpoint's expected image keys
  - Adds the missing batch dimension so images become [1, 3, H, W]
  - Uses an 8D UR5+gripper state by default
"""

import gc
import os
import sys
import time
import traceback
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_CUDAGRAPH_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 0

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

try:
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    HAS_CONTROL_MSGS = True
except ImportError:
    HAS_CONTROL_MSGS = False
    print("⚠️  control_msgs not found — gripper control disabled.")
    print("   Install with: sudo apt install ros-${ROS_DISTRO}-control-msgs")


DEFAULT_STATE_KEY = "observation.state"


def print_gpu_info():
    n_gpus = torch.cuda.device_count()
    print(f"\n{'=' * 60}")
    print(f"  Detected {n_gpus} GPU(s)")
    print(f"{'=' * 60}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(
            f"  GPU {i}: {props.name}  |  "
            f"VRAM: {total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free"
        )
    print(f"{'=' * 60}\n")


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def pick_best_gpu() -> int:
    best_gpu = 0
    best_free = 0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free = free
            best_gpu = i
    return best_gpu


def place_model_on_gpu(policy, num_gpus: int):
    best_gpu = pick_best_gpu()
    target = f"cuda:{best_gpu}"
    free, total = torch.cuda.mem_get_info(best_gpu)
    print(f"  → Target GPU {best_gpu}: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")

    try:
        print(f"  → Moving to {target} in bfloat16 …")
        policy = policy.to(dtype=torch.bfloat16, device=target)
        alloc = torch.cuda.memory_allocated(best_gpu) / 1e9
        print(f"  ✅ Model on {target} in bfloat16  ({alloc:.2f} GB)\n")
        return policy
    except torch.cuda.OutOfMemoryError:
        print("  ⚠️  bfloat16 OOM — trying CPU-offload via accelerate …")
        clear_gpu_memory()

    try:
        from accelerate import dispatch_model, infer_auto_device_map

        print("  → Using accelerate CPU-offload (bfloat16) …")
        policy = policy.to(dtype=torch.bfloat16).cpu()

        max_memory = {best_gpu: int(free - 1 * 1024**3)}
        max_memory["cpu"] = "24GiB"

        device_map = infer_auto_device_map(
            policy,
            max_memory=max_memory,
            no_split_module_classes=["PaliGemmaWithExpertModel"],
        )

        policy = dispatch_model(policy, device_map=device_map)
        print("  ✅ Model with CPU-offload ready\n")
        return policy

    except ImportError:
        print("  ⚠️  accelerate not installed — cannot CPU-offload")
    except Exception as e:
        print(f"  ⚠️  CPU-offload failed: {e}")

    print(f"  → Last resort fallback on {target}")
    policy = policy.to(target)
    return policy


def get_image_feature_keys(policy) -> List[str]:
    try:
        image_features = policy.config.image_features
        if isinstance(image_features, dict) and len(image_features) > 0:
            return list(image_features.keys())
    except Exception:
        pass
    return [
        "observation.images.image",
        "observation.images.hand_image",
        "observation.images.image_with_depth",
    ]


def get_state_feature_key(policy) -> str:
    try:
        input_features = policy.config.input_features
        for key, spec in input_features.items():
            spec_type = spec.get("type") if isinstance(spec, dict) else getattr(spec, "type", None)
            if str(spec_type).upper() == "STATE":
                return key
    except Exception:
        pass
    return DEFAULT_STATE_KEY


def get_expected_state_dim(policy) -> int:
    try:
        input_features = policy.config.input_features
        for _, spec in input_features.items():
            spec_type = spec.get("type") if isinstance(spec, dict) else getattr(spec, "type", None)
            if str(spec_type).upper() == "STATE":
                shape = spec.get("shape") if isinstance(spec, dict) else getattr(spec, "shape", None)
                if shape and len(shape) > 0:
                    return int(shape[-1])
    except Exception:
        pass
    return 8


def get_expected_visual_size(policy) -> Tuple[int, int]:
    try:
        image_features = policy.config.image_features
        if isinstance(image_features, dict):
            for _, spec in image_features.items():
                shape = spec.get("shape") if isinstance(spec, dict) else getattr(spec, "shape", None)
                if shape and len(shape) >= 3:
                    return int(shape[-2]), int(shape[-1])
    except Exception:
        pass

    try:
        input_features = policy.config.input_features
        for _, spec in input_features.items():
            spec_type = spec.get("type") if isinstance(spec, dict) else getattr(spec, "type", None)
            if str(spec_type).upper() == "VISUAL":
                shape = spec.get("shape") if isinstance(spec, dict) else getattr(spec, "shape", None)
                if shape and len(shape) >= 3:
                    return int(shape[-2]), int(shape[-1])
    except Exception:
        pass

    return 480, 640


def decode_rgb_image(msg: RosImage) -> np.ndarray:
    h, w = msg.height, msg.width
    enc = (msg.encoding or "").lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc == "rgb8":
        img = data.reshape((h, w, 3))
    elif enc == "bgr8":
        img = data.reshape((h, w, 3))[:, :, ::-1].copy()
    elif enc == "rgba8":
        img = data.reshape((h, w, 4))[:, :, :3].copy()
    elif enc == "bgra8":
        img = data.reshape((h, w, 4))[:, :, :3][:, :, ::-1].copy()
    else:
        img = data.reshape((h, w, 3))

    return img


def decode_depth_image(msg: RosImage) -> np.ndarray:
    h, w = msg.height, msg.width
    enc = (msg.encoding or "").lower()

    if enc == "32fc1":
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
        return depth.astype(np.float32)

    if enc in ("16uc1", "mono16"):
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
        return depth.astype(np.float32) / 1000.0

    try:
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
        return depth.astype(np.float32)
    except Exception:
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
        return depth.astype(np.float32) / 1000.0


def resize_rgb(image_np: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    image_pil = Image.fromarray(image_np)
    image_resized = image_pil.resize(size, Image.BILINEAR)
    return np.array(image_resized, dtype=np.uint8)


def resize_depth(depth_np: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    depth_t = torch.from_numpy(depth_np).float().unsqueeze(0).unsqueeze(0)
    depth_t = F.interpolate(depth_t, size=size, mode="bilinear", align_corners=False)
    return depth_t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def depth_to_visual(depth_np: np.ndarray, depth_max_m: float = 2.0) -> np.ndarray:
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
    depth_np = np.clip(depth_np, 0.0, depth_max_m)
    depth_np = depth_np / max(depth_max_m, 1e-6)
    depth_3 = np.repeat(depth_np[:, :, None], 3, axis=2)
    return (depth_3 * 255.0).astype(np.uint8)


def load_model(checkpoint_path: str):
    if not torch.cuda.is_available():
        print("❌  No CUDA GPUs detected. Exiting.")
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    print_gpu_info()
    clear_gpu_memory()

    print("Loading SmolVLA policy …")

    try:
        from lerobot.policies import make_pre_post_processors
    except Exception:
        from lerobot.policies.factory import make_pre_post_processors

    try:
        from lerobot.policies.smolvla import SmolVLAPolicy
    except Exception:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)

    if hasattr(policy, "model") and hasattr(policy.model, "_orig_mod"):
        policy.model = policy.model._orig_mod
        print("  ✅ Unwrapped compiled model")

    policy = place_model_on_gpu(policy, num_gpus)
    policy.eval()

    try:
        input_device = next(policy.parameters()).device
    except StopIteration:
        input_device = torch.device("cuda:0")

    print(f"  Input tensors → {input_device}")

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        checkpoint_path,
        preprocessor_overrides={"device_processor": {"device": str(input_device)}},
    )

    print("  ✅ Preprocessors created")
    print(f"  ✅ Image feature keys: {list(getattr(policy.config, 'image_features', {}).keys())}")
    print(f"  ✅ State key         : {get_state_feature_key(policy)}")
    print(f"  ✅ State dim         : {get_expected_state_dim(policy)}")
    print(f"  ✅ chunk_size        : {getattr(policy.config, 'chunk_size', 'unknown')}")
    return policy, preprocess, postprocess


class SmolVLAInferenceNode(Node):
    def __init__(self, policy, preprocess, postprocess):
        super().__init__("smolvla_inference_node")

        self.policy = policy
        self.preprocess = preprocess
        self.postprocess = postprocess

        self.declare_parameter(
            "task",
            "There are two bowls. Sort the fruits so that all fruits of the same type are placed in one bowl",
        )
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("action_horizon", 20)
        self.declare_parameter("velocity_multiplier", 1.0)
        self.declare_parameter("skip_steps", 0)
        self.declare_parameter("gripper_threshold", 0.15)
        self.declare_parameter("depth_max_m", 2.0)

        self.task = self.get_parameter("task").value
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.action_horizon = int(self.get_parameter("action_horizon").value)
        self.velocity_multiplier = float(self.get_parameter("velocity_multiplier").value)
        self.skip_steps = int(self.get_parameter("skip_steps").value)
        self.gripper_threshold = float(self.get_parameter("gripper_threshold").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)

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

        self.image_keys = get_image_feature_keys(policy)
        self.state_key = get_state_feature_key(policy)
        self.expected_state_dim = get_expected_state_dim(policy)
        self.visual_h, self.visual_w = get_expected_visual_size(policy)

        self.latest_rgb = None
        self.latest_rgb_up = None
        self.latest_depth_up = None

        self.latest_rgb_stamp = None
        self.latest_rgb_up_stamp = None
        self.latest_depth_up_stamp = None

        self.latest_joints = {}
        self.latest_joints_stamp = None

        self.action_chunk = None
        self.action_idx = 0
        self.current_gripper_state = 0
        self.force_chunk_completion = False

        self.inference_count = 0
        self.is_running = True

        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )
        default_qos = QoSProfile(depth=10)

        self.create_subscription(RosImage, "/rgb", self.on_rgb, cam_qos)
        self.create_subscription(RosImage, "/rgbUp", self.on_rgb_up, cam_qos)
        self.create_subscription(RosImage, "/depthUp", self.on_depth_up, cam_qos)
        self.create_subscription(JointState, "/joint_states", self.on_joint_states, default_qos)

        self.cmd_pub = self.create_publisher(JointState, "/joint_command", 10)

        if HAS_CONTROL_MSGS:
            self.grip_client = ActionClient(
                self, FollowJointTrajectory, "/ur5_gripper/follow_joint_trajectory"
            )
        else:
            self.grip_client = None

        self.dt = 1.0 / self.rate_hz
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.create_timer(10.0, self.print_status)

        self.get_logger().info("=" * 60)
        self.get_logger().info("SmolVLA Local Inference Node Ready")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Task             : {self.task}")
        self.get_logger().info(f"Rate             : {self.rate_hz} Hz")
        self.get_logger().info(f"Action Horizon   : {self.action_horizon} steps")
        self.get_logger().info(f"Vel Multiplier   : {self.velocity_multiplier}x")
        self.get_logger().info(f"Grip Threshold   : {self.gripper_threshold}")
        self.get_logger().info(f"Depth max (m)    : {self.depth_max_m}")
        self.get_logger().info(f"State dim        : {self.expected_state_dim}")
        self.get_logger().info(f"Image keys       : {self.image_keys}")
        self.get_logger().info(f"State key        : {self.state_key}")
        self.get_logger().info(f"Input size       : {self.visual_h} x {self.visual_w}")
        self.get_logger().info(f"Skipping first   : {self.skip_steps} steps of chunk")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Waiting for /rgb, /rgbUp, /depthUp and /joint_states...")

    def on_rgb(self, msg: RosImage):
        try:
            self.latest_rgb = decode_rgb_image(msg)
            self.latest_rgb_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"RGB decode error (/rgb): {e}")

    def on_rgb_up(self, msg: RosImage):
        try:
            self.latest_rgb_up = decode_rgb_image(msg)
            self.latest_rgb_up_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"RGB decode error (/rgbUp): {e}")

    def on_depth_up(self, msg: RosImage):
        try:
            self.latest_depth_up = decode_depth_image(msg)
            self.latest_depth_up_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"Depth decode error (/depthUp): {e}")

    def on_joint_states(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.latest_joints[name] = float(pos)
        self.latest_joints_stamp = msg.header.stamp

    def get_state_vector(self) -> np.ndarray:
        state_joint_order = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint",
        ]

        state = [self.latest_joints.get(name, 0.0) for name in state_joint_order]

        if len(state) < self.expected_state_dim:
            state.extend([0.0] * (self.expected_state_dim - len(state)))

        return np.array(state[:self.expected_state_dim], dtype=np.float32)

    def prepare_observation(self):
        rgb = resize_rgb(self.latest_rgb, (self.visual_w, self.visual_h))
        rgb_up = resize_rgb(self.latest_rgb_up, (self.visual_w, self.visual_h))
        depth = resize_depth(self.latest_depth_up, (self.visual_w, self.visual_h))
        depth_vis = depth_to_visual(depth, depth_max_m=self.depth_max_m)

        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        rgb_up_tensor = torch.from_numpy(rgb_up).permute(2, 0, 1).float() / 255.0
        depth_tensor = torch.from_numpy(depth_vis).permute(2, 0, 1).float() / 255.0
        state_tensor = torch.from_numpy(self.get_state_vector()).float()

        return {
            "observation.image": rgb_tensor,
            "observation.image2": rgb_up_tensor,
            "observation.image3": depth_tensor,
            self.state_key: state_tensor,
            "task": self.task,
        }

    def remap_batch_to_policy_keys(self, batch: dict) -> dict:
        """
        Copy generic preprocess keys to the exact SmolVLA image feature names.
        """
        expected_keys = list(getattr(self.policy.config, "image_features", {}).keys())
        if len(expected_keys) < 3:
            expected_keys = [
                "observation.images.image",
                "observation.images.hand_image",
                "observation.images.image_with_depth",
            ]

        src_front = "observation.image"
        src_hand = "observation.image2"
        src_depth = "observation.image3"

        remapped = dict(batch)

        for key in expected_keys:
            lower = key.lower()
            if "depth" in lower:
                if src_depth in batch:
                    remapped[key] = batch[src_depth]
            elif "hand" in lower or "wrist" in lower:
                if src_hand in batch:
                    remapped[key] = batch[src_hand]
            else:
                if src_front in batch:
                    remapped[key] = batch[src_front]

        return remapped

    def add_batch_dim_to_policy_images(self, batch: dict) -> dict:
        """
        SmolVLA expects images as [B, C, H, W].
        If preprocess gives [C, H, W], add B=1.
        """
        batched = dict(batch)

        image_keys = list(getattr(self.policy.config, "image_features", {}).keys())
        if not image_keys:
            image_keys = [
                "observation.images.image",
                "observation.images.hand_image",
                "observation.images.image_with_depth",
            ]

        for key in image_keys:
            if key in batched and isinstance(batched[key], torch.Tensor):
                if batched[key].ndim == 3:
                    batched[key] = batched[key].unsqueeze(0)
                elif batched[key].ndim == 4:
                    pass
                else:
                    self.get_logger().warn(f"Unexpected image tensor shape for {key}: {batched[key].shape}")

        return batched

    def run_inference(self) -> Optional[np.ndarray]:
        if self.latest_rgb is None:
            self.get_logger().warn("No /rgb image available yet")
            return None
        if self.latest_rgb_up is None:
            self.get_logger().warn("No /rgbUp image available yet")
            return None
        if self.latest_depth_up is None:
            self.get_logger().warn("No /depthUp image available yet")
            return None

        try:
            t0 = time.time()
            observation = self.prepare_observation()

            model_dtype = torch.bfloat16

            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=True, dtype=model_dtype):
                    batch = self.preprocess(observation)
                    batch = self.remap_batch_to_policy_keys(batch)
                    batch = self.add_batch_dim_to_policy_images(batch)
                    action = self.policy.select_action(batch)
                action = self.postprocess(action)

            if isinstance(action, torch.Tensor):
                action_np = action.detach().cpu().numpy()
            elif isinstance(action, dict):
                key = "action" if "action" in action else list(action.keys())[0]
                val = action[key]
                action_np = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.array(val)
            else:
                action_np = np.array(action)

            action_np = np.squeeze(action_np)
            latency_ms = (time.time() - t0) * 1000.0
            self.inference_count += 1
            self.get_logger().info(
                f"✅ Inference #{self.inference_count}  latency={latency_ms:.1f}ms  chunk_shape={action_np.shape}"
            )
            return action_np

        except Exception as e:
            self.get_logger().error(f"❌ Inference error: {e}")
            self.get_logger().error(traceback.format_exc())
            return None

    def reset_policy(self):
        for obj in (self.policy, self.preprocess, self.postprocess):
            if obj is not None and hasattr(obj, "reset"):
                obj.reset()
        self.action_chunk = None
        self.action_idx = 0
        self.force_chunk_completion = False
        self.get_logger().info("Policy state reset")

    def send_gripper_command(self, close: bool):
        if not HAS_CONTROL_MSGS or self.grip_client is None:
            return

        target_state = 1 if close else 0
        if target_state == self.current_gripper_state:
            return

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
        if not self.is_running:
            return

        if self.latest_rgb is None or self.latest_rgb_up is None or self.latest_depth_up is None:
            return
        if len(self.latest_joints) < 5:
            return

        chunk_len = 0
        if self.action_chunk is not None:
            if hasattr(self.action_chunk, "shape") and len(self.action_chunk.shape) == 2:
                chunk_len = self.action_chunk.shape[0]
            else:
                chunk_len = 1

        if self.force_chunk_completion:
            horizon_limit = chunk_len
        else:
            horizon_limit = min(self.action_horizon, chunk_len)

        if self.action_chunk is None or self.action_idx >= horizon_limit:
            self.action_chunk = self.run_inference()
            self.force_chunk_completion = False

            if self.action_chunk is None:
                return

            new_chunk_len = self.action_chunk.shape[0] if len(self.action_chunk.shape) == 2 else 1
            self.action_idx = min(self.skip_steps, max(0, new_chunk_len - 1))

        if len(self.action_chunk.shape) == 2:
            safe_idx = min(self.action_idx, self.action_chunk.shape[0] - 1)
            action = self.action_chunk[safe_idx]
        elif len(self.action_chunk.shape) == 1:
            action = self.action_chunk
        else:
            self.get_logger().error(f"Unexpected action shape: {self.action_chunk.shape}")
            return

        self.action_idx += 1

        if len(action) >= 6:
            joint_positions = action[:5]
            gripper_cmd = action[5]
        else:
            joint_positions = action[:-1] if len(action) > 1 else action
            gripper_cmd = action[-1] if len(action) > 1 else 0.0

        current_q = self.get_state_vector()[:5]
        predicted_q = np.array([float(p) for p in joint_positions], dtype=np.float32)
        amplified_q = current_q + (predicted_q - current_q) * self.velocity_multiplier

        cmd = JointState()
        cmd.header = Header()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = self.PUBLISH_JOINTS
        cmd.position = [float(x) for x in amplified_q.tolist()] + [float(self.LOCK_WRIST_2)]
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"Step {self.action_idx}/{horizon_limit} | AI Gripper Cmd: {gripper_cmd:.2f}",
            throttle_duration_sec=0.5,
        )

        target_close = True if gripper_cmd > self.gripper_threshold else False

        if target_close != (self.current_gripper_state == 1):
            self.send_gripper_command(close=target_close)
            time.sleep(0.5)

            if target_close:
                self.get_logger().info(
                    f"GRASP INITIATED (Threshold > {self.gripper_threshold})! Forcing chunk completion..."
                )
                self.force_chunk_completion = True

    def print_status(self):
        has_rgb = self.latest_rgb is not None
        has_rgb_up = self.latest_rgb_up is not None
        has_depth_up = self.latest_depth_up is not None
        has_js = len(self.latest_joints) >= 5

        self.get_logger().info(
            f"Status: running={self.is_running}, "
            f"rgb={has_rgb}, rgbUp={has_rgb_up}, depthUp={has_depth_up}, joints={has_js}, "
            f"inferences={self.inference_count}, action_idx={self.action_idx}/{self.action_horizon}"
        )


def main():
    print("\n" + "=" * 60)
    print("  SmolVLA Local Inference")
    print("=" * 60)

    checkpoint_path = input(
        "\nEnter the path to your pretrained model checkpoint:\n>>> "
    ).strip()

    if not checkpoint_path:
        print("❌  No path provided. Exiting.")
        sys.exit(1)

    checkpoint_path = os.path.expanduser(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        print(f"❌  Path does not exist: {checkpoint_path}")
        sys.exit(1)

    print(f"📂  Checkpoint: {checkpoint_path}\n")

    try:
        policy, preprocess, postprocess = load_model(checkpoint_path)
    except Exception as e:
        print(f"❌  Failed to load model: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("✅  Model ready — starting ROS2 node …\n")

    rclpy.init()
    node = SmolVLAInferenceNode(policy, preprocess, postprocess)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down …")
    finally:
        try:
            if rclpy.ok():
                node.destroy_node()
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
