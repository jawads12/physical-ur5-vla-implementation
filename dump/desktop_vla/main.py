#!/usr/bin/env python3
"""
Standalone PI0 inference node for multi-GPU (4x RTX 2080 11GB).

Loads the pretrained PI0 model distributed across all available GPUs,
subscribes to camera + joint state ROS2 topics, runs inference locally,
and publishes joint commands + gripper actions.

No server/client architecture — everything runs in one process.

Usage:
    python3 main.py --ros-args \
        -p task:="There are two bowls. Sort the fruits so that all fruits of the same type are placed in one bowl" \
        -p rate_hz:=20.0 \
        -p action_horizon:=40
"""

import gc
import os
import sys
import traceback
import time

import numpy as np
import torch
from PIL import Image

# ============================================================
# Disable torch.compile / CUDA graphs to avoid runtime issues
# ============================================================
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_CUDAGRAPH_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 0

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState, Image as RosImage
from std_msgs.msg import Header

try:
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    HAS_CONTROL_MSGS = True
except ImportError:
    HAS_CONTROL_MSGS = False
    print("⚠️  control_msgs not found — gripper control disabled.")
    print("   Install with: sudo apt install ros-${ROS_DISTRO}-control-msgs")


# ============================================================
# GPU helpers
# ============================================================
def print_gpu_info():
    n_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"  Detected {n_gpus} GPU(s)")
    print(f"{'='*60}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(
            f"  GPU {i}: {props.name}  |  "
            f"VRAM: {total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free"
        )
    print(f"{'='*60}\n")


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def pick_best_gpu() -> int:
    """Return the GPU index with the most free VRAM."""
    best_gpu = 0
    best_free = 0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free = free
            best_gpu = i
    return best_gpu


def place_model_on_gpu(policy, num_gpus: int):
    """
    Place the PI0 model on GPU.
    We STRICTLY use bfloat16. PaliGemma degrades or overflows in standard float16.
    """
    best_gpu = pick_best_gpu()
    target = f"cuda:{best_gpu}"
    free, total = torch.cuda.mem_get_info(best_gpu)
    print(f"  → Target GPU {best_gpu}: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")

    # --- Strategy 1: bfloat16 on single GPU ---
    try:
        print(f"  → Moving to {target} in bfloat16 …")
        policy = policy.to(dtype=torch.bfloat16, device=target)
        alloc = torch.cuda.memory_allocated(best_gpu) / 1e9
        print(f"  ✅ Model on {target} in bfloat16  ({alloc:.2f} GB)\n")
        return policy
    except torch.cuda.OutOfMemoryError:
        print("  ⚠️  bfloat16 OOM — trying CPU-offload via accelerate …")
        clear_gpu_memory()

    # --- Strategy 2: accelerate CPU-offload (single GPU + RAM overflow) ---
    try:
        from accelerate import dispatch_model, infer_auto_device_map

        print("  → Using accelerate CPU-offload (bfloat16) …")
        policy = policy.to(dtype=torch.bfloat16).cpu()

        max_memory = {best_gpu: int(free - 1 * 1024**3)}  # 1 GB headroom
        max_memory["cpu"] = "24GiB"  # allow spilling into RAM

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

    # --- Last resort: Move directly and hope it fits ---
    print(f"  → Last resort fallback on {target}")
    policy = policy.to(target)
    return policy


# ============================================================
# Load model
# ============================================================
def load_model(checkpoint_path: str):
    """Load PI0 policy, place on GPU, create pre/post processors."""
    if not torch.cuda.is_available():
        print("❌  No CUDA GPUs detected. Exiting.")
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    print_gpu_info()
    clear_gpu_memory()

    print("Loading PI0 policy …")
    from lerobot.policies.pi0 import PI0Policy
    from lerobot.policies.factory import make_pre_post_processors

    policy = PI0Policy.from_pretrained(checkpoint_path)

    # Unwrap torch.compile wrapper if present
    if hasattr(policy, "model") and hasattr(policy.model, "_orig_mod"):
        policy.model = policy.model._orig_mod
        print("  ✅ Unwrapped compiled model")

    policy = place_model_on_gpu(policy, num_gpus)
    policy.eval()

    # Figure out where input tensors should be sent
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
    print(f"  chunk_size: {policy.config.chunk_size}")

    return policy, preprocess, postprocess


# ============================================================
# ROS2 Node
# ============================================================
class PI0InferenceNode(Node):
    """
    All-in-one ROS2 node:
      • subscribes to /rgb and /joint_states
      • runs local PI0 inference on multi-GPU
      • publishes /joint_command and gripper actions
    """

    def __init__(self, policy, preprocess, postprocess):
        super().__init__("pi0_inference_node")

        # Store model objects
        self.policy = policy
        self.preprocess = preprocess
        self.postprocess = postprocess

        # ===============================
        # Parameters
        # ===============================
        self.declare_parameter("task", "There are two bowls. Pick up orange and place in a bowl")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("action_horizon", 40)  # Stop executing blind and look again after N steps

        self.task = self.get_parameter("task").value
        self.rate_hz = self.get_parameter("rate_hz").value
        self.action_horizon = self.get_parameter("action_horizon").value

        # ===============================
        # Joint configuration (matching teleop.py)
        # ===============================
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

        # ===============================
        # State storage
        # ===============================
        self.latest_image = None
        self.latest_image_stamp = None
        self.latest_joints = {}
        self.latest_joints_stamp = None

        self.action_chunk = None
        self.action_idx = 0
        self.current_gripper_state = 0   # 0 = open
        self.force_chunk_completion = False # Used to prevent replanning during grasp

        self.inference_count = 0
        self.is_running = True

        # ===============================
        # ROS2 setup
        # ===============================
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

        if HAS_CONTROL_MSGS:
            self.grip_client = ActionClient(
                self, FollowJointTrajectory, "/ur5_gripper/follow_joint_trajectory"
            )
        else:
            self.grip_client = None

        # Timers
        self.dt = 1.0 / self.rate_hz
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.create_timer(10.0, self.print_status)

        self.get_logger().info("=" * 60)
        self.get_logger().info("PI0 Local Inference Node Ready")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Task             : {self.task}")
        self.get_logger().info(f"Rate             : {self.rate_hz} Hz")
        self.get_logger().info(f"Action Horizon   : {self.action_horizon} steps")
        self.get_logger().info(f"Ctrl joints      : {self.CTRL_JOINTS}")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Waiting for /rgb and /joint_states streams...")

    # ========================== Callbacks ==========================

    def on_image(self, msg: RosImage):
        try:
            h, w = msg.height, msg.width
            enc = (msg.encoding or "").lower()
            data = np.frombuffer(msg.data, dtype=np.uint8)

            if enc == "rgb8":
                self.latest_image = data.reshape((h, w, 3))
            elif enc == "bgr8":
                self.latest_image = data.reshape((h, w, 3))[:, :, ::-1].copy()
            elif enc == "rgba8":
                self.latest_image = data.reshape((h, w, 4))[:, :, :3].copy()
            elif enc == "bgra8":
                self.latest_image = data.reshape((h, w, 4))[:, :, :3][:, :, ::-1].copy()
            else:
                self.latest_image = data.reshape((h, w, 3))

            self.latest_image_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"Image decode error: {e}")

    def on_joint_states(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.latest_joints[name] = float(pos)
        self.latest_joints_stamp = msg.header.stamp

    # ========================== Helpers ==========================

    def get_state_vector(self) -> np.ndarray:
        state = []
        for joint in self.CTRL_JOINTS:
            state.append(self.latest_joints.get(joint, 0.0))
        return np.array(state, dtype=np.float32)

    # ========================== Local inference ==========================

    def run_inference(self) -> np.ndarray:
        """Run PI0 inference locally."""
        if self.latest_image is None:
            self.get_logger().warn("No image available yet")
            return None

        try:
            t0 = time.time()

            # Prepare observation tensors
            image_np = self.latest_image
            state_np = self.get_state_vector()

            # Resize to 224x224
            image_pil = Image.fromarray(image_np)
            image_resized = image_pil.resize((224, 224), Image.BILINEAR)
            image_array = np.array(image_resized, dtype=np.uint8)

            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
            state_tensor = torch.from_numpy(state_np).float()

            observation = {
                "observation.images.rgb": image_tensor,   # [3, 224, 224]
                "observation.state": state_tensor,        # [state_dim]
                "task": self.task,
            }

            model_dtype = torch.bfloat16

            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=True, dtype=model_dtype):
                    batch = self.preprocess(observation)
                    action = self.policy.select_action(batch)
                action = self.postprocess(action)

            # Convert to numpy array
            if isinstance(action, torch.Tensor):
                action_np = action.cpu().numpy()
            elif isinstance(action, dict):
                key = "action" if "action" in action else list(action.keys())[0]
                val = action[key]
                action_np = val.cpu().numpy() if isinstance(val, torch.Tensor) else np.array(val)
            else:
                action_np = np.array(action)

            action_np = np.squeeze(action_np)
            latency_ms = (time.time() - t0) * 1000
            self.inference_count += 1
            self.get_logger().info(
                f"✅ Inference #{self.inference_count}  "
                f"latency={latency_ms:.1f}ms  chunk_shape={action_np.shape}"
            )
            return action_np

        except Exception as e:
            self.get_logger().error(f"❌ Inference error: {e}")
            self.get_logger().error(traceback.format_exc())
            return None

    def reset_policy(self):
        """Reset policy / preprocessor state between episodes."""
        for obj in (self.policy, self.preprocess, self.postprocess):
            if obj is not None and hasattr(obj, "reset"):
                obj.reset()
        self.action_chunk = None
        self.action_idx = 0
        self.force_chunk_completion = False
        self.get_logger().info("Policy state reset")

    # ========================== Gripper ==========================

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

    # ========================== Control loop ==========================

    def control_loop(self):
        if not self.is_running:
            return

        # Wait until we have both observations
        if self.latest_image is None:
            return
        if len(self.latest_joints) < len(self.CTRL_JOINTS):
            return

        # Check chunk sizes to establish our horizon
        chunk_len = 0
        if self.action_chunk is not None:
            if len(self.action_chunk.shape) == 2:
                chunk_len = self.action_chunk.shape[0]
            else:
                chunk_len = 1
        
        # Determine the maximum action step we should take before re-inferring
        if self.force_chunk_completion:
            horizon_limit = chunk_len  # Override to play out the whole animation
        else:
            horizon_limit = min(self.action_horizon, chunk_len)

        # Need new actions? (Action Horizon enforcement)
        if self.action_chunk is None or self.action_idx >= horizon_limit:
            self.action_chunk = self.run_inference()
            self.action_idx = 0
            self.force_chunk_completion = False # Reset the override on fresh inference
            if self.action_chunk is None:
                return

        # Pick current action from the chunk
        if len(self.action_chunk.shape) == 2:
            action = self.action_chunk[self.action_idx]
        elif len(self.action_chunk.shape) == 1:
            action = self.action_chunk
        else:
            self.get_logger().error(f"Unexpected action shape: {self.action_chunk.shape}")
            return

        self.action_idx += 1

        # Parse: [joint1..joint5, gripper_cmd]
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
        cmd.position = [float(p) for p in joint_positions] + [self.LOCK_WRIST_2]
        self.cmd_pub.publish(cmd)

        # DEBUGGING: Print what the AI wants to do with the gripper
        self.get_logger().info(
            f"Step {self.action_idx}/{horizon_limit} | AI Gripper Cmd: {gripper_cmd:.2f}",
            throttle_duration_sec=0.5
        )

        # Gripper Logic with Hysteresis / Dwell time
        target_close = True if gripper_cmd > 0.5 else False
        
        if target_close != (self.current_gripper_state == 1):
            self.send_gripper_command(close=target_close)
            
            # CRITICAL: Give the physical gripper 0.5s to actually close/open
            time.sleep(0.5)
            
            # If the AI just decided to close the gripper, force it to finish the rest of this chunk
            # blindly without replanning. This prevents it from immediately letting go of the lemon.
            if target_close:
                self.get_logger().info("GRASP INITIATED! Forcing chunk completion to prevent drop...")
                self.force_chunk_completion = True

    def print_status(self):
        has_img = self.latest_image is not None
        has_js = len(self.latest_joints) >= len(self.CTRL_JOINTS)
        self.get_logger().info(
            f"Status: running={self.is_running}, "
            f"image={has_img}, joints={has_js}, "
            f"inferences={self.inference_count}, action_idx={self.action_idx}/{self.action_horizon}"
        )


# ============================================================
# Entry point
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  PI0 Local Inference")
    print("=" * 60)

    # Ask for checkpoint path
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

    # Load model
    try:
        policy, preprocess, postprocess = load_model(checkpoint_path)
    except Exception as e:
        print(f"❌  Failed to load model: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("✅  Model ready — starting ROS2 node …\n")

    # Start ROS2 node
    rclpy.init()
    node = PI0InferenceNode(policy, preprocess, postprocess)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down …")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
