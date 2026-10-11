#!/usr/bin/env python3
"""
Standalone SmolVLA inference node for ROS2 + local CUDA inference.

This is adapted from the previous PI0 node, but it loads SmolVLAPolicy,
patches SmolVLA-incompatible checkpoint config keys before from_pretrained(),
subscribes to camera + joint state ROS2 topics, runs inference locally,
and publishes joint commands + gripper actions.
"""

import gc
import os
import sys
import traceback
import time
import json
import shutil

import numpy as np
import torch
from PIL import Image

# ============================================================
# Disable torch.compile / CUDA graphs to avoid runtime issues
# Especially important when older LeRobot/SmolVLA configs reject
# compile_model / compile_mode.
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
def print_gpu_info(logger):
    n_gpus = torch.cuda.device_count()
    logger.info(f"\n{'=' * 60}\n  Detected {n_gpus} GPU(s)\n{'=' * 60}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        logger.info(
            f"  GPU {i}: {props.name}  |  "
            f"VRAM: {total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free"
        )
    logger.info(f"{'=' * 60}\n")


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


def get_model_dtype(logger):
    """
    RTX 2080 Ti does not have native bfloat16 support.
    Use bfloat16 only when CUDA reports support; otherwise use float16.
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        logger.info("  → CUDA bf16 supported: using torch.bfloat16")
        return torch.bfloat16

    logger.info("  → CUDA bf16 not supported/stable on this GPU: using torch.float16")
    return torch.float16


def place_model_on_gpu(policy, logger):
    """Place the SmolVLA model on the GPU with a safe dtype."""
    best_gpu = pick_best_gpu()
    target = f"cuda:{best_gpu}"

    free, total = torch.cuda.mem_get_info(best_gpu)
    model_dtype = get_model_dtype(logger)

    logger.info(
        f"  → Target GPU {best_gpu}: "
        f"{free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total"
    )

    # --- Strategy 1: half precision on single GPU ---
    try:
        logger.info(f"  → Moving SmolVLA to {target} in {model_dtype} …")
        policy = policy.to(dtype=model_dtype, device=target)

        alloc = torch.cuda.memory_allocated(best_gpu) / 1e9
        logger.info(f"  ✅ Model on {target} in {model_dtype}  ({alloc:.2f} GB)\n")
        return policy

    except torch.cuda.OutOfMemoryError:
        logger.warning("  ⚠️  Single-GPU OOM — trying CPU-offload via accelerate …")
        clear_gpu_memory()

    # --- Strategy 2: accelerate CPU-offload ---
    try:
        from accelerate import dispatch_model, infer_auto_device_map

        logger.info(f"  → Using accelerate CPU-offload ({model_dtype}) …")
        policy = policy.to(dtype=model_dtype).cpu()

        max_memory = {best_gpu: int(free - 1 * 1024**3)}
        max_memory["cpu"] = "32GiB"

        device_map = infer_auto_device_map(
            policy,
            max_memory=max_memory,
            no_split_module_classes=[
                "SmolVLMWithExpertModel",
                "VLAFlowMatching",
                "PaliGemmaWithExpertModel",
            ],
        )

        policy = dispatch_model(policy, device_map=device_map)
        logger.info("  ✅ Model with CPU-offload ready\n")
        return policy

    except ImportError:
        logger.warning("  ⚠️  accelerate not installed — cannot CPU-offload")
    except Exception as e:
        logger.error(f"  ⚠️  CPU-offload failed: {e}")

    # --- Last resort ---
    logger.info(f"  → Last resort fallback on {target}")
    policy = policy.to(target)
    return policy


def get_policy_device_and_dtype(policy):
    try:
        p = next(policy.parameters())
        return p.device, p.dtype
    except StopIteration:
        return torch.device("cuda:0"), torch.float16


# ============================================================
# SmolVLA imports
# ============================================================
def import_smolvla_policy():
    """
    LeRobot import paths changed across versions.
    Try the current path first, then older paths.
    """
    try:
        from lerobot.policies.smolvla import SmolVLAPolicy

        return SmolVLAPolicy
    except Exception:
        pass

    try:
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy
    except Exception:
        pass

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    return SmolVLAPolicy


def import_make_pre_post_processors():
    """
    LeRobot moved make_pre_post_processors in some releases.
    """
    try:
        from lerobot.policies import make_pre_post_processors

        return make_pre_post_processors
    except Exception:
        from lerobot.policies.factory import make_pre_post_processors

        return make_pre_post_processors


# ============================================================
# Checkpoint config patching
# ============================================================
def _remove_keys_recursive(obj, keys_to_remove):
    removed = []

    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key in keys_to_remove:
                removed.append(key)
                del obj[key]
            else:
                removed.extend(_remove_keys_recursive(obj[key], keys_to_remove))

    elif isinstance(obj, list):
        for item in obj:
            removed.extend(_remove_keys_recursive(item, keys_to_remove))

    return removed


def patch_checkpoint_config(checkpoint_path: str, logger):
    """
    Automatically removes legacy/invalid keys from config.json to prevent
    draccus DecodingError before SmolVLAPolicy.from_pretrained().

    The current error is:
        fields `compile_model`, `compile_mode` are not valid for SmolVLAConfig

    These keys are safe to remove because this script disables torch.compile anyway.
    """
    config_path = os.path.join(checkpoint_path, "config.json")

    if not os.path.exists(config_path):
        logger.warning(
            f"⚠️  No config.json found at {config_path}. "
            f"If checkpoint is a Hugging Face repo id, this is okay."
        )
        return

    with open(config_path, "r") as f:
        try:
            config_data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"⚠️  Could not parse config.json: {config_path}")
            return

    keys_to_remove = {
        "use_relative_actions",
        "relative_exclude_joints",
        "action_feature_names",
        "compile_model",
        "compile_mode",
    }

    removed = _remove_keys_recursive(config_data, keys_to_remove)

    # If this checkpoint was copied/edited manually, make sure type is smolvla.
    # Do not overwrite if it already says something else intentionally.
    if config_data.get("type") is None:
        config_data["type"] = "smolvla"

    if removed:
        backup_path = config_path + ".bak"

        if not os.path.exists(backup_path):
            shutil.copy(config_path, backup_path)

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        unique_removed = sorted(set(removed))
        logger.info(
            f"🔧 Auto-patched {config_path}. "
            f"Removed invalid keys: {unique_removed}. "
            f"Backup saved as config.json.bak"
        )
    else:
        logger.info("✅ config.json patch check passed: no invalid legacy keys found.")


# ============================================================
# Load model
# ============================================================
def load_model(checkpoint_path: str, logger):
    """Load SmolVLA policy, place on GPU, create pre/post processors."""
    if not torch.cuda.is_available():
        logger.error("❌  No CUDA GPUs detected. Exiting.")
        sys.exit(1)

    patch_checkpoint_config(checkpoint_path, logger)

    print_gpu_info(logger)
    clear_gpu_memory()

    logger.info("Loading SmolVLA policy …")

    SmolVLAPolicy = import_smolvla_policy()
    make_pre_post_processors = import_make_pre_post_processors()

    # Hide ROS args from draccus/LeRobot.
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]

    try:
        policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    finally:
        sys.argv = original_argv

    # Unwrap torch.compile wrapper if present.
    if hasattr(policy, "model") and hasattr(policy.model, "_orig_mod"):
        policy.model = policy.model._orig_mod
        logger.info("  ✅ Unwrapped compiled model")

    policy = place_model_on_gpu(policy, logger)
    policy.eval()

    input_device, input_dtype = get_policy_device_and_dtype(policy)

    logger.info(f"  Input tensors → {input_device}")
    logger.info(f"  Model dtype   → {input_dtype}")

    # Hide ROS args again for processor initialization.
    sys.argv = [sys.argv[0]]

    try:
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            checkpoint_path,
            preprocessor_overrides={
                "device_processor": {
                    "device": str(input_device)
                }
            },
        )
    finally:
        sys.argv = original_argv

    logger.info("  ✅ Preprocessors created")

    if hasattr(policy.config, "chunk_size"):
        logger.info(f"  chunk_size      : {policy.config.chunk_size}")

    if hasattr(policy.config, "n_action_steps"):
        logger.info(f"  n_action_steps  : {policy.config.n_action_steps}")

    return policy, preprocess, postprocess, input_device, input_dtype


# ============================================================
# ROS2 Node
# ============================================================
class SmolVLAInferenceNode(Node):
    def __init__(self):
        super().__init__("smolvla_inference_node")

        # ===============================
        # Parameters
        # ===============================
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("task", "Pick up the lemon and put it on the bowl.")
        self.declare_parameter("robot_type", "")
        self.declare_parameter("rate_hz", 5.0)
        self.declare_parameter("action_horizon", 5)

        # SmolVLA can do its own resize/padding based on config.
        # Keep 0 by default. Set to 224 only if your checkpoint was trained that way
        # and you specifically need to mimic the old PI0 input behavior.
        self.declare_parameter("image_size", 0)

        checkpoint_path = self.get_parameter("checkpoint").value
        self.task = self.get_parameter("task").value
        self.robot_type = self.get_parameter("robot_type").value
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.action_horizon = int(self.get_parameter("action_horizon").value)
        self.image_size = int(self.get_parameter("image_size").value)

        self.get_logger().info("\n" + "=" * 60)
        self.get_logger().info("  SmolVLA Local Inference Node Initialization")
        self.get_logger().info("=" * 60)

        # Fallback if checkpoint parameter is omitted.
        if not checkpoint_path:
            try:
                checkpoint_path = input(
                    "\nEnter the path to your pretrained SmolVLA checkpoint:\n>>> "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                checkpoint_path = ""

        if not checkpoint_path:
            self.get_logger().error("❌ No checkpoint path provided. Exiting.")
            sys.exit(1)

        checkpoint_path = os.path.expanduser(checkpoint_path)

        # Local path or HF repo id is allowed.
        # Only enforce existence for obvious local paths.
        looks_like_local = (
            checkpoint_path.startswith("/")
            or checkpoint_path.startswith(".")
            or os.path.sep in checkpoint_path
        )

        if looks_like_local and not os.path.exists(checkpoint_path):
            self.get_logger().error(f"❌ Checkpoint path does not exist: {checkpoint_path}")
            sys.exit(1)

        self.get_logger().info(f"📂 Checkpoint: {checkpoint_path}")

        # Load the AI model.
        try:
            (
                self.policy,
                self.preprocess,
                self.postprocess,
                self.input_device,
                self.model_dtype,
            ) = load_model(checkpoint_path, self.get_logger())

        except Exception as e:
            self.get_logger().error(f"❌ Failed to load SmolVLA model: {e}")
            traceback.print_exc()
            sys.exit(1)

        # ===============================
        # Joint configuration
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

        self.GRIPPER_JOINTS = [
            "left_inner_finger_joint",
            "right_inner_finger_joint",
        ]

        self.GRIP_OPEN_POS = 0.0
        self.GRIP_CLOSE_POS = 0.5

        # ===============================
        # State storage
        # ===============================
        self.latest_image = None
        self.latest_image_stamp = None

        self.latest_image_up = None
        self.latest_image_up_stamp = None

        self.latest_joints = {}
        self.latest_joints_stamp = None

        self.action_chunk = None
        self.action_idx = 0
        self.current_gripper_state = 0
        self.force_chunk_completion = False

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
        self.create_subscription(RosImage, "/rgbUp", self.on_image_up, cam_qos)
        self.create_subscription(JointState, "/joint_states", self.on_joint_states, default_qos)

        self.cmd_pub = self.create_publisher(JointState, "/joint_command", 10)

        if HAS_CONTROL_MSGS:
            self.grip_client = ActionClient(
                self,
                FollowJointTrajectory,
                "/ur5_gripper/follow_joint_trajectory",
            )
        else:
            self.grip_client = None

        # Timers
        self.dt = 1.0 / self.rate_hz
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.create_timer(10.0, self.print_status)

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Task             : {self.task}")
        self.get_logger().info(f"Robot type       : {self.robot_type}")
        self.get_logger().info(f"Rate             : {self.rate_hz} Hz")
        self.get_logger().info(f"Action Horizon   : {self.action_horizon} steps")
        self.get_logger().info(f"Manual image size: {self.image_size} (0 means no manual resize)")
        self.get_logger().info(
            "✅ NODE READY. Waiting for /rgb, /rgbUp, and /joint_states streams..."
        )

    # ========================== Callbacks ==========================
    def _decode_ros_image(self, msg: RosImage) -> np.ndarray:
        h, w = msg.height, msg.width
        enc = (msg.encoding or "").lower()
        data = np.frombuffer(msg.data, dtype=np.uint8)

        if enc == "rgb8":
            return data.reshape((h, w, 3))

        elif enc == "bgr8":
            return data.reshape((h, w, 3))[:, :, ::-1].copy()

        elif enc == "rgba8":
            return data.reshape((h, w, 4))[:, :, :3].copy()

        elif enc == "bgra8":
            return data.reshape((h, w, 4))[:, :, :3][:, :, ::-1].copy()

        else:
            return data.reshape((h, w, 3))

    def on_image(self, msg: RosImage):
        try:
            self.latest_image = self._decode_ros_image(msg)
            self.latest_image_stamp = msg.header.stamp

        except Exception as e:
            self.get_logger().error(f"Image decode error (/rgb): {e}")

    def on_image_up(self, msg: RosImage):
        try:
            self.latest_image_up = self._decode_ros_image(msg)
            self.latest_image_up_stamp = msg.header.stamp

        except Exception as e:
            self.get_logger().error(f"Image decode error (/rgbUp): {e}")

    def on_joint_states(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.latest_joints[name] = float(pos)

        self.latest_joints_stamp = msg.header.stamp

    def get_state_vector(self) -> np.ndarray:
        state = []

        for joint in self.CTRL_JOINTS:
            state.append(self.latest_joints.get(joint, 0.0))

        return np.array(state, dtype=np.float32)

    def image_to_tensor(self, image_np: np.ndarray) -> torch.Tensor:
        """
        Convert HWC uint8 image to CHW float tensor in [0, 1].
        SmolVLA will resize/pad internally according to its config.
        """
        if self.image_size and self.image_size > 0:
            image_pil = Image.fromarray(image_np)
            image_pil = image_pil.resize(
                (self.image_size, self.image_size),
                Image.BILINEAR,
            )
            image_np = np.array(image_pil, dtype=np.uint8)

        image_tensor = (
            torch.from_numpy(image_np)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            / 255.0
        )

        return image_tensor

    # ========================== Local inference ==========================
    def run_inference(self) -> np.ndarray:
        if self.latest_image is None or self.latest_image_up is None:
            self.get_logger().warning("Waiting for all camera streams to be available...")
            return None

        try:
            t0 = time.time()

            image_tensor = self.image_to_tensor(self.latest_image)
            image_up_tensor = self.image_to_tensor(self.latest_image_up)
            state_tensor = torch.from_numpy(self.get_state_vector()).float()

            # IMPORTANT:
            # These keys must match the input_features/image_features in your SmolVLA checkpoint config.
            # If your checkpoint expects different names, change these two image keys.
            observation = {
                "observation.images.rgb": image_tensor,
                "observation.images.rgbUp": image_up_tensor,
                "observation.state": state_tensor,
                "task": self.task,
                "robot_type": self.robot_type,
            }

            with torch.inference_mode():
                with torch.amp.autocast(
                    "cuda",
                    enabled=self.input_device.type == "cuda",
                    dtype=self.model_dtype,
                ):
                    batch = self.preprocess(observation)
                    action = self.policy.select_action(batch)

                action = self.postprocess(action)

            if isinstance(action, torch.Tensor):
                action_np = action.detach().cpu().numpy()

            elif isinstance(action, dict):
                key = "action" if "action" in action else list(action.keys())[0]
                val = action[key]
                action_np = (
                    val.detach().cpu().numpy()
                    if isinstance(val, torch.Tensor)
                    else np.array(val)
                )

            else:
                action_np = np.array(action)

            action_np = np.squeeze(action_np)

            latency_ms = (time.time() - t0) * 1000
            self.inference_count += 1

            self.get_logger().info(
                f"✅ SmolVLA action #{self.inference_count}  "
                f"latency={latency_ms:.1f}ms  action_shape={action_np.shape}"
            )

            return action_np

        except Exception as e:
            self.get_logger().error(f"❌ Inference error: {e}")
            self.get_logger().error(traceback.format_exc())
            return None

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

        if self.latest_image is None or self.latest_image_up is None:
            return

        if len(self.latest_joints) < len(self.CTRL_JOINTS):
            return

        # SmolVLA select_action has its own internal action queue.
        # We still keep the external action_chunk handling so the rest of your PI0
        # ROS publishing logic remains unchanged.
        chunk_len = 0

        if self.action_chunk is not None:
            if len(self.action_chunk.shape) == 2:
                chunk_len = self.action_chunk.shape[0]
            else:
                chunk_len = 1

        if self.force_chunk_completion:
            horizon_limit = chunk_len
        else:
            horizon_limit = min(self.action_horizon, chunk_len)

        if self.action_chunk is None or self.action_idx >= horizon_limit:
            self.action_chunk = self.run_inference()
            self.action_idx = 0
            self.force_chunk_completion = False

            if self.action_chunk is None:
                return

        if len(self.action_chunk.shape) == 2:
            action = self.action_chunk[self.action_idx]

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

        # Safety guard: publish exactly the five controlled joints + locked wrist_2.
        joint_positions = np.asarray(joint_positions, dtype=np.float32)

        if joint_positions.shape[0] < len(self.CTRL_JOINTS):
            self.get_logger().error(
                f"Action has only {joint_positions.shape[0]} joint values, "
                f"but {len(self.CTRL_JOINTS)} are required. "
                f"Check action feature mapping."
            )
            return

        joint_positions = joint_positions[: len(self.CTRL_JOINTS)]

        cmd = JointState()
        cmd.header = Header()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = self.PUBLISH_JOINTS
        cmd.position = [float(p) for p in joint_positions] + [self.LOCK_WRIST_2]

        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"Step {self.action_idx}/{horizon_limit} | "
            f"AI Gripper Cmd: {float(gripper_cmd):.2f}",
            throttle_duration_sec=0.5,
        )

        target_close = True if float(gripper_cmd) > 0.5 else False

        if target_close != (self.current_gripper_state == 1):
            self.send_gripper_command(close=target_close)
            time.sleep(0.5)

            if target_close:
                self.get_logger().info(
                    "GRASP INITIATED! Forcing chunk completion to prevent drop..."
                )
                self.force_chunk_completion = True

    def print_status(self):
        has_img = self.latest_image is not None
        has_img_up = self.latest_image_up is not None
        has_js = len(self.latest_joints) >= len(self.CTRL_JOINTS)

        self.get_logger().info(
            f"Status: running={self.is_running}, "
            f"rgb={has_img}, rgbUp={has_img_up}, joints={has_js}, "
            f"inferences={self.inference_count}, "
            f"action_idx={self.action_idx}/{self.action_horizon}"
        )


def main(args=None):
    try:
        rclpy.init(args=args)

        node = SmolVLAInferenceNode()

        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\nShutting down cleanly...")

    except Exception as e:
        print(f"\n❌ FATAL STARTUP ERROR: {e}")
        traceback.print_exc()

    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
