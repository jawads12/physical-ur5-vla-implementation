"""Serve a LeRobot policy to the ROS shadow process over localhost only.

This process has no ROS dependency and cannot publish robot commands.
"""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

import numpy as np
import torch

from .shadow_protocol import ImagePayload, ProtocolError, receive_observation, send_error, send_prediction


EXPECTED_IMAGE_KEYS = ("observation.images.base", "observation.images.wrist")
EXPECTED_IMAGE_SHAPE = (3, 480, 640)


def image_tensor(image: ImagePayload) -> torch.Tensor:
    channels_by_encoding = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}
    encoding = image.encoding.lower()
    if encoding not in channels_by_encoding:
        raise ProtocolError(f"unsupported ROS image encoding: {image.encoding}")
    channels = channels_by_encoding[encoding]
    row_bytes = image.width * channels
    if image.height != 480 or image.width != 640:
        raise ProtocolError(f"{image.key} must be 480x640, got {image.height}x{image.width}")
    if image.step < row_bytes or len(image.data) != image.height * image.step:
        raise ProtocolError(f"{image.key} data length/step is inconsistent")
    rows = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.step)
    array = rows[:, :row_bytes].reshape(image.height, image.width, channels)
    if encoding == "bgr8":
        array = array[:, :, ::-1]
    elif encoding == "rgba8":
        array = array[:, :, :3]
    elif encoding == "bgra8":
        array = array[:, :, [2, 1, 0]]
    elif encoding == "mono8":
        array = np.repeat(array, 3, axis=2)
    # The incoming bytes buffer is immutable. Copy before torch.from_numpy so
    # PyTorch never receives a read-only array view.
    array = np.array(array, copy=True, order="C")
    return torch.from_numpy(array).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)


def validate_checkpoint_features(config) -> None:
    inputs = config.input_features
    outputs = config.output_features
    expected_inputs = {*EXPECTED_IMAGE_KEYS, "observation.state"}
    if set(inputs) != expected_inputs:
        raise ValueError(f"checkpoint input features must be {sorted(expected_inputs)}, got {sorted(inputs)}")
    for key in EXPECTED_IMAGE_KEYS:
        if tuple(inputs[key].shape) != EXPECTED_IMAGE_SHAPE:
            raise ValueError(f"{key} checkpoint shape must be {EXPECTED_IMAGE_SHAPE}, got {inputs[key].shape}")
    if tuple(inputs["observation.state"].shape) != (7,):
        raise ValueError("checkpoint observation.state must be 7D")
    if set(outputs) != {"action"} or tuple(outputs["action"].shape) != (7,):
        raise ValueError("checkpoint action must be 7D")


def load_policy(checkpoint: Path, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    validate_checkpoint_features(config)
    config.device = device
    policy = (
        get_policy_class(config.type)
        .from_pretrained(checkpoint, config=config, local_files_only=True)
        .to(device)
        .eval()
    )
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def predict(policy, preprocessor, postprocessor, state, task, images, device: str, inference_seed=None):
    by_key = {image.key: image for image in images}
    if set(by_key) != set(EXPECTED_IMAGE_KEYS):
        raise ProtocolError(f"camera keys must be {list(EXPECTED_IMAGE_KEYS)}")
    gripper = float(state[6])
    if min(abs(gripper), abs(gripper - 1.0)) > 1e-3:
        raise ProtocolError("estimated gripper state must be binary")
    canonical_state = [*map(float, state[:6]), float(round(gripper))]
    batch = {
        key: image_tensor(by_key[key]).unsqueeze(0).to(device) for key in EXPECTED_IMAGE_KEYS
    }
    batch["observation.state"] = torch.tensor(canonical_state, dtype=torch.float32, device=device).unsqueeze(0)
    batch["task"] = [task]
    processed = preprocessor(batch)
    if inference_seed is not None:
        torch.manual_seed(inference_seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(inference_seed)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        predicted = policy.predict_action_chunk(processed)
        predicted = postprocessor(predicted)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    latency = time.perf_counter() - started
    values = predicted.detach().cpu().numpy()
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[-1] != 7:
        raise ProtocolError(f"policy returned invalid action shape {values.shape}")
    if not np.isfinite(values).all():
        raise ProtocolError("policy returned NaN or infinity")
    return values[0].tolist(), latency


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--inference-seed", type=int,
        help="reset policy sampling to this seed before every request",
    )
    parser.add_argument("--request-timeout", type=float, default=5.0)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("The shadow inference server may listen only on localhost")
    if not args.checkpoint.is_dir():
        raise SystemExit(f"checkpoint directory does not exist: {args.checkpoint}")

    policy, preprocessor, postprocessor = load_policy(args.checkpoint, args.device)
    print(f"SHADOW inference server loaded {args.checkpoint}", flush=True)
    print("No ROS imports or command publishers exist in this process", flush=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(1)
        print(f"Listening on {args.host}:{args.port}", flush=True)
        try:
            while True:
                connection, address = listener.accept()
                with connection:
                    connection.settimeout(args.request_timeout)
                    try:
                        state, task, images = receive_observation(connection)
                        chunk, latency = predict(
                            policy, preprocessor, postprocessor, state, task, images, args.device,
                            args.inference_seed,
                        )
                        send_prediction(connection, chunk, latency)
                        first = chunk[0]
                        print(
                            f"shadow prediction from {address[0]} latency={latency:.3f}s "
                            f"first_action={[round(value, 5) for value in first]}",
                            flush=True,
                        )
                    except Exception as exc:
                        try:
                            send_error(connection, str(exc))
                        except Exception:
                            pass
                        print(f"rejected shadow request from {address[0]}: {exc}", flush=True)
        except KeyboardInterrupt:
            print("Shadow inference server stopped", flush=True)


if __name__ == "__main__":
    main()
