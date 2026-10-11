"""Bounded localhost protocol for ROS-to-policy shadow inference.

The protocol deliberately carries observations and predictions only.  It has no
ROS dependency and no concept of a controller command.
"""

from __future__ import annotations

import json
import math
import socket
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


PROTOCOL_VERSION = 1
MAX_HEADER_BYTES = 64 * 1024
MAX_BLOB_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BLOB_BYTES = 24 * 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class ProtocolError(RuntimeError):
    """Raised when a peer sends an invalid or incomplete message."""


@dataclass(frozen=True)
class ImagePayload:
    key: str
    height: int
    width: int
    encoding: str
    step: int
    data: bytes

    def descriptor(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "height": self.height,
            "width": self.width,
            "encoding": self.encoding,
            "step": self.step,
            "length": len(self.data),
        }


def _recv_exact(stream: socket.socket, size: int) -> bytes:
    if size < 0:
        raise ProtocolError("negative payload size")
    parts: list[bytes] = []
    remaining = size
    while remaining:
        part = stream.recv(remaining)
        if not part:
            raise ProtocolError("peer closed the connection before the message completed")
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)


def _json_bytes(value: Mapping[str, Any], limit: int) -> bytes:
    try:
        payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"message is not finite JSON: {exc}") from exc
    if not payload or len(payload) > limit:
        raise ProtocolError(f"JSON message size {len(payload)} is outside the allowed range")
    return payload


def _send_json(stream: socket.socket, value: Mapping[str, Any], limit: int) -> None:
    payload = _json_bytes(value, limit)
    stream.sendall(struct.pack("!I", len(payload)))
    stream.sendall(payload)


def _recv_json(stream: socket.socket, limit: int) -> dict[str, Any]:
    (size,) = struct.unpack("!I", _recv_exact(stream, 4))
    if size == 0 or size > limit:
        raise ProtocolError(f"received JSON size {size} is outside the allowed range")
    try:
        value = json.loads(_recv_exact(stream, size).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid JSON message: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError("JSON message must be an object")
    return value


def send_observation(
    stream: socket.socket,
    state: Sequence[float],
    task: str,
    images: Sequence[ImagePayload],
) -> None:
    values = [float(value) for value in state]
    if len(values) != 7 or not all(math.isfinite(value) for value in values):
        raise ProtocolError("state must contain seven finite values")
    if not task.strip():
        raise ProtocolError("task must not be empty")
    if len(images) != 2:
        raise ProtocolError("exactly two camera images are required")
    descriptors = [image.descriptor() for image in images]
    lengths = [descriptor["length"] for descriptor in descriptors]
    if any(length <= 0 or length > MAX_BLOB_BYTES for length in lengths):
        raise ProtocolError("camera payload size is outside the allowed range")
    if sum(lengths) > MAX_TOTAL_BLOB_BYTES:
        raise ProtocolError("combined camera payload is too large")
    _send_json(
        stream,
        {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "observation",
            "state": values,
            "task": task,
            "images": descriptors,
        },
        MAX_HEADER_BYTES,
    )
    for image in images:
        stream.sendall(image.data)


def receive_observation(stream: socket.socket) -> tuple[list[float], str, list[ImagePayload]]:
    header = _recv_json(stream, MAX_HEADER_BYTES)
    if header.get("protocol_version") != PROTOCOL_VERSION or header.get("kind") != "observation":
        raise ProtocolError("unsupported observation protocol")
    state = header.get("state")
    task = header.get("task")
    descriptors = header.get("images")
    if not isinstance(state, list) or len(state) != 7:
        raise ProtocolError("state must contain seven values")
    try:
        values = [float(value) for value in state]
    except (TypeError, ValueError) as exc:
        raise ProtocolError("state contains a non-numeric value") from exc
    if not all(math.isfinite(value) for value in values):
        raise ProtocolError("state contains NaN or infinity")
    if not isinstance(task, str) or not task.strip():
        raise ProtocolError("task must not be empty")
    if not isinstance(descriptors, list) or len(descriptors) != 2:
        raise ProtocolError("exactly two image descriptors are required")

    images: list[ImagePayload] = []
    total = 0
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise ProtocolError("image descriptor must be an object")
        try:
            key = str(descriptor["key"])
            height = int(descriptor["height"])
            width = int(descriptor["width"])
            encoding = str(descriptor["encoding"])
            step = int(descriptor["step"])
            length = int(descriptor["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("invalid image descriptor") from exc
        if height <= 0 or width <= 0 or step <= 0:
            raise ProtocolError("image dimensions and step must be positive")
        if length <= 0 or length > MAX_BLOB_BYTES:
            raise ProtocolError("camera payload size is outside the allowed range")
        total += length
        if total > MAX_TOTAL_BLOB_BYTES:
            raise ProtocolError("combined camera payload is too large")
        images.append(ImagePayload(key, height, width, encoding, step, _recv_exact(stream, length)))
    return values, task, images


def validate_action_chunk(action_chunk: Any) -> list[list[float]]:
    if not isinstance(action_chunk, list) or not action_chunk:
        raise ProtocolError("action chunk must be a non-empty list")
    if len(action_chunk) > 512:
        raise ProtocolError("action chunk is unexpectedly long")
    result: list[list[float]] = []
    for action in action_chunk:
        if not isinstance(action, list) or len(action) != 7:
            raise ProtocolError("every action must contain seven values")
        try:
            values = [float(value) for value in action]
        except (TypeError, ValueError) as exc:
            raise ProtocolError("action contains a non-numeric value") from exc
        if not all(math.isfinite(value) for value in values):
            raise ProtocolError("action contains a non-finite value")
        result.append(values)
    return result


def send_prediction(stream: socket.socket, action_chunk: Sequence[Sequence[float]], latency_s: float) -> None:
    chunk = validate_action_chunk([list(action) for action in action_chunk])
    latency = float(latency_s)
    if not math.isfinite(latency) or latency < 0:
        raise ProtocolError("inference latency must be finite and non-negative")
    _send_json(
        stream,
        {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "prediction",
            "action_chunk": chunk,
            "latency_s": latency,
        },
        MAX_RESPONSE_BYTES,
    )


def send_error(stream: socket.socket, message: str) -> None:
    _send_json(
        stream,
        {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "error",
            "error": str(message)[:2000],
        },
        MAX_RESPONSE_BYTES,
    )


def receive_prediction(stream: socket.socket) -> tuple[list[list[float]], float]:
    response = _recv_json(stream, MAX_RESPONSE_BYTES)
    if response.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("unsupported prediction protocol")
    if response.get("kind") == "error":
        raise ProtocolError(f"inference server error: {response.get('error', 'unknown error')}")
    if response.get("kind") != "prediction":
        raise ProtocolError("unexpected response kind")
    chunk = validate_action_chunk(response.get("action_chunk"))
    try:
        latency = float(response["latency_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("invalid inference latency") from exc
    if not math.isfinite(latency) or latency < 0:
        raise ProtocolError("invalid inference latency")
    return chunk, latency


def request_prediction(
    host: str,
    port: int,
    timeout_s: float,
    state: Sequence[float],
    task: str,
    images: Sequence[ImagePayload],
) -> tuple[list[list[float]], float]:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ProtocolError("shadow inference is restricted to localhost")
    with socket.create_connection((host, int(port)), timeout=float(timeout_s)) as stream:
        stream.settimeout(float(timeout_s))
        send_observation(stream, state, task, images)
        return receive_prediction(stream)
