import json
import socket
from pathlib import Path

import numpy as np
import pytest

from ur5_vla_dataset_recorder.core import UR5_JOINT_ORDER, ValidationError
from ur5_vla_dataset_recorder.policy_executor import ShadowObservationBuffer, validate_local_artifacts
from ur5_vla_dataset_recorder.policy_inference_server import image_tensor
from ur5_vla_dataset_recorder.shadow_protocol import (
    ImagePayload,
    ProtocolError,
    receive_observation,
    receive_prediction,
    send_observation,
    send_prediction,
)


def _images(data=b"x"):
    return [
        ImagePayload("observation.images.base", 1, 1, "mono8", 1, data),
        ImagePayload("observation.images.wrist", 1, 1, "mono8", 1, data),
    ]


def test_observation_and_prediction_protocol_round_trip():
    sender, receiver = socket.socketpair()
    try:
        send_observation(sender, [0] * 6 + [1], "pick cube", _images())
        state, task, images = receive_observation(receiver)
        assert state == [0] * 6 + [1]
        assert task == "pick cube"
        assert [image.key for image in images] == [
            "observation.images.base",
            "observation.images.wrist",
        ]
        send_prediction(receiver, [[0] * 6 + [1]], 0.25)
        chunk, latency = receive_prediction(sender)
        assert chunk == [[0] * 6 + [1]]
        assert latency == 0.25
    finally:
        sender.close()
        receiver.close()


def test_protocol_rejects_non_finite_state_and_action():
    sender, receiver = socket.socketpair()
    try:
        with pytest.raises(ProtocolError, match="finite"):
            send_observation(sender, [0] * 6 + [float("nan")], "pick cube", _images())
        with pytest.raises(ProtocolError, match="finite"):
            send_prediction(sender, [[0] * 6 + [float("inf")]], 0.1)
    finally:
        sender.close()
        receiver.close()


def test_ros_rgb_image_conversion_preserves_channel_order():
    row = np.zeros((480, 640, 3), dtype=np.uint8)
    row[0, 0] = [10, 20, 30]
    payload = ImagePayload(
        "observation.images.base", 480, 640, "bgr8", 640 * 3, row.tobytes()
    )
    tensor = image_tensor(payload)
    assert tuple(tensor.shape) == (3, 480, 640)
    assert tensor[:, 0, 0].tolist() == pytest.approx([30 / 255, 20 / 255, 10 / 255])


def test_shadow_buffer_requires_complete_fresh_zero_servo_snapshot():
    buffer = ShadowObservationBuffer()
    now = 100.0
    for image in _images():
        buffer.update_image(image.key, 1, 1, image.encoding, image.step, image.data, now)
    buffer.update_joints([0] * 6, now)
    buffer.update_gripper(1, now)
    buffer.update_servo(0, now)
    state, images, ages = buffer.snapshot(now + 0.05, 0.10, 0.20)
    assert state == [0] * 6 + [1]
    assert len(images) == 2 and max(ages.values()) == pytest.approx(0.05)
    with pytest.raises(ValidationError, match="stale"):
        buffer.snapshot(now + 0.30, 0.10, 0.20)
    buffer.update_joints([0] * 6, now + 0.30)
    buffer.update_gripper(1, now + 0.30)
    buffer.update_servo(0, now + 0.30)
    state, _, ages = buffer.snapshot(now + 0.30, 0.10, 0.20, max_camera_age_s=0.50)
    assert state[-1] == 1 and ages["base_image"] == pytest.approx(0.30)
    buffer.update_servo(2, now + 0.30)
    with pytest.raises(ValidationError, match="Servo status"):
        buffer.snapshot(now + 0.30, 1.0, 1.0)


def test_local_artifacts_must_match_physical_ur5_schema(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    dataset = tmp_path / "dataset"
    checkpoint.mkdir()
    (dataset / "meta").mkdir(parents=True)
    inputs = {
        "observation.images.base": {"shape": [3, 480, 640]},
        "observation.images.wrist": {"shape": [3, 480, 640]},
        "observation.state": {"shape": [7]},
    }
    (checkpoint / "config.json").write_text(
        json.dumps({"input_features": inputs, "output_features": {"action": {"shape": [7]}}})
    )
    (checkpoint / "train_config.json").write_text(
        json.dumps({"dataset": {"repo_id": "local/ur5_orange_cube_v1"}})
    )
    (dataset / "meta/info.json").write_text(
        json.dumps({"robot_type": "ur5", "features": {"action": {"shape": [7]}}})
    )
    (dataset / "meta/ur5_vla_conversion.json").write_text(
        json.dumps(
            {
                "joint_order": list(UR5_JOINT_ORDER),
                "action_semantics": "six absolute joint-position targets plus absolute binary gripper command",
            }
        )
    )
    validate_local_artifacts(checkpoint, dataset, "local/ur5_orange_cube_v1")
    with pytest.raises(ValidationError, match="trained on"):
        validate_local_artifacts(checkpoint, dataset, "local/different")


def test_shadow_executor_has_no_publishers_and_live_stays_locked():
    source = (Path(__file__).parents[1] / "ur5_vla_dataset_recorder/policy_executor.py").read_text()
    assert "create_publisher" not in source
    assert "LIVE MODE IS LOCKED" in source
