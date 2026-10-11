import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from ur5_vla_dataset_recorder.core import UR5_JOINT_ORDER
from ur5_vla_dataset_recorder.convert_lerobot import prepare_output
from ur5_vla_dataset_recorder.action_semantics import absolute_to_delta, delta_to_absolute
from ur5_vla_dataset_recorder.raw_dataset import canonical_state_action, load_depth_m, validate_raw_episode
from ur5_vla_dataset_recorder.live_control import LiveReadiness, OneShotArmGate
from ur5_vla_dataset_recorder.safety import (
    PredictionStabilityGate, SafetyConfig, SafetyFault, SafetySnapshot,
    SafetySupervisor, StabilityConfig,
)


def test_joint_reordering_and_seven_dimensional_construction():
    reverse = list(reversed(UR5_JOINT_ORDER))
    frame = {"observation.state": [0, 1, 2, 3, 4, 5, 1], "action": [10, 11, 12, 13, 14, 15, 0]}
    state, action = canonical_state_action(frame, {"joint_order": reverse, "command_joint_order": reverse})
    assert state.tolist() == [5, 4, 3, 2, 1, 0, 1]
    assert action.tolist() == [15, 14, 13, 12, 11, 10, 0]
    assert state.shape == action.shape == (7,)
    assert state.dtype == action.dtype == np.float32


def test_delta_action_round_trip_and_fail_closed_limit():
    state = np.asarray([1, 2, 3, 4, 5, 6, 1], dtype=np.float32)
    target = np.asarray([1.001, 1.998, 3.003, 4, 5, 6, 0], dtype=np.float32)
    delta = absolute_to_delta(state, target)
    assert np.allclose(delta[:6], [.001, -.002, .003, 0, 0, 0], atol=1e-7)
    assert np.allclose(delta_to_absolute(state, delta, .0031), target)
    with pytest.raises(ValueError, match="raw predicted joint delta"):
        delta_to_absolute(state, delta, .0029)


def test_invalid_depth_values_are_masked(tmp_path):
    raw = np.array([[0, 1000], [65535, 250]], dtype=np.uint16)
    path = tmp_path / "depth.png"; Image.fromarray(raw).save(path)
    depth, valid = load_depth_m(path, 0.001)
    assert depth.tolist() == [[0.0, 1.0], [0.0, 0.25]]
    assert valid.tolist() == [[0, 1], [0, 1]]


def test_non_destructive_overwrite(tmp_path):
    output = tmp_path / "existing"; output.mkdir(); marker = output / "keep"; marker.write_text("user")
    with pytest.raises(FileExistsError): prepare_output(output, overwrite=False)
    assert marker.read_text() == "user"


def test_safety_rejects_bad_action_and_latches():
    supervisor = SafetySupervisor(SafetyConfig(max_target_delta_rad=.1, max_velocity_rad_s=2))
    snapshot = SafetySnapshot([0] * 6, {"base_image": 0, "wrist_image": 0, "joint_state": 0, "gripper_state": 0, "inference": 0}, 0, "forward_position_controller", True, True)
    supervisor.manual_enable()
    with pytest.raises(SafetyFault): supervisor.check([1] * 6 + [1], snapshot, .05)
    assert not supervisor.enabled and supervisor.latched_fault


def test_prediction_stability_gate_requires_bounded_consistent_sequence():
    gate = PredictionStabilityGate(StabilityConfig(.05, .01, 3))
    joints = [0.0] * 6
    assert gate.observe([.02] * 6 + [1.0], joints)[0] is False
    assert gate.observe([.021] * 6 + [1.0], joints)[0] is False
    assert gate.observe([.022] * 6 + [1.0], joints)[0] is True
    assert gate.observe([.08] * 6 + [1.0], joints)[0] is False
    assert gate.consecutive == 0


def test_one_shot_gate_consumes_arm_and_fails_closed():
    supervisor = SafetySupervisor(SafetyConfig(max_target_delta_rad=.1, max_velocity_rad_s=2))
    gate = OneShotArmGate(supervisor)
    ready = LiveReadiness(
        joints=[0] * 6,
        ages_s={"base_image": 0, "wrist_image": 0, "joint_state": 0, "gripper_state": 0, "inference": 0},
        servo_status=0,
        controller="forward_position_controller",
        program_running=True,
        deadman_held=True,
        collision_free=True,
    )
    gate.arm()
    assert gate.approve([.01] * 6 + [1], ready, .1) == [.01] * 6 + [1]
    assert not gate.armed and not supervisor.enabled
    with pytest.raises(RuntimeError, match="not armed"):
        gate.approve([0] * 6 + [1], ready, .1)

    gate.arm()
    blocked = LiveReadiness(**{**ready.__dict__, "deadman_held": False})
    with pytest.raises(SafetyFault, match="deadman"):
        gate.approve([0] * 6 + [1], blocked, .1)
    assert not gate.armed and supervisor.latched_fault


def _synthetic_episode(root: Path, missing=False):
    ep = root / "episode_test"; (ep / "images/base").mkdir(parents=True); (ep / "images/wrist").mkdir(parents=True); (ep / "depth/base").mkdir(parents=True); (ep / "depth/wrist").mkdir(parents=True)
    meta = {"outcome":"success", "frames":3, "joint_order":list(UR5_JOINT_ORDER), "command_joint_order":list(UR5_JOINT_ORDER), "depth_unit_m":.001, "task":"test"}
    (ep / "metadata.yaml").write_text(yaml.safe_dump(meta)); rows=[]
    for i,g in enumerate([1,0,1]):
        rgb=np.full((480,640,3), i*20, np.uint8); depth=np.full((480,640),1000,np.uint16)
        Image.fromarray(rgb).save(ep/f"images/base/{i:06}.jpg"); Image.fromarray(255-rgb).save(ep/f"images/wrist/{i:06}.jpg"); Image.fromarray(depth).save(ep/f"depth/base/{i:06}.png"); Image.fromarray(depth+1).save(ep/f"depth/wrist/{i:06}.png")
        row={"frame_index":i,"episode_time_s":i*.05,"observation.state":[0]*6+[g],"action":[0]*6+[g],"gripper.state":g,"gripper.command":g,"servo_status":0,"valid":True,"receive_age_s":{k:0 for k in ["joint_state","arm_command","base_image","wrist_image","base_depth","wrist_depth","gripper_state","gripper_command"]},"observation.images.base":f"images/base/{i:06}.jpg","observation.images.wrist":f"images/wrist/{i:06}.jpg","observation.depth.base":f"depth/base/{i:06}.png","observation.depth.wrist":f"depth/wrist/{i:06}.png"}; rows.append(json.dumps(row))
    if missing: (ep / "images/base/000001.jpg").unlink()
    (ep / "frames.jsonl").write_text("\n".join(rows)+"\n"); return ep


def test_episode_boundaries_gripper_and_missing_files(tmp_path):
    valid = validate_raw_episode(_synthetic_episode(tmp_path), decode_all_media=True)
    assert valid.ok and len(valid.frames) == 3
    bad_root = tmp_path / "bad"; bad = validate_raw_episode(_synthetic_episode(bad_root, missing=True), decode_all_media=False)
    assert not bad.ok and any("missing" in error or "mismatch" in error for error in bad.errors)


def test_raw_validation_rejects_large_timestamp_gap(tmp_path):
    episode = _synthetic_episode(tmp_path)
    rows = [json.loads(line) for line in (episode / "frames.jsonl").read_text().splitlines()]
    rows[2]["episode_time_s"] = 0.40
    (episode / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    validation = validate_raw_episode(episode, decode_all_media=False)

    assert not validation.ok
    assert any("frame interval" in error and "0.350000s" in error for error in validation.errors)


def test_tiny_official_lerobot_round_trip(tmp_path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from ur5_vla_dataset_recorder.convert_lerobot import convert
    from ur5_vla_dataset_recorder.validate_lerobot import validate_dataset

    _synthetic_episode(tmp_path)
    output = tmp_path / "converted"
    assert convert(tmp_path, output, "local/ur5-synthetic", storage="image") == (1, 3)
    assert not validate_dataset(output, "local/ur5-synthetic", tmp_path / "report.md", 1, 3, "test")
    dataset = LeRobotDataset("local/ur5-synthetic", root=output)
    assert dataset.meta.total_episodes == 1
    assert dataset.meta.total_frames == 3
    assert int(dataset[2]["episode_index"]) == 0
