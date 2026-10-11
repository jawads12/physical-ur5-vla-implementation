import pytest

from ur5_vla_dataset_recorder.episode_execution import (
    PulseConfig, PulsePlanningError, gripper_pins, plan_joint_pulse,
)


def test_pulse_is_deadbanded_and_speed_bounded():
    result = plan_joint_pulse([.0001, .001, -.01, 0, 0, 0], [0] * 6, PulseConfig())
    assert result == [0.0, .01, -.02, 0.0, 0.0, 0.0]


def test_pulse_rejects_invalid_input():
    with pytest.raises(PulsePlanningError):
        plan_joint_pulse([0] * 5, [0] * 6, PulseConfig())


def test_gripper_is_break_before_make():
    assert gripper_pins(True) == ((0, 0.0), (1, 1.0))
    assert gripper_pins(False) == ((1, 0.0), (0, 1.0))
