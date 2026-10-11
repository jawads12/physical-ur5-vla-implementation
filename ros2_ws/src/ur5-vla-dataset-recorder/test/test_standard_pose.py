import math
from unittest.mock import Mock

import pytest

from ur5_vla_dataset_recorder.standard_pose_node import (
    MIN_TRAJECTORY_SECONDS,
    STANDARD_POSE_RAD,
    ensure_reset_speed_limit,
    interpolated_joint_path,
    retry_state_validity_query,
    trajectory_duration,
)


def test_standard_pose_is_six_finite_joints():
    assert len(STANDARD_POSE_RAD) == 6
    assert all(math.isfinite(value) for value in STANDARD_POSE_RAD)


def test_interpolated_path_includes_endpoints_and_respects_step():
    start = [0.0] * 6
    target = [0.11, -0.02, 0.0, 0.0, 0.0, 0.0]
    path = interpolated_joint_path(start, target, max_step=0.04)
    assert path[0] == start
    assert path[-1] == pytest.approx(target)
    for first, second in zip(path, path[1:]):
        assert max(abs(a - b) for a, b in zip(first, second)) <= 0.04


def test_trajectory_duration_is_slow_and_has_minimum():
    assert trajectory_duration(STANDARD_POSE_RAD, STANDARD_POSE_RAD) == MIN_TRAJECTORY_SECONDS
    start = [0.0] * 6
    target = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert trajectory_duration(start, target) == pytest.approx(10.0)


def test_state_validity_retries_a_lost_response_after_readiness_check():
    response = object()
    query = Mock(side_effect=[TimeoutError("lost response"), response])
    before_retry = Mock()
    assert retry_state_validity_query(query, before_retry) is response
    assert query.call_count == 2
    before_retry.assert_called_once_with(2)


def test_state_validity_stops_after_three_timeouts():
    query = Mock(side_effect=TimeoutError("no response"))
    before_retry = Mock()
    with pytest.raises(TimeoutError, match="no response"):
        retry_state_validity_query(query, before_retry)
    assert query.call_count == 3
    assert [call.args for call in before_retry.call_args_list] == [(2,), (3,)]


def test_state_validity_does_not_retry_collision_rejection():
    query = Mock(return_value=False)
    before_retry = Mock()
    assert retry_state_validity_query(query, before_retry) is False
    query.assert_called_once_with()
    before_retry.assert_not_called()


def test_state_validity_does_not_retry_non_timeout_failure():
    query = Mock(side_effect=RuntimeError("service unavailable"))
    before_retry = Mock()
    with pytest.raises(RuntimeError, match="service unavailable"):
        retry_state_validity_query(query, before_retry)
    query.assert_called_once_with()
    before_retry.assert_not_called()


def test_state_validity_retry_is_blocked_when_robot_readiness_changes():
    query = Mock(side_effect=TimeoutError("lost response"))
    before_retry = Mock(side_effect=RuntimeError("robot state stale"))
    with pytest.raises(RuntimeError, match="robot state stale"):
        retry_state_validity_query(query, before_retry)
    query.assert_called_once_with()


def test_reset_lowers_high_speed_and_requires_feedback():
    read_speed = Mock(side_effect=[100.0, 20.0])
    request = Mock(return_value=True)
    feedback = Mock()
    assert ensure_reset_speed_limit(read_speed, request, feedback) is True
    request.assert_called_once_with(0.20)
    feedback.assert_called_once_with()
    assert read_speed.call_count == 2


@pytest.mark.parametrize('speed', [1.0, 10.0, 20.0])
def test_reset_never_raises_an_already_low_speed(speed):
    request, feedback = Mock(), Mock()
    assert ensure_reset_speed_limit(lambda: speed, request, feedback) is False
    request.assert_not_called()
    feedback.assert_not_called()


@pytest.mark.parametrize('speed', [0.0, -1.0, 101.0, math.nan, math.inf])
def test_reset_does_not_override_a_stop_or_invalid_speed(speed):
    request, feedback = Mock(), Mock()
    with pytest.raises(RuntimeError):
        ensure_reset_speed_limit(lambda: speed, request, feedback)
    request.assert_not_called()


def test_reset_blocks_when_reduction_is_rejected():
    feedback = Mock()
    with pytest.raises(RuntimeError, match='rejected'):
        ensure_reset_speed_limit(lambda: 100.0, Mock(return_value=False), feedback)
    feedback.assert_not_called()


@pytest.mark.parametrize('confirmed_speed', [100.0, 20.1, 0.0, math.nan])
def test_reset_blocks_if_feedback_does_not_confirm_safe_speed(confirmed_speed):
    with pytest.raises(RuntimeError, match='not confirmed'):
        ensure_reset_speed_limit(Mock(side_effect=[100.0, confirmed_speed]), Mock(return_value=True), Mock())


def test_reset_blocks_on_feedback_timeout_or_changed_robot_readiness():
    with pytest.raises(TimeoutError):
        ensure_reset_speed_limit(lambda: 100.0, Mock(return_value=True), Mock(side_effect=TimeoutError))
    request = Mock()
    with pytest.raises(RuntimeError, match='unsafe'):
        ensure_reset_speed_limit(Mock(side_effect=RuntimeError('unsafe robot')), request, Mock())
    request.assert_not_called()
