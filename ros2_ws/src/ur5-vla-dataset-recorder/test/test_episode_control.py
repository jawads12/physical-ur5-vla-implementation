import pytest

from ur5_vla_dataset_recorder.episode_control import (
    EpisodeFault, EpisodeLimits, EpisodePhase, GuardedEpisodeState,
)


def test_complete_gripper_sequence_requires_debounce_and_time():
    state = GuardedEpisodeState(EpisodeLimits(
        gripper_confirmations=2, earliest_close_s=2, minimum_closed_s=3,
        settle_after_release_s=1,
    ))
    state.arm(0, [0] * 6, True)
    assert state.observe_gripper(1, False) is None
    assert state.observe_gripper(1.1, False) is None
    assert state.observe_gripper(2.1, False) is None
    assert state.observe_gripper(2.2, False) == 0
    assert state.phase == EpisodePhase.CARRYING
    assert state.observe_gripper(3, True) is None
    assert state.observe_gripper(3.1, True) is None
    assert state.observe_gripper(5.3, True) is None
    assert state.observe_gripper(5.4, True) == 1
    assert not state.update_completion(6.3)
    assert state.update_completion(6.5)


def test_episode_faults_on_timeout_and_travel():
    timed = GuardedEpisodeState(EpisodeLimits(max_duration_s=1))
    timed.arm(0, [0] * 6, True)
    with pytest.raises(EpisodeFault, match="runtime"):
        timed.check_time(1.1)

    travel = GuardedEpisodeState(EpisodeLimits(max_cumulative_joint_travel_rad=.1))
    travel.arm(0, [0] * 6, True)
    with pytest.raises(EpisodeFault, match="cumulative"):
        travel.record_pulse(.1, [.2, 0, 0, 0, 0, 0])
