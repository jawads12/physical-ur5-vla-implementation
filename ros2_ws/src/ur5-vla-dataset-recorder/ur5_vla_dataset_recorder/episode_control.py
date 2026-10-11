"""ROS-independent fail-closed state for a guarded pick-and-place episode."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class EpisodePhase(str, Enum):
    DISARMED = "disarmed"
    SEEKING_PICK = "seeking_pick"
    CARRYING = "carrying"
    RELEASED = "released"
    COMPLETE = "complete"
    FAULT = "fault"


@dataclass(frozen=True)
class EpisodeLimits:
    max_duration_s: float = 210.0
    max_pulses: int = 1000
    max_cumulative_joint_travel_rad: float = 12.0
    max_joint_displacement_from_start_rad: float = 2.5
    gripper_confirmations: int = 5
    earliest_close_s: float = 10.0
    minimum_closed_s: float = 5.0
    settle_after_release_s: float = 3.0


class EpisodeFault(RuntimeError):
    pass


class GuardedEpisodeState:
    """Track task progress and reject unbounded or invalid episode evolution."""

    def __init__(self, limits: EpisodeLimits = EpisodeLimits()):
        if limits.max_duration_s <= 0 or limits.max_pulses < 1:
            raise ValueError("episode duration and pulse limit must be positive")
        if limits.gripper_confirmations < 1:
            raise ValueError("gripper_confirmations must be positive")
        self.limits = limits
        self.phase = EpisodePhase.DISARMED
        self.started_at = 0.0
        self.phase_at = 0.0
        self.start_joints: list[float] = []
        self.previous_joints: list[float] = []
        self.cumulative_travel = 0.0
        self.pulses = 0
        self.candidate_gripper: int | None = None
        self.candidate_count = 0
        self.fault_reason: str | None = None

    @staticmethod
    def _joints(values: Sequence[float]) -> list[float]:
        result = [float(value) for value in values]
        if len(result) != 6 or not all(math.isfinite(value) for value in result):
            raise EpisodeFault("joint state must contain six finite values")
        return result

    def arm(self, now: float, joints: Sequence[float], gripper_open: bool) -> None:
        if self.phase not in {EpisodePhase.DISARMED, EpisodePhase.COMPLETE}:
            raise EpisodeFault(f"cannot arm episode from {self.phase.value}")
        if not gripper_open:
            raise EpisodeFault("episode must start with the gripper open")
        current = self._joints(joints)
        self.phase = EpisodePhase.SEEKING_PICK
        self.started_at = self.phase_at = float(now)
        self.start_joints = current
        self.previous_joints = current
        self.cumulative_travel = 0.0
        self.pulses = 0
        self.candidate_gripper = None
        self.candidate_count = 0
        self.fault_reason = None

    def fault(self, reason: str) -> None:
        self.phase = EpisodePhase.FAULT
        self.fault_reason = reason

    def check_time(self, now: float) -> None:
        if self.phase in {EpisodePhase.DISARMED, EpisodePhase.COMPLETE, EpisodePhase.FAULT}:
            return
        if now - self.started_at > self.limits.max_duration_s:
            self.fault("episode runtime limit exceeded")
            raise EpisodeFault(self.fault_reason)

    def record_pulse(self, now: float, joints: Sequence[float]) -> None:
        self.check_time(now)
        current = self._joints(joints)
        self.pulses += 1
        self.cumulative_travel += max(
            abs(value - previous) for value, previous in zip(current, self.previous_joints)
        )
        displacement = max(
            abs(value - start) for value, start in zip(current, self.start_joints)
        )
        self.previous_joints = current
        if self.pulses > self.limits.max_pulses:
            self.fault("episode pulse limit exceeded")
        elif self.cumulative_travel > self.limits.max_cumulative_joint_travel_rad:
            self.fault("cumulative joint travel limit exceeded")
        elif displacement > self.limits.max_joint_displacement_from_start_rad:
            self.fault("joint displacement from start exceeded")
        if self.phase == EpisodePhase.FAULT:
            raise EpisodeFault(self.fault_reason)

    def observe_gripper(self, now: float, predicted_open: bool) -> int | None:
        """Return a confirmed new binary command, or None while debouncing."""
        self.check_time(now)
        value = int(bool(predicted_open))
        if self.candidate_gripper == value:
            self.candidate_count += 1
        else:
            self.candidate_gripper = value
            self.candidate_count = 1
        if self.candidate_count < self.limits.gripper_confirmations:
            return None
        self.candidate_count = 0
        if self.phase == EpisodePhase.SEEKING_PICK and value == 0:
            if now - self.started_at < self.limits.earliest_close_s:
                return None
            self.phase = EpisodePhase.CARRYING
            self.phase_at = now
            return 0
        if self.phase == EpisodePhase.CARRYING and value == 1:
            if now - self.phase_at < self.limits.minimum_closed_s:
                return None
            self.phase = EpisodePhase.RELEASED
            self.phase_at = now
            return 1
        return None

    def update_completion(self, now: float) -> bool:
        self.check_time(now)
        if (
            self.phase == EpisodePhase.RELEASED
            and now - self.phase_at >= self.limits.settle_after_release_s
        ):
            self.phase = EpisodePhase.COMPLETE
            return True
        return self.phase == EpisodePhase.COMPLETE
