"""ROS-independent helpers for bounded receding-horizon episode execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


class PulsePlanningError(ValueError):
    pass


@dataclass(frozen=True)
class PulseConfig:
    duration_s: float = 0.10
    max_joint_speed_rad_s: float = 0.02
    deadband_rad: float = 0.0002

    def __post_init__(self) -> None:
        if not 0.0 < self.duration_s <= 0.10:
            raise ValueError("pulse duration must be in (0, 0.10]")
        if not 0.0 < self.max_joint_speed_rad_s <= 0.02:
            raise ValueError("joint speed must be in (0, 0.02]")
        if self.deadband_rad < 0.0:
            raise ValueError("deadband cannot be negative")


def plan_joint_pulse(
    target: Sequence[float], current: Sequence[float], config: PulseConfig
) -> list[float]:
    """Return bounded JointJog velocities; never rescales or clips a target."""
    goal = [float(value) for value in target]
    actual = [float(value) for value in current]
    if len(goal) != 6 or len(actual) != 6 or not all(map(math.isfinite, goal + actual)):
        raise PulsePlanningError("target and current must each contain six finite joints")
    velocities = []
    for wanted, present in zip(goal, actual):
        error = wanted - present
        if abs(error) <= config.deadband_rad:
            velocities.append(0.0)
        else:
            velocities.append(max(-config.max_joint_speed_rad_s, min(config.max_joint_speed_rad_s, error / config.duration_s)))
    return velocities


def gripper_pins(open_gripper: bool) -> tuple[tuple[int, float], tuple[int, float]]:
    """Return break-before-make pneumatic output writes (close=0, open=1)."""
    return ((0, 0.0), (1, 1.0)) if open_gripper else ((1, 0.0), (0, 1.0))
