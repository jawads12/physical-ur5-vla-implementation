"""ROS-independent, fail-closed state machine for one-shot physical commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .safety import SafetySnapshot, SafetySupervisor


@dataclass(frozen=True)
class LiveReadiness:
    joints: Sequence[float]
    ages_s: dict[str, float]
    servo_status: int
    controller: str
    program_running: bool
    deadman_held: bool
    collision_free: bool


class OneShotArmGate:
    """Allow exactly one checked command after each deliberate arm request."""

    def __init__(self, supervisor: SafetySupervisor):
        self.supervisor = supervisor
        self.armed = False

    def arm(self) -> None:
        if self.armed:
            raise RuntimeError("one-shot gate is already armed")
        self.supervisor.manual_enable()
        self.armed = True

    def cancel(self, reason: str | None = None) -> None:
        self.armed = False
        self.supervisor.disable(reason)

    def approve(self, action: Sequence[float], readiness: LiveReadiness, period_s: float) -> list[float]:
        if not self.armed:
            raise RuntimeError("one-shot gate is not armed")
        snapshot = SafetySnapshot(
            joint_positions=readiness.joints,
            ages_s=readiness.ages_s,
            servo_status=readiness.servo_status,
            controller=readiness.controller,
            external_control_ready=readiness.program_running,
            deadman_held=readiness.deadman_held,
            workspace_ok=readiness.collision_free,
        )
        try:
            return self.supervisor.check(action, snapshot, period_s)
        finally:
            # Consumption is unconditional: success or failure requires a new,
            # deliberate human arm request.
            self.armed = False
            self.supervisor.enabled = False
