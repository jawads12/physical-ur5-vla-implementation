"""ROS-independent safety supervisor for bounded policy requests."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .core import UR5_JOINT_ORDER


@dataclass(frozen=True)
class SafetyConfig:
    joint_min: tuple[float, ...] = (-6.283,) * 6
    joint_max: tuple[float, ...] = (6.283,) * 6
    max_target_delta_rad: float = 0.012566
    max_velocity_rad_s: float = 0.10
    max_sensor_age_s: float = 0.10
    max_gripper_age_s: float = 0.20
    max_inference_age_s: float = 0.25
    expected_controller: str = "forward_position_controller"


@dataclass
class SafetySnapshot:
    joint_positions: Sequence[float]
    ages_s: dict[str, float]
    servo_status: int
    controller: str
    external_control_ready: bool
    deadman_held: bool
    workspace_ok: bool = True


class SafetyFault(RuntimeError):
    pass


@dataclass(frozen=True)
class StabilityConfig:
    max_target_delta_rad: float = 0.012566
    max_prediction_jitter_rad: float = 0.005
    required_consecutive: int = 100


class PredictionStabilityGate:
    """Require several bounded, mutually consistent predictions before eligibility."""

    def __init__(self, config: StabilityConfig = StabilityConfig()):
        if config.required_consecutive < 1:
            raise ValueError("required_consecutive must be positive")
        self.config = config
        self.previous: list[float] | None = None
        self.consecutive = 0

    def reset(self) -> None:
        self.previous = None
        self.consecutive = 0

    def observe(self, action: Sequence[float], joints: Sequence[float]) -> tuple[bool, str]:
        values = [float(value) for value in action]
        current = [float(value) for value in joints]
        if len(values) != 7 or len(current) != 6 or not all(map(math.isfinite, values + current)):
            self.reset()
            return False, "invalid prediction or joint state"
        if min(abs(values[6]), abs(values[6] - 1.0)) > 1e-6:
            self.reset()
            return False, "gripper prediction is not binary"
        target_delta = max(abs(values[index] - current[index]) for index in range(6))
        if target_delta > self.config.max_target_delta_rad:
            self.reset()
            return False, f"target delta {target_delta:.5f} exceeds limit"
        if self.previous is not None:
            jitter = max(abs(values[index] - self.previous[index]) for index in range(6))
            if jitter > self.config.max_prediction_jitter_rad or values[6] != self.previous[6]:
                self.reset()
                self.previous = values
                self.consecutive = 1
                return False, f"prediction jitter {jitter:.5f} exceeds limit"
        else:
            self.consecutive = 1
        self.previous = values
        if self.consecutive < self.config.required_consecutive:
            self.consecutive += 1
            return False, f"stable predictions {self.consecutive - 1}/{self.config.required_consecutive}"
        return True, f"stable predictions {self.consecutive}/{self.config.required_consecutive}"


class SafetySupervisor:
    def __init__(self, config: SafetyConfig = SafetyConfig()):
        self.config = config
        self.enabled = False
        self.latched_fault: str | None = None

    def disable(self, reason: str | None = None) -> None:
        self.enabled = False
        if reason:
            self.latched_fault = reason

    def manual_enable(self) -> None:
        if self.latched_fault:
            raise SafetyFault("fault is latched; clear it explicitly after inspection")
        self.enabled = True

    def clear_fault(self) -> None:
        self.enabled = False
        self.latched_fault = None

    def check(self, action: Sequence[float], snapshot: SafetySnapshot, period_s: float) -> list[float]:
        values = [float(x) for x in action]
        failures = []
        if len(values) != 7 or not all(math.isfinite(x) for x in values):
            failures.append("action must contain seven finite values")
        if len(snapshot.joint_positions) != 6 or not all(math.isfinite(float(x)) for x in snapshot.joint_positions):
            failures.append("joint state is missing or invalid")
        if snapshot.servo_status != 0: failures.append("Servo status is nonzero")
        if snapshot.controller != self.config.expected_controller: failures.append("controller mismatch")
        if not snapshot.external_control_ready: failures.append("External Control is not ready")
        if not snapshot.deadman_held: failures.append("deadman is not held")
        if not snapshot.workspace_ok: failures.append("workspace constraint violated")
        limits = {"base_image": self.config.max_sensor_age_s, "wrist_image": self.config.max_sensor_age_s,
                  "joint_state": self.config.max_sensor_age_s, "gripper_state": self.config.max_gripper_age_s,
                  "inference": self.config.max_inference_age_s}
        for key, limit in limits.items():
            if snapshot.ages_s.get(key, math.inf) > limit: failures.append(f"stale {key}")
        if len(values) == 7 and len(snapshot.joint_positions) == 6:
            for i, (target, current, lower, upper) in enumerate(zip(values[:6], snapshot.joint_positions, self.config.joint_min, self.config.joint_max)):
                delta = abs(target - float(current))
                if not lower <= target <= upper: failures.append(f"{UR5_JOINT_ORDER[i]} target outside limits")
                if delta > self.config.max_target_delta_rad: failures.append(f"{UR5_JOINT_ORDER[i]} target delta too large")
                if period_s <= 0 or delta / period_s > self.config.max_velocity_rad_s: failures.append(f"{UR5_JOINT_ORDER[i]} velocity too large")
            if min(abs(values[6]), abs(values[6] - 1.0)) > 1e-6: failures.append("gripper action is not binary")
        if failures:
            reason = "; ".join(dict.fromkeys(failures))
            self.disable(reason)
            raise SafetyFault(reason)
        if not self.enabled:
            raise SafetyFault("supervisor is disabled")
        return values
