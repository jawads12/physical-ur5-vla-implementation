"""Explicit, fail-closed transforms between UR5 action representations."""

from __future__ import annotations

import numpy as np


ABSOLUTE_ACTION_SEMANTICS = (
    "six absolute joint-position targets plus absolute binary gripper command"
)
DELTA_ACTION_SEMANTICS = (
    "six joint-position deltas relative to the same-frame measured state plus "
    "absolute binary gripper command"
)


def _action7(values, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain seven finite values")
    return result


def absolute_to_delta(state, absolute_action) -> np.ndarray:
    """Encode a demonstrated absolute target as a same-frame joint delta."""
    measured = _action7(state, "state")
    target = _action7(absolute_action, "absolute action")
    result = target.copy()
    result[:6] -= measured[:6]
    return result


def delta_to_absolute(state, delta_action, max_delta_rad: float) -> np.ndarray:
    """Decode a policy delta, rejecting (never clipping) out-of-envelope output."""
    if not np.isfinite(max_delta_rad) or max_delta_rad <= 0:
        raise ValueError("max_delta_rad must be finite and positive")
    measured = _action7(state, "state")
    delta = _action7(delta_action, "delta action")
    largest = float(np.max(np.abs(delta[:6])))
    if largest > max_delta_rad:
        raise ValueError(
            f"raw predicted joint delta {largest:.6f} exceeds {max_delta_rad:.6f} rad"
        )
    target = delta.copy()
    target[:6] += measured[:6]
    return target

