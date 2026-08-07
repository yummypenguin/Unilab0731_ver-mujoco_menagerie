"""Pure teacher-skill diagnostic helpers for the LEAP HORA task."""

from __future__ import annotations

import numpy as np

from .deploy_contract import validate_axis

ROTATION_DIAGNOSTIC_KEYS = (
    "rotation/axis_speed_mean",
    "rotation/axis_speed_abs_mean",
    "rotation/positive_rate",
    "rotation/reverse_rate",
    "rotation/high_clip_rate",
    "rotation/low_clip_rate",
)

CONTROL_DIAGNOSTIC_KEYS = (
    "control/action_abs_mean",
    "control/action_saturation_rate",
    "control/target_saturation_rate",
    "control/target_lower_saturation_rate",
    "control/target_upper_saturation_rate",
)


def _finite_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must be a real numeric array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def compute_rotation_diagnostics(
    ball_angvel: object,
    reward_axis: object,
    *,
    clip_min: float,
    clip_max: float,
) -> dict[str, float]:
    """Summarize raw reward-axis angular velocity without clipping its mean."""

    angular_velocity = _finite_array(ball_angvel, name="ball_angvel")
    if angular_velocity.ndim != 2 or angular_velocity.shape[1] != 3:
        raise ValueError("ball_angvel must have shape (N, 3)")
    axis = validate_axis(reward_axis)
    if axis.shape != (3,):
        raise ValueError("reward_axis must have shape (3,)")
    if not np.isfinite(clip_min) or not np.isfinite(clip_max) or clip_min > clip_max:
        raise ValueError("rotation clip bounds must be finite and ordered")

    axis_speed = angular_velocity @ axis
    return {
        "rotation/axis_speed_mean": float(np.mean(axis_speed)),
        "rotation/axis_speed_abs_mean": float(np.mean(np.abs(axis_speed))),
        "rotation/positive_rate": float(np.mean(axis_speed > 0.0)),
        "rotation/reverse_rate": float(np.mean(axis_speed < 0.0)),
        "rotation/high_clip_rate": float(np.mean(axis_speed >= clip_max)),
        "rotation/low_clip_rate": float(np.mean(axis_speed <= clip_min)),
    }


def compute_control_diagnostics(
    applied_raw_action: object,
    applied_target: object,
    ctrl_lower: object,
    ctrl_upper: object,
    *,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """Summarize the delayed action and target actually applied this control step."""

    action = _finite_array(applied_raw_action, name="applied_raw_action")
    target = _finite_array(applied_target, name="applied_target")
    lower = _finite_array(ctrl_lower, name="ctrl_lower")
    upper = _finite_array(ctrl_upper, name="ctrl_upper")
    if action.ndim != 2 or target.shape != action.shape:
        raise ValueError("applied action and target must have matching shape (N, A)")
    if lower.shape != (action.shape[1],) or upper.shape != lower.shape:
        raise ValueError("control bounds must have shape (A,)")
    if np.any(upper <= lower):
        raise ValueError("ctrl_upper must be greater than ctrl_lower")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    lower_saturation = target <= lower + tolerance
    upper_saturation = target >= upper - tolerance
    return {
        "control/action_abs_mean": float(np.mean(np.abs(action))),
        "control/action_saturation_rate": float(np.mean(np.abs(action) >= 1.0)),
        "control/target_saturation_rate": float(
            np.mean(lower_saturation | upper_saturation)
        ),
        "control/target_lower_saturation_rate": float(np.mean(lower_saturation)),
        "control/target_upper_saturation_rate": float(np.mean(upper_saturation)),
    }


def validate_diagnostic_metrics(metrics: dict[str, float]) -> None:
    """Fail closed when a diagnostic payload contains non-finite or invalid rates."""

    for key, value in metrics.items():
        if not np.isfinite(value):
            raise ValueError(f"diagnostic metric {key!r} must be finite")
        if key.endswith("_rate") and not 0.0 <= value <= 1.0:
            raise ValueError(f"diagnostic rate {key!r} must be in [0, 1]")


__all__ = [
    "CONTROL_DIAGNOSTIC_KEYS",
    "ROTATION_DIAGNOSTIC_KEYS",
    "compute_control_diagnostics",
    "compute_rotation_diagnostics",
    "validate_diagnostic_metrics",
]
