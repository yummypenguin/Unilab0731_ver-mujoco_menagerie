"""Pure NumPy deployment contract for LEAP-hand HORA policies.

This module is intentionally independent of an environment or simulator backend.  It
defines the measurable policy inputs, oldest-first history behavior, incremental action
integration, and generic joint-order permutation helpers shared by later deployment phases.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

NUM_JOINTS = 16

ACTOR_FRAME_DIM = 35
ACTOR_HISTORY_LEN = 3
ACTOR_OBS_DIM = 105

PROPRIO_FRAME_DIM = 32
PROPRIO_HISTORY_LEN = 30

PRIV_INFO_DIM = 9

CONTROL_DT = 0.05
ACTION_SCALE = 1.0 / 24.0

_NORMALIZATION_EPSILON = 1.0e-8
_AXIS_NORM_EPSILON = 1.0e-8


def _as_real_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be a real numeric array")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must be a real numeric array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _floating_dtype(*arrays: np.ndarray) -> np.dtype:
    return np.dtype(np.result_type(*(array.dtype for array in arrays), np.float32))


def _validate_joint_values_and_bounds(
    values: object,
    lower: object,
    upper: object,
    *,
    values_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values_array = _as_real_array(values, name=values_name)
    lower_array = _as_real_array(lower, name="lower")
    upper_array = _as_real_array(upper, name="upper")

    if values_array.ndim < 1 or values_array.shape[-1] != NUM_JOINTS:
        raise ValueError(f"{values_name} must have shape (..., {NUM_JOINTS})")
    expected_bounds_shape = (NUM_JOINTS,)
    if lower_array.shape != expected_bounds_shape:
        raise ValueError(f"lower must have shape {expected_bounds_shape}")
    if upper_array.shape != expected_bounds_shape:
        raise ValueError(f"upper must have shape {expected_bounds_shape}")
    if np.any(upper_array <= lower_array):
        raise ValueError("upper must be greater than lower for every joint")

    dtype = _floating_dtype(values_array, lower_array, upper_array)
    return (
        np.asarray(values_array, dtype=dtype),
        np.asarray(lower_array, dtype=dtype),
        np.asarray(upper_array, dtype=dtype),
    )


def normalize_to_joint_range(
    values: object,
    lower: object,
    upper: object,
    *,
    clip: bool = False,
) -> np.ndarray:
    """Normalize joint values with training actuator bounds.

    Inputs are not clipped unless ``clip=True`` is explicitly requested.  Values may be
    unbatched ``(16,)`` or batched ``(..., 16)``; bounds are always exactly ``(16,)``.
    """

    values_array, lower_array, upper_array = _validate_joint_values_and_bounds(
        values,
        lower,
        upper,
        values_name="values",
    )
    epsilon = np.asarray(_NORMALIZATION_EPSILON, dtype=values_array.dtype)
    normalized = 2.0 * (values_array - lower_array) / (
        upper_array - lower_array + epsilon
    ) - 1.0
    if clip:
        normalized = np.clip(normalized, -1.0, 1.0)
    return np.asarray(normalized, dtype=values_array.dtype)


def denormalize_from_joint_range(
    values_norm: object,
    lower: object,
    upper: object,
    *,
    clip: bool = False,
) -> np.ndarray:
    """Convert normalized joint values back to radians using training bounds."""

    normalized, lower_array, upper_array = _validate_joint_values_and_bounds(
        values_norm,
        lower,
        upper,
        values_name="values_norm",
    )
    if clip:
        normalized = np.clip(normalized, -1.0, 1.0)
    values = 0.5 * (normalized + 1.0) * (upper_array - lower_array) + lower_array
    return np.asarray(values, dtype=normalized.dtype)


def validate_axis(rotation_axis: object) -> np.ndarray:
    """Validate and unit-normalize one or more rotation axes with shape ``(..., 3)``."""

    axis = _as_real_array(rotation_axis, name="rotation_axis")
    if axis.ndim < 1 or axis.shape[-1] != 3:
        raise ValueError("rotation_axis must have shape (..., 3)")
    dtype = _floating_dtype(axis)
    axis = np.asarray(axis, dtype=dtype)
    norms = np.linalg.norm(axis, axis=-1, keepdims=True)
    if np.any(norms <= _AXIS_NORM_EPSILON):
        raise ValueError("rotation_axis norm must be greater than epsilon")
    return np.asarray(axis / norms, dtype=dtype)


def _validate_frame_joint_inputs(
    measured_q: object,
    previous_target: object,
    joint_lower: object,
    joint_upper: object,
) -> tuple[np.ndarray, np.ndarray]:
    q_norm = normalize_to_joint_range(measured_q, joint_lower, joint_upper)
    target_norm = normalize_to_joint_range(previous_target, joint_lower, joint_upper)
    if q_norm.shape != target_norm.shape:
        raise ValueError("measured_q and previous_target must have the same shape")
    return q_norm, target_norm


def build_actor_frame(
    measured_q: object,
    previous_target: object,
    rotation_axis: object,
    joint_lower: object,
    joint_upper: object,
) -> np.ndarray:
    """Build ``[q_t, target_(t-1), axis_t]`` in the deployable actor layout."""

    q_norm, target_norm = _validate_frame_joint_inputs(
        measured_q,
        previous_target,
        joint_lower,
        joint_upper,
    )
    axis = validate_axis(rotation_axis)
    axis_shape = q_norm.shape[:-1] + (3,)
    try:
        axis = np.broadcast_to(axis, axis_shape)
    except ValueError as exc:
        raise ValueError(
            f"rotation_axis shape {axis.shape} cannot broadcast to {axis_shape}"
        ) from exc

    dtype = _floating_dtype(q_norm, target_norm, axis)
    frame = np.concatenate(
        [
            np.asarray(q_norm, dtype=dtype),
            np.asarray(target_norm, dtype=dtype),
            np.asarray(axis, dtype=dtype),
        ],
        axis=-1,
    )
    if frame.shape[-1] != ACTOR_FRAME_DIM:  # pragma: no cover - constant invariant
        raise RuntimeError(f"actor frame width must be {ACTOR_FRAME_DIM}")
    return np.asarray(frame, dtype=dtype)


def build_proprio_frame(
    measured_q: object,
    previous_target: object,
    joint_lower: object,
    joint_upper: object,
) -> np.ndarray:
    """Build ``[q_t, target_(t-1)]`` in the deployable proprio layout."""

    q_norm, target_norm = _validate_frame_joint_inputs(
        measured_q,
        previous_target,
        joint_lower,
        joint_upper,
    )
    dtype = _floating_dtype(q_norm, target_norm)
    frame = np.concatenate(
        [np.asarray(q_norm, dtype=dtype), np.asarray(target_norm, dtype=dtype)],
        axis=-1,
    )
    if frame.shape[-1] != PROPRIO_FRAME_DIM:  # pragma: no cover - constant invariant
        raise RuntimeError(f"proprio frame width must be {PROPRIO_FRAME_DIM}")
    return np.asarray(frame, dtype=dtype)


class OldestFirstHistoryBuffer:
    """Single-environment finite history whose first element is always the oldest."""

    def __init__(self, history_length: int, frame_shape: int | Sequence[int]) -> None:
        if isinstance(history_length, bool) or not isinstance(history_length, int):
            raise TypeError("history_length must be an integer")
        if history_length <= 0:
            raise ValueError("history_length must be positive")

        if isinstance(frame_shape, int):
            resolved_shape = (frame_shape,)
        else:
            resolved_shape = tuple(frame_shape)
        if not resolved_shape or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in resolved_shape
        ):
            raise ValueError("frame_shape must contain positive integer dimensions")

        self.history_length = history_length
        self.frame_shape = resolved_shape
        self._buffer: np.ndarray | None = None

    def _validated_frame(self, current_frame: object) -> np.ndarray:
        frame = _as_real_array(current_frame, name="current_frame")
        if frame.shape != self.frame_shape:
            raise ValueError(f"current_frame must have shape {self.frame_shape}")
        return frame

    def reset(self, current_frame: object) -> None:
        """Fill every history slot with the current observable frame."""

        frame = self._validated_frame(current_frame)
        self._buffer = np.repeat(frame[np.newaxis, ...], self.history_length, axis=0)

    def push(self, current_frame: object) -> None:
        """Drop the oldest frame and append the current frame as newest."""

        if self._buffer is None:
            raise RuntimeError("history must be reset before push")
        frame = self._validated_frame(current_frame)
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = frame

    def as_array_oldest_first(self) -> np.ndarray:
        """Return a copy with shape ``(history_length, *frame_shape)``."""

        if self._buffer is None:
            raise RuntimeError("history must be reset before it can be read")
        return self._buffer.copy()

    def flatten_oldest_first(self) -> np.ndarray:
        """Return an independent oldest-first flattened copy."""

        return self.as_array_oldest_first().reshape(-1).copy()


def integrate_incremental_action(
    previous_target: object,
    action: object,
    target_lower: object,
    target_upper: object,
    *,
    action_scale: float = ACTION_SCALE,
    rollout_gain: float = 1.0,
) -> np.ndarray:
    """Integrate a clipped relative action and clip only to actuator target bounds."""

    previous, lower, upper = _validate_joint_values_and_bounds(
        previous_target,
        target_lower,
        target_upper,
        values_name="previous_target",
    )
    action_array = _as_real_array(action, name="action")
    if action_array.shape != previous.shape:
        raise ValueError("action and previous_target must have the same shape")
    if not np.isfinite(action_scale) or action_scale <= 0.0:
        raise ValueError("action_scale must be positive and finite")
    if not np.isfinite(rollout_gain) or not 0.0 < rollout_gain <= 1.0:
        raise ValueError("rollout_gain must be finite and in (0, 1]")

    dtype = _floating_dtype(previous, action_array, lower, upper)
    previous = np.asarray(previous, dtype=dtype)
    clipped_action = np.clip(np.asarray(action_array, dtype=dtype), -1.0, 1.0)
    target = previous + rollout_gain * action_scale * clipped_action
    return np.asarray(np.clip(target, lower, upper), dtype=dtype)


def validate_permutation(mapping: object, *, size: int = NUM_JOINTS) -> np.ndarray:
    """Validate ``mapping[source_index] == destination_index`` and return a copy."""

    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size <= 0:
        raise ValueError("size must be positive")
    permutation = np.asarray(mapping)
    if permutation.shape != (size,):
        raise ValueError(f"mapping must have shape ({size},)")
    if np.issubdtype(permutation.dtype, np.bool_) or not np.issubdtype(
        permutation.dtype, np.integer
    ):
        raise TypeError("mapping must contain integer indices")
    permutation = np.asarray(permutation, dtype=np.intp)
    if not np.array_equal(np.sort(permutation), np.arange(size, dtype=np.intp)):
        raise ValueError(f"mapping must contain every index from 0 to {size - 1} exactly once")
    return permutation.copy()


def invert_permutation(mapping: object, *, size: int = NUM_JOINTS) -> np.ndarray:
    """Return the inverse of a validated source-to-destination permutation."""

    permutation = validate_permutation(mapping, size=size)
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(size, dtype=np.intp)
    return inverse


def reorder_source_to_destination(
    values: object,
    mapping: object,
) -> np.ndarray:
    """Scatter the last source axis into destination order using ``mapping``.

    For each source index ``i``, the result at ``mapping[i]`` receives ``values[i]``.
    This is generic permutation machinery, not a verified real-hardware joint mapping.
    """

    values_array = np.asarray(values)
    if values_array.ndim < 1:
        raise ValueError("values must have at least one dimension")
    permutation = validate_permutation(mapping, size=values_array.shape[-1])
    result = np.empty_like(values_array)
    result[..., permutation] = values_array
    return result


__all__ = [
    "ACTION_SCALE",
    "ACTOR_FRAME_DIM",
    "ACTOR_HISTORY_LEN",
    "ACTOR_OBS_DIM",
    "CONTROL_DT",
    "NUM_JOINTS",
    "PRIV_INFO_DIM",
    "PROPRIO_FRAME_DIM",
    "PROPRIO_HISTORY_LEN",
    "OldestFirstHistoryBuffer",
    "build_actor_frame",
    "build_proprio_frame",
    "denormalize_from_joint_range",
    "integrate_incremental_action",
    "invert_permutation",
    "normalize_to_joint_range",
    "reorder_source_to_destination",
    "validate_axis",
    "validate_permutation",
]
