from __future__ import annotations

import numpy as np
import pytest

from unilab.envs.manipulation.leap_inhand.deploy_contract import (
    ACTION_SCALE,
    NUM_JOINTS,
    integrate_incremental_action,
)


@pytest.fixture
def action_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    previous = np.zeros(NUM_JOINTS, dtype=np.float64)
    lower = np.full(NUM_JOINTS, -0.1)
    upper = np.full(NUM_JOINTS, 0.1)
    return previous, lower, upper


def test_zero_action_keeps_previous_target(action_inputs) -> None:
    previous, lower, upper = action_inputs
    target = integrate_incremental_action(previous, np.zeros(NUM_JOINTS), lower, upper)
    np.testing.assert_array_equal(target, previous)


def test_action_is_clipped_before_integration(action_inputs) -> None:
    previous, lower, upper = action_inputs
    action = np.array([2.0, -3.0] + [0.0] * 14)

    target = integrate_incremental_action(previous, action, lower, upper)

    assert target[0] == pytest.approx(ACTION_SCALE)
    assert target[1] == pytest.approx(-ACTION_SCALE)


def test_target_is_clipped_to_upper_and_lower_bounds(action_inputs) -> None:
    _, lower, upper = action_inputs
    previous = np.array([0.09, -0.09] + [0.0] * 14)
    action = np.array([1.0, -1.0] + [0.0] * 14)

    target = integrate_incremental_action(previous, action, lower, upper)

    assert target[0] == pytest.approx(upper[0])
    assert target[1] == pytest.approx(lower[1])


@pytest.mark.parametrize("gain", [0.05, 1.0])
def test_rollout_gain_scales_increment_only(action_inputs, gain) -> None:
    previous, lower, upper = action_inputs
    action = np.ones(NUM_JOINTS)

    target = integrate_incremental_action(
        previous,
        action,
        lower,
        upper,
        rollout_gain=gain,
    )

    np.testing.assert_allclose(target, previous + gain * ACTION_SCALE)


@pytest.mark.parametrize("gain", [0.0, -0.1, 1.01, np.nan, np.inf])
def test_invalid_rollout_gain_is_rejected(action_inputs, gain) -> None:
    previous, lower, upper = action_inputs
    with pytest.raises(ValueError, match="rollout_gain"):
        integrate_incremental_action(
            previous,
            np.zeros(NUM_JOINTS),
            lower,
            upper,
            rollout_gain=gain,
        )


@pytest.mark.parametrize("scale", [0.0, -0.1, np.nan, np.inf])
def test_invalid_action_scale_is_rejected(action_inputs, scale) -> None:
    previous, lower, upper = action_inputs
    with pytest.raises(ValueError, match="action_scale"):
        integrate_incremental_action(
            previous,
            np.zeros(NUM_JOINTS),
            lower,
            upper,
            action_scale=scale,
        )


def test_action_integration_rejects_nonfinite_and_shape_mismatch(action_inputs) -> None:
    previous, lower, upper = action_inputs
    with pytest.raises(ValueError, match="finite"):
        integrate_incremental_action(previous, np.full(NUM_JOINTS, np.nan), lower, upper)
    with pytest.raises(ValueError, match="same shape"):
        integrate_incremental_action(previous, np.zeros(NUM_JOINTS - 1), lower, upper)
    with pytest.raises(ValueError, match="greater"):
        integrate_incremental_action(previous, np.zeros(NUM_JOINTS), upper, lower)


def test_action_integration_supports_batch_and_preserves_float32() -> None:
    previous = np.zeros((2, NUM_JOINTS), dtype=np.float32)
    action = np.stack(
        [np.ones(NUM_JOINTS, dtype=np.float32), -np.ones(NUM_JOINTS, dtype=np.float32)]
    )
    lower = np.full(NUM_JOINTS, -1.0, dtype=np.float32)
    upper = np.full(NUM_JOINTS, 1.0, dtype=np.float32)

    target = integrate_incremental_action(previous, action, lower, upper)

    assert target.shape == (2, NUM_JOINTS)
    assert target.dtype == np.float32
    np.testing.assert_allclose(target[0], ACTION_SCALE)
    np.testing.assert_allclose(target[1], -ACTION_SCALE)
