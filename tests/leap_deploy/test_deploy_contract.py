from __future__ import annotations

import numpy as np
import pytest

from unilab.envs.manipulation.leap_inhand.deploy_contract import (
    ACTION_SCALE,
    ACTOR_FRAME_DIM,
    ACTOR_HISTORY_LEN,
    ACTOR_OBS_DIM,
    CONTROL_DT,
    NUM_JOINTS,
    PRIV_INFO_DIM,
    PROPRIO_FRAME_DIM,
    PROPRIO_HISTORY_LEN,
    build_actor_frame,
    build_proprio_frame,
    denormalize_from_joint_range,
    normalize_to_joint_range,
)


@pytest.fixture
def bounds() -> tuple[np.ndarray, np.ndarray]:
    lower = np.linspace(-1.5, -0.5, NUM_JOINTS)
    upper = np.linspace(0.75, 2.25, NUM_JOINTS)
    return lower, upper


def test_fixed_contract_dimensions() -> None:
    assert NUM_JOINTS == 16
    assert ACTOR_FRAME_DIM == 35
    assert ACTOR_HISTORY_LEN == 3
    assert ACTOR_OBS_DIM == 105
    assert PROPRIO_FRAME_DIM == 32
    assert PROPRIO_HISTORY_LEN == 30
    assert PRIV_INFO_DIM == 9
    assert CONTROL_DT == pytest.approx(0.05)
    assert ACTION_SCALE == pytest.approx(1.0 / 24.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_normalization_endpoints_midpoint_and_round_trip(dtype: type[np.floating]) -> None:
    lower = np.linspace(-2.0, -0.5, NUM_JOINTS, dtype=dtype)
    upper = np.linspace(0.5, 2.0, NUM_JOINTS, dtype=dtype)
    midpoint = (lower + upper) * dtype(0.5)
    values = np.stack([lower, midpoint, upper])

    normalized = normalize_to_joint_range(values, lower, upper)
    restored = denormalize_from_joint_range(normalized, lower, upper)

    assert normalized.dtype == dtype
    np.testing.assert_allclose(normalized[0], -1.0, atol=1e-6)
    np.testing.assert_allclose(normalized[1], 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized[2], 1.0, atol=1e-6)
    assert float(np.max(np.abs(restored - values))) < 1.0e-6


def test_normalization_supports_batch_and_only_clips_when_requested(bounds) -> None:
    lower, upper = bounds
    values = np.stack([lower - 0.25, upper + 0.25])

    unclipped = normalize_to_joint_range(values, lower, upper)
    clipped = normalize_to_joint_range(values, lower, upper, clip=True)

    assert unclipped.shape == (2, NUM_JOINTS)
    assert np.any(unclipped[0] < -1.0)
    assert np.any(unclipped[1] > 1.0)
    assert np.all(clipped >= -1.0)
    assert np.all(clipped <= 1.0)


@pytest.mark.parametrize(
    ("values", "lower", "upper", "error"),
    [
        (np.zeros(15), np.zeros(16), np.ones(16), "values"),
        (np.zeros(16), np.zeros(15), np.ones(16), "lower"),
        (np.zeros(16), np.zeros(16), np.ones(15), "upper"),
        (np.zeros(16), np.zeros(16), np.zeros(16), "greater"),
        (np.full(16, np.nan), np.zeros(16), np.ones(16), "finite"),
        (np.zeros(16), np.zeros(16), np.full(16, np.inf), "finite"),
    ],
)
def test_normalization_rejects_invalid_inputs(values, lower, upper, error) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        normalize_to_joint_range(values, lower, upper)


def test_actor_and_proprio_frames_have_exact_field_order(bounds) -> None:
    lower, upper = bounds
    measured_q = lower + 0.2 * (upper - lower)
    previous_target = lower + 0.8 * (upper - lower)
    axis = np.array([0.0, 0.0, 4.0])

    q_norm = normalize_to_joint_range(measured_q, lower, upper)
    target_norm = normalize_to_joint_range(previous_target, lower, upper)
    actor = build_actor_frame(measured_q, previous_target, axis, lower, upper)
    proprio = build_proprio_frame(measured_q, previous_target, lower, upper)

    assert actor.shape == (ACTOR_FRAME_DIM,)
    assert proprio.shape == (PROPRIO_FRAME_DIM,)
    np.testing.assert_allclose(actor[:16], q_norm)
    np.testing.assert_allclose(actor[16:32], target_norm)
    np.testing.assert_allclose(actor[32:35], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(proprio[:16], q_norm)
    np.testing.assert_allclose(proprio[16:32], target_norm)


def test_frame_builders_support_batch_and_axis_broadcast(bounds) -> None:
    lower, upper = bounds
    measured_q = np.stack([lower, upper])
    previous_target = np.stack([(lower + upper) / 2.0, lower])

    actor = build_actor_frame(measured_q, previous_target, [1.0, 0.0, 0.0], lower, upper)
    proprio = build_proprio_frame(measured_q, previous_target, lower, upper)

    assert actor.shape == (2, ACTOR_FRAME_DIM)
    assert proprio.shape == (2, PROPRIO_FRAME_DIM)
    np.testing.assert_allclose(actor[:, -3:], [[1.0, 0.0, 0.0]] * 2)


@pytest.mark.parametrize(
    ("axis", "error"),
    [
        (np.zeros(3), "norm"),
        (np.array([np.nan, 0.0, 1.0]), "finite"),
        (np.array([np.inf, 0.0, 1.0]), "finite"),
        (np.zeros(2), "shape"),
    ],
)
def test_actor_frame_rejects_invalid_axis(bounds, axis, error) -> None:
    lower, upper = bounds
    with pytest.raises((TypeError, ValueError), match=error):
        build_actor_frame(lower, lower, axis, lower, upper)


def test_frame_builders_reject_mismatched_joint_shapes(bounds) -> None:
    lower, upper = bounds
    with pytest.raises(ValueError, match="same shape"):
        build_proprio_frame(np.zeros((2, 16)), np.zeros(16), lower, upper)
