from __future__ import annotations

import numpy as np
import pytest

from unilab.envs.manipulation.leap_inhand.deploy_contract import (
    ACTOR_FRAME_DIM,
    ACTOR_HISTORY_LEN,
    ACTOR_OBS_DIM,
    NUM_JOINTS,
    OldestFirstHistoryBuffer,
    build_actor_frame,
    integrate_incremental_action,
    normalize_to_joint_range,
)


def test_reset_fills_every_slot_with_current_frame() -> None:
    history = OldestFirstHistoryBuffer(ACTOR_HISTORY_LEN, ACTOR_FRAME_DIM)
    frame = np.arange(ACTOR_FRAME_DIM, dtype=np.float32)

    history.reset(frame)

    expected = np.repeat(frame[None, :], ACTOR_HISTORY_LEN, axis=0)
    np.testing.assert_array_equal(history.as_array_oldest_first(), expected)


def test_push_drops_oldest_and_appends_newest() -> None:
    history = OldestFirstHistoryBuffer(3, 2)
    history.reset(np.array([0.0, 0.5]))

    history.push(np.array([1.0, 1.5]))
    np.testing.assert_array_equal(
        history.as_array_oldest_first(),
        [[0.0, 0.5], [0.0, 0.5], [1.0, 1.5]],
    )

    history.push(np.array([2.0, 2.5]))
    history.push(np.array([3.0, 3.5]))
    np.testing.assert_array_equal(
        history.as_array_oldest_first(),
        [[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]],
    )


def test_flatten_is_oldest_first_and_returns_an_independent_copy() -> None:
    history = OldestFirstHistoryBuffer(3, 2)
    history.reset(np.array([0.0, 1.0]))
    history.push(np.array([2.0, 3.0]))
    history.push(np.array([4.0, 5.0]))

    flattened = history.flatten_oldest_first()
    array_copy = history.as_array_oldest_first()
    flattened[:] = -1.0
    array_copy[:] = -2.0

    np.testing.assert_array_equal(history.flatten_oldest_first(), [0, 1, 2, 3, 4, 5])


@pytest.mark.parametrize(
    ("frame", "error"),
    [
        (np.zeros(3), "shape"),
        (np.array([np.nan, 0.0]), "finite"),
        (np.array([np.inf, 0.0]), "finite"),
    ],
)
def test_history_rejects_invalid_frames(frame, error) -> None:
    history = OldestFirstHistoryBuffer(3, 2)
    with pytest.raises((TypeError, ValueError), match=error):
        history.reset(frame)


def test_history_requires_reset_before_push_or_read() -> None:
    history = OldestFirstHistoryBuffer(3, 2)
    with pytest.raises(RuntimeError, match="reset"):
        history.push(np.zeros(2))
    with pytest.raises(RuntimeError, match="reset"):
        history.as_array_oldest_first()


def test_timestep_frame_uses_previous_sent_target_not_new_target() -> None:
    lower = np.full(NUM_JOINTS, -1.0)
    upper = np.full(NUM_JOINTS, 1.0)
    measured_q_t = np.linspace(-0.4, 0.4, NUM_JOINTS)
    target_t_minus_1 = np.linspace(-0.3, 0.3, NUM_JOINTS)
    action_t = np.ones(NUM_JOINTS)

    frame_t = build_actor_frame(
        measured_q_t,
        target_t_minus_1,
        [0.0, 0.0, 1.0],
        lower,
        upper,
    )
    target_t = integrate_incremental_action(target_t_minus_1, action_t, lower, upper)

    expected_previous = normalize_to_joint_range(target_t_minus_1, lower, upper)
    new_target_norm = normalize_to_joint_range(target_t, lower, upper)
    np.testing.assert_allclose(frame_t[16:32], expected_previous)
    assert not np.allclose(frame_t[16:32], new_target_norm)

    history = OldestFirstHistoryBuffer(ACTOR_HISTORY_LEN, ACTOR_FRAME_DIM)
    history.reset(frame_t)
    assert history.flatten_oldest_first().shape == (ACTOR_OBS_DIM,)
