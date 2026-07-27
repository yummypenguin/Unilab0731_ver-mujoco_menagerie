"""Contracts for the independent LEAP Ready/A/B state-cycle task."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.envs.manipulation.leap_inhand.state_cycle_pose_library import POSE_LIBRARY
from unilab.envs.manipulation.leap_inhand.state_cycle_rotation import (
    NEXT_PHASE,
    RESET_POSE_NAMES,
    SOURCE_PHASE,
    LeapInhandBallStateCycleRotationEnv,
    StateCycleConfig,
    StateCyclePhase,
    StateCycleRewardConfig,
    advance_state_cycle,
    build_state_cycle_reset_arrays,
    compute_pose_distance,
    compute_pose_progress,
    compute_state_cycle_reward,
    compute_timeout_event,
    reset_phase_for_pose,
    rotation_condition,
)


def test_pose_library_contains_valid_immutable_waypoints() -> None:
    assert tuple(POSE_LIBRARY) == ("ready", "A", "B", "C", "D")
    for waypoint in POSE_LIBRARY.values():
        assert waypoint.qpos.shape == (23,)
        assert waypoint.hand_qpos.shape == (16,)
        assert waypoint.ball_pos.shape == (3,)
        assert waypoint.ball_quat.shape == (4,)
        assert waypoint.ctrl.shape == (16,)
        assert np.isfinite(waypoint.qpos).all()
        assert np.isfinite(waypoint.ctrl).all()
        assert np.linalg.norm(waypoint.ball_quat) == pytest.approx(1.0)
        assert not waypoint.qpos.flags.writeable
        assert not waypoint.hand_qpos.flags.writeable
        assert not waypoint.ball_pos.flags.writeable
        assert not waypoint.ball_quat.flags.writeable
        assert not waypoint.ctrl.flags.writeable


def test_fsm_cycles_ready_a_b_ready_and_counts_cycle() -> None:
    required = np.ones(3, dtype=np.uint32)
    phase = np.asarray([StateCyclePhase.READY_TO_A], dtype=np.int8)
    hold = np.zeros(1, dtype=np.uint32)
    cycles = 0

    for expected in (
        StateCyclePhase.A_TO_B,
        StateCyclePhase.B_TO_READY,
        StateCyclePhase.READY_TO_A,
    ):
        update = advance_state_cycle(phase, hold, np.asarray([True]), required)
        phase, hold = update.phase, update.hold_steps
        cycles += int(update.cycle_event[0])
        assert phase[0] == int(expected)

    assert cycles == 1
    assert NEXT_PHASE[StateCyclePhase.B_TO_READY] == StateCyclePhase.READY_TO_A


def test_reset_sources_select_matching_phase_zero_qvel_and_source_ctrl() -> None:
    qpos, qvel, ctrl, phases = build_state_cycle_reset_arrays(RESET_POSE_NAMES, nv=22)

    assert qpos.shape == (3, 23)
    assert qvel.shape == (3, 22)
    assert not np.any(qvel)
    np.testing.assert_array_equal(
        phases,
        [
            StateCyclePhase.READY_TO_A,
            StateCyclePhase.A_TO_B,
            StateCyclePhase.B_TO_READY,
        ],
    )
    for row, name in enumerate(RESET_POSE_NAMES):
        np.testing.assert_allclose(qpos[row], POSE_LIBRARY[name].qpos)
        np.testing.assert_allclose(ctrl[row], POSE_LIBRARY[name].ctrl)

    np.testing.assert_array_equal(
        reset_phase_for_pose(list(RESET_POSE_NAMES)),
        [int(SOURCE_PHASE[name]) for name in RESET_POSE_NAMES],
    )


def test_pose_distance_and_progress_sign() -> None:
    lower = np.full(16, -2.0)
    upper = np.full(16, 2.0)
    target = np.zeros((1, 16))
    far = np.full((1, 16), 1.0)
    near = np.full((1, 16), 0.25)

    target_distance = compute_pose_distance(target, target, lower, upper)
    far_distance = compute_pose_distance(far, target, lower, upper)
    near_distance = compute_pose_distance(near, target, lower, upper)

    np.testing.assert_allclose(target_distance, 0.0)
    assert compute_pose_progress(far_distance, near_distance)[0] > 0.0
    assert compute_pose_progress(near_distance, far_distance)[0] < 0.0


def test_a_to_b_requires_positive_net_rotation_threshold() -> None:
    phases = np.asarray([StateCyclePhase.A_TO_B] * 3, dtype=np.int8)
    minimum_angles = np.asarray([0.0, 0.03, 0.0])
    result = rotation_condition(
        phases,
        np.asarray([0.029, 0.030, -0.10]),
        minimum_angles,
    )

    np.testing.assert_array_equal(result, [False, True, False])


def test_zero_rotation_requirement_accepts_negative_drift() -> None:
    phases = np.asarray(
        [StateCyclePhase.READY_TO_A, StateCyclePhase.B_TO_READY],
        dtype=np.int8,
    )
    minimum_angles = np.asarray([0.0, 0.03, 0.0])

    result = rotation_condition(
        phases,
        np.asarray([-0.01, -0.02]),
        minimum_angles,
    )

    np.testing.assert_array_equal(result, [True, True])


def _reward_terms(**overrides: np.ndarray):
    values = {
        "phases": np.asarray([StateCyclePhase.A_TO_B], dtype=np.int8),
        "pose_progress": np.zeros(1),
        "pose_distance": np.zeros(1),
        "phase_start_pose_distance": np.zeros(1),
        "axis_delta": np.zeros(1),
        "edge_net_angle_before": np.zeros(1),
        "required_angle": np.asarray([0.03]),
        "position_error": np.zeros(1),
        "ball_linvel": np.zeros((1, 3)),
        "transition_event": np.zeros(1, dtype=bool),
        "cycle_event": np.zeros(1, dtype=bool),
        "timeout": np.zeros(1, dtype=bool),
        "workspace_failure": np.zeros(1, dtype=bool),
    }
    values.update(overrides)
    return compute_state_cycle_reward(
        reward_cfg=StateCycleRewardConfig(),
        ctrl_dt=0.05,
        **values,
    )


def test_pose_progress_reward_is_not_scaled_by_ctrl_dt() -> None:
    terms = _reward_terms(pose_progress=np.asarray([0.10]))

    np.testing.assert_allclose(terms.pose_progress, [0.40])
    np.testing.assert_allclose(terms.total, [0.40])


def test_pose_tracking_reward_has_zero_upper_bound_at_target() -> None:
    at_target = _reward_terms(pose_distance=np.asarray([0.0]))
    far = _reward_terms(pose_distance=np.asarray([1.0]))

    np.testing.assert_allclose(at_target.pose_tracking, [0.0])
    assert far.pose_tracking[0] < 0.0
    assert far.pose_tracking[0] >= -0.25 * 0.05


def test_positive_rotation_reward_is_capped_at_remaining_angle() -> None:
    terms = _reward_terms(
        axis_delta=np.asarray([0.20]),
        edge_net_angle_before=np.asarray([0.0]),
        required_angle=np.asarray([0.03]),
    )

    np.testing.assert_allclose(terms.rotation_progress, [3.0 * 0.03])


def test_reverse_rotation_reward_is_negative() -> None:
    terms = _reward_terms(axis_delta=np.asarray([-0.01]))

    np.testing.assert_allclose(terms.reverse_rotation, [-4.0 * 0.01])
    assert terms.total[0] < 0.0


def test_workspace_failure_suppresses_timeout_penalty() -> None:
    both = _reward_terms(
        timeout=np.asarray([True]),
        workspace_failure=np.asarray([True]),
    )
    timeout_only = _reward_terms(timeout=np.asarray([True]))

    np.testing.assert_allclose(both.timeout, [0.0])
    np.testing.assert_allclose(both.failure, [-1.0])
    np.testing.assert_allclose(both.total, [-1.0])
    np.testing.assert_allclose(timeout_only.timeout, [-0.25])


def test_timeout_claws_back_net_phase_pose_progress() -> None:
    terms = _reward_terms(
        pose_progress=np.asarray([0.10]),
        pose_distance=np.asarray([0.0]),
        phase_start_pose_distance=np.asarray([0.10]),
        timeout=np.asarray([True]),
    )

    np.testing.assert_allclose(terms.pose_progress, [0.40])
    np.testing.assert_allclose(terms.timeout, [-0.65])
    np.testing.assert_allclose(terms.total, [-0.25])


def test_last_legal_step_can_succeed_before_timeout() -> None:
    phase_steps = np.asarray([30, 30], dtype=np.uint32)
    timeout_steps = np.asarray([30, 30], dtype=np.uint32)
    timeout = compute_timeout_event(
        phase_steps,
        timeout_steps,
        transition_event=np.asarray([True, False]),
        workspace_failure=np.asarray([False, False]),
    )

    np.testing.assert_array_equal(timeout, [False, True])


def test_hold_requires_four_consecutive_valid_steps() -> None:
    phase = np.asarray([StateCyclePhase.READY_TO_A], dtype=np.int8)
    hold = np.zeros(1, dtype=np.uint32)
    required = np.full(3, 4, dtype=np.uint32)

    for _ in range(3):
        update = advance_state_cycle(phase, hold, np.asarray([True]), required)
        phase, hold = update.phase, update.hold_steps
        assert not update.transition_event[0]
        assert phase[0] == int(StateCyclePhase.READY_TO_A)

    update = advance_state_cycle(phase, hold, np.asarray([True]), required)
    assert update.transition_event[0]
    assert update.phase[0] == int(StateCyclePhase.A_TO_B)


def test_hold_must_be_consecutive() -> None:
    phase = np.asarray([StateCyclePhase.READY_TO_A], dtype=np.int8)
    required = np.full(3, 4, dtype=np.uint32)
    update = advance_state_cycle(phase, np.asarray([2], dtype=np.uint32), np.asarray([False]), required)
    assert update.hold_steps[0] == 0
    assert not update.transition_event[0]


def test_timeout_cannot_be_counted_as_transition_success() -> None:
    phase = np.asarray([StateCyclePhase.A_TO_B], dtype=np.int8)
    timeout = np.asarray([True])
    workspace_failure = np.asarray([False])
    otherwise_valid = np.asarray([True])
    update = advance_state_cycle(
        phase,
        np.asarray([3], dtype=np.uint32),
        otherwise_valid & ~timeout & ~workspace_failure,
        np.full(3, 4, dtype=np.uint32),
    )

    assert not update.transition_event[0]
    assert timeout[0]
    assert not workspace_failure[0]


def test_registry_exposes_independent_state_cycle_task() -> None:
    ensure_registries()
    registered = registry.list_registered_envs()
    assert registered["LeapInhandBallStateCycleRotation"]["available_backends"] == [
        "mujoco",
        "motrix",
    ]


def test_state_cycle_environment_reset_and_step() -> None:
    pytest.importorskip("mujoco")
    try:
        from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env is unavailable")

    ensure_registries()
    env = registry.make(
        "LeapInhandBallStateCycleRotation",
        sim_backend="mujoco",
        num_envs=3,
        env_cfg_override={
            "sim_dt": 0.005,
            "ctrl_dt": 0.05,
            "reward_config": asdict(StateCycleRewardConfig()),
            "state_cycle": asdict(StateCycleConfig()),
        },
    )
    assert isinstance(env, LeapInhandBallStateCycleRotationEnv)
    try:
        env._sample_reset_pose_names = lambda num_reset: list(RESET_POSE_NAMES[:num_reset])
        obs, info = env.reset(np.arange(3, dtype=np.int32))
        assert obs["obs"].shape == (3, 140)
        assert np.isfinite(obs["obs"]).all()
        np.testing.assert_allclose(info["prev_ctrl"], np.stack([
            POSE_LIBRARY[name].ctrl for name in RESET_POSE_NAMES
        ]))
        np.testing.assert_array_equal(
            info["state_cycle_phase"],
            [
                StateCyclePhase.READY_TO_A,
                StateCyclePhase.A_TO_B,
                StateCyclePhase.B_TO_READY,
            ],
        )
        np.testing.assert_array_equal(
            obs["obs"][:, 132:135],
            np.eye(3),
        )
        np.testing.assert_allclose(obs["obs"][:, 136], [0.0, 1.0, 0.0])

        next_state = env.step(np.zeros((3, 16), dtype=np.float32))
        assert next_state.obs["obs"].shape == (3, 140)
        assert np.isfinite(next_state.obs["obs"]).all()
        assert np.isfinite(next_state.reward).all()
        assert "termination/workspace_rate" in next_state.info["log"]
        assert "termination/timeout_rate" in next_state.info["log"]
        assert "reward/timeout" in next_state.info["log"]
    finally:
        env.close()


def test_recorded_waypoints_settle_without_palm_contact() -> None:
    pytest.importorskip("mujoco")
    try:
        from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env is unavailable")

    ensure_registries()
    env = registry.make(
        "LeapInhandBallStateCycleRotation",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "sim_dt": 0.005,
            "ctrl_dt": 0.05,
            "reward_config": asdict(StateCycleRewardConfig()),
            "state_cycle": asdict(StateCycleConfig()),
        },
    )
    try:
        for pose_name in RESET_POSE_NAMES:
            env._sample_reset_pose_names = (
                lambda num_reset, name=pose_name: [name] * num_reset
            )
            env.reset(np.asarray([0], dtype=np.int32))
            for _ in range(5):
                env.step(np.zeros((1, 16), dtype=np.float32))
            fingertip_contacts = env._contacts(np.asarray([0], dtype=np.int32))[0]
            palm_contact = bool(
                env._palm_contacts(np.asarray([0], dtype=np.int32))[0] > 0.5
            )
            print(
                f"{pose_name}: fingertips={fingertip_contacts.astype(int).tolist()} "
                f"palm={int(palm_contact)}"
            )
            assert np.sum(fingertip_contacts) >= 2, pose_name
            assert not palm_contact, pose_name
    finally:
        env.close()
