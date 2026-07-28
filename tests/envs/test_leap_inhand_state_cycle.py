"""Contracts for the independent LEAP Ready/A/B state-cycle task."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.envs.manipulation.leap_inhand.state_cycle_pose_library import POSE_LIBRARY
from unilab.envs.manipulation.leap_inhand.state_cycle_rotation import (
    NEXT_PHASE,
    RESET_POSE_NAMES,
    SOURCE_PHASE,
    LeapInhandBallStateCycleRotationCfg,
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
    update_state_cycle_rotation,
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
    minimum_angles = np.asarray([0.0, 0.08, 0.0])
    result = rotation_condition(
        phases,
        np.asarray([0.079, 0.080, -0.10]),
        minimum_angles,
    )

    np.testing.assert_array_equal(result, [False, True, False])


def test_zero_rotation_requirement_accepts_negative_drift() -> None:
    phases = np.asarray(
        [StateCyclePhase.READY_TO_A, StateCyclePhase.B_TO_READY],
        dtype=np.int8,
    )
    minimum_angles = np.asarray([0.0, 0.08, 0.0])

    result = rotation_condition(
        phases,
        np.asarray([-0.01, -0.02]),
        minimum_angles,
    )

    np.testing.assert_array_equal(result, [True, True])


def _reward_terms(**overrides: np.ndarray):
    values = {
        "pose_progress": np.zeros(1),
        "pose_distance": np.zeros(1),
        "phase_start_pose_distance": np.zeros(1),
        "phase_positive_angle": np.zeros(1),
        "axis_delta": np.zeros(1),
        "position_error": np.zeros(1),
        "ball_linvel": np.zeros((1, 3)),
        "transition_event": np.zeros(1, dtype=bool),
        "rotation_cycle_success_event": np.zeros(1, dtype=bool),
        "invalid_rotation_cycle_event": np.zeros(1, dtype=bool),
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


@pytest.mark.parametrize("phase", list(StateCyclePhase))
def test_positive_rotation_reward_applies_in_every_phase(phase: StateCyclePhase) -> None:
    del phase
    terms = _reward_terms(axis_delta=np.asarray([0.01]))

    np.testing.assert_allclose(terms.rotation_progress, [3.0 * 0.01])


def test_positive_rotation_reward_is_not_capped_at_transition_angle() -> None:
    terms = _reward_terms(axis_delta=np.asarray([0.20]))

    np.testing.assert_allclose(terms.rotation_progress, [3.0 * 0.20])


@pytest.mark.parametrize("phase", list(StateCyclePhase))
def test_reverse_rotation_reward_applies_in_every_phase(phase: StateCyclePhase) -> None:
    del phase
    terms = _reward_terms(axis_delta=np.asarray([-0.01]))

    np.testing.assert_allclose(terms.reverse_rotation, [-5.0 * 0.01])
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


def test_timeout_claws_back_all_phase_positive_rotation_reward() -> None:
    terms = _reward_terms(
        phase_positive_angle=np.asarray([0.10]),
        timeout=np.asarray([True]),
    )

    np.testing.assert_allclose(terms.timeout, [-(0.25 + 3.0 * 0.10)])
    np.testing.assert_allclose(terms.total, [-(0.25 + 3.0 * 0.10)])

    transitioned = _reward_terms(
        phase_positive_angle=np.asarray([0.10]),
        transition_event=np.asarray([True]),
        timeout=np.asarray([False]),
    )
    np.testing.assert_allclose(transitioned.timeout, [0.0])
    np.testing.assert_allclose(transitioned.transition_event, [0.05])


def _rotation_update(**overrides: np.ndarray | float):
    values = {
        "axis_delta": np.zeros(1),
        "cycle_net_angle": np.zeros(1),
        "episode_net_angle": np.zeros(1),
        "phase_positive_angle": np.zeros(1),
        "last_completed_cycle_net_angle": np.zeros(1),
        "completed_cycle_angle_sum": np.zeros(1),
        "completed_cycle_count": np.zeros(1, dtype=np.uint32),
        "valid_rotation_cycles_completed": np.zeros(1, dtype=np.uint32),
        "transition_event": np.zeros(1, dtype=bool),
        "cycle_event": np.zeros(1, dtype=bool),
        "cycle_target_net_angle_rad": 0.10,
    }
    values.update(overrides)
    return update_state_cycle_rotation(**values)


def test_cycle_rotation_accumulates_across_phases_and_resets_only_on_cycle() -> None:
    ready_to_a = _rotation_update(
        axis_delta=np.asarray([0.02]),
        transition_event=np.asarray([True]),
    )
    np.testing.assert_allclose(ready_to_a.cycle_net_angle, [0.02])
    np.testing.assert_allclose(ready_to_a.episode_net_angle, [0.02])

    a_to_b = _rotation_update(
        axis_delta=np.asarray([0.07]),
        cycle_net_angle=ready_to_a.cycle_net_angle,
        episode_net_angle=ready_to_a.episode_net_angle,
        phase_positive_angle=ready_to_a.phase_positive_angle,
        last_completed_cycle_net_angle=ready_to_a.last_completed_cycle_net_angle,
        completed_cycle_angle_sum=ready_to_a.completed_cycle_angle_sum,
        completed_cycle_count=ready_to_a.completed_cycle_count,
        valid_rotation_cycles_completed=ready_to_a.valid_rotation_cycles_completed,
        transition_event=np.asarray([True]),
    )
    np.testing.assert_allclose(a_to_b.cycle_net_angle, [0.09])
    np.testing.assert_allclose(a_to_b.episode_net_angle, [0.09])

    b_to_ready = _rotation_update(
        axis_delta=np.asarray([0.03]),
        cycle_net_angle=a_to_b.cycle_net_angle,
        episode_net_angle=a_to_b.episode_net_angle,
        phase_positive_angle=a_to_b.phase_positive_angle,
        last_completed_cycle_net_angle=a_to_b.last_completed_cycle_net_angle,
        completed_cycle_angle_sum=a_to_b.completed_cycle_angle_sum,
        completed_cycle_count=a_to_b.completed_cycle_count,
        valid_rotation_cycles_completed=a_to_b.valid_rotation_cycles_completed,
        transition_event=np.asarray([True]),
        cycle_event=np.asarray([True]),
    )
    np.testing.assert_allclose(b_to_ready.completed_cycle_net_angle, [0.12])
    np.testing.assert_allclose(b_to_ready.last_completed_cycle_net_angle, [0.12])
    np.testing.assert_allclose(b_to_ready.completed_cycle_angle_sum, [0.12])
    np.testing.assert_array_equal(b_to_ready.completed_cycle_count, [1])
    np.testing.assert_array_equal(b_to_ready.rotation_cycle_success_event, [True])
    np.testing.assert_array_equal(b_to_ready.invalid_rotation_cycle_event, [False])
    np.testing.assert_array_equal(b_to_ready.valid_rotation_cycles_completed, [1])
    np.testing.assert_allclose(b_to_ready.cycle_net_angle, [0.0])
    np.testing.assert_allclose(b_to_ready.episode_net_angle, [0.12])


def test_invalid_rotation_cycle_has_penalty_and_no_cycle_bonus() -> None:
    update = _rotation_update(
        axis_delta=np.asarray([0.04]),
        transition_event=np.asarray([True]),
        cycle_event=np.asarray([True]),
    )
    np.testing.assert_array_equal(update.rotation_cycle_success_event, [False])
    np.testing.assert_array_equal(update.invalid_rotation_cycle_event, [True])

    terms = _reward_terms(
        transition_event=np.asarray([True]),
        rotation_cycle_success_event=update.rotation_cycle_success_event,
        invalid_rotation_cycle_event=update.invalid_rotation_cycle_event,
    )
    np.testing.assert_allclose(terms.transition_event, [0.05])
    np.testing.assert_allclose(terms.cycle_event, [0.0])
    np.testing.assert_allclose(terms.invalid_cycle, [-0.10])


def test_only_valid_completed_rotation_cycle_receives_cycle_bonus() -> None:
    ordinary_transition = _reward_terms(transition_event=np.asarray([True]))
    valid_cycle = _reward_terms(
        transition_event=np.asarray([True]),
        rotation_cycle_success_event=np.asarray([True]),
    )

    np.testing.assert_allclose(ordinary_transition.transition_event, [0.05])
    np.testing.assert_allclose(ordinary_transition.cycle_event, [0.0])
    np.testing.assert_allclose(valid_cycle.cycle_event, [0.20])


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


def test_state_cycle_uses_render_only_beach_ball_scene() -> None:
    mujoco = pytest.importorskip("mujoco")
    cfg = LeapInhandBallStateCycleRotationCfg()

    assert Path(cfg.scene.model_file).name == "scene_ball.xml"
    assert cfg.scene.visual_model_file is not None
    assert Path(cfg.scene.visual_model_file).name == "scene_ball_state_cycle_visual.xml"

    physics_model = mujoco.MjModel.from_xml_path(cfg.scene.model_file)
    visual_model = mujoco.MjModel.from_xml_path(cfg.scene.visual_model_file)
    assert (visual_model.nq, visual_model.nv, visual_model.nu) == (
        physics_model.nq,
        physics_model.nv,
        physics_model.nu,
    )

    object_body = mujoco.mj_name2id(
        physics_model,
        mujoco.mjtObj.mjOBJ_BODY,
        "leap_object",
    )
    visual_object_body = mujoco.mj_name2id(
        visual_model,
        mujoco.mjtObj.mjOBJ_BODY,
        "leap_object",
    )
    np.testing.assert_allclose(
        visual_model.body_mass[visual_object_body],
        physics_model.body_mass[object_body],
    )
    np.testing.assert_allclose(
        visual_model.body_inertia[visual_object_body],
        physics_model.body_inertia[object_body],
    )

    object_geom = mujoco.mj_name2id(
        physics_model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "leap_object_col",
    )
    visual_object_geom = mujoco.mj_name2id(
        visual_model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "leap_object_col",
    )
    np.testing.assert_allclose(
        visual_model.geom_size[visual_object_geom],
        physics_model.geom_size[object_geom],
    )
    np.testing.assert_allclose(
        visual_model.geom_friction[visual_object_geom],
        physics_model.geom_friction[object_geom],
    )

    visual_data = mujoco.MjData(visual_model)
    mujoco.mj_resetDataKeyframe(visual_model, visual_data, 0)
    mujoco.mj_forward(visual_model, visual_data)
    texture_pole_world_axis = visual_data.geom_xmat[visual_object_geom].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(texture_pole_world_axis, [0.0, 0.0, 1.0], atol=1e-6)


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
        assert obs["obs"].shape == (3, 142)
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
        np.testing.assert_allclose(obs["obs"][:, -2], 0.0)
        np.testing.assert_allclose(obs["obs"][:, -1], 1.0)
        for key in (
            "state_cycle_cycle_net_angle",
            "state_cycle_episode_net_angle",
            "state_cycle_phase_positive_angle",
            "state_cycle_last_completed_cycle_net_angle",
            "state_cycle_completed_cycle_angle_sum",
            "state_cycle_completed_cycle_count",
            "state_cycle_valid_rotation_cycles_completed",
        ):
            np.testing.assert_array_equal(info[key], np.zeros(3))

        next_state = env.step(np.zeros((3, 16), dtype=np.float32))
        assert next_state.obs["obs"].shape == (3, 142)
        assert np.isfinite(next_state.obs["obs"]).all()
        assert np.all((next_state.obs["obs"][:, -1] >= 0.0))
        assert np.all((next_state.obs["obs"][:, -1] <= 1.0))
        assert np.isfinite(next_state.reward).all()
        assert "termination/workspace_rate" in next_state.info["log"]
        assert "termination/timeout_rate" in next_state.info["log"]
        assert "reward/timeout" in next_state.info["log"]
        for key in (
            "rotation/episode_net_angle_mean",
            "rotation/episode_net_turns_mean",
            "rotation/cycle_net_angle_mean",
            "rotation/cycle_net_turns_mean",
            "rotation/last_completed_cycle_net_angle_mean",
            "rotation/completed_cycle_net_angle_mean",
            "rotation/valid_rotation_cycle_rate",
            "rotation/valid_rotation_cycles_completed_mean",
            "rotation/axis_delta_READY_TO_A_mean",
            "rotation/axis_delta_A_TO_B_mean",
            "rotation/axis_delta_B_TO_READY_mean",
            "rotation/reverse_fraction_READY_TO_A",
            "rotation/reverse_fraction_A_TO_B",
            "rotation/reverse_fraction_B_TO_READY",
            "state_cycle/pose_ok_rate",
            "state_cycle/position_ok_rate",
            "state_cycle/speed_ok_rate",
            "state_cycle/contact_ok_rate",
            "state_cycle/rotation_ok_rate",
            "state_cycle/no_palm_contact_rate",
            "state_cycle/hold_steps_mean",
            "state_cycle/timeout_READY_TO_A_rate",
            "state_cycle/timeout_A_TO_B_rate",
            "state_cycle/timeout_B_TO_READY_rate",
            "reward/invalid_cycle",
        ):
            assert key in next_state.info["log"]
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
