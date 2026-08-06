from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base.np_env import NpEnvState
from unilab.dr import DomainRandomizationCapabilities
from unilab.dr.types import (
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_GRAVITY,
)
from unilab.envs.manipulation.leap_inhand.ball_rotation_0730_hora import (
    LeapInhandBall0730HoraRotationEnv,
)
from unilab.envs.manipulation.leap_inhand.deploy_contract import (
    ACTION_SCALE,
    ACTOR_FRAME_DIM,
    ACTOR_HISTORY_LEN,
    PROPRIO_FRAME_DIM,
    denormalize_from_joint_range,
    normalize_to_joint_range,
)
from unilab.envs.manipulation.leap_inhand.hora_domain_randomization import (
    PRIV_ACTION_DELAY,
    PRIV_COM_SLICE,
    PRIV_FRICTION_SCALE,
    PRIV_GRAVITY_SLICE,
    PRIV_MASS_RATIO,
    LeapHoraDomainRandomizationConfig,
    build_hora_critic_info,
    build_hora_reset_payload,
    sample_hora_reset_values,
    validate_hora_backend_capabilities,
)
from unilab.training import BackendAdapter, create_env

ROOT = Path(__file__).parent.parent.parent
CONF_DIR = ROOT / "conf" / "ppo"
NOMINAL_GRAVITY = np.asarray([0.0, 0.0, -9.81], dtype=np.float64)


def _compose_hora_cfg(overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(
            "config",
            overrides=["task=leap_inhand_ball_0730/mujoco_hora", *(overrides or [])],
        )


def _create_hora_env(num_envs: int, overrides: list[str] | None = None):
    cfg = _compose_hora_cfg(overrides)
    env_override = BackendAdapter(
        cfg, root_dir=ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    env = create_env(
        cfg,
        num_envs=num_envs,
        env_cfg_override=env_override,
        sim_backend="mujoco",
        task_name="LeapInhandBall0730HoraRotation",
    )
    return cfg, env


@pytest.mark.parametrize(
    "updates",
    [
        {"object_mass_ratio_lower": 0.0},
        {"object_mass_ratio_lower": 1.2, "object_mass_ratio_upper": 1.1},
        {"object_friction_scale_lower": -0.1},
        {"object_friction_scale_lower": 1.3, "object_friction_scale_upper": 1.2},
        {"object_com_offset_lower": (0.0, 0.0)},
        {
            "object_com_offset_lower": (0.0, 0.0, 0.1),
            "object_com_offset_upper": (0.0, 0.0, 0.0),
        },
        {"joint_measurement_noise_rad": -0.001},
        {"gravity_tilt_max_deg": np.nan},
        {"gravity_tilt_max_deg": 15.1},
        {"action_delay_min_steps": -1},
        {"action_delay_max_steps": 2},
        {"action_delay_min_steps": 1, "action_delay_max_steps": 0},
    ],
)
def test_hora_domain_randomization_config_rejects_invalid_values(
    updates: dict[str, object],
) -> None:
    cfg = LeapHoraDomainRandomizationConfig(**updates)
    with pytest.raises((TypeError, ValueError)):
        cfg.validate()


def test_hora_domain_randomization_default_config_is_valid() -> None:
    LeapHoraDomainRandomizationConfig().validate()


def test_hora_sampling_ranges_and_seed_reproducibility() -> None:
    cfg = LeapHoraDomainRandomizationConfig()
    np.random.seed(123)
    first = sample_hora_reset_values(cfg, 1024, NOMINAL_GRAVITY)
    np.random.seed(123)
    repeated = sample_hora_reset_values(cfg, 1024, NOMINAL_GRAVITY)
    np.random.seed(124)
    different = sample_hora_reset_values(cfg, 1024, NOMINAL_GRAVITY)

    np.testing.assert_array_equal(first.mass_ratio, repeated.mass_ratio)
    np.testing.assert_array_equal(first.friction_scale, repeated.friction_scale)
    np.testing.assert_array_equal(first.com_offset, repeated.com_offset)
    np.testing.assert_array_equal(first.gravity, repeated.gravity)
    np.testing.assert_array_equal(first.action_delay_steps, repeated.action_delay_steps)
    assert not np.array_equal(first.mass_ratio, different.mass_ratio)

    assert np.all((first.mass_ratio >= 0.9) & (first.mass_ratio <= 1.1))
    assert np.all((first.friction_scale >= 0.8) & (first.friction_scale <= 1.2))
    assert np.all((first.com_offset >= -0.001) & (first.com_offset <= 0.001))
    np.testing.assert_allclose(np.linalg.norm(first.gravity, axis=1), 9.81, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(first.gravity_direction, axis=1), 1.0, atol=1e-12
    )
    tilt_deg = np.rad2deg(
        np.arccos(np.clip(-first.gravity_direction[:, 2], -1.0, 1.0))
    )
    assert float(np.max(tilt_deg)) <= 3.0 + 1e-10
    assert set(np.unique(first.action_delay_steps)).issubset({0, 1})


def test_disabled_sampling_and_critic_info_are_exactly_nominal() -> None:
    cfg = LeapHoraDomainRandomizationConfig(enabled=False)
    samples = sample_hora_reset_values(cfg, 4, NOMINAL_GRAVITY)
    critic_info = build_hora_critic_info(samples, action_delay_max_steps=1)

    np.testing.assert_array_equal(
        critic_info,
        np.tile([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0], (4, 1)),
    )


def test_backend_capability_validation_fails_closed_with_config_field() -> None:
    cfg = LeapHoraDomainRandomizationConfig()
    capabilities = DomainRandomizationCapabilities(
        supported_reset_terms=frozenset(
            {RESET_TERM_BODY_IPOS, RESET_TERM_GEOM_FRICTION, RESET_TERM_GRAVITY}
        )
    )
    with pytest.raises(
        NotImplementedError,
        match=r"fake.*body_mass.*env\.hora_domain_rand\.randomize_object_mass",
    ):
        validate_hora_backend_capabilities(cfg, capabilities, "fake")


def test_reset_payload_is_object_only_and_matches_critic_info() -> None:
    cfg = LeapHoraDomainRandomizationConfig()
    np.random.seed(11)
    samples = sample_hora_reset_values(cfg, 512, NOMINAL_GRAVITY)
    body_mass = np.asarray([0.0, 0.4, 0.08, 0.01])
    body_ipos = np.arange(12, dtype=np.float64).reshape(4, 3) * 0.01
    geom_friction = np.asarray(
        [[1.0, 0.01, 0.001], [0.8, 0.02, 0.002], [0.6, 0.03, 0.003]]
    )
    payload = build_hora_reset_payload(
        samples,
        cfg=cfg,
        object_body_id=2,
        object_geom_id=1,
        nominal_body_mass=body_mass,
        nominal_body_ipos=body_ipos,
        nominal_geom_friction=geom_friction,
    )
    assert payload is not None
    critic_info = build_hora_critic_info(samples, action_delay_max_steps=1)

    np.testing.assert_allclose(
        payload.body_mass[:, 2] / body_mass[2],
        critic_info[:, PRIV_MASS_RATIO],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        payload.geom_friction[:, 1, 0] / geom_friction[1, 0],
        critic_info[:, PRIV_FRICTION_SCALE],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        payload.geom_friction[:, 1, :] / geom_friction[1, :],
        np.repeat(
            critic_info[:, PRIV_FRICTION_SCALE, None], 3, axis=1
        ),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        payload.body_ipos[:, 2] - body_ipos[2],
        critic_info[:, PRIV_COM_SLICE],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        payload.gravity / np.linalg.norm(payload.gravity, axis=1, keepdims=True),
        critic_info[:, PRIV_GRAVITY_SLICE],
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        critic_info[:, PRIV_ACTION_DELAY], samples.action_delay_steps
    )
    np.testing.assert_array_equal(
        payload.body_mass[:, [0, 1, 3]],
        np.broadcast_to(body_mass[[0, 1, 3]], (512, 3)),
    )
    np.testing.assert_array_equal(
        payload.body_ipos[:, [0, 1, 3]],
        np.broadcast_to(body_ipos[[0, 1, 3]], (512, 3, 3)),
    )
    np.testing.assert_array_equal(
        payload.geom_friction[:, [0, 2]],
        np.broadcast_to(geom_friction[[0, 2]], (512, 2, 3)),
    )


def test_actual_reset_plan_payload_matches_critic_info() -> None:
    np.random.seed(19)
    _, env = _create_hora_env(4)
    try:
        provider = env._dr_manager._provider
        plan = provider.build_reset_plan(env, np.arange(4, dtype=np.int32))
        payload = plan.randomization
        critic_info = plan.info_updates["critic_info"]
        assert payload is not None

        mass_ratio = (
            payload.body_mass[:, env._hora_object_body_id]
            / env._hora_nominal_body_mass[env._hora_object_body_id]
        )
        friction_scale = (
            payload.geom_friction[:, env._hora_object_geom_id, 0]
            / env._hora_nominal_geom_friction[env._hora_object_geom_id, 0]
        )
        com_offset = (
            payload.body_ipos[:, env._hora_object_body_id]
            - env._hora_nominal_body_ipos[env._hora_object_body_id]
        )
        gravity_direction = payload.gravity / np.linalg.norm(
            payload.gravity, axis=1, keepdims=True
        )

        assert float(np.max(np.abs(mass_ratio - critic_info[:, PRIV_MASS_RATIO]))) < 1e-6
        assert (
            float(
                np.max(
                    np.abs(friction_scale - critic_info[:, PRIV_FRICTION_SCALE])
                )
            )
            < 1e-6
        )
        assert float(np.max(np.abs(com_offset - critic_info[:, PRIV_COM_SLICE]))) < 1e-6
        assert (
            float(
                np.max(
                    np.abs(
                        gravity_direction - critic_info[:, PRIV_GRAVITY_SLICE]
                    )
                )
            )
            < 1e-6
        )
        np.testing.assert_array_equal(
            critic_info[:, PRIV_ACTION_DELAY],
            plan.info_updates["hora_action_delay_steps"],
        )

        non_object_bodies = np.arange(env._hora_nominal_body_mass.size) != env._hora_object_body_id
        non_object_geoms = (
            np.arange(env._hora_nominal_geom_friction.shape[0])
            != env._hora_object_geom_id
        )
        np.testing.assert_array_equal(
            payload.body_mass[:, non_object_bodies],
            np.broadcast_to(
                env._hora_nominal_body_mass[non_object_bodies],
                payload.body_mass[:, non_object_bodies].shape,
            ),
        )
        np.testing.assert_array_equal(
            payload.body_ipos[:, non_object_bodies],
            np.broadcast_to(
                env._hora_nominal_body_ipos[non_object_bodies],
                payload.body_ipos[:, non_object_bodies].shape,
            ),
        )
        np.testing.assert_array_equal(
            payload.geom_friction[:, non_object_geoms],
            np.broadcast_to(
                env._hora_nominal_geom_friction[non_object_geoms],
                payload.geom_friction[:, non_object_geoms].shape,
            ),
        )
    finally:
        env.close()


def test_disabled_hora_reset_plan_has_no_physics_payload() -> None:
    _, env = _create_hora_env(2, ["env.hora_domain_rand.enabled=false"])
    try:
        plan = env._dr_manager._provider.build_reset_plan(
            env, np.arange(2, dtype=np.int32)
        )
        assert plan.randomization is None
        np.testing.assert_array_equal(
            plan.info_updates["critic_info"],
            np.tile(
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                (2, 1),
            ),
        )
        np.testing.assert_array_equal(plan.info_updates["hora_action_queue"], 0.0)
        np.testing.assert_array_equal(
            plan.info_updates["hora_action_delay_steps"], 0
        )
    finally:
        env.close()


def _action_only_env(delay_steps: np.ndarray) -> tuple[LeapInhandBall0730HoraRotationEnv, NpEnvState]:
    num_envs = len(delay_steps)
    env = object.__new__(LeapInhandBall0730HoraRotationEnv)
    env._np_dtype = np.dtype(np.float32)
    env._num_action = 16
    env._ctrl_lower = np.full(16, -10.0, dtype=np.float32)
    env._ctrl_upper = np.full(16, 10.0, dtype=np.float32)
    env.default_angles = np.zeros(16, dtype=np.float32)
    env._cfg = SimpleNamespace(
        control_config=SimpleNamespace(action_scale=ACTION_SCALE)
    )
    info = {
        "prev_ctrl": np.zeros((num_envs, 16), dtype=np.float32),
        "current_actions": np.zeros((num_envs, 16), dtype=np.float32),
        "hora_action_delay_steps": np.asarray(delay_steps, dtype=np.int32),
        "hora_action_queue": np.zeros((num_envs, 2, 16), dtype=np.float32),
    }
    state = NpEnvState(
        obs={},
        reward=np.zeros(num_envs, dtype=np.float32),
        terminated=np.zeros(num_envs, dtype=bool),
        truncated=np.zeros(num_envs, dtype=bool),
        info=info,
    )
    return env, state


def test_action_delay_sequence_changes_the_applied_target() -> None:
    env, state = _action_only_env(np.asarray([0, 1]))
    action_0 = np.full((2, 16), 0.5, dtype=np.float32)
    action_1 = np.full((2, 16), -0.25, dtype=np.float32)

    target_0 = env.apply_action(action_0, state).copy()
    target_1 = env.apply_action(action_1, state).copy()
    target_2 = env.apply_action(np.zeros((2, 16), dtype=np.float32), state).copy()

    np.testing.assert_allclose(target_0[0], ACTION_SCALE * 0.5)
    np.testing.assert_allclose(target_1[0], ACTION_SCALE * (0.5 - 0.25))
    np.testing.assert_allclose(target_2[0], ACTION_SCALE * (0.5 - 0.25))
    np.testing.assert_allclose(target_0[1], 0.0)
    np.testing.assert_allclose(target_1[1], ACTION_SCALE * 0.5)
    np.testing.assert_allclose(target_2[1], ACTION_SCALE * (0.5 - 0.25))


def test_joint_noise_changes_only_observed_q_channels() -> None:
    env = object.__new__(LeapInhandBall0730HoraRotationEnv)
    env._ctrl_lower = np.full(16, -1.0, dtype=np.float32)
    env._ctrl_upper = np.full(16, 1.0, dtype=np.float32)
    env._hora_rotation_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    env._num_envs = 2
    env._cfg = SimpleNamespace(
        hora_domain_rand=LeapHoraDomainRandomizationConfig(),
        termination_drop_distance=0.03,
    )
    true_q = np.linspace(-0.5, 0.5, 32, dtype=np.float32).reshape(2, 16)
    true_q_copy = true_q.copy()
    target = np.linspace(-0.2, 0.2, 32, dtype=np.float32).reshape(2, 16)

    zeros_3 = np.zeros((2, 3), dtype=np.float32)
    zeros_16 = np.zeros((2, 16), dtype=np.float32)
    reward_info = {"init_pose": zeros_16.copy()}
    reward_before = env._reward_pose_diff(
        reward_info,
        true_q,
        zeros_16,
        zeros_3,
        zeros_3,
        zeros_3,
        zeros_16,
        np.zeros(2, dtype=bool),
    )
    env._state = NpEnvState(
        obs={},
        reward=np.zeros(2),
        terminated=np.zeros(2, dtype=bool),
        truncated=np.zeros(2, dtype=bool),
        info={"initial_ball_z": np.asarray([0.65, 0.66])},
    )
    ball_pos = np.asarray([[0.0, 0.0, 0.64], [0.0, 0.0, 0.62]])
    terminated_before = env._compute_terminated(ball_pos)

    np.random.seed(7)
    actor, proprio = env._build_observation_frames(true_q, target)
    observed_q = denormalize_from_joint_range(
        actor[:, :16], env._ctrl_lower, env._ctrl_upper
    )

    np.testing.assert_array_equal(true_q, true_q_copy)
    assert float(np.max(np.abs(observed_q - true_q))) <= 0.003 + 1e-7
    assert not np.array_equal(observed_q, true_q)
    np.testing.assert_allclose(actor[:, :16], proprio[:, :16])
    np.testing.assert_allclose(
        actor[:, 16:32],
        normalize_to_joint_range(target, env._ctrl_lower, env._ctrl_upper),
    )
    np.testing.assert_allclose(
        actor[:, 32:35], np.tile([0.0, 0.0, 1.0], (2, 1))
    )
    np.testing.assert_allclose(proprio[:, 16:32], actor[:, 16:32])
    reward_after = env._reward_pose_diff(
        reward_info,
        true_q,
        zeros_16,
        zeros_3,
        zeros_3,
        zeros_3,
        zeros_16,
        np.zeros(2, dtype=bool),
    )
    np.testing.assert_array_equal(reward_after, reward_before)
    np.testing.assert_array_equal(env._compute_terminated(ball_pos), terminated_before)


def test_delay_one_preserves_observation_timing_in_mujoco() -> None:
    overrides = [
        "env.hora_domain_rand.randomize_object_mass=false",
        "env.hora_domain_rand.randomize_object_friction=false",
        "env.hora_domain_rand.randomize_object_com=false",
        "env.hora_domain_rand.randomize_gravity_direction=false",
        "env.hora_domain_rand.joint_measurement_noise_rad=0.0",
        "env.hora_domain_rand.action_delay_min_steps=1",
        "env.hora_domain_rand.action_delay_max_steps=1",
    ]
    _, env = _create_hora_env(2, overrides)
    try:
        state = env.init_state()
        state.info["steps"][:] = 0
        initial_target = np.asarray(state.info["prev_ctrl"]).copy()
        action_0 = np.full((2, 16), 0.5, dtype=np.float32)
        action_1 = np.full((2, 16), -0.25, dtype=np.float32)

        state = env.step(action_0)
        np.testing.assert_allclose(state.info["prev_ctrl"], initial_target)
        frame_1 = state.obs["obs"].reshape(2, ACTOR_HISTORY_LEN, ACTOR_FRAME_DIM)[:, -1]
        np.testing.assert_allclose(
            frame_1[:, 16:32],
            normalize_to_joint_range(initial_target, env._ctrl_lower, env._ctrl_upper),
            atol=1e-6,
        )

        state = env.step(action_1)
        applied_target_1 = np.clip(
            initial_target + ACTION_SCALE * action_0,
            env._ctrl_lower,
            env._ctrl_upper,
        )
        np.testing.assert_allclose(state.info["prev_ctrl"], applied_target_1, atol=1e-6)
        frame_2 = state.obs["obs"].reshape(2, ACTOR_HISTORY_LEN, ACTOR_FRAME_DIM)[:, -1]
        np.testing.assert_allclose(
            frame_2[:, 16:32],
            normalize_to_joint_range(initial_target, env._ctrl_lower, env._ctrl_upper),
            atol=1e-6,
        )

        state = env.step(np.zeros((2, 16), dtype=np.float32))
        frame_3 = state.obs["obs"].reshape(2, ACTOR_HISTORY_LEN, ACTOR_FRAME_DIM)[:, -1]
        np.testing.assert_allclose(
            frame_3[:, 16:32],
            normalize_to_joint_range(applied_target_1, env._ctrl_lower, env._ctrl_upper),
            atol=1e-6,
        )
    finally:
        env.close()


def test_vectorized_subset_reset_only_replaces_selected_rows() -> None:
    np.random.seed(51)
    _, env = _create_hora_env(4)
    try:
        state = env.init_state()
        state.info["hora_action_queue"][:] = np.arange(
            state.info["hora_action_queue"].size, dtype=np.float32
        ).reshape(state.info["hora_action_queue"].shape)
        state.info["hora_actor_history"][:] += np.arange(4)[:, None, None]
        queue_before = state.info["hora_action_queue"].copy()
        history_before = state.info["hora_actor_history"].copy()
        critic_before = state.info["critic_info"].copy()

        state.terminated[:] = False
        state.truncated[:] = False
        state.terminated[[1, 3]] = True
        env._reset_done_envs()

        np.testing.assert_array_equal(
            state.info["hora_action_queue"][[0, 2]], queue_before[[0, 2]]
        )
        np.testing.assert_array_equal(
            state.info["hora_actor_history"][[0, 2]], history_before[[0, 2]]
        )
        np.testing.assert_array_equal(
            state.info["critic_info"][[0, 2]], critic_before[[0, 2]]
        )
        np.testing.assert_array_equal(state.info["hora_action_queue"][[1, 3]], 0.0)
        np.testing.assert_allclose(
            state.info["hora_actor_history"][[1, 3], 0],
            state.info["hora_actor_history"][[1, 3], -1],
        )
        assert not np.array_equal(
            state.info["critic_info"][[1, 3]], critic_before[[1, 3]]
        )
    finally:
        env.close()


def test_dr_enabled_four_env_random_action_smoke() -> None:
    np.random.seed(61)
    _, env = _create_hora_env(4)
    try:
        state = env.init_state()
        rng = np.random.default_rng(62)
        for _ in range(200):
            state = env.step(rng.uniform(-0.25, 0.25, size=(4, 16)).astype(np.float32))
            assert state.info["hora_action_queue"].shape == (4, 2, 16)
            assert state.info["proprio_hist"].shape == (4, 30, PROPRIO_FRAME_DIM)
            assert state.obs["obs"].shape == (4, ACTOR_HISTORY_LEN * ACTOR_FRAME_DIM)
            assert np.isfinite(state.obs["obs"]).all()
            assert np.isfinite(state.info["proprio_hist"]).all()
            assert np.isfinite(state.info["critic_info"]).all()
            assert np.isfinite(state.info["prev_ctrl"]).all()
            assert np.isfinite(state.reward).all()
    finally:
        env.close()
