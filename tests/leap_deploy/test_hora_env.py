from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.envs.manipulation.allegro_inhand.rotation import AllegroRotationPPO
from unilab.envs.manipulation.leap_inhand.ball_rotation_0730 import (
    LeapInhandBall0730RotationEnv,
)
from unilab.envs.manipulation.leap_inhand.ball_rotation_0730_hora import (
    LeapInhandBall0730HoraRotationCfg,
    LeapInhandBall0730HoraRotationEnv,
)
from unilab.envs.manipulation.leap_inhand.deploy_contract import (
    ACTOR_FRAME_DIM,
    ACTOR_HISTORY_LEN,
    ACTOR_OBS_DIM,
    PRIV_INFO_DIM,
    PROPRIO_FRAME_DIM,
    PROPRIO_HISTORY_LEN,
    normalize_to_joint_range,
)
from unilab.envs.manipulation.leap_inhand.hora_domain_randomization import (
    LeapHoraDomainRandomizationConfig,
)
from unilab.training import BackendAdapter, create_env

ROOT = Path(__file__).parent.parent.parent
CONF_DIR = ROOT / "conf" / "ppo"


def _compose_hora_cfg(overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(
            "config",
            overrides=["task=leap_inhand_ball_0730/mujoco_hora", *(overrides or [])],
        )


def _create_hora_env(num_envs: int = 2):
    cfg = _compose_hora_cfg(["env.hora_domain_rand.enabled=false"])
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


def test_hora_owner_composes_the_deployable_contract() -> None:
    cfg = _compose_hora_cfg()

    assert cfg.training.task_name == "LeapInhandBall0730HoraRotation"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.env.ctrl_dt == pytest.approx(0.05)
    assert cfg.env.max_episode_seconds == pytest.approx(20.0)
    assert cfg.env.control_config.action_scale == pytest.approx(1.0 / 24.0)
    assert cfg.env.actor_history_len == ACTOR_HISTORY_LEN
    assert cfg.env.proprio_history_len == PROPRIO_HISTORY_LEN
    assert cfg.env.critic_info_dim == PRIV_INFO_DIM
    assert cfg.env.rotation_axis_command == [0.0, 0.0, 1.0]
    assert cfg.env.hora_domain_rand.enabled is True
    assert cfg.env.hora_domain_rand.object_mass_ratio_lower == pytest.approx(0.9)
    assert cfg.env.hora_domain_rand.object_mass_ratio_upper == pytest.approx(1.1)
    assert cfg.env.hora_domain_rand.object_friction_scale_lower == pytest.approx(0.8)
    assert cfg.env.hora_domain_rand.object_friction_scale_upper == pytest.approx(1.2)
    assert cfg.env.hora_domain_rand.gravity_tilt_max_deg == pytest.approx(3.0)
    assert cfg.env.hora_domain_rand.joint_measurement_noise_rad == pytest.approx(0.003)
    assert cfg.env.hora_domain_rand.action_delay_min_steps == 0
    assert cfg.env.hora_domain_rand.action_delay_max_steps == 1
    assert cfg.algo.runtime_impl == "hora_ppo"
    assert cfg.algo.actor.class_name == "unilab.algos.torch.hora:HoraActorModel"
    assert cfg.algo.critic.class_name == "unilab.algos.torch.hora:HoraCriticModel"
    assert cfg.algo.obs_groups.actor == ["actor"]
    assert cfg.algo.obs_groups.critic == ["actor"]


def test_hora_registry_and_inheritance_preserve_owner_task_semantics() -> None:
    registered = registry.list_registered_envs()
    assert registered["LeapInhandBall0730HoraRotation"]["available_backends"] == [
        "mujoco"
    ]
    assert issubclass(
        LeapInhandBall0730HoraRotationEnv, LeapInhandBall0730RotationEnv
    )
    for method_name in (
        "_compute_reward",
        "_reward_rotate",
        "_reward_obj_linvel",
        "_reward_pose_diff",
        "_reward_torque",
        "_reward_work",
        "_reward_drop",
        "_compute_terminated",
    ):
        assert method_name not in LeapInhandBall0730HoraRotationEnv.__dict__
    assert LeapInhandBall0730HoraRotationEnv._compute_reward is AllegroRotationPPO._compute_reward
    assert (
        LeapInhandBall0730HoraRotationEnv._compute_terminated
        is LeapInhandBall0730RotationEnv._compute_terminated
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("actor_history_len", 2),
        ("proprio_history_len", 29),
        ("critic_info_dim", 8),
        ("rotation_axis_command", (0.0, 0.0, 0.0)),
    ],
)
def test_hora_config_rejects_contract_drift(field: str, value: object) -> None:
    cfg = LeapInhandBall0730HoraRotationCfg(**{field: value})
    with pytest.raises(ValueError):
        cfg.validate()


def test_hora_actor_observation_does_not_depend_on_ball_state() -> None:
    env = object.__new__(LeapInhandBall0730HoraRotationEnv)
    env._ctrl_lower = np.full(16, -1.0, dtype=np.float32)
    env._ctrl_upper = np.full(16, 1.0, dtype=np.float32)
    env._hora_rotation_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    env._nominal_gravity_direction = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    env._cfg = SimpleNamespace(
        hora_domain_rand=LeapHoraDomainRandomizationConfig(enabled=False)
    )
    dof_pos = np.linspace(-0.5, 0.5, 32, dtype=np.float32).reshape(2, 16)
    target = np.linspace(-0.25, 0.25, 32, dtype=np.float32).reshape(2, 16)
    info = {"prev_ctrl": target, "observation_previous_target": target.copy()}

    obs_a = env._compute_obs(
        deepcopy(info), dof_pos, np.zeros((2, 3), dtype=np.float32)
    )
    obs_b = env._compute_obs(
        deepcopy(info), dof_pos, np.full((2, 3), 123.0, dtype=np.float32)
    )

    np.testing.assert_array_equal(obs_a["obs"], obs_b["obs"])


def test_hora_mujoco_reset_timing_and_smoke_rollout() -> None:
    cfg, env = _create_hora_env(num_envs=2)
    try:
        state = env.init_state()
        state.info["steps"][:] = 0
        initial_target = np.asarray(state.info["prev_ctrl"]).copy()

        assert set(state.obs) == {"obs"}
        assert state.obs["obs"].shape == (2, ACTOR_OBS_DIM)
        assert env.obs_groups_spec == {"obs": ACTOR_OBS_DIM}
        assert state.info["proprio_hist"].shape == (
            2,
            PROPRIO_HISTORY_LEN,
            PROPRIO_FRAME_DIM,
        )
        assert state.info["critic_info"].shape == (2, PRIV_INFO_DIM)
        assert state.info["hora_action_queue"].shape == (2, 2, 16)
        np.testing.assert_array_equal(state.info["hora_action_delay_steps"], 0)
        np.testing.assert_allclose(
            state.info["critic_info"],
            np.tile([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0], (2, 1)),
        )
        actor_history = state.obs["obs"].reshape(
            2, ACTOR_HISTORY_LEN, ACTOR_FRAME_DIM
        )
        np.testing.assert_allclose(actor_history[:, 0], actor_history[:, -1])
        np.testing.assert_allclose(
            state.info["proprio_hist"][:, 0], state.info["proprio_hist"][:, -1]
        )
        np.testing.assert_allclose(
            state.info["initial_ball_z"], env.get_ball_pos()[:, 2], atol=1e-6
        )

        action_t = np.full((2, 16), 0.5, dtype=np.float32)
        state = env.step(action_t)
        target_t = np.asarray(state.info["prev_ctrl"]).copy()
        actor_frame_t = state.obs["obs"].reshape(
            2, ACTOR_HISTORY_LEN, ACTOR_FRAME_DIM
        )[:, -1]
        np.testing.assert_allclose(
            actor_frame_t[:, 16:32],
            normalize_to_joint_range(initial_target, env._ctrl_lower, env._ctrl_upper),
            atol=1e-6,
        )
        assert not np.allclose(target_t, initial_target)

        state = env.step(np.zeros((2, 16), dtype=np.float32))
        actor_frame_next = state.obs["obs"].reshape(
            2, ACTOR_HISTORY_LEN, ACTOR_FRAME_DIM
        )[:, -1]
        np.testing.assert_allclose(
            actor_frame_next[:, 16:32],
            normalize_to_joint_range(target_t, env._ctrl_lower, env._ctrl_upper),
            atol=1e-6,
        )

        for _ in range(198):
            state = env.step(np.zeros((2, 16), dtype=np.float32))
            assert set(state.obs) == {"obs"}
            assert state.obs["obs"].shape == (2, ACTOR_OBS_DIM)
            assert state.info["proprio_hist"].shape == (
                2,
                PROPRIO_HISTORY_LEN,
                PROPRIO_FRAME_DIM,
            )
            assert np.isfinite(state.obs["obs"]).all()
            assert np.isfinite(state.info["critic_info"]).all()
            assert np.isfinite(state.info["proprio_hist"]).all()
            assert np.isfinite(state.reward).all()
    finally:
        env.close()


def test_hora_owner_retains_0730_reward_and_reset_configuration() -> None:
    cfg = _compose_hora_cfg()
    assert cfg.reward.scales == {
        "rotate": 1.25,
        "obj_linvel": -0.3,
        "pose_diff": -0.3,
        "torque": -0.1,
        "work": -2.0,
        "drop": 0.0,
    }
    assert cfg.env.grasp_cache_path.endswith(
        "ball_grasp_allegro_new_physics_0731_50k.npy"
    )
    assert cfg.env.termination_drop_distance == pytest.approx(0.03)
