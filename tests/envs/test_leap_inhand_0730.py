from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.envs.manipulation.allegro_inhand.rotation import (
    AllegroRotationPPO,
)
from unilab.envs.manipulation.leap_inhand.ball_rotation import (
    LeapInhandBallRotationCfg,
    LeapInhandBallRotationEnv,
)
from unilab.envs.manipulation.leap_inhand.ball_rotation_0730 import (
    Leap0730RewardConfig,
    LeapBall0730ResetProvider,
    LeapInhandBall0730RotationCfg,
    LeapInhandBall0730RotationEnv,
    apply_middle_contact_rotation_share,
)
from unilab.training import BackendAdapter, create_env

ROOT = Path(__file__).parent.parent.parent
CONF_DIR = ROOT / "conf" / "ppo"
APPO_CONF_DIR = ROOT / "conf" / "appo"


def _compose_cfg():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose("config", overrides=["task=leap_inhand_ball_0730/mujoco"])


def _compose_appo_cfg():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(APPO_CONF_DIR), version_base="1.3"):
        return compose("config", overrides=["task=leap_inhand_ball_0730/mujoco"])


def _termination_env(initial_ball_z: np.ndarray):
    env = object.__new__(LeapInhandBall0730RotationEnv)
    env._num_envs = len(initial_ball_z)
    env._cfg = SimpleNamespace(termination_drop_distance=0.03)
    env._state = NpEnvState(
        obs={},
        reward=np.zeros(len(initial_ball_z)),
        terminated=np.zeros(len(initial_ball_z), dtype=bool),
        truncated=np.zeros(len(initial_ball_z), dtype=bool),
        info={"initial_ball_z": np.asarray(initial_ball_z).copy()},
    )
    return env


def test_0730_owner_is_independent_and_matches_allegro_reward_contract() -> None:
    cfg = _compose_cfg()

    assert cfg.training.task_name == "LeapInhandBall0730Rotation"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.reward.scales == {
        "rotate": 1.25,
        "obj_linvel": -0.3,
        "pose_diff": -0.3,
        "torque": -0.1,
        "work": -2.0,
        "drop": 0.0,
    }
    assert cfg.reward.angvel_clip_min == pytest.approx(-0.5)
    assert cfg.reward.angvel_clip_max == pytest.approx(0.5)
    assert cfg.reward.middle_contact_rotation_fraction == pytest.approx(0.2)
    assert cfg.env.grasp_cache_path.endswith(
        "ball_grasp_allegro_new_physics_0731_50k.npy"
    )
    np.testing.assert_allclose(
        cfg.env.pose_diff_target_qpos,
        [
            1.5152045040427635,
            0.11430147259750476,
            0.2876406730815961,
            0.19280835997306603,
            1.4188457206477074,
            0.025681830807677088,
            -0.26717932336688344,
            0.5369823550831088,
            1.5294890485315962,
            -0.01798386011739139,
            0.27558019211759954,
            0.19821762108233876,
            1.9245445859343515,
            0.04788276935232176,
            -0.021885380331691334,
            0.19524630120127295,
        ],
    )
    assert cfg.env.termination_drop_distance == pytest.approx(0.03)
    assert cfg.env.max_episode_seconds == pytest.approx(20.0)
    assert OmegaConf.select(cfg, "env.scene.joint_dynamics") is None
    owner = (
        ROOT / "conf" / "ppo" / "task" / "leap_inhand_ball_0730" / "mujoco.yaml"
    ).read_text(encoding="utf-8")
    assert "defaults:" not in owner
    assert "joint_dynamics:" not in owner


def test_0730_appo_owner_preserves_task_contract_and_materializes_env() -> None:
    ppo_cfg = _compose_cfg()
    appo_cfg = _compose_appo_cfg()

    assert appo_cfg.training.task_name == "LeapInhandBall0730Rotation"
    assert appo_cfg.training.sim_backend == "mujoco"
    assert appo_cfg.algo.algo == "appo"
    assert appo_cfg.algo.num_envs == 2048
    assert appo_cfg.algo.steps_per_env == 32
    assert appo_cfg.algo.max_iterations == 5000
    assert appo_cfg.algo.actor.obs_normalization is True
    assert appo_cfg.algo.critic.obs_normalization is True
    assert appo_cfg.algo.algorithm.learning_rate == pytest.approx(5e-4)
    assert appo_cfg.algo.algorithm.entropy_coef == pytest.approx(0.0)
    assert appo_cfg.training.replay_queue_size == 1
    assert appo_cfg.training.device == "cpu"
    assert appo_cfg.training.collector_device == "cpu"
    assert appo_cfg.training.no_play is True
    assert OmegaConf.to_container(appo_cfg.reward, resolve=True) == OmegaConf.to_container(
        ppo_cfg.reward, resolve=True
    )
    assert appo_cfg.env.grasp_cache_path == ppo_cfg.env.grasp_cache_path
    np.testing.assert_allclose(
        appo_cfg.env.pose_diff_target_qpos,
        ppo_cfg.env.pose_diff_target_qpos,
    )
    assert appo_cfg.env.termination_drop_distance == pytest.approx(0.03)
    assert appo_cfg.env.control_config.action_scale == pytest.approx(1.0 / 24.0)
    assert OmegaConf.select(appo_cfg, "env.scene.joint_dynamics") is None

    owner_path = (
        ROOT / "conf" / "appo" / "task" / "leap_inhand_ball_0730" / "mujoco.yaml"
    )
    owner = owner_path.read_text(encoding="utf-8")
    assert "defaults:" not in owner
    assert "joint_dynamics:" not in owner
    assert "\n    kp:" not in owner
    assert "\n    kd:" not in owner

    env_override = BackendAdapter(
        appo_cfg,
        root_dir=ROOT,
        algo_name="appo",
    ).build_task_env_cfg_override()
    env = create_env(
        appo_cfg,
        num_envs=2,
        env_cfg_override=env_override,
        sim_backend="mujoco",
        task_name="LeapInhandBall0730Rotation",
    )
    try:
        state = env.init_state()

        assert isinstance(state.obs, dict)
        assert env.action_space.shape == (16,)
        assert env.obs_groups_spec == {"obs": 105}
        np.testing.assert_allclose(
            state.info["initial_ball_z"],
            env.get_ball_pos()[:, 2],
            atol=1e-6,
        )
        state = env.step(np.zeros((2, 16), dtype=np.float32))
        assert np.isfinite(state.reward).all()
        assert 0.0 <= state.info["log"]["contact/middle_rate"] <= 1.0
        assert np.isfinite(
            state.info["log"]["reward/middle_contact_rotation_adjustment"]
        )
    finally:
        env.close()


def test_0730_task_is_not_coupled_to_existing_leap_rotation_task() -> None:
    assert not issubclass(LeapInhandBall0730RotationCfg, LeapInhandBallRotationCfg)
    assert not issubclass(LeapInhandBall0730RotationEnv, LeapInhandBallRotationEnv)
    for reward_method in (
        "_reward_rotate",
        "_reward_obj_linvel",
        "_reward_pose_diff",
        "_reward_torque",
        "_reward_work",
        "_reward_drop",
    ):
        assert reward_method not in LeapInhandBall0730RotationEnv.__dict__
        assert getattr(LeapInhandBall0730RotationEnv, reward_method) is getattr(
            AllegroRotationPPO, reward_method
        )
    assert LeapInhandBall0730RotationEnv._compute_reward is not AllegroRotationPPO._compute_reward


def test_middle_contact_is_required_for_only_part_of_positive_rotation_reward() -> None:
    weighted_rotate = np.asarray([-0.625, 0.0, 0.625, 0.625], dtype=np.float32)
    middle_contact = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    adjusted, adjustment = apply_middle_contact_rotation_share(
        weighted_rotate,
        middle_contact,
        fraction=0.2,
    )

    np.testing.assert_allclose(adjusted, [-0.625, 0.0, 0.5, 0.625])
    np.testing.assert_allclose(adjustment, [0.0, 0.0, -0.125, 0.0])


def test_0730_registry_is_mujoco_only() -> None:
    registered = registry.list_registered_envs()
    assert registered["LeapInhandBall0730Rotation"]["available_backends"] == ["mujoco"]


def test_0730_enables_training_episode_diagnostics() -> None:
    assert LeapInhandBall0730RotationEnv.enable_training_episode_diagnostics is True


def test_0730_mujoco_reset_records_cache_relative_drop_anchor() -> None:
    cfg = _compose_cfg()
    env_override = BackendAdapter(cfg, root_dir=ROOT, algo_name="ppo").build_task_env_cfg_override()
    env = create_env(
        cfg,
        num_envs=2,
        env_cfg_override=env_override,
        sim_backend="mujoco",
        task_name="LeapInhandBall0730Rotation",
    )
    try:
        state = env.init_state()
        initial_ball_z = np.asarray(state.info["initial_ball_z"])

        assert initial_ball_z.shape == (2,)
        np.testing.assert_allclose(initial_ball_z, env.get_ball_pos()[:, 2], atol=1e-6)
        assert env.cfg.max_episode_steps == 400
        assert env.cfg.grasp_cache_path.endswith(
            "ball_grasp_allegro_new_physics_0731_50k.npy"
        )
    finally:
        env.close()


def test_reset_provider_records_each_cache_rows_initial_ball_height() -> None:
    provider = LeapBall0730ResetProvider()
    hand = np.zeros((2, 16), dtype=np.float64)
    ball_pos = np.asarray([[0.0, 0.0, 0.661], [0.0, 0.0, 0.674]])
    ball_quat = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            pose_diff_target_qpos=[0.25] * 16,
        ),
        _dof_mid=np.zeros(16),
        _dof_range=np.ones(16),
        _NUM_LAG_STEPS=3,
        _NUM_OBS_PER_STEP=35,
        _num_action=16,
    )

    updates = provider._build_info_updates(env, hand, ball_pos, ball_quat)

    np.testing.assert_allclose(updates["initial_ball_z"], [0.661, 0.674])
    np.testing.assert_allclose(updates["init_pose"], np.full((2, 16), 0.25))
    np.testing.assert_allclose(updates["prev_ctrl"], hand)
    assert not np.shares_memory(updates["initial_ball_z"], ball_pos)


def test_relative_drop_termination_uses_per_environment_initial_height() -> None:
    env = _termination_env(np.asarray([0.661, 0.674, 0.690]))
    threshold = np.asarray(env.state.info["initial_ball_z"]) - 0.03
    ball_pos = np.asarray(
        [
            [0.0, 0.0, threshold[0] - 0.0001],
            [0.0, 0.0, threshold[1]],
            [0.0, 0.0, threshold[2] + 0.0001],
        ]
    )

    terminated = env._compute_terminated(ball_pos)

    # First fell more than 30 mm; second is exactly at 30 mm; third fell less.
    np.testing.assert_array_equal(terminated, [True, True, False])


def test_relative_drop_anchor_changes_when_reset_info_changes() -> None:
    env = _termination_env(np.asarray([0.661]))
    ball_pos = np.asarray([[0.0, 0.0, 0.630]])
    assert env._compute_terminated(ball_pos)[0]

    env.state.info["initial_ball_z"][:] = 0.659
    assert not env._compute_terminated(ball_pos)[0]


@pytest.mark.parametrize("distance", [0.0, -0.001, np.inf, np.nan])
def test_0730_config_rejects_invalid_drop_distance(distance: float) -> None:
    cfg = LeapInhandBall0730RotationCfg(termination_drop_distance=distance)
    with pytest.raises(ValueError, match="termination_drop_distance"):
        cfg.validate()


@pytest.mark.parametrize("fraction", [-0.01, 1.01, np.inf, np.nan])
def test_0730_config_rejects_invalid_middle_contact_rotation_fraction(
    fraction: float,
) -> None:
    reward = Leap0730RewardConfig(
        scales={"rotate": 1.25},
        angvel_clip_min=-0.5,
        angvel_clip_max=0.5,
        reset_z_threshold=0.0,
        middle_contact_rotation_fraction=fraction,
    )
    cfg = LeapInhandBall0730RotationCfg(reward_config=reward)

    with pytest.raises(ValueError, match="middle_contact_rotation_fraction"):
        cfg.validate()
