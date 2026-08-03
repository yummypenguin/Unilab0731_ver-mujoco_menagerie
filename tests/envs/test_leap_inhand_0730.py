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
    LeapBall0730ResetProvider,
    LeapInhandBall0730RotationCfg,
    LeapInhandBall0730RotationEnv,
)
from unilab.training import BackendAdapter, create_env

ROOT = Path(__file__).parent.parent.parent
CONF_DIR = ROOT / "conf" / "ppo"


def _compose_cfg():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose("config", overrides=["task=leap_inhand_ball_0730/mujoco"])


def _termination_env(initial_ball_z: np.ndarray):
    env = object.__new__(LeapInhandBall0730RotationEnv)
    env._num_envs = len(initial_ball_z)
    env._cfg = SimpleNamespace(termination_drop_distance=0.005)
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
        "drop": -10.0,
    }
    assert cfg.reward.angvel_clip_min == pytest.approx(-0.5)
    assert cfg.reward.angvel_clip_max == pytest.approx(0.5)
    assert cfg.env.grasp_cache_path.endswith(
        "ball_grasp_allegro_new_physics_0731_50k.npy"
    )
    assert cfg.env.termination_drop_distance == pytest.approx(0.005)
    assert cfg.env.max_episode_seconds == pytest.approx(20.0)
    assert OmegaConf.select(cfg, "env.scene.joint_dynamics") is None
    owner = (
        ROOT / "conf" / "ppo" / "task" / "leap_inhand_ball_0730" / "mujoco.yaml"
    ).read_text(encoding="utf-8")
    assert "defaults:" not in owner
    assert "joint_dynamics:" not in owner


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
        "_compute_reward",
    ):
        assert reward_method not in LeapInhandBall0730RotationEnv.__dict__
        assert getattr(LeapInhandBall0730RotationEnv, reward_method) is getattr(
            AllegroRotationPPO, reward_method
        )


def test_0730_registry_is_mujoco_only() -> None:
    registered = registry.list_registered_envs()
    assert registered["LeapInhandBall0730Rotation"]["available_backends"] == ["mujoco"]


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
        _dof_mid=np.zeros(16),
        _dof_range=np.ones(16),
        _NUM_LAG_STEPS=3,
        _NUM_OBS_PER_STEP=35,
        _num_action=16,
    )

    updates = provider._build_info_updates(env, hand, ball_pos, ball_quat)

    np.testing.assert_allclose(updates["initial_ball_z"], [0.661, 0.674])
    assert not np.shares_memory(updates["initial_ball_z"], ball_pos)


def test_relative_drop_termination_uses_per_environment_initial_height() -> None:
    env = _termination_env(np.asarray([0.661, 0.674, 0.690]))
    threshold = np.asarray(env.state.info["initial_ball_z"]) - 0.005
    ball_pos = np.asarray(
        [
            [0.0, 0.0, threshold[0] - 0.0001],
            [0.0, 0.0, threshold[1]],
            [0.0, 0.0, threshold[2] + 0.0001],
        ]
    )

    terminated = env._compute_terminated(ball_pos)

    # First fell more than 5 mm; second is exactly at 5 mm; third fell less.
    np.testing.assert_array_equal(terminated, [True, True, False])


def test_relative_drop_anchor_changes_when_reset_info_changes() -> None:
    env = _termination_env(np.asarray([0.661]))
    ball_pos = np.asarray([[0.0, 0.0, 0.655]])
    assert env._compute_terminated(ball_pos)[0]

    env.state.info["initial_ball_z"][:] = 0.659
    assert not env._compute_terminated(ball_pos)[0]


@pytest.mark.parametrize("distance", [0.0, -0.001, np.inf, np.nan])
def test_0730_config_rejects_invalid_drop_distance(distance: float) -> None:
    cfg = LeapInhandBall0730RotationCfg(termination_drop_distance=distance)
    with pytest.raises(ValueError, match="termination_drop_distance"):
        cfg.validate()
