from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper
from unilab.base.np_env import NpEnvState
from unilab.envs.manipulation.leap_inhand.ball_rotation_0730_hora import (
    LeapInhandBall0730HoraRotationCfg,
    LeapInhandBall0730HoraRotationEnv,
)
from unilab.envs.manipulation.leap_inhand.hora_diagnostics import (
    CONTROL_DIAGNOSTIC_KEYS,
    ROTATION_DIAGNOSTIC_KEYS,
    compute_control_diagnostics,
    compute_rotation_diagnostics,
)
from unilab.training import BackendAdapter, create_env

ROOT = Path(__file__).parent.parent.parent
CONF_DIR = ROOT / "conf" / "ppo"
ALL_DIAGNOSTIC_KEYS = {
    *ROTATION_DIAGNOSTIC_KEYS,
    *CONTROL_DIAGNOSTIC_KEYS,
    "termination/rate",
    "termination/drop_count",
    "termination/truncation_count",
    "episode/length_seconds",
    "episode/full_length_seconds",
}


def _create_hora_wrapper(num_envs: int) -> HoraRslRlVecEnvWrapper:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=["task=leap_inhand_ball_0730/mujoco_hora"])
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
    return HoraRslRlVecEnvWrapper(env, device="cpu")


@pytest.mark.parametrize(
    ("command_axis", "reward_axis"),
    [
        ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        ((0.0, 0.0, 5.0), (0.0, 0.0, 2.0)),
    ],
)
def test_rotation_command_axis_accepts_matching_normalized_axis(
    command_axis: tuple[float, float, float],
    reward_axis: tuple[float, float, float],
) -> None:
    cfg = LeapInhandBall0730HoraRotationCfg(
        rotation_axis_command=command_axis,
        rotation_axis=reward_axis,
    )
    cfg.validate()


@pytest.mark.parametrize(
    ("command_axis", "reward_axis"),
    [
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
    ],
)
def test_rotation_command_axis_rejects_mismatch_or_zero(
    command_axis: tuple[float, float, float],
    reward_axis: tuple[float, float, float],
) -> None:
    cfg = LeapInhandBall0730HoraRotationCfg(
        rotation_axis_command=command_axis,
        rotation_axis=reward_axis,
    )
    with pytest.raises(
        ValueError, match=r"rotation_axis_command.*rotation_axis"
    ):
        cfg.validate()


def test_rotation_diagnostics_cover_direction_zero_and_raw_clip_edges() -> None:
    metrics = compute_rotation_diagnostics(
        np.asarray(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, -3.0],
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.25],
            ]
        ),
        np.asarray([0.0, 0.0, 1.0]),
        clip_min=-0.5,
        clip_max=0.5,
    )

    assert metrics["rotation/axis_speed_mean"] == pytest.approx(-0.1875)
    assert metrics["rotation/axis_speed_abs_mean"] == pytest.approx(1.3125)
    assert metrics["rotation/positive_rate"] == pytest.approx(0.5)
    assert metrics["rotation/reverse_rate"] == pytest.approx(0.25)
    assert metrics["rotation/high_clip_rate"] == pytest.approx(0.25)
    assert metrics["rotation/low_clip_rate"] == pytest.approx(0.25)


def test_control_diagnostics_cover_action_and_both_target_limits() -> None:
    metrics = compute_control_diagnostics(
        np.asarray([[1.0, -1.25, 0.25], [0.5, 0.0, -0.75]]),
        np.asarray([[1.0, -1.0, 0.0], [0.0, 0.25, -0.5]]),
        np.asarray([-1.0, -1.0, -1.0]),
        np.asarray([1.0, 1.0, 1.0]),
    )

    assert metrics["control/action_abs_mean"] == pytest.approx(0.625)
    assert metrics["control/action_saturation_rate"] == pytest.approx(2.0 / 6.0)
    assert metrics["control/target_lower_saturation_rate"] == pytest.approx(1.0 / 6.0)
    assert metrics["control/target_upper_saturation_rate"] == pytest.approx(1.0 / 6.0)
    assert metrics["control/target_saturation_rate"] == pytest.approx(2.0 / 6.0)


def test_delay_one_diagnostics_use_applied_previous_raw_action() -> None:
    env = object.__new__(LeapInhandBall0730HoraRotationEnv)
    env._np_dtype = np.dtype(np.float32)
    env._num_action = 2
    env._ctrl_lower = np.full(2, -1.0, dtype=np.float32)
    env._ctrl_upper = np.full(2, 1.0, dtype=np.float32)
    env.default_angles = np.zeros(2, dtype=np.float32)
    env._cfg = SimpleNamespace(
        control_config=SimpleNamespace(action_scale=0.5)
    )
    state = NpEnvState(
        obs={},
        reward=np.zeros(1, dtype=np.float32),
        terminated=np.zeros(1, dtype=bool),
        truncated=np.zeros(1, dtype=bool),
        info={
            "prev_ctrl": np.zeros((1, 2), dtype=np.float32),
            "current_actions": np.zeros((1, 2), dtype=np.float32),
            "hora_action_delay_steps": np.ones(1, dtype=np.int32),
            "hora_action_queue": np.zeros((1, 2, 2), dtype=np.float32),
        },
    )

    env.apply_action(np.asarray([[1.2, -0.5]], dtype=np.float32), state)
    np.testing.assert_array_equal(state.info["hora_applied_raw_action"], 0.0)
    env.apply_action(np.asarray([[-0.25, 0.25]], dtype=np.float32), state)

    np.testing.assert_allclose(
        state.info["hora_applied_raw_action"], [[1.2, -0.5]]
    )
    np.testing.assert_allclose(state.info["hora_applied_target"], [[0.5, -0.25]])


class _FixedDoneHoraEnv:
    enable_training_episode_diagnostics = True
    episode_static_critic_info = True

    def __init__(self) -> None:
        self.num_envs = 3
        self.cfg = SimpleNamespace(max_episode_seconds=2.0, ctrl_dt=0.5)
        self.observation_space = SimpleNamespace(shape=(105,))
        self.action_space = SimpleNamespace(shape=(16,))
        self.obs_groups_spec = {"obs": 105}
        self._step_index = 0
        self.state = self._make_state(np.zeros(3, dtype=bool), np.zeros(3, dtype=bool))

    @staticmethod
    def _info() -> dict[str, object]:
        return {
            "critic_info": np.zeros((3, 9), dtype=np.float32),
            "proprio_hist": np.zeros((3, 30, 32), dtype=np.float32),
            "log": {},
        }

    def _make_state(self, terminated: np.ndarray, truncated: np.ndarray) -> NpEnvState:
        return NpEnvState(
            obs={"obs": np.zeros((3, 105), dtype=np.float32)},
            reward=np.zeros(3, dtype=np.float32),
            terminated=terminated,
            truncated=truncated,
            info=self._info(),
        )

    def init_state(self) -> NpEnvState:
        return self.state

    def reset(self, env_indices: np.ndarray):
        del env_indices
        return self.state.obs, self.state.info

    def step(self, actions: np.ndarray) -> NpEnvState:
        del actions
        if self._step_index == 0:
            self.state = self._make_state(
                np.asarray([True, False, False]),
                np.asarray([False, True, False]),
            )
        else:
            self.state = self._make_state(
                np.zeros(3, dtype=bool), np.zeros(3, dtype=bool)
            )
        self._step_index += 1
        return self.state

    def close(self) -> None:
        pass


def test_terminated_and_truncated_counts_are_separate_and_not_repeated() -> None:
    wrapper = HoraRslRlVecEnvWrapper(_FixedDoneHoraEnv(), device="cpu")
    try:
        _, _, _, first_infos = wrapper.step(torch.zeros((3, 16)))
        assert first_infos["log"]["termination/drop_count"] == 1.0
        assert first_infos["log"]["termination/truncation_count"] == 1.0
        assert first_infos["log"]["termination/rate"] == pytest.approx(0.5)

        _, _, _, second_infos = wrapper.step(torch.zeros((3, 16)))
        assert second_infos["log"]["termination/drop_count"] == 1.0
        assert second_infos["log"]["termination/truncation_count"] == 1.0
    finally:
        wrapper.close()


def test_real_mujoco_diagnostics_smoke_is_complete_and_finite() -> None:
    np.random.seed(704)
    torch.manual_seed(704)
    wrapper = _create_hora_wrapper(4)
    seen: set[str] = set()
    try:
        rng = np.random.default_rng(705)
        for _ in range(100):
            actions = torch.from_numpy(
                rng.uniform(-1.25, 1.25, size=(4, 16)).astype(np.float32)
            )
            _, _, _, infos = wrapper.step(actions)
            log = infos.get("log", {})
            seen.update(log)
            for key, value in log.items():
                if key in ALL_DIAGNOSTIC_KEYS:
                    assert np.isfinite(value), key
                    if key.endswith("_rate") or key == "termination/rate":
                        assert 0.0 <= value <= 1.0, key
        assert ALL_DIAGNOSTIC_KEYS <= seen
    finally:
        wrapper.close()
