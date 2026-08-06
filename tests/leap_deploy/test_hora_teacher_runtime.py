from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from unilab.algos.torch.hora.models import (
    HoraActorModel,
    HoraCriticModel,
    HoraSharedActorCritic,
)
from unilab.algos.torch.hora.ppo import HoraPPO
from unilab.algos.torch.hora.rsl_rl import (
    HoraRslRlVecEnvWrapper,
    resolve_hora_ppo_runtime,
)
from unilab.algos.torch.rsl_rl_runtime import resolve_rsl_rl_ppo_runtime
from unilab.training import BackendAdapter, create_env
from unilab.training.rsl_rl import normalize_ppo_train_cfg

ROOT = Path(__file__).parent.parent.parent
CONF_DIR = ROOT / "conf" / "ppo"


def _compose_hora_cfg(overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(
            "config",
            overrides=["task=leap_inhand_ball_0730/mujoco_hora", *(overrides or [])],
        )


def _algo_cfg_dict(cfg) -> dict[str, Any]:
    value = OmegaConf.to_container(cfg.algo, resolve=True)
    assert isinstance(value, dict)
    return value


def _create_wrapper(
    num_envs: int,
    overrides: list[str] | None = None,
) -> tuple[Any, HoraRslRlVecEnvWrapper, dict[str, Any]]:
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
    wrapper = HoraRslRlVecEnvWrapper(env, device="cpu")
    return cfg, wrapper, _algo_cfg_dict(cfg)


def _small_train_cfg(
    rl_cfg: dict[str, Any],
    *,
    num_steps: int,
) -> dict[str, Any]:
    train_cfg = normalize_ppo_train_cfg(deepcopy(rl_cfg))
    train_cfg["num_steps_per_env"] = num_steps
    train_cfg["num_learning_epochs"] = 1
    train_cfg["num_mini_batches"] = 1
    train_cfg["save_interval"] = 1
    train_cfg["algorithm"]["num_learning_epochs"] = 1
    train_cfg["algorithm"]["num_mini_batches"] = 1
    train_cfg["multi_gpu"] = None
    train_cfg["logger"] = "tensorboard"
    train_cfg.setdefault("runner", {})["logger"] = "tensorboard"
    return train_cfg


def _all_tensor_values_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_all_tensor_values_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_tensor_values_finite(item) for item in value)
    return True


def test_hora_runtime_resolver_uses_real_owner_classes() -> None:
    cfg = _compose_hora_cfg()
    rl_cfg = _algo_cfg_dict(cfg)

    runtime = resolve_hora_ppo_runtime(rl_cfg)
    generic_runtime = resolve_rsl_rl_ppo_runtime(
        rl_cfg,
        default_wrapper_cls=object,  # type: ignore[arg-type]
    )

    assert runtime is not None
    assert runtime.wrapper_cls is HoraRslRlVecEnvWrapper
    assert generic_runtime.wrapper_cls is HoraRslRlVecEnvWrapper
    assert resolve_callable(rl_cfg["algorithm"]["class_name"]) is HoraPPO
    assert resolve_callable(rl_cfg["actor"]["class_name"]) is HoraActorModel
    assert resolve_callable(rl_cfg["critic"]["class_name"]) is HoraCriticModel


def test_hora_wrapper_reset_has_complete_finite_teacher_tensordict() -> None:
    _, wrapper, _ = _create_wrapper(4)
    try:
        obs, _ = wrapper.reset()

        assert obs["actor"].shape == (4, 105)
        assert obs["policy"].shape == (4, 105)
        assert obs["critic"].shape == (4, 105)
        assert obs["priv_info"].shape == (4, 9)
        assert obs["proprio_hist"].shape == (4, 30, 32)
        assert all(torch.isfinite(value).all() for value in obs.values())
    finally:
        wrapper.close()


def test_timeout_bootstrap_uses_terminal_episode_privileged_info() -> None:
    np.random.seed(401)
    _, wrapper, _ = _create_wrapper(4)
    try:
        assert wrapper.env.state is not None
        state = wrapper.env.state
        terminal_priv_info = np.asarray(state.info["critic_info"]).copy()
        state.info["steps"][:] = 0
        state.info["steps"][[0, 3]] = wrapper.env.cfg.max_episode_steps - 1

        ball_z = wrapper.env.get_ball_pos()[:, 2]
        state.info["initial_ball_z"][1] = ball_z[1] + 1.0

        _, _, dones, infos = wrapper.step(torch.zeros((4, 16)))

        torch.testing.assert_close(dones, torch.tensor([True, True, False, True]))
        torch.testing.assert_close(
            infos["time_outs"], torch.tensor([True, False, False, True])
        )
        bootstrap_obs = infos["time_out_bootstrap_obs"]
        assert isinstance(bootstrap_obs, TensorDict)
        assert bootstrap_obs["actor"].shape == (4, 105)
        assert bootstrap_obs["priv_info"].shape == (4, 9)
        np.testing.assert_allclose(
            bootstrap_obs["priv_info"].cpu().numpy(),
            terminal_priv_info,
            atol=0.0,
        )

        reset_priv_info = np.asarray(wrapper.env.state.info["critic_info"])
        assert not np.array_equal(reset_priv_info[[0, 1, 3]], terminal_priv_info[[0, 1, 3]])
        np.testing.assert_array_equal(reset_priv_info[2], terminal_priv_info[2])
    finally:
        wrapper.close()


def test_real_hora_model_construction_and_one_ppo_update() -> None:
    torch.manual_seed(402)
    np.random.seed(402)
    _, wrapper, rl_cfg = _create_wrapper(8)
    try:
        obs = wrapper.get_observations()
        train_cfg = _small_train_cfg(rl_cfg, num_steps=4)
        algorithm = HoraPPO.construct_algorithm(obs, wrapper, train_cfg, "cpu")
        assert isinstance(algorithm, HoraPPO)
        assert algorithm.actor.shared is algorithm.critic.shared
        assert isinstance(algorithm.actor.shared, HoraSharedActorCritic)

        with torch.inference_mode():
            actions = algorithm.actor(obs, stochastic_output=True)
            values = algorithm.critic(obs)
        assert actions.shape == (8, 16)
        assert values.shape == (8, 1)
        assert algorithm.actor.output_mean.shape == (8, 16)
        assert algorithm.actor.output_std.shape[-1] in (1, 16)
        assert torch.isfinite(algorithm.actor.output_mean).all()
        assert torch.isfinite(algorithm.actor.output_std).all()
        assert torch.all(algorithm.actor.output_std > 0)
        assert torch.isfinite(values).all()

        params_before = {
            name: parameter.detach().clone()
            for name, parameter in algorithm.actor.shared.named_parameters()
            if parameter.requires_grad
        }
        algorithm.train_mode()
        for _ in range(4):
            with torch.inference_mode():
                rollout_actions = algorithm.act(obs)
            obs, rewards, dones, extras = wrapper.step(rollout_actions)
            assert rewards.dtype.is_floating_point
            assert dones.dtype is torch.bool
            assert torch.isfinite(rewards).all()
            assert torch.isfinite(obs["priv_info"]).all()
            assert torch.isfinite(obs["proprio_hist"]).all()
            algorithm.process_env_step(obs, rewards, dones, extras)

        with torch.inference_mode():
            algorithm.compute_returns(obs)
            algorithm.actor(obs, stochastic_output=True)
            old_params = tuple(
                value.detach().clone()
                for value in algorithm.actor.output_distribution_params
            )
        loss_dict = algorithm.update()

        assert all(np.isfinite(float(value)) for value in loss_dict.values())
        assert any(
            not torch.equal(params_before[name], parameter.detach())
            for name, parameter in algorithm.actor.shared.named_parameters()
            if parameter.requires_grad
        )
        assert all(torch.isfinite(parameter).all() for parameter in algorithm.actor.parameters())
        assert all(torch.isfinite(parameter).all() for parameter in algorithm.critic.parameters())
        assert _all_tensor_values_finite(algorithm.optimizer.state_dict())

        with torch.inference_mode():
            algorithm.actor(obs, stochastic_output=True)
            new_params = algorithm.actor.output_distribution_params
            kl = algorithm.actor.get_kl_divergence(old_params, new_params)
        assert torch.isfinite(kl).all()
        assert torch.isfinite(algorithm.transition.values).all() if algorithm.transition.values is not None else True
    finally:
        wrapper.close()


def test_real_runner_checkpoint_save_reload_and_legacy_rejection(tmp_path: Path) -> None:
    torch.manual_seed(403)
    np.random.seed(403)
    _, wrapper, rl_cfg = _create_wrapper(4, ["env.hora_domain_rand.enabled=false"])
    runner = OnPolicyRunner(
        wrapper,
        _small_train_cfg(rl_cfg, num_steps=2),
        log_dir=str(tmp_path / "source"),
        device="cpu",
    )
    try:
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        checkpoint = tmp_path / "hora_teacher.pt"
        runner.save(str(checkpoint))
        fixed_obs = wrapper.get_observations().clone()
        with torch.inference_mode():
            expected_actions = runner.get_inference_policy("cpu")(fixed_obs).clone()

        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        assert {
            "actor_state_dict",
            "critic_state_dict",
            "optimizer_state_dict",
            "iter",
        }.issubset(saved)
        actor_keys = set(saved["actor_state_dict"])
        assert any("obs_normalizer" in key for key in actor_keys)
        assert any("distribution" in key for key in actor_keys)
        assert _all_tensor_values_finite(saved["optimizer_state_dict"])
    finally:
        wrapper.close()

    _, reloaded_wrapper, reloaded_rl_cfg = _create_wrapper(
        4, ["env.hora_domain_rand.enabled=false"]
    )
    reloaded_runner = OnPolicyRunner(
        reloaded_wrapper,
        _small_train_cfg(reloaded_rl_cfg, num_steps=2),
        log_dir=None,
        device="cpu",
    )
    try:
        reloaded_runner.load(str(checkpoint), map_location="cpu")
        with torch.inference_mode():
            actual_actions = reloaded_runner.get_inference_policy("cpu")(fixed_obs)
        max_abs_diff = torch.max(torch.abs(expected_actions - actual_actions)).item()
        assert max_abs_diff < 1e-6
        assert reloaded_runner.alg.actor.shared.obs_dim == 105
        assert reloaded_runner.alg.actor.shared.action_dim == 16
        assert reloaded_runner.alg.actor.shared.priv_info_dim == 9

        legacy_checkpoint = tmp_path / "legacy_105d_ppo.pt"
        legacy = deepcopy(saved)
        legacy["actor_state_dict"] = {
            "model.0.weight": torch.zeros((16, 105)),
            "model.0.bias": torch.zeros(16),
        }
        torch.save(legacy, legacy_checkpoint)
        with pytest.raises(RuntimeError):
            reloaded_runner.load(str(legacy_checkpoint), map_location="cpu")

        malformed_checkpoint = tmp_path / "malformed.pt"
        torch.save({"iter": 0}, malformed_checkpoint)
        with pytest.raises(KeyError):
            reloaded_runner.load(str(malformed_checkpoint), map_location="cpu")
    finally:
        reloaded_wrapper.close()
