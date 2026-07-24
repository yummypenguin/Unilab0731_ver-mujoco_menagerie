from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from unilab.training.rsl_rl import load_rsl_rl_training_checkpoint


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.obs_normalizer = nn.Linear(2, 2, bias=False)
        self.distribution = nn.Linear(1, 1, bias=False)
        self.mlp = nn.Linear(2, 1)


class _Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.obs_normalizer = nn.Linear(2, 2, bias=False)
        self.mlp = nn.Linear(2, 1)


class _Algorithm:
    def __init__(self, policy: _Policy, critic: _Critic | None = None) -> None:
        self.policy = policy
        self.critic = critic if critic is not None else _Critic()

    def get_policy(self) -> _Policy:
        return self.policy


class _Runner:
    def __init__(self, policy: _Policy, critic: _Critic | None = None) -> None:
        self.alg = _Algorithm(policy, critic)
        self.load_calls: list[tuple[str, str | None]] = []

    def load(self, path: str, *, map_location: str | None = None) -> None:
        self.load_calls.append((path, map_location))


def test_resume_delegates_to_runner_load() -> None:
    runner = _Runner(_Policy())

    load_rsl_rl_training_checkpoint(runner, "model.pt", map_location="cpu")

    assert runner.load_calls == [("model.pt", "cpu")]


def test_policy_warm_start_excludes_distribution_and_training_state(tmp_path) -> None:
    source = _Policy()
    target = _Policy()
    with torch.no_grad():
        source.obs_normalizer.weight.fill_(2.0)
        source.distribution.weight.fill_(0.09)
        source.mlp.weight.fill_(3.0)
        source.mlp.bias.fill_(4.0)
        target.obs_normalizer.weight.fill_(-1.0)
        target.distribution.weight.fill_(0.30)
        target.mlp.weight.fill_(-1.0)
        target.mlp.bias.fill_(-1.0)

    checkpoint_path = tmp_path / "model_1150.pt"
    torch.save(
        {
            "actor_state_dict": source.state_dict(),
            "critic_state_dict": {"sentinel": torch.tensor(1.0)},
            "optimizer_state_dict": {"sentinel": 1},
            "iter": 1150,
        },
        checkpoint_path,
    )
    runner = _Runner(target)
    runner.current_learning_iteration = 0
    runner.logger = SimpleNamespace(tot_timesteps=0)

    load_rsl_rl_training_checkpoint(
        runner,
        str(checkpoint_path),
        load_mode="warm_start_policy",
        map_location="cpu",
    )

    assert torch.all(target.obs_normalizer.weight == 2.0)
    assert torch.all(target.mlp.weight == 3.0)
    assert torch.all(target.mlp.bias == 4.0)
    assert torch.all(target.distribution.weight == 0.30)
    assert runner.current_learning_iteration == 0
    assert runner.logger.tot_timesteps == 0
    assert runner.load_calls == []


def test_policy_warm_start_rejects_non_distribution_mismatch(tmp_path) -> None:
    checkpoint_path = tmp_path / "bad.pt"
    torch.save({"actor_state_dict": {"distribution.weight": torch.ones(1, 1)}}, checkpoint_path)

    with pytest.raises(RuntimeError, match="incompatible"):
        load_rsl_rl_training_checkpoint(
            _Runner(_Policy()),
            str(checkpoint_path),
            load_mode="warm_start_policy",
        )


def test_actor_critic_warm_start_transfers_models_but_not_training_state(tmp_path) -> None:
    source_actor = _Policy()
    source_critic = _Critic()
    target_actor = _Policy()
    target_critic = _Critic()
    with torch.no_grad():
        source_actor.obs_normalizer.weight.fill_(2.0)
        source_actor.distribution.weight.fill_(0.09)
        source_actor.mlp.weight.fill_(3.0)
        source_actor.mlp.bias.fill_(4.0)
        source_critic.obs_normalizer.weight.fill_(5.0)
        source_critic.mlp.weight.fill_(6.0)
        source_critic.mlp.bias.fill_(7.0)
        target_actor.obs_normalizer.weight.fill_(-1.0)
        target_actor.distribution.weight.fill_(0.125)
        target_actor.mlp.weight.fill_(-1.0)
        target_actor.mlp.bias.fill_(-1.0)
        target_critic.obs_normalizer.weight.fill_(-1.0)
        target_critic.mlp.weight.fill_(-1.0)
        target_critic.mlp.bias.fill_(-1.0)

    checkpoint_path = tmp_path / "model_250.pt"
    torch.save(
        {
            "actor_state_dict": source_actor.state_dict(),
            "critic_state_dict": source_critic.state_dict(),
            "optimizer_state_dict": {"sentinel": 1},
            "iter": 250,
        },
        checkpoint_path,
    )
    runner = _Runner(target_actor, target_critic)
    runner.current_learning_iteration = 0
    runner.logger = SimpleNamespace(tot_timesteps=0)

    load_rsl_rl_training_checkpoint(
        runner,
        str(checkpoint_path),
        load_mode="warm_start_actor_critic",
        map_location="cpu",
    )

    assert torch.all(target_actor.obs_normalizer.weight == 2.0)
    assert torch.all(target_actor.mlp.weight == 3.0)
    assert torch.all(target_actor.mlp.bias == 4.0)
    assert torch.all(target_actor.distribution.weight == 0.125)
    assert torch.all(target_critic.obs_normalizer.weight == 5.0)
    assert torch.all(target_critic.mlp.weight == 6.0)
    assert torch.all(target_critic.mlp.bias == 7.0)
    assert runner.current_learning_iteration == 0
    assert runner.logger.tot_timesteps == 0
    assert runner.load_calls == []


def test_actor_critic_warm_start_requires_compatible_critic(tmp_path) -> None:
    checkpoint_path = tmp_path / "bad_critic.pt"
    torch.save(
        {
            "actor_state_dict": _Policy().state_dict(),
            "critic_state_dict": {"sentinel": torch.tensor(1.0)},
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="Critic warm-start checkpoint is incompatible"):
        load_rsl_rl_training_checkpoint(
            _Runner(_Policy()),
            str(checkpoint_path),
            load_mode="warm_start_actor_critic",
        )


def test_checkpoint_load_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported RSL-RL checkpoint load_mode"):
        load_rsl_rl_training_checkpoint(_Runner(_Policy()), "model.pt", load_mode="weights")
