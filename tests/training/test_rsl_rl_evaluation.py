"""Contracts for PPO batch instrumented evaluation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


def _load_train_rsl_rl() -> Any:
    path = Path(__file__).parents[2] / "scripts" / "train_rsl_rl.py"
    spec = importlib.util.spec_from_file_location("train_rsl_rl_evaluation", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def evaluation_module() -> Any:
    return _load_train_rsl_rl()


def test_aggregate_step_mean(evaluation_module: Any) -> None:
    rows = [{"metric": 0.2}, {"metric": 0.4}, {"metric": 0.6}]

    assert evaluation_module.aggregate_step_mean(rows, "metric") == pytest.approx(0.4)


def test_aggregate_count(evaluation_module: Any) -> None:
    rows = [
        {"timeout/READY_TO_A_count": 2.0},
        {"timeout/READY_TO_A_count": 5.0},
    ]

    assert evaluation_module.aggregate_count(
        rows, "timeout/READY_TO_A_count"
    ) == pytest.approx(7.0)


def test_aggregate_count_weighted_mean(evaluation_module: Any) -> None:
    rows = [
        {"metric": 0.1, "count": 9.0},
        {"metric": 0.9, "count": 1.0},
    ]

    result = evaluation_module.aggregate_count_weighted_mean(
        rows,
        metric_key="metric",
        count_key="count",
    )

    assert result == pytest.approx(0.18)
    assert result != pytest.approx(0.5)


def test_aggregate_count_weighted_mean_without_events_is_zero(
    evaluation_module: Any,
) -> None:
    rows = [
        {"metric": 0.1, "count": 0.0},
        {"metric": 0.9, "count": 0.0},
    ]

    result = evaluation_module.aggregate_count_weighted_mean(
        rows,
        metric_key="metric",
        count_key="count",
    )

    assert result == 0.0
    assert np.isfinite(result)


def test_summary_timeout_contributions_and_episode_statistics(
    evaluation_module: Any,
) -> None:
    rows = [
        {
            "timeout/READY_TO_A_count": 60.0,
            "timeout/A_TO_B_count": 40.0,
            "timeout/B_TO_READY_count": 0.0,
            "timeout/READY_TO_A_pose_distance_mean": 0.1,
            "timeout/A_TO_B_pose_distance_mean": 0.2,
            "timeout/B_TO_READY_pose_distance_mean": 0.0,
        }
    ]

    summary = evaluation_module.build_evaluation_summary(
        metric_rows=rows,
        completed_returns=[1.0, 2.0, 5.0],
        completed_lengths=[10, 20, 30],
        ctrl_dt=0.05,
    )

    assert summary["timeout/READY_TO_A_total_count"] == pytest.approx(60.0)
    assert summary["timeout/A_TO_B_total_count"] == pytest.approx(40.0)
    assert summary["timeout/B_TO_READY_total_count"] == pytest.approx(0.0)
    assert summary["timeout/READY_TO_A_contribution"] == pytest.approx(0.6)
    assert summary["timeout/A_TO_B_contribution"] == pytest.approx(0.4)
    assert summary["timeout/B_TO_READY_contribution"] == pytest.approx(0.0)
    assert summary["episode/count"] == 3
    assert summary["episode/return_mean"] == pytest.approx(8.0 / 3.0)
    assert summary["episode/return_p50"] == pytest.approx(2.0)
    assert summary["episode/return_p90"] == pytest.approx(4.4)
    assert summary["episode/length_mean"] == pytest.approx(20.0)
    assert summary["episode/length_p50"] == pytest.approx(20.0)
    assert summary["episode/length_p90"] == pytest.approx(28.0)
    assert summary["episode/duration_seconds_mean"] == pytest.approx(1.0)


def test_policy_state_hash_changes_only_when_tensor_changes(
    evaluation_module: Any,
) -> None:
    state = {
        "weight": torch.asarray([[1.0, 2.0]]),
        "bias": torch.asarray([3.0]),
    }
    same = {key: value.clone() for key, value in state.items()}
    changed = {key: value.clone() for key, value in state.items()}
    changed["weight"][0, 0] += 1.0

    original_hash = evaluation_module._state_dict_sha256(state)

    assert evaluation_module._state_dict_sha256(same) == original_hash
    assert evaluation_module._state_dict_sha256(changed) != original_hash


def test_evaluation_policy_actions_selects_mean_or_sample(
    evaluation_module: Any,
) -> None:
    calls: list[bool] = []

    def policy(obs: torch.Tensor, *, stochastic_output: bool) -> torch.Tensor:
        calls.append(stochastic_output)
        return obs

    obs = torch.zeros((2, 3))
    evaluation_module.evaluation_policy_actions(policy, obs, deterministic=True)
    evaluation_module.evaluation_policy_actions(policy, obs, deterministic=False)

    assert calls == [False, True]


def _evaluation_cfg(*, eval_only: bool, play_only: bool) -> Any:
    return OmegaConf.create(
        {
            "training": {"eval_only": eval_only, "play_only": play_only},
            "evaluation": {
                "num_envs": 64,
                "num_steps": 100,
                "deterministic": True,
                "write_tensorboard": True,
                "output_dir": None,
            },
        }
    )


def test_eval_and_play_are_mutually_exclusive(evaluation_module: Any) -> None:
    cfg = _evaluation_cfg(eval_only=True, play_only=True)

    with pytest.raises(ValueError, match="cannot both be true"):
        evaluation_module.validate_evaluation_config(cfg)


def test_evaluation_config_validation(evaluation_module: Any) -> None:
    evaluation_module.validate_evaluation_config(
        _evaluation_cfg(eval_only=True, play_only=False)
    )

    cfg = _evaluation_cfg(eval_only=True, play_only=False)
    cfg.evaluation.num_envs = 0
    with pytest.raises(ValueError, match="num_envs"):
        evaluation_module.validate_evaluation_config(cfg)


def test_checkpoint_config_recovery_preserves_evaluator_settings(
    evaluation_module: Any,
    tmp_path: Path,
) -> None:
    source_config = {
        "algo": {"load_run": "-1", "checkpoint": -1, "seed": 9},
        "training": {
            "eval_only": False,
            "play_only": False,
            "no_play": False,
        },
        "env": {
            "state_cycle": {
                "ready_to_a": {"timeout_seconds": 1.5},
            }
        },
    }
    (tmp_path / "run_config.json").write_text(
        json.dumps({"config": source_config}),
        encoding="utf-8",
    )
    target = OmegaConf.create(
        {
            "algo": {
                "load_run": str(tmp_path),
                "checkpoint": 299,
                "seed": 1,
            },
            "training": {
                "eval_only": True,
                "play_only": False,
                "no_play": True,
            },
            "evaluation": {
                "num_envs": 64,
                "num_steps": 100,
                "deterministic": False,
                "write_tensorboard": True,
                "output_dir": "eval-output",
            },
            "env": {
                "state_cycle": {
                    "ready_to_a": {"timeout_seconds": 2.0},
                }
            },
        }
    )

    recovered = evaluation_module.recover_evaluation_config(tmp_path, target)

    assert recovered.env.state_cycle.ready_to_a.timeout_seconds == pytest.approx(1.5)
    assert recovered.evaluation.num_envs == 64
    assert recovered.evaluation.num_steps == 100
    assert recovered.evaluation.deterministic is False
    assert recovered.evaluation.output_dir == "eval-output"
    assert recovered.training.eval_only is True
    assert recovered.training.play_only is False
    assert recovered.training.no_play is True
    assert recovered.algo.load_run == str(tmp_path)
    assert recovered.algo.checkpoint == 299
    assert recovered.algo.seed == 1


def test_state_cycle_observation_guard_rejects_mismatch(
    evaluation_module: Any,
) -> None:
    cfg = OmegaConf.create(
        {
            "training": {"task_name": "LeapInhandBallStateCycleRotation"},
            "env": {
                "ctrl_dt": 0.05,
                "termination_workspace_radius": 0.05,
                "state_cycle": {
                    "ready_to_a": {"timeout_seconds": 1.5},
                    "a_to_b": {"timeout_seconds": 2.0},
                    "b_to_ready": {"timeout_seconds": 1.3},
                },
            },
            "reward": {
                "pose_tracking_scale": 0.5,
                "rotation_progress_scale": 3.0,
                "rotation_target_axis_speed_rad_s": 0.5,
                "rotation_overspeed_scale": 1.0,
                "failure_penalty": 3.0,
                "failure_rotation_clawback_cap": 5.0,
            },
        }
    )

    with pytest.raises(AssertionError, match="dimension 142"):
        evaluation_module._assert_v4_state_cycle_evaluation_contract(cfg, 141)
