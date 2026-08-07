from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from rsl_rl.runners import OnPolicyRunner
from scripts.evaluate_leap_hora_teacher import (
    METRIC_FIELDS,
    assign_checkpoint_ranks,
    config_for_suite,
    evaluate_checkpoint,
    evaluate_run,
    load_run_config,
    resolve_checkpoints,
    validate_checkpoint,
)

from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper
from unilab.training import BackendAdapter, create_env, ensure_registries
from unilab.training.rsl_rl import normalize_ppo_train_cfg

ROOT = Path(__file__).parent.parent.parent
CONF_DIR = ROOT / "conf" / "ppo"


def _compose_hora_cfg() -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(
            "config", overrides=["task=leap_inhand_ball_0730/mujoco_hora"]
        )


def _write_run_config(run_dir: Path, cfg: DictConfig) -> None:
    payload = {"config": OmegaConf.to_container(cfg, resolve=True)}
    (run_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _dummy_checkpoint(path: Path, iteration: int) -> None:
    torch.save(
        {
            "actor_state_dict": {"shared.weight": torch.tensor([float(iteration)])},
            "critic_state_dict": {"shared.weight": torch.tensor([float(iteration)])},
            "optimizer_state_dict": {},
            "iter": iteration,
        },
        path,
    )


def _metric_row(
    *,
    run: str,
    checkpoint: str,
    iteration: int,
    suite: str,
    seed: int,
    termination_rate: float = 0.1,
    axis_speed: float = 0.2,
    duration: float = 10.0,
    reverse_rate: float = 0.1,
    episode_return: float = 1.0,
) -> dict[str, Any]:
    return {
        "run": run,
        "checkpoint": checkpoint,
        "iteration": iteration,
        "suite": suite,
        "seed": seed,
        "episode_count": 2,
        "episode_return_mean": episode_return,
        "episode_duration_mean": duration,
        "termination_rate": termination_rate,
        "rotation_axis_speed_mean": axis_speed,
        "rotation_axis_speed_abs_mean": abs(axis_speed),
        "rotation_positive_rate": 0.6,
        "rotation_reverse_rate": reverse_rate,
        "rotation_high_clip_rate": 0.2,
        "rotation_low_clip_rate": 0.1,
        "action_abs_mean": 0.3,
        "action_saturation_rate": 0.01,
        "target_saturation_rate": 0.02,
        "target_lower_saturation_rate": 0.01,
        "target_upper_saturation_rate": 0.01,
        "actor_hash_before": "abc",
        "actor_hash_after": "abc",
        "rank": 0,
    }


@pytest.fixture
def dummy_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "dummy_run"
    run_dir.mkdir()
    _write_run_config(run_dir, _compose_hora_cfg())
    _dummy_checkpoint(run_dir / "model_10.pt", 10)
    _dummy_checkpoint(run_dir / "model_20.pt", 20)
    return run_dir


def test_single_checkpoint_and_suite_config_contract(dummy_run: Path) -> None:
    source = load_run_config(dummy_run)
    assert resolve_checkpoints(
        dummy_run, checkpoint="10", all_checkpoints=False
    ) == [dummy_run / "model_10.pt"]

    nominal = config_for_suite(source, "nominal")
    train_dr = config_for_suite(source, "train_dr")
    assert nominal.env.hora_domain_rand.enabled is False
    assert train_dr.env.hora_domain_rand.enabled is True
    assert source.env.hora_domain_rand.enabled is True


def test_multi_checkpoint_lexicographic_ranking_is_per_suite() -> None:
    rows = [
        _metric_row(
            run="r",
            checkpoint="model_10.pt",
            iteration=10,
            suite="nominal",
            seed=101,
            termination_rate=0.2,
            axis_speed=1.0,
        ),
        _metric_row(
            run="r",
            checkpoint="model_20.pt",
            iteration=20,
            suite="nominal",
            seed=101,
            termination_rate=0.1,
            axis_speed=0.1,
        ),
        _metric_row(
            run="r",
            checkpoint="model_10.pt",
            iteration=10,
            suite="train_dr",
            seed=101,
            termination_rate=0.0,
            axis_speed=0.1,
        ),
        _metric_row(
            run="r",
            checkpoint="model_20.pt",
            iteration=20,
            suite="train_dr",
            seed=101,
            termination_rate=0.0,
            axis_speed=0.2,
        ),
    ]

    assign_checkpoint_ranks(rows)

    ranks = {(row["suite"], row["checkpoint"]): row["rank"] for row in rows}
    assert ranks[("nominal", "model_20.pt")] == 1
    assert ranks[("nominal", "model_10.pt")] == 2
    assert ranks[("train_dr", "model_20.pt")] == 1
    assert ranks[("train_dr", "model_10.pt")] == 2


def test_evaluate_run_writes_fixed_csv_json_schema_and_is_seed_reproducible(
    dummy_run: Path,
    tmp_path: Path,
) -> None:
    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        rng = np.random.default_rng(kwargs["seed"])
        iteration = int(kwargs["checkpoint"].stem.split("_")[1])
        return _metric_row(
            run=kwargs["run_dir"].name,
            checkpoint=kwargs["checkpoint"].name,
            iteration=iteration,
            suite=kwargs["suite"],
            seed=kwargs["seed"],
            axis_speed=float(rng.uniform(0.0, 1.0)),
        )

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = evaluate_run(
        run_dir=dummy_run,
        checkpoint="model_10.pt",
        all_checkpoints=False,
        num_envs=2,
        num_steps=400,
        seeds=[101],
        suites=["nominal", "train_dr"],
        output_dir=first_dir,
        evaluate_fn=fake_evaluate,
    )
    second = evaluate_run(
        run_dir=dummy_run,
        checkpoint="model_10.pt",
        all_checkpoints=False,
        num_envs=2,
        num_steps=400,
        seeds=[101],
        suites=["nominal", "train_dr"],
        output_dir=second_dir,
        evaluate_fn=fake_evaluate,
    )

    assert first == second
    assert json.loads((first_dir / "checkpoint_metrics.json").read_text()) == first
    manifest = json.loads((first_dir / "evaluation_manifest.json").read_text())
    assert manifest["deterministic"] is True
    assert manifest["render"] is False
    assert manifest["export"] is False
    with (first_dir / "checkpoint_metrics.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == METRIC_FIELDS
        assert len(list(reader)) == 2


def test_no_checkpoint_wrong_checkpoint_and_legacy_fail_closed(
    dummy_run: Path,
) -> None:
    empty = dummy_run.parent / "empty"
    empty.mkdir()
    _write_run_config(empty, _compose_hora_cfg())
    with pytest.raises(FileNotFoundError, match="no model"):
        resolve_checkpoints(empty, checkpoint=None, all_checkpoints=True)
    with pytest.raises(FileNotFoundError, match="requested checkpoint"):
        resolve_checkpoints(dummy_run, checkpoint="999", all_checkpoints=False)

    malformed = dummy_run / "model_30.pt"
    torch.save({"iter": 30}, malformed)
    with pytest.raises(KeyError, match="required keys"):
        validate_checkpoint(malformed)

    legacy = dummy_run / "model_40.pt"
    torch.save(
        {
            "actor_state_dict": {"model.0.weight": torch.zeros((16, 105))},
            "critic_state_dict": {},
            "optimizer_state_dict": {},
            "iter": 40,
        },
        legacy,
    )
    with pytest.raises(ValueError, match="not a HORA"):
        validate_checkpoint(legacy)


def test_nan_metric_fails_closed_before_output(dummy_run: Path, tmp_path: Path) -> None:
    def nan_evaluate(**kwargs: Any) -> dict[str, Any]:
        return _metric_row(
            run=kwargs["run_dir"].name,
            checkpoint=kwargs["checkpoint"].name,
            iteration=10,
            suite=kwargs["suite"],
            seed=kwargs["seed"],
            axis_speed=float("nan"),
        )

    with pytest.raises(ValueError, match="must be finite"):
        evaluate_run(
            run_dir=dummy_run,
            checkpoint="model_10.pt",
            all_checkpoints=False,
            num_envs=2,
            num_steps=400,
            seeds=[101],
            suites=["nominal"],
            output_dir=tmp_path / "nan-output",
            evaluate_fn=nan_evaluate,
        )


@pytest.fixture(scope="module")
def real_hora_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("hora-evaluator") / "run"
    run_dir.mkdir()
    cfg = _compose_hora_cfg()
    _write_run_config(run_dir, cfg)
    ensure_registries()
    env_override = BackendAdapter(
        cfg, root_dir=ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    env = create_env(
        cfg,
        num_envs=2,
        env_cfg_override=env_override,
        sim_backend="mujoco",
        task_name="LeapInhandBall0730HoraRotation",
    )
    wrapper = HoraRslRlVecEnvWrapper(env, device="cpu")
    algo_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
    assert isinstance(algo_cfg, dict)
    train_cfg = normalize_ppo_train_cfg(deepcopy(algo_cfg))
    train_cfg["multi_gpu"] = None
    train_cfg["logger"] = "tensorboard"
    train_cfg.setdefault("runner", {})["logger"] = "tensorboard"
    runner = OnPolicyRunner(
        wrapper, train_cfg, log_dir=None, device="cpu"
    )
    try:
        torch.save(
            {
                "actor_state_dict": runner.alg.actor.state_dict(),
                "critic_state_dict": runner.alg.critic.state_dict(),
                "optimizer_state_dict": runner.alg.optimizer.state_dict(),
                "iter": 1,
                "infos": None,
            },
            run_dir / "model_1.pt",
        )
    finally:
        wrapper.close()
    return run_dir


def test_actual_checkpoint_evaluation_is_deterministic_and_preserves_actor_hash(
    real_hora_run: Path,
) -> None:
    cfg = load_run_config(real_hora_run)
    checkpoint = real_hora_run / "model_1.pt"
    first = evaluate_checkpoint(
        run_dir=real_hora_run,
        checkpoint=checkpoint,
        source_cfg=cfg,
        suite="nominal",
        seed=909,
        num_envs=2,
        num_steps=400,
    )
    second = evaluate_checkpoint(
        run_dir=real_hora_run,
        checkpoint=checkpoint,
        source_cfg=cfg,
        suite="nominal",
        seed=909,
        num_envs=2,
        num_steps=400,
    )

    assert first["actor_hash_before"] == first["actor_hash_after"]
    assert first == second
    assert first["episode_count"] >= 2
    assert set(first) == set(METRIC_FIELDS)
