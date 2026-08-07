"""Deterministic, no-render checkpoint evaluator for LEAP HORA teachers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rsl_rl.runners import OnPolicyRunner

from unilab.algos.torch.rsl_rl_runtime import resolve_rsl_rl_ppo_runtime
from unilab.training import BackendAdapter, create_env, ensure_registries
from unilab.training.rsl_rl import normalize_ppo_train_cfg
from unilab.training.sim2sim import policy_load_dim_guard

SUITES = ("nominal", "train_dr")
RANKING_RULE = (
    "termination_rate ascending",
    "rotation_axis_speed_mean descending",
    "episode_duration_mean descending",
    "rotation_reverse_rate ascending",
    "episode_return_mean descending",
)
METRIC_FIELDS = (
    "run",
    "checkpoint",
    "iteration",
    "suite",
    "seed",
    "episode_count",
    "episode_return_mean",
    "episode_duration_mean",
    "termination_rate",
    "rotation_axis_speed_mean",
    "rotation_axis_speed_abs_mean",
    "rotation_positive_rate",
    "rotation_reverse_rate",
    "rotation_high_clip_rate",
    "rotation_low_clip_rate",
    "action_abs_mean",
    "action_saturation_rate",
    "target_saturation_rate",
    "target_lower_saturation_rate",
    "target_upper_saturation_rate",
    "actor_hash_before",
    "actor_hash_after",
    "rank",
)
_LOG_TO_ROW = {
    "rotation/axis_speed_mean": "rotation_axis_speed_mean",
    "rotation/axis_speed_abs_mean": "rotation_axis_speed_abs_mean",
    "rotation/positive_rate": "rotation_positive_rate",
    "rotation/reverse_rate": "rotation_reverse_rate",
    "rotation/high_clip_rate": "rotation_high_clip_rate",
    "rotation/low_clip_rate": "rotation_low_clip_rate",
    "control/action_abs_mean": "action_abs_mean",
    "control/action_saturation_rate": "action_saturation_rate",
    "control/target_saturation_rate": "target_saturation_rate",
    "control/target_lower_saturation_rate": "target_lower_saturation_rate",
    "control/target_upper_saturation_rate": "target_upper_saturation_rate",
}
_ITERATION_RE = re.compile(r"^model_(\d+)\.pt$")


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash tensor names, shapes, dtypes and bytes deterministically."""

    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_iteration(path: Path) -> int:
    match = _ITERATION_RE.match(path.name)
    if match is None:
        raise ValueError(f"checkpoint must be named model_<iteration>.pt: {path}")
    return int(match.group(1))


def validate_checkpoint(path: Path) -> dict[str, Any]:
    """Validate the outer RSL-RL checkpoint contract before constructing an env."""

    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint payload must be a mapping: {path}")
    required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter"}
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"checkpoint is missing required keys {missing}: {path}")
    actor_state = payload["actor_state_dict"]
    if not isinstance(actor_state, dict) or not actor_state:
        raise ValueError(f"checkpoint actor_state_dict must be non-empty: {path}")
    if not any(str(key).startswith("shared.") for key in actor_state):
        raise ValueError(f"checkpoint is not a HORA shared-actor checkpoint: {path}")
    return payload


def load_run_config(run_dir: Path) -> DictConfig:
    path = run_dir / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"run config does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"run_config.json must contain a config mapping: {path}")
    cfg = OmegaConf.create(payload["config"])
    if str(OmegaConf.select(cfg, "training.task_name")) != "LeapInhandBall0730HoraRotation":
        raise ValueError("run config is not a LEAP HORA teacher run")
    return cfg


def config_for_suite(source_cfg: DictConfig, suite: str) -> DictConfig:
    if suite not in SUITES:
        raise ValueError(f"unsupported suite {suite!r}; expected one of {SUITES}")
    cfg = OmegaConf.create(OmegaConf.to_container(source_cfg, resolve=True))
    if suite == "nominal":
        OmegaConf.update(cfg, "env.hora_domain_rand.enabled", False, merge=False)
    return cfg


def resolve_checkpoints(
    run_dir: Path,
    *,
    checkpoint: str | Sequence[str] | None,
    all_checkpoints: bool,
) -> list[Path]:
    if checkpoint is not None and all_checkpoints:
        raise ValueError("--checkpoint and --all-checkpoints are mutually exclusive")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    available: list[tuple[int, Path]] = []
    for path in run_dir.glob("model_*.pt"):
        try:
            available.append((checkpoint_iteration(path), path))
        except ValueError:
            continue
    available.sort(key=lambda item: item[0])
    if not available:
        raise FileNotFoundError(f"no model_<iteration>.pt checkpoints found in {run_dir}")
    if all_checkpoints:
        return [path for _, path in available]
    if checkpoint is None:
        return [available[-1][1]]
    selected = [checkpoint] if isinstance(checkpoint, str) else list(checkpoint)
    resolved: list[Path] = []
    for value in selected:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = run_dir / candidate
        if candidate.suffix != ".pt" and value.isdigit():
            candidate = run_dir / f"model_{value}.pt"
        if not candidate.is_file():
            raise FileNotFoundError(f"requested checkpoint does not exist: {candidate}")
        checkpoint_iteration(candidate)
        resolved.append(candidate)
    return sorted(set(resolved), key=checkpoint_iteration)


def _algo_config_dict(cfg: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("cfg.algo must resolve to a mapping")
    return value


def _mean_metric(samples: dict[str, list[float]], key: str) -> float:
    values = samples.get(key, [])
    if not values:
        raise RuntimeError(f"evaluation did not observe required diagnostic {key!r}")
    value = float(np.mean(values))
    if not np.isfinite(value):
        raise ValueError(f"evaluation metric {key!r} is not finite")
    return value


def _validate_metric_row(row: dict[str, Any]) -> None:
    if set(row) != set(METRIC_FIELDS):
        missing = sorted(set(METRIC_FIELDS) - set(row))
        extra = sorted(set(row) - set(METRIC_FIELDS))
        raise ValueError(f"metric row schema mismatch; missing={missing}, extra={extra}")
    for key, value in row.items():
        if key in {"run", "checkpoint", "suite", "actor_hash_before", "actor_hash_after"}:
            if not isinstance(value, str) or not value:
                raise ValueError(f"metric row field {key!r} must be a non-empty string")
        elif not np.isfinite(value):
            raise ValueError(f"metric row field {key!r} must be finite")
    for key in (
        "termination_rate",
        "rotation_positive_rate",
        "rotation_reverse_rate",
        "rotation_high_clip_rate",
        "rotation_low_clip_rate",
        "action_saturation_rate",
        "target_saturation_rate",
        "target_lower_saturation_rate",
        "target_upper_saturation_rate",
    ):
        if not 0.0 <= float(row[key]) <= 1.0:
            raise ValueError(f"metric row rate {key!r} must be in [0, 1]")
    if row["actor_hash_before"] != row["actor_hash_after"]:
        raise AssertionError("evaluation modified actor weights")


def evaluate_checkpoint(
    *,
    run_dir: Path,
    checkpoint: Path,
    source_cfg: DictConfig,
    suite: str,
    seed: int,
    num_envs: int,
    num_steps: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate one checkpoint without optimizer, normalizer, render or export writes."""

    if num_envs <= 0 or num_steps <= 0:
        raise ValueError("num_envs and num_steps must be positive")
    validate_checkpoint(checkpoint)
    cfg = config_for_suite(source_cfg, suite)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ensure_registries()

    env_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="ppo"
    ).build_task_env_cfg_override()
    env = create_env(
        cfg,
        num_envs=num_envs,
        env_cfg_override=env_override,
        sim_backend=str(cfg.training.sim_backend),
        task_name=str(cfg.training.task_name),
    )
    rl_cfg = _algo_config_dict(cfg)
    runtime = resolve_rsl_rl_ppo_runtime(rl_cfg, default_wrapper_cls=None)
    if runtime is None:
        env.close()
        raise RuntimeError("run config does not resolve the HORA PPO runtime")
    wrapper = runtime.wrapper_cls(env, device=device)
    train_cfg = normalize_ppo_train_cfg(deepcopy(rl_cfg))
    train_cfg["multi_gpu"] = None
    train_cfg.setdefault("runner", {})["logger"] = "tensorboard"
    train_cfg["logger"] = "tensorboard"
    runner = cast(
        Any,
        OnPolicyRunner(cast(Any, wrapper), train_cfg, log_dir=None, device=device),
    )

    try:
        with policy_load_dim_guard(
            env_obs_dim=getattr(wrapper, "num_obs", None),
            env_action_dim=getattr(wrapper, "num_actions", None),
            algo_name="ppo",
        ):
            runner.load(str(checkpoint), map_location=device)
        actor = runner.alg.actor
        actor.eval()
        runner.alg.critic.eval()
        actor_hash_before = state_dict_sha256(actor.state_dict())
        obs, _ = wrapper.reset()
        episode_returns = np.zeros(num_envs, dtype=np.float64)
        episode_lengths = np.zeros(num_envs, dtype=np.int64)
        completed_returns: list[float] = []
        completed_durations: list[float] = []
        termination_count = 0
        metric_samples: dict[str, list[float]] = defaultdict(list)

        with torch.inference_mode():
            for step_index in range(num_steps):
                actions = actor(obs, stochastic_output=False)
                if not torch.isfinite(actions).all():
                    raise FloatingPointError(f"non-finite actions at step {step_index}")
                obs, rewards, dones, infos = wrapper.step(actions)
                if not torch.isfinite(rewards).all():
                    raise FloatingPointError(f"non-finite rewards at step {step_index}")
                for key, value in obs.items():
                    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                        raise FloatingPointError(
                            f"non-finite observation {key!r} at step {step_index}"
                        )
                reward_np = rewards.detach().cpu().numpy().astype(np.float64)
                done_np = dones.detach().cpu().numpy().astype(bool)
                episode_returns += reward_np
                episode_lengths += 1
                if np.any(done_np):
                    timeout_tensor = infos.get("time_outs")
                    timeout_np = (
                        timeout_tensor.detach().cpu().numpy().astype(bool)
                        if isinstance(timeout_tensor, torch.Tensor)
                        else np.zeros(num_envs, dtype=bool)
                    )
                    done_ids = np.flatnonzero(done_np)
                    completed_returns.extend(episode_returns[done_ids].tolist())
                    completed_durations.extend(
                        (episode_lengths[done_ids] * float(cfg.env.ctrl_dt)).tolist()
                    )
                    termination_count += int(np.count_nonzero(done_np & ~timeout_np))
                    episode_returns[done_ids] = 0.0
                    episode_lengths[done_ids] = 0
                log = infos.get("log", {})
                for log_key in _LOG_TO_ROW:
                    if log_key in log:
                        value = float(log[log_key])
                        if not np.isfinite(value):
                            raise ValueError(f"diagnostic {log_key!r} is not finite")
                        metric_samples[log_key].append(value)

        actor_hash_after = state_dict_sha256(actor.state_dict())
        if actor_hash_before != actor_hash_after:
            raise AssertionError("evaluation modified actor weights or normalizer state")
        episode_count = len(completed_returns)
        if episode_count == 0:
            raise RuntimeError(
                "evaluation completed no episodes; increase num_steps to cover the horizon"
            )
        row: dict[str, Any] = {
            "run": run_dir.name,
            "checkpoint": checkpoint.name,
            "iteration": checkpoint_iteration(checkpoint),
            "suite": suite,
            "seed": int(seed),
            "episode_count": episode_count,
            "episode_return_mean": float(np.mean(completed_returns)),
            "episode_duration_mean": float(np.mean(completed_durations)),
            "termination_rate": float(termination_count / episode_count),
            "actor_hash_before": actor_hash_before,
            "actor_hash_after": actor_hash_after,
            "rank": 0,
        }
        for log_key, row_key in _LOG_TO_ROW.items():
            row[row_key] = _mean_metric(metric_samples, log_key)
        _validate_metric_row(row)
        return row
    finally:
        wrapper.close()


def _ranking_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(row["termination_rate"]),
        -float(row["rotation_axis_speed_mean"]),
        -float(row["episode_duration_mean"]),
        float(row["rotation_reverse_rate"]),
        -float(row["episode_return_mean"]),
    )


def assign_checkpoint_ranks(rows: list[dict[str, Any]]) -> None:
    """Rank checkpoints per suite using seed-mean metrics and the published key."""

    by_suite_checkpoint: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        _validate_metric_row(row)
        by_suite_checkpoint[(str(row["suite"]), str(row["checkpoint"]))].append(row)

    for suite in sorted({key[0] for key in by_suite_checkpoint}):
        aggregates: list[dict[str, Any]] = []
        checkpoint_names: list[str] = []
        for (row_suite, checkpoint), group in by_suite_checkpoint.items():
            if row_suite != suite:
                continue
            aggregate = {
                field: float(np.mean([float(row[field]) for row in group]))
                for field in (
                    "termination_rate",
                    "rotation_axis_speed_mean",
                    "episode_duration_mean",
                    "rotation_reverse_rate",
                    "episode_return_mean",
                )
            }
            aggregates.append(aggregate)
            checkpoint_names.append(checkpoint)
        ordering = sorted(
            range(len(aggregates)),
            key=lambda index: (_ranking_key(aggregates[index]), checkpoint_names[index]),
        )
        rank_by_checkpoint = {
            checkpoint_names[index]: rank
            for rank, index in enumerate(ordering, start=1)
        }
        for (row_suite, checkpoint), group in by_suite_checkpoint.items():
            if row_suite == suite:
                for row in group:
                    row["rank"] = rank_by_checkpoint[checkpoint]


def write_evaluation_outputs(
    *,
    output_dir: Path,
    run_dir: Path,
    checkpoints: Sequence[Path],
    suites: Sequence[str],
    seeds: Sequence[int],
    num_envs: int,
    num_steps: int,
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        _validate_metric_row(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "run_dir": str(run_dir.resolve()),
        "checkpoints": [str(path.resolve()) for path in checkpoints],
        "suites": list(suites),
        "seeds": list(seeds),
        "num_envs": num_envs,
        "num_steps": num_steps,
        "deterministic": True,
        "render": False,
        "export": False,
        "ranking": list(RANKING_RULE),
        "row_count": len(rows),
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "checkpoint_metrics.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "checkpoint_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_run(
    *,
    run_dir: Path,
    checkpoint: str | Sequence[str] | None,
    all_checkpoints: bool,
    num_envs: int,
    num_steps: int,
    seeds: Sequence[int],
    suites: Sequence[str],
    output_dir: Path,
    evaluate_fn: Callable[..., dict[str, Any]] = evaluate_checkpoint,
) -> list[dict[str, Any]]:
    if num_steps < 400:
        raise ValueError("num_steps must be at least one complete 400-step episode horizon")
    if not seeds or not suites:
        raise ValueError("at least one seed and suite are required")
    source_cfg = load_run_config(run_dir)
    checkpoints = resolve_checkpoints(
        run_dir, checkpoint=checkpoint, all_checkpoints=all_checkpoints
    )
    for path in checkpoints:
        validate_checkpoint(path)
    rows = [
        evaluate_fn(
            run_dir=run_dir,
            checkpoint=path,
            source_cfg=source_cfg,
            suite=suite,
            seed=seed,
            num_envs=num_envs,
            num_steps=num_steps,
        )
        for path in checkpoints
        for suite in suites
        for seed in seeds
    ]
    assign_checkpoint_ranks(rows)
    write_evaluation_outputs(
        output_dir=output_dir,
        run_dir=run_dir,
        checkpoints=checkpoints,
        suites=suites,
        seeds=seeds,
        num_envs=num_envs,
        num_steps=num_steps,
        rows=rows,
    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", nargs="+")
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=800)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--suite", choices=SUITES, nargs="+", default=list(SUITES))
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "teacher_evaluation"
    )
    rows = evaluate_run(
        run_dir=run_dir,
        checkpoint=args.checkpoint,
        all_checkpoints=args.all_checkpoints,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        seeds=args.seeds,
        suites=args.suite,
        output_dir=output_dir,
    )
    print(f"Evaluated {len(rows)} checkpoint/suite/seed rows into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
