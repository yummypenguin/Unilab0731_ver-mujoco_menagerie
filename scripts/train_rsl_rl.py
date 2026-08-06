import contextlib
import csv
import datetime
import hashlib
import io
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.algos.torch.rsl_rl_runtime import resolve_rsl_rl_ppo_runtime
from unilab.base.backend.mujoco.xml import materialize_scene_visual_override
from unilab.training import (
    BackendAdapter,
    apply_configured_training_seed,
    create_env,
    ensure_registries,
    get_latest_checkpoint,
    get_latest_run,
    get_log_root,
    log_playback_plan,
    parse_checkpoint_path,
    should_run_playback,
)
from unilab.training.experiment import (
    ExperimentTracker,
    get_git_info,
    patch_rsl_rl_resume_state,
    patch_rsl_rl_wandb_writer,
)
from unilab.training.rsl_rl import (
    RslRlVecEnvWrapper,
    load_rsl_rl_training_checkpoint,
    normalize_ppo_train_cfg,
)
from unilab.training.sim2sim import policy_load_dim_guard, resolve_sim2sim_config
from unilab.utils.device import get_default_device

try:
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    print("Could not import rsl_rl. Please ensure it is installed.")
    sys.exit(1)


def _patch_runner_action_std_logging(
    runner: Any,
    *,
    suppress_console: bool = False,
) -> None:
    original_log = runner.logger.log

    def _safe_log(self, *args, **kwargs):
        policy = runner.alg.get_policy()
        dist = policy.distribution
        if dist.std_type == "scalar":
            std = dist.std_param
        else:
            std = torch.exp(dist.log_std_param)
        kwargs["action_std"] = std.detach().clone()
        if suppress_console:
            with contextlib.redirect_stdout(io.StringIO()):
                return original_log(*args, **kwargs)
        return original_log(*args, **kwargs)

    runner.logger.log = _safe_log.__get__(runner.logger, type(runner.logger))


def _resolve_rsl_rl_logger(log_backend: str) -> tuple[str, bool]:
    """Map UniLab logging modes to RSL-RL while retaining checkpoint writes."""

    normalized = str(log_backend).strip().lower()
    if normalized == "no_print":
        return "tensorboard", True
    if normalized in {"tensorboard", "wandb"}:
        return normalized, False
    return "tensorboard", False


def _backend_adapter(cfg: DictConfig) -> BackendAdapter:
    return BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="ppo",
        scene_materializer=materialize_scene_visual_override,
    )


def build_ppo_env_cfg_override(cfg: DictConfig) -> dict[str, Any]:
    return cast(dict[str, Any], _backend_adapter(cfg).build_task_env_cfg_override())


def build_ppo_play_env_cfg_override(cfg: DictConfig) -> dict[str, Any]:
    return cast(dict[str, Any], _backend_adapter(cfg).build_play_env_cfg_override())


def run_motrix_rsl_play_loop(
    wrapped_env,
    policy,
    *,
    render_spacing: float,
    render_offset_mode: str,
    num_steps: int | None = None,
) -> None:
    env = wrapped_env.env

    with torch.inference_mode():
        env.run_playback(
            render_spacing=render_spacing,
            render_offset_mode=render_offset_mode,
            num_steps=num_steps,
            initialize=lambda: wrapped_env.reset()[0],
            step=lambda obs: wrapped_env.step(policy(obs))[0],
        )


def _get_log_root(cfg: DictConfig) -> str:
    return str(get_log_root(ROOT_DIR, cfg))


def _algo_config_dict(cfg: DictConfig) -> dict[str, Any]:
    train_cfg_raw = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(train_cfg_raw, dict):
        raise TypeError("cfg.algo must resolve to a dict")
    return cast(dict[str, Any], train_cfg_raw)


def _resolve_ppo_wrapper_cls(rl_cfg: dict[str, Any]) -> type[RslRlVecEnvWrapper]:
    """Resolve the VecEnv wrapper class from the owner-selected PPO runtime.

    Args:
        rl_cfg: Resolved algorithm config dictionary from Hydra composition.

    Returns:
        Wrapper class used to adapt the UniLab env contract to the active
        RSL-RL PPO runtime.
    """
    return resolve_rsl_rl_ppo_runtime(
        rl_cfg,
        default_wrapper_cls=RslRlVecEnvWrapper,
    ).wrapper_cls


def apply_ppo_runtime_flags(
    train_cfg: dict[str, Any],
    cfg: DictConfig,
    *,
    training_enabled: bool,
) -> None:
    algorithm_cfg = train_cfg.setdefault("algorithm", {})
    if not isinstance(algorithm_cfg, dict):
        return
    if not training_enabled:
        algorithm_cfg["enable_compile"] = False


def aggregate_step_mean(rows: list[dict[str, float]], key: str) -> float:
    """Average a scalar metric over evaluation logging points."""
    values = [row[key] for row in rows if key in row]
    return float(np.mean(values)) if values else 0.0


def aggregate_count(rows: list[dict[str, float]], key: str) -> float:
    """Sum event counts over evaluation logging points."""
    return float(np.sum([row[key] for row in rows if key in row]))


def aggregate_count_weighted_mean(
    rows: list[dict[str, float]],
    *,
    metric_key: str,
    count_key: str,
) -> float:
    """Aggregate terminal metrics using the corresponding event count."""
    weighted_sum = 0.0
    total_count = 0.0
    for row in rows:
        if metric_key not in row or count_key not in row:
            continue
        count = float(row[count_key])
        weighted_sum += float(row[metric_key]) * count
        total_count += count
    return weighted_sum / total_count if total_count > 0.0 else 0.0


def _distribution_stats(values: list[float] | list[int], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_p50": float(np.percentile(array, 50)),
        f"{prefix}_p90": float(np.percentile(array, 90)),
    }


def build_evaluation_summary(
    *,
    metric_rows: list[dict[str, float]],
    completed_returns: list[float],
    completed_lengths: list[int],
    ctrl_dt: float,
) -> dict[str, float | int]:
    """Build correctly weighted episode, phase-timeout, and step summaries."""
    summary: dict[str, float | int] = {
        "episode/count": len(completed_returns),
    }
    summary.update(_distribution_stats(completed_returns, "episode/return"))
    summary.update(_distribution_stats(completed_lengths, "episode/length"))
    summary["episode/duration_seconds_mean"] = (
        float(np.mean(completed_lengths)) * ctrl_dt if completed_lengths else 0.0
    )

    step_mean_prefixes = ("state_cycle/", "rotation/", "termination/")
    step_keys = sorted(
        {
            key
            for row in metric_rows
            for key in row
            if key != "evaluation_step" and key.startswith(step_mean_prefixes)
        }
    )
    for key in step_keys:
        summary[key] = aggregate_step_mean(metric_rows, key)

    phases = ("READY_TO_A", "A_TO_B", "B_TO_READY")
    total_timeout_count = 0.0
    for phase in phases:
        source_count_key = f"timeout/{phase}_count"
        total_count_key = f"timeout/{phase}_total_count"
        count = aggregate_count(metric_rows, source_count_key)
        summary[total_count_key] = count
        total_timeout_count += count
    summary["timeout/total_count"] = total_timeout_count

    terminal_metric_names = (
        "pose_distance_mean",
        "position_error_m_mean",
        "ball_speed_m_s_mean",
        "contact_count_mean",
        "edge_net_angle_mean",
        "remaining_angle_mean",
        "hold_steps_mean",
        "pose_ok_rate",
        "position_ok_rate",
        "speed_ok_rate",
        "contact_ok_rate",
        "rotation_ok_rate",
        "no_palm_contact_rate",
        "pose_blocked_rate",
        "position_blocked_rate",
        "speed_blocked_rate",
        "contact_blocked_rate",
        "rotation_blocked_rate",
        "palm_blocked_rate",
    )
    for phase in phases:
        count_key = f"timeout/{phase}_count"
        phase_count = float(summary[f"timeout/{phase}_total_count"])
        summary[f"timeout/{phase}_contribution"] = (
            phase_count / total_timeout_count if total_timeout_count > 0.0 else 0.0
        )
        for metric_name in terminal_metric_names:
            metric_key = f"timeout/{phase}_{metric_name}"
            summary[metric_key] = aggregate_count_weighted_mean(
                metric_rows,
                metric_key=metric_key,
                count_key=count_key,
            )
    return summary


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode("utf-8"))
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def evaluation_policy_actions(
    policy: Any,
    obs: Any,
    *,
    deterministic: bool,
) -> torch.Tensor:
    """Select policy mean actions or explicit distribution samples."""
    return cast(torch.Tensor, policy(obs, stochastic_output=not deterministic))


def validate_evaluation_config(cfg: DictConfig) -> None:
    """Validate evaluator-only values and mutually exclusive entry modes."""
    eval_only = bool(OmegaConf.select(cfg, "training.eval_only", default=False))
    play_only = bool(OmegaConf.select(cfg, "training.play_only", default=False))
    if eval_only and play_only:
        raise ValueError("training.eval_only and training.play_only cannot both be true.")
    if int(cfg.evaluation.num_envs) <= 0:
        raise ValueError("evaluation.num_envs must be positive")
    if int(cfg.evaluation.num_steps) <= 0:
        raise ValueError("evaluation.num_steps must be positive")
    if not isinstance(cfg.evaluation.deterministic, bool):
        raise TypeError("evaluation.deterministic must be bool")


def _read_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(f"Evaluation run config could not be read: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"Evaluation run config has no full config mapping: {path}")
    return payload


def recover_evaluation_config(
    source_run_dir: Path,
    target_cfg: DictConfig,
) -> DictConfig:
    """Restore checkpoint config while preserving evaluator-only CLI settings."""
    resolve_sim2sim_config(
        source_run_dir,
        target_cfg,
        algo_name="ppo",
        strict=True,
    )
    evaluation_settings = OmegaConf.to_container(target_cfg.evaluation, resolve=True)
    selector = {
        "load_run": OmegaConf.select(target_cfg, "algo.load_run"),
        "checkpoint": OmegaConf.select(target_cfg, "algo.checkpoint"),
        "seed": OmegaConf.select(target_cfg, "algo.seed"),
    }
    payload = _read_run_config(source_run_dir)
    recovered = OmegaConf.create(payload["config"])
    recovered.evaluation = OmegaConf.create(evaluation_settings)
    recovered.training.eval_only = True
    recovered.training.play_only = False
    recovered.training.no_play = True
    recovered.algo.load_run = selector["load_run"]
    recovered.algo.checkpoint = selector["checkpoint"]
    recovered.algo.seed = selector["seed"]
    return recovered


def _assert_state_cycle_evaluation_contract(cfg: DictConfig, num_obs: int) -> None:
    if str(cfg.training.task_name) != "LeapInhandBallStateCycleRotation":
        return
    assert float(cfg.env.state_cycle.ready_to_a.timeout_seconds) == 1.5
    assert float(cfg.env.state_cycle.a_to_b.timeout_seconds) == 2.0
    assert float(cfg.env.state_cycle.b_to_ready.timeout_seconds) == 1.3
    assert float(cfg.env.ctrl_dt) == 0.05
    assert float(cfg.env.termination_workspace_radius) == 0.05
    pose_tracking_scale = float(cfg.reward.pose_tracking_scale)
    if not np.isfinite(pose_tracking_scale) or pose_tracking_scale <= 0.0:
        raise AssertionError(
            "Expected a finite positive state-cycle pose tracking scale, "
            f"got {pose_tracking_scale}"
        )
    assert float(cfg.reward.rotation_progress_scale) == 3.0
    assert float(cfg.reward.rotation_target_axis_speed_rad_s) == 0.50
    assert float(cfg.reward.rotation_overspeed_scale) == 1.0
    assert float(cfg.reward.failure_penalty) == 3.0
    assert float(cfg.reward.failure_rotation_clawback_cap) == 5.0
    if num_obs != 142:
        raise AssertionError(f"Expected state-cycle observation dimension 142, got {num_obs}")


def _evaluation_output_dir(
    cfg: DictConfig,
    *,
    load_path: Path,
    load_path_dir: Path,
) -> Path:
    configured = OmegaConf.select(cfg, "evaluation.output_dir", default=None)
    if configured not in (None, ""):
        return Path(str(configured)).resolve()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mode = "deterministic" if bool(cfg.evaluation.deterministic) else "stochastic"
    name = f"{load_path_dir.name}_{load_path.stem}_instrumented_{mode}_{timestamp}"
    return (
        ROOT_DIR
        / "logs"
        / "evaluation"
        / str(cfg.training.task_name)
        / name
    )


def _required_state_cycle_evaluation_keys() -> set[str]:
    keys: set[str] = set()
    phases = ("READY_TO_A", "A_TO_B", "B_TO_READY")
    for phase in phases:
        keys.update(
            {
                f"state_cycle/pose_distance_{phase}_mean",
                f"state_cycle/pose_ok_{phase}_rate",
                f"state_cycle/position_ok_{phase}_rate",
                f"state_cycle/contact_ok_{phase}_rate",
                f"state_cycle/rotation_ok_{phase}_rate",
                f"state_cycle/hold_steps_{phase}_mean",
                f"state_cycle/remaining_angle_{phase}_mean",
                f"state_cycle/timeout_{phase}_rate",
                f"state_cycle/timeout_{phase}_conditional_rate",
                f"timeout/{phase}_count",
                f"timeout/{phase}_pose_distance_mean",
                f"timeout/{phase}_position_error_m_mean",
                f"timeout/{phase}_ball_speed_m_s_mean",
                f"timeout/{phase}_contact_count_mean",
                f"timeout/{phase}_edge_net_angle_mean",
                f"timeout/{phase}_remaining_angle_mean",
                f"timeout/{phase}_hold_steps_mean",
                f"timeout/{phase}_pose_blocked_rate",
                f"timeout/{phase}_position_blocked_rate",
                f"timeout/{phase}_speed_blocked_rate",
                f"timeout/{phase}_contact_blocked_rate",
                f"timeout/{phase}_rotation_blocked_rate",
                f"timeout/{phase}_palm_blocked_rate",
            }
        )
    return keys


def _format_play_checkpoint_error(
    cfg: DictConfig,
    *,
    task_log_root: Path,
    load_path: Path | None,
    load_path_dir: Path | None,
) -> str:
    selected_checkpoint = OmegaConf.select(cfg, "algo.checkpoint", default=-1)
    checkpoint_hint = (
        f" algo.checkpoint={selected_checkpoint!r}"
        if selected_checkpoint not in (None, "", -1, "-1")
        else ""
    )

    if load_path_dir is not None and load_path is None and checkpoint_hint:
        reason = f"Requested checkpoint was not found under resolved_run={load_path_dir}."
    elif not task_log_root.exists():
        reason = "Task log root does not exist."
    else:
        latest_run = get_latest_run(task_log_root)
        if latest_run is None:
            reason = "No run directories were found under the task log root."
        elif get_latest_checkpoint(latest_run) is None:
            reason = f"Resolved latest run has no model_*.pt checkpoint files: {latest_run}."
        else:
            reason = "Requested run or checkpoint could not be resolved."

    return (
        "Could not resolve a checkpoint for play mode. "
        f"{reason} task={cfg.training.task_name} task_log_root={task_log_root} "
        f"algo.load_run={cfg.algo.load_run!r}{checkpoint_hint}."
        " Use algo.load_run=<run-dir-or-checkpoint-path> "
        "and optionally algo.checkpoint=<iteration-or-filename>."
    )


def _resolve_play_num_steps(cfg: DictConfig) -> int | None:
    play_steps = OmegaConf.select(cfg, "training.play_steps", default=None)
    if play_steps is None:
        return None
    return int(play_steps)


def play_rsl_rl(cfg: DictConfig, device: str) -> str | None:
    """Play mode for RSL-RL."""
    rl_cfg = _algo_config_dict(cfg)
    wrapper_cls = _resolve_ppo_wrapper_cls(rl_cfg)

    task_log_root = get_log_root(ROOT_DIR, cfg) / str(cfg.training.task_name)
    load_path, load_path_dir = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
    if load_path is None or load_path_dir is None or not load_path.exists():
        print(
            _format_play_checkpoint_error(
                cfg,
                task_log_root=task_log_root,
                load_path=load_path,
                load_path_dir=load_path_dir,
            )
        )
        return None

    print(f"Loading latest model: {load_path}")
    _ckpt_keys = set(torch.load(load_path, map_location="cpu", weights_only=True).keys())
    if "actor_state_dict" not in _ckpt_keys:
        print(
            f"Checkpoint at {load_path} is not an rsl-rl checkpoint "
            f"(found keys: {_ckpt_keys}). Aborting play."
        )
        return None

    cfg = (
        resolve_sim2sim_config(
            load_path_dir,
            cfg,
            algo_name="ppo",
            strict=bool(getattr(cfg.training, "sim2sim_strict", True)),
        )
        or cfg
    )

    env_cfg_override = build_ppo_play_env_cfg_override(cfg)

    env = create_env(
        cfg,
        num_envs=cfg.training.play_env_num,
        env_cfg_override=env_cfg_override,
    )
    wrapped_env = wrapper_cls(env, device=device)
    train_cfg = normalize_ppo_train_cfg(rl_cfg)
    apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
    if "runner" not in train_cfg:
        train_cfg["runner"] = {}
    train_cfg["runner"]["logger"] = "none"

    runner = cast(
        Any,
        OnPolicyRunner(cast(Any, wrapped_env), train_cfg, log_dir=None, device=device),
    )
    with policy_load_dim_guard(
        env_obs_dim=getattr(wrapped_env, "num_obs", None),
        env_action_dim=getattr(wrapped_env, "num_actions", None),
        algo_name="ppo",
    ):
        runner.load(str(load_path), map_location=device)
    policy = runner.get_inference_policy(device=device)
    num_steps = _resolve_play_num_steps(cfg)
    play_render_mode = str(getattr(cfg.training, "play_render_mode", "auto")).lower()
    if play_render_mode == "none":
        if num_steps is None:
            raise ValueError(
                "Headless checkpoint reload requires a finite training.play_steps value."
            )
        actor_hash_before = _state_dict_sha256(policy.state_dict())
        obs, _ = wrapped_env.reset()
        try:
            with torch.inference_mode():
                for step_index in range(num_steps):
                    actions = policy(obs)
                    if not torch.isfinite(actions).all():
                        raise FloatingPointError(
                            f"Non-finite checkpoint-reload action at step {step_index}"
                        )
                    obs, rewards, _, _ = wrapped_env.step(actions)
                    if not torch.isfinite(rewards).all():
                        raise FloatingPointError(
                            f"Non-finite checkpoint-reload reward at step {step_index}"
                        )
                    for key, value in obs.items():
                        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                            raise FloatingPointError(
                                f"Non-finite checkpoint-reload observation {key!r} "
                                f"at step {step_index}"
                            )
            actor_hash_after = _state_dict_sha256(policy.state_dict())
            if actor_hash_before != actor_hash_after:
                raise AssertionError("Headless checkpoint reload modified policy weights")
            print(
                f"Completed {num_steps} deterministic headless checkpoint-reload steps; "
                "policy weights unchanged."
            )
            return None
        finally:
            wrapped_env.close()

    if EXPORT_POLICY:
        runner.export_policy_to_onnx(path=str(load_path_dir))
        runner.export_policy_to_jit(path=str(load_path_dir))
    output_video = Path(load_path_dir) / "play_video.mp4"
    playback_mode: str | None = None

    def _log_plan(plan) -> None:
        nonlocal playback_mode
        playback_mode = plan.mode
        log_playback_plan(plan)

    try:
        with torch.inference_mode():
            play_video_path = env.run_playback_mode(
                play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
                play_steps=num_steps,
                output_video=output_video,
                render_spacing=float(
                    getattr(cfg.training, "render_spacing", getattr(env.cfg, "render_spacing", 1.0))
                ),
                render_offset_mode=str(getattr(env.cfg, "render_offset_mode", "grid")),
                initialize=lambda: wrapped_env.reset()[0],
                step=lambda obs: wrapped_env.step(policy(obs))[0],
                camera_kwargs={
                    "cam_distance": cfg.training.cam_distance,
                    "cam_elevation": cfg.training.cam_elevation,
                    "cam_azimuth": cfg.training.cam_azimuth,
                    "cam_lookat": getattr(cfg.training, "cam_lookat", None),
                    "cam_tracking": getattr(cfg.training, "cam_tracking", False),
                    "cam_tracking_env_idx": getattr(cfg.training, "cam_tracking_env_idx", 0),
                    "cam_tracking_extra_envs": getattr(cfg.training, "cam_tracking_extra_envs", 2),
                },
                on_plan=_log_plan,
                extra_data_getter=(
                    (lambda: getattr(env, "curr_ee_goal_world", None))
                    if hasattr(env, "curr_ee_goal_world")
                    else None
                ),
            )
    except Exception as e:
        if cfg.training.sim_backend == "motrix" and "RenderClosedError" in str(type(e).__name__):
            print("Render window closed.")
        else:
            raise
    if playback_mode != "none" and num_steps is not None:
        print("Done.")
    return play_video_path


def evaluate_rsl_rl(cfg: DictConfig, device: str) -> Path:
    """Run a checkpoint-only batch rollout with instrumented aggregation."""
    validate_evaluation_config(cfg)
    load_path, load_path_dir = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
    if load_path is None or load_path_dir is None or not load_path.exists():
        raise FileNotFoundError("Evaluation checkpoint could not be resolved.")
    load_path = load_path.resolve()
    load_path_dir = load_path_dir.resolve()
    cfg = recover_evaluation_config(load_path_dir, cfg)
    validate_evaluation_config(cfg)

    num_envs = int(cfg.evaluation.num_envs)
    num_steps = int(cfg.evaluation.num_steps)
    deterministic = bool(cfg.evaluation.deterministic)
    output_dir = _evaluation_output_dir(
        cfg,
        load_path=load_path,
        load_path_dir=load_path_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    rl_cfg = _algo_config_dict(cfg)
    wrapper_cls = _resolve_ppo_wrapper_cls(rl_cfg)
    env_cfg_override = build_ppo_env_cfg_override(cfg)
    env = create_env(
        cfg,
        num_envs=num_envs,
        env_cfg_override=env_cfg_override,
    )
    if str(cfg.training.task_name) == "LeapInhandBallStateCycleRotation":
        env.set_diagnostic_log_interval(1)
    wrapped_env = wrapper_cls(env, device=device)
    try:
        _assert_state_cycle_evaluation_contract(cfg, int(wrapped_env.num_obs))
        train_cfg = normalize_ppo_train_cfg(rl_cfg)
        apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
        train_cfg.setdefault("runner", {})
        train_cfg["runner"]["logger"] = "none"
        train_cfg["logger"] = "none"
        runner = cast(
            Any,
            OnPolicyRunner(
                cast(Any, wrapped_env),
                train_cfg,
                log_dir=None,
                device=device,
            ),
        )
        with policy_load_dim_guard(
            env_obs_dim=getattr(wrapped_env, "num_obs", None),
            env_action_dim=getattr(wrapped_env, "num_actions", None),
            algo_name="ppo",
        ):
            runner.load(str(load_path), map_location=device)
        policy = runner.get_inference_policy(device=device)
        actor_hash_before = _state_dict_sha256(policy.state_dict())

        ready_seconds = float(cfg.env.state_cycle.ready_to_a.timeout_seconds)
        a_to_b_seconds = float(cfg.env.state_cycle.a_to_b.timeout_seconds)
        b_to_ready_seconds = float(cfg.env.state_cycle.b_to_ready.timeout_seconds)
        ctrl_dt = float(cfg.env.ctrl_dt)
        pose_tracking_scale = float(cfg.reward.pose_tracking_scale)
        print(f"Resolved evaluation checkpoint:\n{load_path}\n")
        print("Resolved state-cycle task config:")
        print(f"Pose tracking scale: {pose_tracking_scale:.2f}")
        print(
            f"Ready->A timeout: {ready_seconds:.1f} s / "
            f"{round(ready_seconds / ctrl_dt)} steps"
        )
        print(
            f"A->B timeout:     {a_to_b_seconds:.1f} s / "
            f"{round(a_to_b_seconds / ctrl_dt)} steps"
        )
        print(
            f"B->Ready timeout: {b_to_ready_seconds:.1f} s / "
            f"{round(b_to_ready_seconds / ctrl_dt)} steps\n"
        )
        print(f"Observation dim: {wrapped_env.num_obs}")
        print("Training updates: disabled")
        print(f"Deterministic policy: {str(deterministic).lower()}")

        obs, _ = wrapped_env.reset()
        episode_returns = np.zeros(num_envs, dtype=np.float64)
        episode_lengths = np.zeros(num_envs, dtype=np.int64)
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        metric_rows: list[dict[str, float]] = []

        with torch.inference_mode():
            for step_index in range(num_steps):
                actions = evaluation_policy_actions(
                    policy,
                    obs,
                    deterministic=deterministic,
                )
                next_obs, rewards, dones, infos = wrapped_env.step(actions)
                rewards_np = rewards.detach().cpu().numpy()
                dones_np = dones.detach().cpu().numpy().astype(bool)
                if not np.isfinite(rewards_np).all():
                    raise FloatingPointError(
                        f"Non-finite evaluation reward at step {step_index}"
                    )
                episode_returns += rewards_np
                episode_lengths += 1
                if np.any(dones_np):
                    completed_returns.extend(episode_returns[dones_np].tolist())
                    completed_lengths.extend(episode_lengths[dones_np].tolist())
                    episode_returns[dones_np] = 0.0
                    episode_lengths[dones_np] = 0

                raw_log = infos.get("log")
                if isinstance(raw_log, dict):
                    row = {"evaluation_step": float(step_index)}
                    for key, value in raw_log.items():
                        if isinstance(value, torch.Tensor):
                            value = value.detach().cpu().item()
                        elif isinstance(value, np.ndarray):
                            value = value.item()
                        scalar = float(value)
                        if not np.isfinite(scalar):
                            raise FloatingPointError(
                                f"Non-finite evaluation metric {key!r} "
                                f"at step {step_index}"
                            )
                        row[str(key)] = scalar
                    metric_rows.append(row)
                obs = next_obs

        present_keys = {key for row in metric_rows for key in row}
        if str(cfg.training.task_name) == "LeapInhandBallStateCycleRotation":
            missing = sorted(_required_state_cycle_evaluation_keys() - present_keys)
            if missing:
                raise KeyError(f"Evaluation diagnostics are missing keys: {missing}")

        actor_hash_after = _state_dict_sha256(policy.state_dict())
        if actor_hash_before != actor_hash_after:
            raise AssertionError("Evaluation modified actor policy weights")

        summary = build_evaluation_summary(
            metric_rows=metric_rows,
            completed_returns=completed_returns,
            completed_lengths=completed_lengths,
            ctrl_dt=ctrl_dt,
        )
        run_payload = _read_run_config(load_path_dir)
        source_git = run_payload.get("run", {}).get("git", {})
        eval_config = {
            "checkpoint": str(load_path),
            "checkpoint_run": load_path_dir.name,
            "checkpoint_commit": source_git.get("commit"),
            "evaluation_code_commit": get_git_info(ROOT_DIR).get("commit"),
            "num_envs": num_envs,
            "num_steps": num_steps,
            "seed": int(cfg.algo.seed),
            "deterministic": deterministic,
            "observation_dim": int(wrapped_env.num_obs),
            "ctrl_dt": ctrl_dt,
            "pose_tracking_scale": pose_tracking_scale,
            "ready_to_a_timeout_seconds": ready_seconds,
            "a_to_b_timeout_seconds": a_to_b_seconds,
            "b_to_ready_timeout_seconds": b_to_ready_seconds,
            "actor_hash_before": actor_hash_before,
            "actor_hash_after": actor_hash_after,
            "training_updates": False,
        }
        (output_dir / "eval_config.json").write_text(
            json.dumps(eval_config, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        (output_dir / "eval_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        fieldnames = ["evaluation_step"] + sorted(
            present_keys - {"evaluation_step"}
        )
        with (output_dir / "eval_step_metrics.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metric_rows)

        if bool(cfg.evaluation.write_tensorboard):
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(output_dir))
            try:
                for row in metric_rows:
                    eval_step = int(row["evaluation_step"])
                    for key, value in row.items():
                        if key != "evaluation_step":
                            writer.add_scalar(key, value, eval_step)
                for key, value in summary.items():
                    writer.add_scalar(key, value, 0)
            finally:
                writer.close()
        print(f"Evaluation output: {output_dir}")
        print(f"Actor SHA256 before: {actor_hash_before}")
        print(f"Actor SHA256 after:  {actor_hash_after}")
        return output_dir
    finally:
        wrapped_env.close()


@hydra.main(version_base="1.3", config_path="../conf/ppo", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()
    validate_evaluation_config(cfg)

    seed_info = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)

    device = get_default_device()
    print(f"Using device: {device}")

    if bool(cfg.training.eval_only):
        evaluate_rsl_rl(cfg, device)
        return

    env_cfg_override = build_ppo_env_cfg_override(cfg)

    # Compute effective max_iterations (supports num_timesteps override)
    max_iterations = cfg.algo.max_iterations
    if cfg.training.num_timesteps:
        n_steps_per_iter = cfg.algo.num_steps_per_env * cfg.algo.num_envs
        max_iterations = max(1, int(cfg.training.num_timesteps / n_steps_per_iter))
        print(
            f"Overriding max_iterations to {max_iterations} based on "
            f"num_timesteps {cfg.training.num_timesteps}"
        )

    if not cfg.training.play_only:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_root = _get_log_root(cfg)
        log_dir = str(
            Path(log_root) / cfg.training.task_name / f"{timestamp}_{cfg.training.sim_backend}"
        )
    else:
        log_dir = None

    tracker = None
    if not cfg.training.play_only and log_dir is not None:
        tracker = ExperimentTracker(
            root_dir=ROOT_DIR,
            log_dir=log_dir,
            algo_name="ppo",
            task_name=cfg.training.task_name,
            sim_backend=cfg.training.sim_backend,
            training_cfg=cfg.training,
            full_cfg=cfg,
            device=device,
            seed_info=seed_info,
        )
        tracker.start()

    try:
        if not cfg.training.play_only:
            env = create_env(
                cfg,
                num_envs=cfg.algo.num_envs,
                env_cfg_override=env_cfg_override,
            )
            rl_cfg = _algo_config_dict(cfg)
            wrapper_cls = _resolve_ppo_wrapper_cls(rl_cfg)

            nan_guard_cfg = getattr(cfg.training, "nan_guard", None)
            if nan_guard_cfg is not None and getattr(nan_guard_cfg, "enabled", False):
                from unilab.utils.nan_guard import NanGuard, NanGuardCfg

                guard = NanGuard(
                    NanGuardCfg(
                        enabled=True,
                        buffer_size=int(getattr(nan_guard_cfg, "buffer_size", 100)),
                        max_envs_to_dump=int(getattr(nan_guard_cfg, "max_envs_to_dump", 5)),
                        output_dir=getattr(nan_guard_cfg, "output_dir", None),
                    ),
                    num_envs=env.num_envs,
                    supports_state_playback=env.play_capabilities.supports_physics_state_playback,
                )
                env.set_nan_guard(guard)

            wrapped_env = wrapper_cls(env, device=device)

            train_cfg = normalize_ppo_train_cfg(rl_cfg)
            apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=True)
            if "runner" not in train_cfg:
                train_cfg["runner"] = {}

            logger_type, suppress_console_log = _resolve_rsl_rl_logger(
                str(cfg.training.logger)
            )
            train_cfg["runner"]["logger"] = logger_type
            train_cfg["logger"] = logger_type

            patch_rsl_rl_resume_state()

            if tracker is not None and logger_type == "wandb":
                patch_rsl_rl_wandb_writer()
                wandb_settings = tracker.wandb_settings
                train_cfg["wandb_project"] = wandb_settings["project"]
                train_cfg["wandb_entity"] = wandb_settings["entity"]
                train_cfg["wandb_group"] = wandb_settings["group"]
                train_cfg["wandb_job_type"] = wandb_settings["job_type"]
                train_cfg["wandb_tags"] = wandb_settings["tags"]
                train_cfg["wandb_notes"] = wandb_settings["notes"]
                train_cfg["wandb_mode"] = wandb_settings["mode"]

            runner = cast(
                Any,
                OnPolicyRunner(cast(Any, wrapped_env), train_cfg, log_dir=log_dir, device=device),
            )
            _patch_runner_action_std_logging(
                runner,
                suppress_console=suppress_console_log,
            )

            if cfg.algo.load_run != "-1":
                resume_path, _ = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
                if resume_path:
                    load_mode = str(getattr(cfg.algo, "load_mode", "resume"))
                    if load_mode == "warm_start_policy":
                        print(
                            f"Warm-starting policy from {resume_path}; critic, optimizer, "
                            "iteration, logger state, and action std remain freshly initialized."
                        )
                    elif load_mode == "warm_start_actor_critic":
                        print(
                            f"Warm-starting actor and critic from {resume_path}; optimizer, "
                            "iteration, logger state, and action std remain freshly initialized."
                        )
                    else:
                        print(f"Resuming from {resume_path}")
                    load_rsl_rl_training_checkpoint(
                        runner,
                        str(resume_path),
                        load_mode=load_mode,
                        map_location=device,
                    )

            train_start_wall = time.time()
            runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
            assert log_dir is not None
            train_summary = {
                "status": "completed",
                "completed_iterations": int(runner.current_learning_iteration),
                "total_env_steps": int(getattr(runner.logger, "tot_timesteps", 0)),
                "final_mean_reward": (
                    float(statistics.mean(runner.logger.rewbuffer))
                    if len(getattr(runner.logger, "rewbuffer", [])) > 0
                    else None
                ),
                "best_mean_reward": (
                    float(max(runner.logger.rewbuffer))
                    if len(getattr(runner.logger, "rewbuffer", [])) > 0
                    else None
                ),
                "mean_episode_length": (
                    float(statistics.mean(runner.logger.lenbuffer))
                    if len(getattr(runner.logger, "lenbuffer", [])) > 0
                    else None
                ),
                "last_checkpoint": str(
                    Path(log_dir) / f"model_{int(runner.current_learning_iteration)}.pt"
                ),
                "training_wall_time_sec": time.time() - train_start_wall,
            }
            if tracker is not None:
                tracker.update_summary(train_summary)
            env.close()

        if bool(cfg.training.play_only) or should_run_playback(
            play_only=cfg.training.play_only,
            no_play=cfg.training.no_play,
            play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
        ):
            play_video_path = play_rsl_rl(cfg, device)
            if tracker is not None:
                tracker.log_video(play_video_path)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    EXPORT_POLICY = True
    main()
