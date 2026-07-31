"""Read-only physics replay validator for a LEAP Allegro-style grasp cache."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unilab.base.np_env import NpEnv
from unilab.envs.manipulation.leap_inhand.ball_grasp_allegro import (
    LeapAllegroGraspResetProvider,
)
from unilab.training import BackendAdapter, create_env, ensure_registries

_TASK_VARIANT = "leap_inhand_ball_grasp_allegro/mujoco"
_TASK_NAME = "LeapInhandBallGraspAllegro"
_CONTACT_NAMES = ("index", "middle", "ring", "thumb")
_ROW_WIDTH = 23
_HAND_DOF = 16


class ValidationInputError(ValueError):
    """Invalid CLI argument or cache schema."""


@dataclass
class CacheSaveGuard:
    """Fail closed if any generator lifecycle attempts to publish a cache."""

    called: bool = False

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.called = True
        raise RuntimeError("Replay validator forbids all cache save attempts")


@dataclass(frozen=True)
class ConditionSnapshot:
    cond1: np.ndarray
    cond2: np.ndarray
    cond3: np.ndarray
    valid: np.ndarray
    contact_count: np.ndarray
    contact_flags: np.ndarray
    fingertip_distances: np.ndarray
    max_fingertip_distance: np.ndarray
    ball_center_z: np.ndarray
    height_margin: np.ndarray
    distance_margin: np.ndarray


def validate_cache_array(
    rows: np.ndarray,
    *,
    batch_size: int,
    settle_seconds: float,
) -> np.ndarray:
    """Validate without casting, normalizing, or mutating the source array."""
    if batch_size <= 0:
        raise ValidationInputError("batch-size must be positive")
    if not np.isfinite(settle_seconds) or settle_seconds <= 0.0:
        raise ValidationInputError("settle-seconds must be positive and finite")
    if rows.ndim != 2 or rows.shape[1] != _ROW_WIDTH:
        raise ValidationInputError(f"cache must have shape (?, {_ROW_WIDTH}), got {rows.shape}")
    if rows.shape[0] == 0:
        raise ValidationInputError("cache must contain at least one row")
    if not np.isfinite(rows).all():
        raise ValidationInputError("cache contains NaN or Inf")
    quaternion_norm = np.linalg.norm(np.asarray(rows[:, 19:23], dtype=np.float64), axis=1)
    if np.any(quaternion_norm <= 1e-8):
        raise ValidationInputError("cache contains a quaternion with norm <= 1e-8")
    return rows


def load_cache(
    path: Path,
    *,
    batch_size: int,
    settle_seconds: float,
) -> np.ndarray:
    if path.suffix.lower() != ".npy":
        raise ValidationInputError("cache path must use a .npy suffix")
    if not path.is_file():
        raise ValidationInputError(f"cache does not exist: {path}")
    try:
        rows = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise ValidationInputError(f"failed to read cache: {exc}") from exc
    return validate_cache_array(rows, batch_size=batch_size, settle_seconds=settle_seconds)


def _compose_task_cfg() -> Any:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "ppo"), version_base="1.3"):
        return compose("config", overrides=[f"task={_TASK_VARIANT}"])


def _create_replay_env(path: Path, batch_size: int) -> Any:
    ensure_registries()
    cfg = _compose_task_cfg()
    adapter = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="ppo")
    env_override = adapter.build_task_env_cfg_override()
    env_override.update(
        {
            "gen_grasp": False,
            "grasp_auto_save": False,
            "grasp_cache_path": str(path.resolve()),
        }
    )
    return create_env(
        cfg,
        num_envs=batch_size,
        env_cfg_override=env_override,
        sim_backend="mujoco",
        task_name=_TASK_NAME,
    )


def _contact_flags(env: Any) -> np.ndarray:
    return np.stack(
        [
            np.asarray(env._sensor_scalar(env.get_sensor_data(name)) > 0.5, dtype=bool)
            for name in env._CONTACT_SENSORS
        ],
        axis=1,
    )


def measure_conditions(env: Any, active_count: int) -> ConditionSnapshot:
    """Use the production conditions and collect diagnostics without new gates."""
    cond1, cond2, cond3 = env._compute_grasp_conditions()
    ball_pos = np.asarray(env.get_ball_pos(), dtype=np.float64)
    fingertip_pos = np.asarray(env.get_fingertip_pos(), dtype=np.float64)
    distances = np.linalg.norm(fingertip_pos - ball_pos[:, None, :], axis=-1)
    flags = _contact_flags(env)
    contact_count = np.count_nonzero(flags, axis=1).astype(np.int32)
    maximum = np.max(distances, axis=1)
    limit = float(env.cfg.grasp_max_fingertip_distance)
    height = float(env._reward_cfg.reset_z_threshold)
    active = slice(0, active_count)
    c1 = np.asarray(cond1[active], dtype=bool).copy()
    c2 = np.asarray(cond2[active], dtype=bool).copy()
    c3 = np.asarray(cond3[active], dtype=bool).copy()
    return ConditionSnapshot(
        cond1=c1,
        cond2=c2,
        cond3=c3,
        valid=c1 & c2 & c3,
        contact_count=contact_count[active].copy(),
        contact_flags=flags[active].copy(),
        fingertip_distances=distances[active].copy(),
        max_fingertip_distance=maximum[active].copy(),
        ball_center_z=ball_pos[active, 2].copy(),
        height_margin=(ball_pos[active, 2] - height).copy(),
        distance_margin=(limit - maximum[active]).copy(),
    )


def _pad_final_batch(rows: np.ndarray, batch_size: int) -> np.ndarray:
    if rows.shape[0] == batch_size:
        return np.asarray(rows, dtype=np.float64).copy()
    padded = np.empty((batch_size, _ROW_WIDTH), dtype=np.float64)
    padded[: rows.shape[0]] = np.asarray(rows, dtype=np.float64)
    padded[rows.shape[0] :] = padded[rows.shape[0] - 1]
    return padded


def initialize_replay_batch(
    env: Any,
    provider: LeapAllegroGraspResetProvider,
    rows: np.ndarray,
) -> None:
    """Install cache rows and synchronize the production control/observation state."""
    replay_rows = np.asarray(rows, dtype=np.float64)
    num_envs = replay_rows.shape[0]
    hand_qpos = replay_rows[:, :_HAND_DOF]
    ball_pos = replay_rows[:, 16:19]
    ball_quat = replay_rows[:, 19:23]
    qvel = np.zeros((num_envs, env.nv), dtype=np.float64)
    env_ids = np.arange(num_envs, dtype=np.int32)
    env._backend.set_state(env_ids, replay_rows, qvel)

    info_updates = provider._build_info_updates(env, hand_qpos, ball_pos, ball_quat)
    # Preserve the cache row values exactly; in particular, never normalize quaternions
    # and never replace the held control target with the nominal seed.
    info_updates.update(
        {
            "prev_ctrl": hand_qpos.copy(),
            "init_pose": hand_qpos.copy(),
            "prev_dof_pos": hand_qpos.copy(),
            "prev_ball_pos": ball_pos.copy(),
            "prev_ball_quat": ball_quat.copy(),
            "current_actions": np.zeros((num_envs, _HAND_DOF), dtype=np.float64),
            "last_actions": np.zeros((num_envs, _HAND_DOF), dtype=np.float64),
        }
    )
    if env.state is None:
        raise RuntimeError("environment state must be initialized before replay")
    env.state.info.update(info_updates)
    env.state.info["steps"] = np.zeros((num_envs,), dtype=np.uint32)
    env.state.terminated[:] = False
    env.state.truncated[:] = False


def compute_timeout_success(
    *,
    final_truncated: np.ndarray,
    ever_terminated: np.ndarray,
    final_cond1: np.ndarray,
    final_cond2: np.ndarray,
    final_cond3: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        final_truncated
        & ~ever_terminated
        & final_cond1
        & final_cond2
        & final_cond3,
        dtype=bool,
    )


def _distribution(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)}


def _global_failed_indices(failed: np.ndarray, start: int) -> list[int]:
    return (np.flatnonzero(np.asarray(failed, dtype=bool)) + int(start)).astype(int).tolist()


def _snapshot_summary(snapshot: ConditionSnapshot) -> dict[str, float | int]:
    count = int(snapshot.valid.size)
    return {
        "valid_count": int(np.count_nonzero(snapshot.valid)),
        "valid_rate": float(np.mean(snapshot.valid)),
        "fingertip_distance_failure_count": int(np.count_nonzero(~snapshot.cond1)),
        "contact_failure_count": int(np.count_nonzero(~snapshot.cond2)),
        "height_failure_count": int(np.count_nonzero(~snapshot.cond3)),
        "row_count": count,
    }


def _row_snapshot(snapshot: ConditionSnapshot, index: int) -> dict[str, Any]:
    return {
        "cond1": bool(snapshot.cond1[index]),
        "cond2": bool(snapshot.cond2[index]),
        "cond3": bool(snapshot.cond3[index]),
        "valid": bool(snapshot.valid[index]),
        "contact_count": int(snapshot.contact_count[index]),
        "contact_flags": {
            name: bool(value)
            for name, value in zip(
                _CONTACT_NAMES, snapshot.contact_flags[index], strict=True
            )
        },
        "fingertip_body_origin_distances": snapshot.fingertip_distances[index].tolist(),
        "max_fingertip_distance": float(snapshot.max_fingertip_distance[index]),
        "ball_center_z": float(snapshot.ball_center_z[index]),
        "height_margin": float(snapshot.height_margin[index]),
        "fingertip_distance_margin": float(snapshot.distance_margin[index]),
    }


def _close_without_generator_lifecycle(env: Any) -> None:
    if isinstance(env, NpEnv):
        NpEnv.close(env)
        return
    cleanup = getattr(getattr(env, "_backend", None), "cleanup_scene_assets", None)
    if callable(cleanup):
        cleanup()


def replay_validate(
    path: Path,
    *,
    batch_size: int = 256,
    settle_seconds: float = 3.0,
    env_factory: Callable[[Path, int], Any] = _create_replay_env,
) -> dict[str, Any]:
    rows = load_cache(path, batch_size=batch_size, settle_seconds=settle_seconds)
    env = env_factory(path, batch_size)
    save_guard = CacheSaveGuard()
    env._save_grasp_cache = save_guard
    provider = LeapAllegroGraspResetProvider()
    try:
        env.init_state()
        env.set_autoreset(False)
        settle_steps = int(np.ceil(settle_seconds / float(env.cfg.ctrl_dt)))
        if settle_steps <= 0:
            raise ValidationInputError("settle-seconds resolves to zero control steps")

        initial_snapshots: list[ConditionSnapshot] = []
        final_snapshots: list[ConditionSnapshot] = []
        row_records: list[dict[str, Any]] = []
        initial_invalid: list[int] = []
        ever_terminated_indices: list[int] = []
        final_invalid: list[int] = []
        timeout_failure: list[int] = []
        all_timeout_success: list[np.ndarray] = []
        all_ever_terminated: list[np.ndarray] = []
        all_ever_failed_cond1: list[np.ndarray] = []
        all_ever_failed_cond2: list[np.ndarray] = []
        all_ever_failed_cond3: list[np.ndarray] = []

        for start in range(0, rows.shape[0], batch_size):
            source_batch = rows[start : start + batch_size]
            active_count = int(source_batch.shape[0])
            replay_batch = _pad_final_batch(source_batch, batch_size)
            initialize_replay_batch(env, provider, replay_batch)

            initial = measure_conditions(env, active_count)
            ever_terminated = np.zeros((active_count,), dtype=bool)
            ever_failed_cond1 = ~initial.cond1.copy()
            ever_failed_cond2 = ~initial.cond2.copy()
            ever_failed_cond3 = ~initial.cond3.copy()
            zero_actions = np.zeros((batch_size, _HAND_DOF), dtype=np.float64)
            for _ in range(settle_steps):
                state = env.step(zero_actions)
                ever_terminated |= np.asarray(state.terminated[:active_count], dtype=bool)
                current = measure_conditions(env, active_count)
                ever_failed_cond1 |= ~current.cond1
                ever_failed_cond2 |= ~current.cond2
                ever_failed_cond3 |= ~current.cond3

            final = measure_conditions(env, active_count)
            if env.state is None:
                raise RuntimeError("environment state disappeared during replay")
            final_terminated = np.asarray(
                env.state.terminated[:active_count], dtype=bool
            ).copy()
            final_truncated = np.asarray(
                env.state.truncated[:active_count], dtype=bool
            ).copy()
            timeout_success = compute_timeout_success(
                final_truncated=final_truncated,
                ever_terminated=ever_terminated,
                final_cond1=final.cond1,
                final_cond2=final.cond2,
                final_cond3=final.cond3,
            )

            initial_snapshots.append(initial)
            final_snapshots.append(final)
            all_timeout_success.append(timeout_success)
            all_ever_terminated.append(ever_terminated)
            all_ever_failed_cond1.append(ever_failed_cond1)
            all_ever_failed_cond2.append(ever_failed_cond2)
            all_ever_failed_cond3.append(ever_failed_cond3)

            initial_invalid.extend(_global_failed_indices(~initial.valid, start))
            ever_terminated_indices.extend(_global_failed_indices(ever_terminated, start))
            final_invalid.extend(_global_failed_indices(~final.valid, start))
            timeout_failure.extend(_global_failed_indices(~timeout_success, start))
            for local_index in range(active_count):
                global_index = start + local_index
                row_records.append(
                    {
                        "index": global_index,
                        "initial": _row_snapshot(initial, local_index),
                        "replay": {
                            "ever_terminated": bool(ever_terminated[local_index]),
                            "ever_failed_cond1": bool(ever_failed_cond1[local_index]),
                            "ever_failed_cond2": bool(ever_failed_cond2[local_index]),
                            "ever_failed_cond3": bool(ever_failed_cond3[local_index]),
                        },
                        "final": {
                            **_row_snapshot(final, local_index),
                            "terminated": bool(final_terminated[local_index]),
                            "truncated": bool(final_truncated[local_index]),
                        },
                        "timeout_success": bool(timeout_success[local_index]),
                    }
                )

        initial_all = _concatenate_snapshots(initial_snapshots)
        final_all = _concatenate_snapshots(final_snapshots)
        timeout_success_all = np.concatenate(all_timeout_success)
        ever_terminated_all = np.concatenate(all_ever_terminated)
        ever_failed_cond1_all = np.concatenate(all_ever_failed_cond1)
        ever_failed_cond2_all = np.concatenate(all_ever_failed_cond2)
        ever_failed_cond3_all = np.concatenate(all_ever_failed_cond3)
        report: dict[str, Any] = {
            "path": str(path),
            "row_count": int(rows.shape[0]),
            "batch_size": int(batch_size),
            "settle_seconds": float(settle_seconds),
            "settle_steps": settle_steps,
            "cache_save_called": bool(save_guard.called),
            "initial": _snapshot_summary(initial_all),
            "replay": {
                "ever_terminated_count": int(np.count_nonzero(ever_terminated_all)),
                "ever_terminated_rate": float(np.mean(ever_terminated_all)),
                "ever_fingertip_failure_count": int(np.count_nonzero(ever_failed_cond1_all)),
                "ever_contact_failure_count": int(np.count_nonzero(ever_failed_cond2_all)),
                "ever_height_failure_count": int(np.count_nonzero(ever_failed_cond3_all)),
            },
            "final": _snapshot_summary(final_all),
            "timeout_success_count": int(np.count_nonzero(timeout_success_all)),
            "timeout_success_rate": float(np.mean(timeout_success_all)),
            "contact_count_distribution_initial": _distribution(initial_all.contact_count),
            "contact_count_distribution_final": _distribution(final_all.contact_count),
            "height_margin": {
                "initial_min": float(np.min(initial_all.height_margin)),
                "initial_mean": float(np.mean(initial_all.height_margin)),
                "final_min": float(np.min(final_all.height_margin)),
                "final_mean": float(np.mean(final_all.height_margin)),
            },
            "fingertip_distance_margin": {
                "initial_min": float(np.min(initial_all.distance_margin)),
                "initial_mean": float(np.mean(initial_all.distance_margin)),
                "final_min": float(np.min(final_all.distance_margin)),
                "final_mean": float(np.mean(final_all.distance_margin)),
            },
            "near_height_threshold": {
                "below_0_1_mm_count": int(np.count_nonzero(final_all.height_margin < 0.0001)),
                "below_0_5_mm_count": int(np.count_nonzero(final_all.height_margin < 0.0005)),
                "below_1_0_mm_count": int(np.count_nonzero(final_all.height_margin < 0.0010)),
            },
            "failed_row_indices": {
                "initial_invalid": initial_invalid,
                "ever_terminated": ever_terminated_indices,
                "final_invalid": final_invalid,
                "timeout_failure": timeout_failure,
            },
            "rows": row_records,
        }
        return report
    finally:
        _close_without_generator_lifecycle(env)


def _concatenate_snapshots(snapshots: list[ConditionSnapshot]) -> ConditionSnapshot:
    return ConditionSnapshot(
        **{
            field: np.concatenate([getattr(snapshot, field) for snapshot in snapshots], axis=0)
            for field in ConditionSnapshot.__dataclass_fields__
        }
    )


def _console_report(report: dict[str, Any]) -> dict[str, Any]:
    compact = {key: value for key, value in report.items() if key != "rows"}
    compact["failed_row_indices"] = {
        key: {
            "count": len(indices),
            "first_100": indices[:100],
        }
        for key, indices in report["failed_row_indices"].items()
    }
    return compact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.output_json is not None and args.output_json.suffix.lower() != ".json":
            raise ValidationInputError("output-json must use a .json suffix")
        report = replay_validate(
            args.path,
            batch_size=args.batch_size,
            settle_seconds=args.settle_seconds,
        )
        if args.output_json is not None:
            args.output_json.write_text(
                json.dumps(report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        print(json.dumps(_console_report(report), indent=2, sort_keys=True))
    except (ValidationInputError, OSError) as exc:
        print(f"Validation input error: {exc}", file=sys.stderr)
        return 2
    if (
        report["cache_save_called"]
        or report["initial"]["valid_count"] != report["row_count"]
        or report["timeout_success_rate"] != 1.0
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
