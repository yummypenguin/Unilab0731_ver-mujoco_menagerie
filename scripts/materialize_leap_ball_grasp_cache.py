"""Materialize a reload-stable LEAP ball-grasp cache from raw generator output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from unilab.envs.manipulation.allegro_inhand.rotation import RewardConfigPPO
from unilab.envs.manipulation.leap_inhand.ball_grasp_gen import (
    GraspCacheReplayResult,
    LeapInhandBallGraspCfg,
    LeapInhandBallGraspEnv,
    deduplicate_grasp_cache_rows,
    normalize_grasp_cache_rows,
    save_grasp_cache_atomic,
)

_CONTACT_NAMES = ("index", "middle", "ring", "thumb")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--target", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--settle-seconds", type=float, default=2.5)
    parser.add_argument("--replay-passes", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--max-self-penetration", type=float, default=0.001)
    parser.add_argument("--max-object-penetration", type=float, default=0.001)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.target <= 0:
        parser.error("--target must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.settle_seconds <= 0.0:
        parser.error("--settle-seconds must be positive")
    if args.replay_passes < 2:
        parser.error("--replay-passes must be at least 2")
    if args.max_self_penetration < 0.0 or args.max_object_penetration < 0.0:
        parser.error("penetration limits must be non-negative")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validation_cfg(args: argparse.Namespace) -> LeapInhandBallGraspCfg:
    return LeapInhandBallGraspCfg(
        grasp_auto_save=False,
        grasp_collection_target=0,
        grasp_max_self_penetration=float(args.max_self_penetration),
        grasp_max_object_penetration=float(args.max_object_penetration),
        max_episode_seconds=max(20.0, float(args.settle_seconds) * 2.0),
        reward_config=RewardConfigPPO(
            scales={
                "rotate": 0.0,
                "obj_linvel": 0.0,
                "pose_diff": 0.0,
                "torque": 0.0,
                "work": 0.0,
                "drop": 0.0,
            },
            angvel_clip_min=-0.5,
            angvel_clip_max=0.5,
            reset_z_threshold=0.4,
        ),
    )


def _combine_results(results: list[GraspCacheReplayResult]) -> GraspCacheReplayResult:
    if not results:
        raise ValueError("Cannot combine an empty replay result list")
    condition_names = tuple(results[0].conditions)
    measurement_names = tuple(results[0].measurements)
    return GraspCacheReplayResult(
        settle_steps=results[0].settle_steps,
        settle_seconds=results[0].settle_seconds,
        accepted=np.concatenate([result.accepted for result in results]),
        terminated_during_settle=np.concatenate(
            [result.terminated_during_settle for result in results]
        ),
        final_quality=np.concatenate([result.final_quality for result in results]),
        penetration_valid=np.concatenate([result.penetration_valid for result in results]),
        conditions={
            name: np.concatenate([result.conditions[name] for result in results])
            for name in condition_names
        },
        contacts=np.concatenate([result.contacts for result in results]),
        measurements={
            name: np.concatenate([result.measurements[name] for result in results])
            for name in measurement_names
        },
        self_penetration=np.concatenate([result.self_penetration for result in results]),
        object_penetration=np.concatenate([result.object_penetration for result in results]),
    )


def _pass_report(result: GraspCacheReplayResult) -> dict[str, Any]:
    accepted_contacts = result.contacts[result.accepted]
    contact_counts = np.sum(accepted_contacts, axis=1) if accepted_contacts.size else np.array([])
    return {
        "input_rows": int(result.accepted.size),
        "accepted_rows": int(np.count_nonzero(result.accepted)),
        "rejected_rows": int(np.count_nonzero(~result.accepted)),
        "settle_steps": result.settle_steps,
        "settle_seconds": result.settle_seconds,
        "transient_termination_rows": int(
            np.count_nonzero(result.terminated_during_settle)
        ),
        "final_quality_failures": int(np.count_nonzero(~result.final_quality)),
        "penetration_failures": int(np.count_nonzero(~result.penetration_valid)),
        "condition_failures": {
            name: int(np.count_nonzero(~values))
            for name, values in result.conditions.items()
        },
        "max_self_penetration_mm": float(
            np.max(result.self_penetration, initial=0.0) * 1_000.0
        ),
        "max_object_penetration_mm": float(
            np.max(result.object_penetration, initial=0.0) * 1_000.0
        ),
        "accepted_contact_count": {
            str(count): int(np.count_nonzero(contact_counts == count)) for count in range(5)
        },
        "accepted_finger_contact_ratio": {
            name: float(np.mean(accepted_contacts[:, index])) if accepted_contacts.size else 0.0
            for index, name in enumerate(_CONTACT_NAMES)
        },
        "measurement_max": {
            name: float(np.max(values, initial=0.0))
            for name, values in result.measurements.items()
        },
    }


def _validate_pass(
    rows: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    env = LeapInhandBallGraspEnv(
        _validation_cfg(args),
        num_envs=min(int(args.batch_size), len(rows)),
        backend_type="mujoco",
    )
    results: list[GraspCacheReplayResult] = []
    try:
        for start in range(0, len(rows), int(args.batch_size)):
            batch = rows[start : start + int(args.batch_size)]
            results.append(
                env.replay_validate_grasp_cache_rows(
                    batch,
                    settle_seconds=float(args.settle_seconds),
                )
            )
    finally:
        env.close()
    combined = _combine_results(results)
    return rows[combined.accepted], combined.contacts[combined.accepted], _pass_report(combined)


def _joint_statistics(rows: np.ndarray) -> list[dict[str, float | int]]:
    stats: list[dict[str, float | int]] = []
    for index in range(16):
        values = np.asarray(rows[:, index], dtype=np.float64)
        stats.append(
            {
                "index": index,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "q05": float(np.quantile(values, 0.05)),
                "q50": float(np.quantile(values, 0.50)),
                "q95": float(np.quantile(values, 0.95)),
            }
        )
    return stats


def materialize(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    report_path = args.report or args.output.with_suffix(".json")
    output_resolved = args.output.resolve()
    report_resolved = report_path.resolve()
    input_resolved = {path.resolve() for path in args.input}
    if output_resolved in input_resolved:
        raise ValueError("Materialized output must be different from every raw input")
    if report_resolved == output_resolved:
        raise ValueError("Report path must be different from cache output path")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing grasp cache: {args.output}")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")

    source_rows: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    for path in args.input:
        rows = normalize_grasp_cache_rows(np.load(path))
        source_rows.append(rows)
        sources.append({"path": str(path.resolve()), "rows": len(rows), "sha256": _sha256(path)})
    raw_rows = np.concatenate(source_rows, axis=0)
    unique_rows, unique_indices = deduplicate_grasp_cache_rows(raw_rows)

    report: dict[str, Any] = {
        "contract": {
            "layout": "hand_qpos[16] + ball_pos[3] + ball_quat_wxyz[4]",
            "settle_seconds": float(args.settle_seconds),
            "replay_passes": int(args.replay_passes),
            "max_self_penetration_m": float(args.max_self_penetration),
            "max_object_penetration_m": float(args.max_object_penetration),
            "requires_no_transient_termination": True,
        },
        "sources": sources,
        "raw_rows": len(raw_rows),
        "unique_rows": len(unique_rows),
        "duplicate_rows": len(raw_rows) - len(unique_indices),
        "passes": [],
        "target_rows": int(args.target),
        "selection_seed": int(args.selection_seed),
    }

    rows = unique_rows
    contacts = np.empty((len(rows), 4), dtype=bool)
    for replay_pass in range(1, int(args.replay_passes) + 1):
        rows, contacts, pass_report = _validate_pass(rows, args)
        pass_report["pass"] = replay_pass
        report["passes"].append(pass_report)
        if len(rows) == 0:
            break

    report["stable_rows_before_selection"] = len(rows)
    enough_rows = len(rows) >= int(args.target)
    if enough_rows:
        rng = np.random.default_rng(int(args.selection_seed))
        selected = rng.permutation(len(rows))[: int(args.target)]
        rows = rows[selected]
        contacts = contacts[selected]
        save_grasp_cache_atomic(args.output, rows, overwrite=bool(args.overwrite))
        report["output"] = {
            "path": str(args.output.resolve()),
            "rows": len(rows),
            "shape": list(rows.shape),
            "dtype": str(rows.dtype),
            "sha256": _sha256(args.output),
        }
        report["final_contact_count"] = {
            str(count): int(np.count_nonzero(np.sum(contacts, axis=1) == count))
            for count in range(5)
        }
        report["final_finger_contact_ratio"] = {
            name: float(np.mean(contacts[:, index]))
            for index, name in enumerate(_CONTACT_NAMES)
        }
        report["joint_statistics"] = _joint_statistics(rows)
    else:
        report["output"] = None
        report["shortfall_rows"] = int(args.target) - len(rows)

    report["report_path"] = str(report_resolved)
    _write_json_atomic(report_path, report, overwrite=bool(args.overwrite))
    return report, enough_rows


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report, complete = materialize(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
