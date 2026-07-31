"""Read-only inspection for a LEAP Allegro-equivalent deduplicated cache."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def _axis_stats(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": np.min(values, axis=0).astype(np.float64).tolist(),
        "max": np.max(values, axis=0).astype(np.float64).tolist(),
        "mean": np.mean(values, axis=0).astype(np.float64).tolist(),
        "std": np.std(values, axis=0).astype(np.float64).tolist(),
    }


def inspect_cache(
    path: Path,
    *,
    expected_rows: int,
    joint_resolution: float,
    ball_position_resolution: float,
) -> dict[str, object]:
    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    if not np.isfinite(joint_resolution) or joint_resolution <= 0.0:
        raise ValueError("joint_resolution must be positive and finite")
    if not np.isfinite(ball_position_resolution) or ball_position_resolution <= 0.0:
        raise ValueError("ball_position_resolution must be positive and finite")

    report: dict[str, object] = {"path": str(path), "file_exists": path.is_file()}
    if not path.is_file():
        report["expected_row_count_pass"] = False
        return report

    rows = np.load(path, allow_pickle=False)
    report["shape"] = list(rows.shape)
    report["dtype"] = str(rows.dtype)
    report["dtype_valid"] = rows.dtype == np.float32
    valid_shape = rows.ndim == 2 and rows.shape[1] == 23
    report["schema_valid"] = valid_shape
    if not valid_shape:
        report["expected_row_count_pass"] = False
        return report

    row_count = int(rows.shape[0])
    finite = bool(np.isfinite(rows).all())
    report["finite"] = finite
    report["row_count"] = row_count
    report["expected_rows"] = int(expected_rows)
    report["expected_row_count_pass"] = row_count == expected_rows
    if row_count == 0:
        report.update(
            {
                "quaternion_norm": {"min": None, "max": None, "mean": None},
                "hand_joint_stats": {},
                "ball_xyz_stats": {},
                "exact_duplicate_rows": 0,
                "quantized_unique_key_count": 0,
                "quantized_duplicate_count": 0,
                "unique_total_ratio": 0.0,
                "quaternion_only_duplicate_group_count": 0,
            }
        )
        return report

    quaternion_norms = np.linalg.norm(rows[:, 19:23].astype(np.float64), axis=1)
    report["quaternion_norm"] = {
        "min": float(np.min(quaternion_norms)),
        "max": float(np.max(quaternion_norms)),
        "mean": float(np.mean(quaternion_norms)),
    }
    report["hand_joint_stats"] = _axis_stats(rows[:, :16].astype(np.float64))
    report["ball_xyz_stats"] = _axis_stats(rows[:, 16:19].astype(np.float64))
    exact_unique = int(np.unique(rows, axis=0).shape[0])
    report["exact_duplicate_rows"] = row_count - exact_unique

    float32_rows = np.asarray(rows, dtype=np.float32)
    hand_keys = np.rint(float32_rows[:, :16] / joint_resolution).astype(np.int64)
    ball_keys = np.rint(float32_rows[:, 16:19] / ball_position_resolution).astype(np.int64)
    keys = np.concatenate([hand_keys, ball_keys], axis=1)
    _, inverse, counts = np.unique(
        keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    quantized_unique = int(counts.size)
    report["quantized_unique_key_count"] = quantized_unique
    report["quantized_duplicate_count"] = row_count - quantized_unique
    report["unique_total_ratio"] = quantized_unique / row_count

    quaternion_only_groups = 0
    for group_id in np.flatnonzero(counts > 1):
        group_quaternions = float32_rows[inverse == group_id, 19:23]
        if np.unique(group_quaternions, axis=0).shape[0] > 1:
            quaternion_only_groups += 1
    report["quaternion_only_duplicate_group_count"] = quaternion_only_groups
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--joint-resolution", type=float, default=0.01)
    parser.add_argument("--ball-position-resolution", type=float, default=0.001)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = inspect_cache(
        args.path,
        expected_rows=args.expected_rows,
        joint_resolution=args.joint_resolution,
        ball_position_resolution=args.ball_position_resolution,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(
        not (
            report.get("file_exists")
            and report.get("schema_valid")
            and report.get("dtype_valid")
            and report.get("finite")
            and report.get("expected_row_count_pass")
            and report.get("quantized_duplicate_count") == 0
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
