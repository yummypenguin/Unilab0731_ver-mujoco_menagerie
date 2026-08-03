"""Read-only inspection for a LEAP Allegro-equivalent deduplicated cache."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent.parent
DEFAULT_SCENE = (
    ROOT_DIR / "src" / "unilab" / "assets" / "robots" / "leap_hand" / "scene_ball.xml"
)
FINGERTIP_GEOMS = (
    "index_tip_col",
    "middle_tip_col",
    "ring_tip_col",
    "thumb_tip_col",
)


def inspect_fingertip_surface_gaps(
    rows: np.ndarray,
    *,
    scene_path: Path,
    max_gap: float,
) -> dict[str, object]:
    """Run static FK and signed geom-distance checks without physics stepping."""
    import mujoco

    if not scene_path.is_file():
        raise FileNotFoundError(f"scene does not exist: {scene_path}")
    if not np.isfinite(max_gap) or max_gap <= 0.0:
        raise ValueError("max_gap must be positive and finite")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    object_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "leap_object_col")
    fingertip_geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in FINGERTIP_GEOMS
    ]
    if object_geom_id < 0 or any(geom_id < 0 for geom_id in fingertip_geom_ids):
        raise RuntimeError("scene is missing a fingertip or leap_object_col geom")

    signed_distances = np.empty((rows.shape[0], 4), dtype=np.float64)
    for row_index, row in enumerate(rows):
        data.qpos[:23] = np.asarray(row, dtype=np.float64)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for finger_index, fingertip_geom_id in enumerate(fingertip_geom_ids):
            signed_distances[row_index, finger_index] = mujoco.mj_geomDistance(
                model,
                data,
                int(fingertip_geom_id),
                int(object_geom_id),
                max(0.2, 2.0 * max_gap),
                None,
            )

    surface_gaps = np.maximum(signed_distances, 0.0)
    row_max = np.max(surface_gaps, axis=1)
    valid = np.all(np.isfinite(signed_distances), axis=1) & (row_max < max_gap)
    rejected = np.flatnonzero(~valid)
    per_finger_rejected = np.count_nonzero(
        ~np.isfinite(signed_distances) | (surface_gaps >= max_gap),
        axis=0,
    )
    return {
        "fingertip_surface_gap_threshold": float(max_gap),
        "fingertip_surface_gap_strictly_less": True,
        "fingertip_surface_gap_valid": bool(np.all(valid)),
        "fingertip_surface_gap_rejected_row_count": int(rejected.size),
        "fingertip_surface_gap_rejected_row_indices_first_100": rejected[:100].tolist(),
        "fingertip_surface_gap_per_finger_rejected_count": {
            name: int(count)
            for name, count in zip(FINGERTIP_GEOMS, per_finger_rejected, strict=True)
        },
        "fingertip_surface_gap_max": float(np.max(row_max, initial=0.0)),
        "fingertip_surface_gap_percentiles": {
            str(percentile): float(np.percentile(row_max, percentile))
            for percentile in (50, 90, 95, 99)
        },
    }


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
    nominal_ball_z: float,
    max_drop_distance: float,
    scene_path: Path | None = None,
    max_fingertip_surface_gap: float = 0.005,
) -> dict[str, object]:
    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    if not np.isfinite(joint_resolution) or joint_resolution <= 0.0:
        raise ValueError("joint_resolution must be positive and finite")
    if not np.isfinite(ball_position_resolution) or ball_position_resolution <= 0.0:
        raise ValueError("ball_position_resolution must be positive and finite")
    if not np.isfinite(nominal_ball_z):
        raise ValueError("nominal_ball_z must be finite")
    if not np.isfinite(max_drop_distance) or max_drop_distance <= 0.0:
        raise ValueError("max_drop_distance must be positive and finite")

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
    # Cache rows are published as float32. Quantize the threshold to the same
    # representation so an exactly-5-mm drop cannot pass due only to comparing
    # a rounded float32 row against an unrounded float64 threshold.
    height_threshold = float(np.float32(nominal_ball_z - max_drop_distance))
    ball_z = rows[:, 18].astype(np.float64)
    dropped_mask = ball_z <= height_threshold
    report["nominal_ball_z"] = float(nominal_ball_z)
    report["max_drop_distance"] = float(max_drop_distance)
    report["height_threshold"] = height_threshold
    report["height_rejected_row_count"] = int(np.count_nonzero(dropped_mask))
    report["height_valid"] = not bool(np.any(dropped_mask))
    report["minimum_height_margin"] = (
        float(np.min(ball_z - height_threshold)) if row_count else None
    )
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
    if scene_path is not None:
        report.update(
            inspect_fingertip_surface_gaps(
                rows,
                scene_path=scene_path,
                max_gap=max_fingertip_surface_gap,
            )
        )
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--joint-resolution", type=float, default=0.001)
    parser.add_argument("--ball-position-resolution", type=float, default=0.0005)
    parser.add_argument("--nominal-ball-z", type=float, default=0.664301098275159)
    parser.add_argument("--max-drop-distance", type=float, default=0.005)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--max-fingertip-surface-gap", type=float, default=0.005)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = inspect_cache(
        args.path,
        expected_rows=args.expected_rows,
        joint_resolution=args.joint_resolution,
        ball_position_resolution=args.ball_position_resolution,
        nominal_ball_z=args.nominal_ball_z,
        max_drop_distance=args.max_drop_distance,
        scene_path=args.scene,
        max_fingertip_surface_gap=args.max_fingertip_surface_gap,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(
        not (
            report.get("file_exists")
            and report.get("schema_valid")
            and report.get("dtype_valid")
            and report.get("finite")
            and report.get("expected_row_count_pass")
            and report.get("height_valid")
            and report.get("fingertip_surface_gap_valid")
            and report.get("quantized_duplicate_count") == 0
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
