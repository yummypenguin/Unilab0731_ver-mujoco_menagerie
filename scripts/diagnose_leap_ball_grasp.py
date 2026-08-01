"""Probe LEAP ball-grasp self penetration along selected joint coordinates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.scene import SceneCfg
from unilab.envs.manipulation.leap_inhand.ball_grasp_gen import (
    build_joint_coordinate_probes,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qpos",
        type=float,
        nargs=23,
        required=True,
        metavar="VALUE",
        help="Complete LEAP hand and ball qpos from a frontier diagnostic line.",
    )
    parser.add_argument(
        "--joints",
        nargs="+",
        default=["rf_rot", "rf_pip"],
        help="LEAP joint names to probe (default: rf_rot rf_pip).",
    )
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.10],
        help="Positive probe magnitudes in radians (default: 0.02 0.05 0.10).",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=None,
        help="Run the full env-owned settling and quality diagnostic instead of coordinate probes.",
    )
    parser.add_argument(
        "--settle-probes",
        action="store_true",
        help="Apply --joints/--deltas as independent full-settling coordinate probes.",
    )
    return parser.parse_args(argv)


def _run_settling_diagnostic(args: argparse.Namespace) -> int:
    from unilab.envs.manipulation.allegro_inhand.rotation import RewardConfigPPO
    from unilab.envs.manipulation.leap_inhand.ball_grasp_gen import (
        LeapInhandBallGraspCfg,
        LeapInhandBallGraspEnv,
    )

    cfg = LeapInhandBallGraspCfg(
        grasp_auto_save=False,
        grasp_collection_target=0,
        grasp_max_self_penetration=0.001,
        grasp_max_object_penetration=0.001,
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
    env = LeapInhandBallGraspEnv(cfg, num_envs=1, backend_type="mujoco")
    try:
        qpos = np.asarray(args.qpos, dtype=np.float64)
        if args.settle_probes:
            report = env.diagnose_joint_coordinate_probes(
                qpos,
                joint_names=[str(name) for name in args.joints],
                delta_magnitudes=np.asarray(args.deltas, dtype=np.float64),
                settle_seconds=float(args.settle_seconds),
            )
        else:
            report = env.diagnose_grasp_state(
                qpos,
                settle_seconds=float(args.settle_seconds),
            )
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        env.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.settle_seconds is not None:
        return _run_settling_diagnostic(args)
    scene_path = Path(ASSETS_ROOT_PATH) / "robots" / "leap_hand" / "scene_ball.xml"
    joint_names = [str(name) for name in args.joints]

    probe_count = 1 + 2 * len(joint_names) * len(args.deltas)
    backend = MuJoCoBackend(
        SceneCfg(model_file=str(scene_path)),
        probe_count,
        0.005,
        base_name="palm_lower",
    )
    try:
        backend.materialize()
        coordinate_indices = backend.get_joint_dof_pos_indices(joint_names)
        joint_range = backend.get_joint_range()
        if joint_range is None or joint_range.shape != (16, 2):
            raise RuntimeError(f"Expected LEAP joint range shape (16, 2), got {joint_range}")

        base_qpos = np.asarray(args.qpos, dtype=np.float64)
        hand_probes, probe_indices, applied_deltas = build_joint_coordinate_probes(
            base_qpos[:16],
            coordinate_indices,
            np.asarray(args.deltas, dtype=np.float64),
            joint_lower=joint_range[:, 0],
            joint_upper=joint_range[:, 1],
        )
        qpos = np.broadcast_to(base_qpos, (probe_count, 23)).copy()
        qpos[:, :16] = hand_probes
        env_ids = np.arange(probe_count, dtype=np.int32)
        backend.set_state(env_ids, qpos, np.zeros((probe_count, backend.model.nv)))

        palm_id = backend.get_body_id("palm_lower")
        object_id = backend.get_body_id("leap_object")
        details = backend.get_contact_penetration_details(
            env_ids,
            self_collision_body_ids=backend.get_body_subtree_ids(palm_id),
            object_body_id=object_id,
        )
        joint_by_index = dict(zip(coordinate_indices.tolist(), joint_names, strict=True))
        print("probe,joint,delta_rad,joint_value,self_mm,self_geoms,object_mm,object_geoms")
        for row, coordinate_index, delta, detail in zip(
            hand_probes, probe_indices, applied_deltas, details, strict=True
        ):
            if coordinate_index < 0:
                joint_name = "baseline"
                joint_value = float("nan")
            else:
                joint_name = joint_by_index[int(coordinate_index)]
                joint_value = row[coordinate_index]
            print(
                f"{detail.env_id},{joint_name},{delta:.6f},{joint_value:.6f},"
                f"{detail.self_depth * 1_000.0:.6f},{detail.self_geom_pair},"
                f"{detail.object_depth * 1_000.0:.6f},{detail.object_geom_pair}"
            )
    finally:
        backend.cleanup_scene_assets()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
