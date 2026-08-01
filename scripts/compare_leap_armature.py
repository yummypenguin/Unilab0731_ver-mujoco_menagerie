"""Compare LEAP Hand joint armature using the official MuJoCo viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


HAND_DOF = 16
POSE_SIZE = 23


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--armature",
        type=float,
        required=True,
        help="Armature value applied to all 16 LEAP Hand joints.",
    )
    parser.add_argument(
        "--cache-row",
        type=int,
        default=0,
        help="Initial row from the stable grasp cache.",
    )
    args = parser.parse_args()

    if not np.isfinite(args.armature) or args.armature < 0.0:
        parser.error("--armature must be finite and non-negative")

    if args.cache_row < 0:
        parser.error("--cache-row must be non-negative")

    return args


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    scene_path = (
        root
        / "src"
        / "unilab"
        / "assets"
        / "robots"
        / "leap_hand"
        / "scene_ball.xml"
    )
    cache_path = (
        root
        / "src"
        / "unilab"
        / "assets"
        / "robots"
        / "leap_hand"
        / "caches"
        / "ball_grasp_allegro_dedup_50k.npy"
    )

    if not scene_path.exists():
        raise FileNotFoundError(f"Scene not found: {scene_path}")
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # Match leap_inhand_ball_0730 physics.
    model.opt.timestep = 0.005

    hand_joint_ids: list[int] = []
    for joint_name in map(str, range(HAND_DOF)):
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            raise RuntimeError(f"Joint not found: {joint_name}")

        dof_id = int(model.jnt_dofadr[joint_id])

        model.dof_damping[dof_id] = 0.03
        model.dof_frictionloss[dof_id] = 0.001
        model.dof_armature[dof_id] = args.armature

        hand_joint_ids.append(joint_id)

    # Match task position-controller gains: kp=3.0, kd=0.01.
    for actuator_id in range(model.nu):
        model.actuator_gainprm[actuator_id, 0] = 3.0
        model.actuator_biasprm[actuator_id, 1] = -3.0
        model.actuator_biasprm[actuator_id, 2] = -0.01

    cache = np.load(cache_path, mmap_mode="r")

    if cache.ndim != 2 or cache.shape[1] < POSE_SIZE:
        raise RuntimeError(f"Unexpected cache shape: {cache.shape}")
    if args.cache_row >= cache.shape[0]:
        raise IndexError(
            f"cache row {args.cache_row} is outside 0..{cache.shape[0] - 1}"
        )

    pose = np.asarray(cache[args.cache_row, :POSE_SIZE], dtype=np.float64).copy()

    quat_norm = float(np.linalg.norm(pose[19:23]))
    if quat_norm <= 1e-8:
        raise RuntimeError("Cache row contains an invalid ball quaternion")
    pose[19:23] /= quat_norm

    data.qpos[:POSE_SIZE] = pose
    data.qvel[:] = 0.0

    # Set every position actuator target to its corresponding current qpos.
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        qpos_address = int(model.jnt_qposadr[joint_id])
        data.ctrl[actuator_id] = data.qpos[qpos_address]

    mujoco.mj_forward(model, data)

    effective_armatures = [
        float(model.dof_armature[int(model.jnt_dofadr[joint_id])])
        for joint_id in hand_joint_ids
    ]

    print("=" * 64)
    print(f"Armature       : {args.armature}")
    print(f"Cache row      : {args.cache_row}")
    print(f"Physics dt     : {model.opt.timestep}")
    print("Joint damping  : 0.03")
    print("Friction loss  : 0.001")
    print("Actuator gains : kp=3.0, kd=0.01")
    print(f"Force limit    : {model.actuator_forcerange[0].tolist()}")
    print(f"Verified values: {sorted(set(effective_armatures))}")
    print()
    print("Use the right-side Control panel to move actuator sliders.")
    print("Move the same joint by the same amount in both tests.")
    print("=" * 64)

    # Official MuJoCo viewer with its native actuator-control panel.
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
