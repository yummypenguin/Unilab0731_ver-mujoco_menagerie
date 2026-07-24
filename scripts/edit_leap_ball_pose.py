"""Interactively edit a LEAP-hand ball pose without actuator-side stiction.

This is a cold-path visualization tool.  It reads the existing ball-grasp seed,
updates ``MjData.qpos`` directly while paused, and never writes task assets,
configuration, caches, or training state.
"""

from __future__ import annotations

import argparse
import json
import queue
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent.parent
DEFAULT_SCENE = (
    ROOT_DIR / "src" / "unilab" / "assets" / "robots" / "leap_hand" / "scene_ball.xml"
)
DEFAULT_SEED_CONFIG = ROOT_DIR / "conf" / "ppo" / "task" / "leap_inhand_ball_grasp" / "mujoco.yaml"

HAND_DOF = 16
POSE_QPOS_SIZE = 23
BALL_POS_OFFSET = 16
JOINT_STEPS = (0.001, 0.005, 0.01, 0.05)
BALL_STEPS = (0.0005, 0.001, 0.002, 0.005)


def normalize_pose_qpos(values: Sequence[float]) -> np.ndarray:
    """Validate a 23-value LEAP-hand/ball pose and normalize its quaternion."""
    qpos = np.asarray(values, dtype=np.float64)
    if qpos.shape != (POSE_QPOS_SIZE,):
        raise ValueError(f"Expected qpos shape ({POSE_QPOS_SIZE},), got {qpos.shape}")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos contains non-finite values")
    qpos = qpos.copy()
    quat_norm = float(np.linalg.norm(qpos[19:23]))
    if quat_norm <= 1e-8:
        raise ValueError("Ball quaternion must have non-zero length")
    qpos[19:23] /= quat_norm
    return qpos


def apply_joint_delta(
    qpos: np.ndarray,
    ctrl: np.ndarray,
    qpos_index: int,
    delta: float,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
) -> float:
    """Directly edit one hand joint and synchronize its position target."""
    if not 0 <= qpos_index < HAND_DOF:
        raise IndexError(f"Hand qpos index must be in [0, {HAND_DOF}), got {qpos_index}")
    value = float(
        np.clip(
            qpos[qpos_index] + delta,
            joint_lower[qpos_index],
            joint_upper[qpos_index],
        )
    )
    qpos[qpos_index] = value
    ctrl[qpos_index] = value
    return value


def apply_ball_delta(qpos: np.ndarray, axis: int, delta: float) -> float:
    """Translate the ball along world X/Y/Z without changing its orientation."""
    if not 0 <= axis < 3:
        raise IndexError(f"Ball position axis must be in [0, 3), got {axis}")
    qpos_index = BALL_POS_OFFSET + axis
    qpos[qpos_index] += delta
    return float(qpos[qpos_index])


def state_payload(data) -> dict[str, object]:
    """Build the copy/paste payload printed by P and on viewer exit."""
    return {
        "qpos": np.asarray(data.qpos, dtype=np.float64).tolist(),
        "ctrl": np.asarray(data.ctrl, dtype=np.float64).tolist(),
        "qvel": np.asarray(data.qvel, dtype=np.float64).tolist(),
        "time": float(data.time),
    }


def _load_seed_qpos(path: Path) -> np.ndarray:
    from omegaconf import OmegaConf

    config = OmegaConf.load(path)
    values = OmegaConf.to_container(config.env.grasp_seed_qpos, resolve=True)
    if not isinstance(values, list):
        raise ValueError(f"env.grasp_seed_qpos in {path} must be a list")
    return normalize_pose_qpos(values)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--seed-config", type=Path, default=DEFAULT_SEED_CONFIG)
    parser.add_argument("--qpos", type=float, nargs=POSE_QPOS_SIZE)
    parser.add_argument("--ctrl", type=float, nargs=HAND_DOF)
    parser.add_argument(
        "--selected-joint",
        default="13",
        help="Initial MuJoCo joint name (default: 13).",
    )
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-5.0)
    parser.add_argument("--camera-distance", type=float, default=0.22)
    return parser.parse_args(argv)


def _hand_joint_metadata(model) -> tuple[list[str], np.ndarray, np.ndarray]:
    import mujoco

    names_by_qpos: list[str | None] = [None] * HAND_DOF
    lower = np.empty(HAND_DOF, dtype=np.float64)
    upper = np.empty(HAND_DOF, dtype=np.float64)
    for joint_id in range(model.njnt):
        qpos_index = int(model.jnt_qposadr[joint_id])
        if not 0 <= qpos_index < HAND_DOF:
            continue
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        names_by_qpos[qpos_index] = str(name)
        lower[qpos_index], upper[qpos_index] = model.jnt_range[joint_id]
    if any(name is None for name in names_by_qpos):
        raise RuntimeError(f"Could not resolve all {HAND_DOF} LEAP hand joints")
    return [str(name) for name in names_by_qpos], lower, upper


def _overlay_text(
    *,
    selected_index: int,
    joint_names: list[str],
    data,
    running: bool,
    joint_step_index: int,
    ball_step_index: int,
) -> tuple[str, str]:
    labels = "\n".join(
        (
            "Mode",
            "Selected joint",
            "Joint qpos / ctrl",
            "Joint step",
            "Ball XYZ",
            "Ball step",
            "Keys",
        )
    )
    values = "\n".join(
        (
            "SETTLING" if running else "EDIT (direct qpos)",
            f"{joint_names[selected_index]}  [qpos {selected_index}]",
            f"{data.qpos[selected_index]: .6f} / {data.ctrl[selected_index]: .6f}",
            f"{JOINT_STEPS[joint_step_index]:.4f} rad",
            " ".join(f"{value: .6f}" for value in data.qpos[16:19]),
            f"{BALL_STEPS[ball_step_index] * 1_000.0:.1f} mm",
            "[ ] select | - = joint | , . joint step\n"
            "U/J X | I/K Y | O/L Z | N/M ball step\n"
            "Space settle | R freeze | P print state",
        )
    )
    return labels, values


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    import mujoco
    import mujoco.viewer

    qpos = normalize_pose_qpos(args.qpos) if args.qpos else _load_seed_qpos(args.seed_config)
    ctrl = (
        np.asarray(args.ctrl, dtype=np.float64).copy()
        if args.ctrl
        else qpos[:HAND_DOF].copy()
    )
    if ctrl.shape != (HAND_DOF,) or not np.isfinite(ctrl).all():
        raise ValueError(f"Expected finite ctrl shape ({HAND_DOF},), got {ctrl.shape}")

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    joint_names, joint_lower, joint_upper = _hand_joint_metadata(model)
    try:
        selected_index = joint_names.index(str(args.selected_joint))
    except ValueError as exc:
        raise ValueError(
            f"Unknown hand joint {args.selected_joint!r}; available: {joint_names}"
        ) from exc

    data.qpos[:POSE_QPOS_SIZE] = qpos
    data.ctrl[:HAND_DOF] = np.clip(ctrl, joint_lower, joint_upper)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    key_events: queue.SimpleQueue[int] = queue.SimpleQueue()
    running = False
    joint_step_index = 1
    ball_step_index = 1
    changed = True

    def on_key(keycode: int) -> None:
        key_events.put(int(keycode))

    print("LEAP ball pose editor: direct-qpos edit mode is active.")
    print("Press P to print state; closing the viewer also prints final state.")

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=on_key,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [-0.02, 0.03, 0.55]
        viewer.cam.distance = float(args.camera_distance)
        viewer.cam.azimuth = float(args.camera_azimuth)
        viewer.cam.elevation = float(args.camera_elevation)

        target_frame_seconds = 1.0 / 60.0
        simulation_steps_per_frame = max(1, round(target_frame_seconds / model.opt.timestep))

        while viewer.is_running():
            frame_start = time.perf_counter()
            while not key_events.empty():
                keycode = key_events.get()
                # Pull in any mouse perturbation before applying a keyboard edit.
                qpos[:] = data.qpos[:POSE_QPOS_SIZE]
                ctrl[:] = data.ctrl[:HAND_DOF]

                if keycode == ord("["):
                    selected_index = (selected_index - 1) % HAND_DOF
                elif keycode == ord("]"):
                    selected_index = (selected_index + 1) % HAND_DOF
                elif keycode in (ord("-"), ord("=")):
                    delta = JOINT_STEPS[joint_step_index]
                    if keycode == ord("-"):
                        delta = -delta
                    apply_joint_delta(
                        qpos,
                        ctrl,
                        selected_index,
                        delta,
                        joint_lower,
                        joint_upper,
                    )
                    running = False
                elif keycode == ord(","):
                    joint_step_index = max(0, joint_step_index - 1)
                elif keycode == ord("."):
                    joint_step_index = min(len(JOINT_STEPS) - 1, joint_step_index + 1)
                elif keycode in (ord("N"), ord("M")):
                    direction = -1 if keycode == ord("N") else 1
                    ball_step_index = int(
                        np.clip(ball_step_index + direction, 0, len(BALL_STEPS) - 1)
                    )
                elif keycode in (ord("U"), ord("J"), ord("I"), ord("K"), ord("O"), ord("L")):
                    axis_and_sign = {
                        ord("U"): (0, 1.0),
                        ord("J"): (0, -1.0),
                        ord("I"): (1, 1.0),
                        ord("K"): (1, -1.0),
                        ord("O"): (2, 1.0),
                        ord("L"): (2, -1.0),
                    }
                    axis, sign = axis_and_sign[keycode]
                    apply_ball_delta(qpos, axis, sign * BALL_STEPS[ball_step_index])
                    running = False
                elif keycode == ord(" "):
                    running = not running
                elif keycode == ord("R"):
                    running = False
                    data.qvel[:] = 0.0
                elif keycode == ord("P"):
                    print(
                        "UNILAB_POSE_STATE="
                        + json.dumps(state_payload(data), separators=(",", ":")),
                        flush=True,
                    )

                if not running:
                    data.qpos[:POSE_QPOS_SIZE] = qpos
                    data.ctrl[:HAND_DOF] = ctrl
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                changed = True

            if running:
                for _ in range(simulation_steps_per_frame):
                    mujoco.mj_step(model, data)
                changed = True

            if changed:
                labels, values = _overlay_text(
                    selected_index=selected_index,
                    joint_names=joint_names,
                    data=data,
                    running=running,
                    joint_step_index=joint_step_index,
                    ball_step_index=ball_step_index,
                )
                viewer.set_texts(
                    (
                        int(mujoco.mjtFontScale.mjFONTSCALE_100),
                        int(mujoco.mjtGridPos.mjGRID_TOPLEFT),
                        labels,
                        values,
                    )
                )
                changed = False
            viewer.sync()
            remaining = target_frame_seconds - (time.perf_counter() - frame_start)
            if remaining > 0.0:
                time.sleep(remaining)

    print(
        "UNILAB_FINAL_STATE=" + json.dumps(state_payload(data), separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
