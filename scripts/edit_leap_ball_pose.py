"""Interactively edit a LEAP-hand ball pose with MuJoCo and Tkinter controls.

This is a cold-path visualization tool. It updates ``MjData`` only from the
main thread and never writes task assets, configuration, caches, or training
state.
"""

from __future__ import annotations

import argparse
import json
import queue
import time
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT_DIR = Path(__file__).parent.parent
DEFAULT_SCENE = ROOT_DIR / "src" / "unilab" / "assets" / "robots" / "leap_hand" / "scene_ball.xml"
DEFAULT_SEED_CONFIG = ROOT_DIR / "conf" / "ppo" / "task" / "leap_inhand_ball_grasp" / "mujoco.yaml"

HAND_DOF = 16
POSE_QPOS_SIZE = 23
BALL_POS_OFFSET = 16
JOINT_STEPS = (0.001, 0.005, 0.01, 0.05)
BALL_STEPS = (0.0005, 0.001, 0.002, 0.005)


class HandJointMetadata(NamedTuple):
    """Resolved MuJoCo addresses for one independently actuated hand joint."""

    name: str
    joint_id: int
    qpos_address: int
    actuator_id: int
    lower: float
    upper: float


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


def euler_zyx_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX yaw-pitch-roll radians to a normalized WXYZ quaternion."""
    half_roll = 0.5 * float(roll)
    half_pitch = 0.5 * float(pitch)
    half_yaw = 0.5 * float(yaw)
    cr, sr = np.cos(half_roll), np.sin(half_roll)
    cp, sp = np.cos(half_pitch), np.sin(half_pitch)
    cy, sy = np.cos(half_yaw), np.sin(half_yaw)
    quat = np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    quat /= np.linalg.norm(quat)
    return quat


def quat_wxyz_to_euler_zyx(quat: Sequence[float]) -> np.ndarray:
    """Convert a WXYZ quaternion to ZYX roll-pitch-yaw radians."""
    normalized = np.asarray(quat, dtype=np.float64)
    if normalized.shape != (4,) or not np.isfinite(normalized).all():
        raise ValueError("Expected a finite WXYZ quaternion with shape (4,)")
    norm = float(np.linalg.norm(normalized))
    if norm <= 1e-8:
        raise ValueError("Quaternion must have non-zero length")
    w, x, y, z = normalized / norm
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def apply_joint_delta(
    qpos: np.ndarray,
    ctrl: np.ndarray,
    qpos_index: int,
    delta: float,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
) -> float:
    """Backward-compatible direct edit for the legacy contiguous 1:1 layout."""
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


def set_hand_joint_radians(
    qpos: np.ndarray,
    ctrl: np.ndarray,
    joint: HandJointMetadata,
    value: float,
) -> float:
    """Clamp a joint value and synchronize its resolved actuator target."""
    clamped = float(np.clip(value, joint.lower, joint.upper))
    qpos[joint.qpos_address] = clamped
    ctrl[joint.actuator_id] = clamped
    return clamped


def set_hand_joint_degrees(
    qpos: np.ndarray,
    ctrl: np.ndarray,
    joint: HandJointMetadata,
    degrees: float,
) -> float:
    """Set a hand joint from UI degrees and return the applied radians."""
    return set_hand_joint_radians(qpos, ctrl, joint, np.deg2rad(float(degrees)))


def apply_ball_delta(
    qpos: np.ndarray,
    axis: int,
    delta: float,
    ball_qpos_address: int = BALL_POS_OFFSET,
) -> float:
    """Translate the ball along world X/Y/Z without changing orientation."""
    if not 0 <= axis < 3:
        raise IndexError(f"Ball position axis must be in [0, 3), got {axis}")
    qpos_index = int(ball_qpos_address) + axis
    qpos[qpos_index] += delta
    return float(qpos[qpos_index])


def set_ball_position(
    qpos: np.ndarray,
    ball_qpos_address: int,
    position: Sequence[float],
) -> None:
    """Set only a freejoint's XYZ position."""
    xyz = np.asarray(position, dtype=np.float64)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError("Expected a finite ball position with shape (3,)")
    qpos[ball_qpos_address : ball_qpos_address + 3] = xyz


def set_ball_euler_zyx(
    qpos: np.ndarray,
    ball_qpos_address: int,
    euler_radians: Sequence[float],
) -> np.ndarray:
    """Set only a freejoint's WXYZ quaternion from absolute ZYX Euler values."""
    euler = np.asarray(euler_radians, dtype=np.float64)
    if euler.shape != (3,) or not np.isfinite(euler).all():
        raise ValueError("Expected finite roll, pitch, yaw with shape (3,)")
    quat = euler_zyx_to_quat_wxyz(*euler)
    qpos[ball_qpos_address + 3 : ball_qpos_address + 7] = quat
    return quat


def reset_pose_arrays(
    qpos: np.ndarray,
    ctrl: np.ndarray,
    qvel: np.ndarray,
    initial_qpos: np.ndarray,
    initial_ctrl: np.ndarray,
) -> None:
    """Restore the loaded pose and targets without rebuilding the model."""
    qpos[:] = initial_qpos
    ctrl[:] = initial_ctrl
    qvel[:] = 0.0


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
        default="th_axl",
        help="Initial MuJoCo joint name (default: th_axl).",
    )
    parser.add_argument("--ball-position-span", type=float, default=0.08)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-5.0)
    parser.add_argument("--camera-distance", type=float, default=0.22)
    args = parser.parse_args(argv)
    if not np.isfinite(args.ball_position_span) or args.ball_position_span <= 0.0:
        parser.error("--ball-position-span must be a positive finite distance")
    return args


def find_ball_freejoint_qpos_address(model, joint_name: str = "leap_object_joint") -> int:
    """Resolve the ball freejoint qpos address by name."""
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise RuntimeError(f"Could not find ball freejoint {joint_name!r}")
    if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise RuntimeError(f"Joint {joint_name!r} is not a freejoint")
    qpos_address = int(model.jnt_qposadr[joint_id])
    if qpos_address < 0 or qpos_address + 7 > model.nq:
        raise RuntimeError(f"Invalid qpos address {qpos_address} for {joint_name!r}")
    return qpos_address


def find_joint_actuator_mapping(model, joint_ids: Sequence[int]) -> dict[int, int]:
    """Resolve each joint to exactly one joint-transmission actuator."""
    import mujoco

    requested = {int(joint_id) for joint_id in joint_ids}
    candidates: dict[int, list[int]] = {joint_id: [] for joint_id in requested}
    joint_transmission = int(mujoco.mjtTrn.mjTRN_JOINT)
    for actuator_id in range(model.nu):
        if int(model.actuator_trntype[actuator_id]) != joint_transmission:
            continue
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id in candidates:
            candidates[joint_id].append(actuator_id)
    invalid = {
        joint_id: actuator_ids
        for joint_id, actuator_ids in candidates.items()
        if len(actuator_ids) != 1
    }
    if invalid:
        raise RuntimeError(
            "Every LEAP hand joint must map to exactly one joint actuator; "
            f"invalid mappings: {invalid}"
        )
    return {joint_id: actuator_ids[0] for joint_id, actuator_ids in candidates.items()}


def find_hand_joint_metadata(model) -> list[HandJointMetadata]:
    """Resolve all 16 hinge joints, limits, qpos addresses, and actuators."""
    import mujoco

    ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "leap_object_joint")
    hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
    joint_ids = [
        joint_id
        for joint_id in range(model.njnt)
        if joint_id != ball_joint_id and int(model.jnt_type[joint_id]) == hinge_type
    ]
    joint_ids.sort(key=lambda joint_id: int(model.jnt_qposadr[joint_id]))
    if len(joint_ids) != HAND_DOF:
        raise RuntimeError(f"Expected {HAND_DOF} LEAP hand hinge joints, found {len(joint_ids)}")
    actuator_by_joint = find_joint_actuator_mapping(model, joint_ids)
    metadata: list[HandJointMetadata] = []
    for joint_id in joint_ids:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            raise RuntimeError(f"LEAP hand joint {joint_id} has no name")
        lower, upper = model.jnt_range[joint_id]
        metadata.append(
            HandJointMetadata(
                name=str(name),
                joint_id=joint_id,
                qpos_address=int(model.jnt_qposadr[joint_id]),
                actuator_id=actuator_by_joint[joint_id],
                lower=float(lower),
                upper=float(upper),
            )
        )
    if len({joint.qpos_address for joint in metadata}) != HAND_DOF:
        raise RuntimeError("LEAP hand joint qpos addresses are not unique")
    if len({joint.actuator_id for joint in metadata}) != HAND_DOF:
        raise RuntimeError("LEAP hand actuator mappings are not unique")
    return metadata


def _hand_joint_metadata(model) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return the legacy metadata tuple for callers that only need limits."""
    metadata = find_hand_joint_metadata(model)
    return (
        [joint.name for joint in metadata],
        np.asarray([joint.lower for joint in metadata]),
        np.asarray([joint.upper for joint in metadata]),
    )


class PoseEditorState:
    """Desired UI pose and main-loop requests; contains no ``MjData`` access."""

    def __init__(
        self,
        hand_radians: Sequence[float],
        ball_position: Sequence[float],
        ball_euler_radians: Sequence[float],
    ) -> None:
        self.desired_hand = np.asarray(hand_radians, dtype=np.float64).copy()
        self.desired_ball_position = np.asarray(ball_position, dtype=np.float64).copy()
        self.desired_ball_euler = np.asarray(ball_euler_radians, dtype=np.float64).copy()
        self.pose_dirty = False
        self.running = False
        self.updating_ui = False
        self.close_requested = False
        self.toggle_settling_requested = False
        self.freeze_requested = False
        self.reset_requested = False
        self.print_requested = False
        self.copy_requested = False


def sync_editor_state_from_data(
    state: PoseEditorState,
    data,
    joints: Sequence[HandJointMetadata],
    ball_qpos_address: int,
) -> None:
    """Refresh desired values from the current physical state."""
    state.desired_hand[:] = [data.qpos[joint.qpos_address] for joint in joints]
    state.desired_ball_position[:] = data.qpos[ball_qpos_address : ball_qpos_address + 3]
    state.desired_ball_euler[:] = quat_wxyz_to_euler_zyx(
        data.qpos[ball_qpos_address + 3 : ball_qpos_address + 7]
    )
    state.pose_dirty = False


def apply_desired_pose(
    data,
    joints: Sequence[HandJointMetadata],
    ball_qpos_address: int,
    state: PoseEditorState,
) -> None:
    """Apply one coalesced desired pose update to ``MjData`` arrays."""
    for index, joint in enumerate(joints):
        state.desired_hand[index] = set_hand_joint_radians(
            data.qpos,
            data.ctrl,
            joint,
            state.desired_hand[index],
        )
    set_ball_position(data.qpos, ball_qpos_address, state.desired_ball_position)
    set_ball_euler_zyx(data.qpos, ball_qpos_address, state.desired_ball_euler)
    data.qvel[:] = 0.0
    state.pose_dirty = False
    state.running = False


class PoseControlPanel:
    """Tkinter widgets that update only :class:`PoseEditorState`."""

    def __init__(
        self,
        root,
        state: PoseEditorState,
        joints: Sequence[HandJointMetadata],
        initial_ball_position: np.ndarray,
        ball_position_span: float,
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.state = state
        self.joints = list(joints)
        self.tk = tk
        self.pose_scales: list[tk.Scale] = []
        self.joint_variables: list[tk.DoubleVar] = []
        self.joint_value_labels: list[tk.StringVar] = []
        self.position_variables: list[tk.DoubleVar] = []
        self.position_value_labels: list[tk.StringVar] = []
        self.euler_variables: list[tk.DoubleVar] = []
        self.euler_value_labels: list[tk.StringVar] = []
        self.quaternion_labels = [tk.StringVar() for _ in range(4)]

        root.title("LEAP Pose Controls")
        root.geometry("780x820+20+40")
        root.protocol("WM_DELETE_WINDOW", self._request_close)

        controls = ttk.Frame(root, padding=8)
        controls.pack(fill=tk.X)
        self.mode_label = ttk.Label(controls, text="Mode: EDIT")
        self.mode_label.pack(side=tk.LEFT, padx=(0, 12))
        self.settle_button = ttk.Button(
            controls, text="Start Settling", command=self._request_toggle_settling
        )
        self.settle_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="Freeze", command=self._request_freeze).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="Reset Pose", command=self._request_reset).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(controls, text="Print State", command=self._request_print).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(controls, text="Copy State", command=self._request_copy).pack(
            side=tk.LEFT, padx=2
        )

        joint_group = ttk.LabelFrame(root, text="Hand Joints", padding=6)
        joint_group.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))
        canvas = tk.Canvas(joint_group, height=390, highlightthickness=0)
        scrollbar = ttk.Scrollbar(joint_group, orient=tk.VERTICAL, command=canvas.yview)
        joint_frame = ttk.Frame(canvas)
        joint_frame.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=joint_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for index, joint in enumerate(self.joints):
            ttk.Label(joint_frame, text=joint.name, width=13).grid(row=index, column=0, sticky="w")
            variable = tk.DoubleVar(value=np.rad2deg(state.desired_hand[index]))
            value_label = tk.StringVar()
            scale = tk.Scale(
                joint_frame,
                from_=np.rad2deg(joint.lower),
                to=np.rad2deg(joint.upper),
                orient=tk.HORIZONTAL,
                resolution=0.1,
                showvalue=False,
                length=470,
                variable=variable,
                command=lambda value, i=index: self._on_joint_changed(i, value),
            )
            scale.grid(row=index, column=1, sticky="ew")
            ttk.Label(joint_frame, textvariable=value_label, width=13).grid(
                row=index, column=2, sticky="e"
            )
            self.joint_variables.append(variable)
            self.joint_value_labels.append(value_label)
            self.pose_scales.append(scale)
        joint_frame.columnconfigure(1, weight=1)

        position_group = ttk.LabelFrame(root, text="Ball Position", padding=6)
        position_group.pack(fill=tk.X, padx=8, pady=3)
        for axis, axis_name in enumerate(("X", "Y", "Z")):
            ttk.Label(position_group, text=axis_name, width=8).grid(row=axis, column=0, sticky="w")
            center = float(initial_ball_position[axis])
            variable = tk.DoubleVar(value=state.desired_ball_position[axis])
            value_label = tk.StringVar()
            scale = tk.Scale(
                position_group,
                from_=center - ball_position_span,
                to=center + ball_position_span,
                orient=tk.HORIZONTAL,
                resolution=0.0001,
                showvalue=False,
                length=480,
                variable=variable,
                command=lambda value, i=axis: self._on_position_changed(i, value),
            )
            scale.grid(row=axis, column=1, sticky="ew")
            ttk.Label(position_group, textvariable=value_label, width=24).grid(
                row=axis, column=2, sticky="e"
            )
            self.position_variables.append(variable)
            self.position_value_labels.append(value_label)
            self.pose_scales.append(scale)
        position_group.columnconfigure(1, weight=1)

        orientation_group = ttk.LabelFrame(root, text="Ball Orientation", padding=6)
        orientation_group.pack(fill=tk.X, padx=8, pady=3)
        for axis, axis_name in enumerate(("Roll X", "Pitch Y", "Yaw Z")):
            ttk.Label(orientation_group, text=axis_name, width=8).grid(
                row=axis, column=0, sticky="w"
            )
            variable = tk.DoubleVar(value=np.rad2deg(state.desired_ball_euler[axis]))
            value_label = tk.StringVar()
            scale = tk.Scale(
                orientation_group,
                from_=-180.0,
                to=180.0,
                orient=tk.HORIZONTAL,
                resolution=0.1,
                showvalue=False,
                length=480,
                variable=variable,
                command=lambda value, i=axis: self._on_orientation_changed(i, value),
            )
            scale.grid(row=axis, column=1, sticky="ew")
            ttk.Label(orientation_group, textvariable=value_label, width=13).grid(
                row=axis, column=2, sticky="e"
            )
            self.euler_variables.append(variable)
            self.euler_value_labels.append(value_label)
            self.pose_scales.append(scale)
        orientation_group.columnconfigure(1, weight=1)

        quat_group = ttk.LabelFrame(root, text="Quaternion WXYZ", padding=6)
        quat_group.pack(fill=tk.X, padx=8, pady=(3, 8))
        for index, name in enumerate(("w", "x", "y", "z")):
            ttk.Label(quat_group, textvariable=self.quaternion_labels[index]).grid(
                row=0, column=index, padx=12, sticky="w"
            )
            self.quaternion_labels[index].set(f"{name} = 0.000000")

        self.sync_from_state()
        self.set_running(False)

    def _request_close(self) -> None:
        self.state.close_requested = True
        self.root.withdraw()

    def _request_toggle_settling(self) -> None:
        self.state.toggle_settling_requested = True

    def _request_freeze(self) -> None:
        self.state.freeze_requested = True

    def _request_reset(self) -> None:
        self.state.reset_requested = True

    def _request_print(self) -> None:
        self.state.print_requested = True

    def _request_copy(self) -> None:
        self.state.copy_requested = True

    def _begin_edit(self) -> None:
        self.state.running = False
        self.state.pose_dirty = True
        self.set_running(False)

    def _on_joint_changed(self, index: int, value: str) -> None:
        if self.state.updating_ui:
            return
        degrees = float(value)
        self.state.desired_hand[index] = np.deg2rad(degrees)
        self.joint_value_labels[index].set(f"{degrees:.1f} deg")
        self._begin_edit()

    def _on_position_changed(self, axis: int, value: str) -> None:
        if self.state.updating_ui:
            return
        meters = float(value)
        self.state.desired_ball_position[axis] = meters
        self.position_value_labels[axis].set(f"{meters:.4f} m  ({meters * 1_000.0:.1f} mm)")
        self._begin_edit()

    def _on_orientation_changed(self, axis: int, value: str) -> None:
        if self.state.updating_ui:
            return
        degrees = float(value)
        self.state.desired_ball_euler[axis] = np.deg2rad(degrees)
        self.euler_value_labels[axis].set(f"{degrees:.1f} deg")
        self._update_quaternion_labels()
        self._begin_edit()

    def _update_quaternion_labels(self) -> None:
        quat = euler_zyx_to_quat_wxyz(*self.state.desired_ball_euler)
        for label, name, value in zip(
            self.quaternion_labels, ("w", "x", "y", "z"), quat, strict=True
        ):
            label.set(f"{name} = {value:+.6f}")

    def sync_from_state(self) -> None:
        """Update widgets without feeding values back into desired pose."""
        self.state.updating_ui = True
        try:
            for index, radians in enumerate(self.state.desired_hand):
                degrees = float(np.rad2deg(radians))
                self.joint_variables[index].set(degrees)
                self.joint_value_labels[index].set(f"{degrees:.1f} deg")
            for axis, meters in enumerate(self.state.desired_ball_position):
                value = float(meters)
                self.position_variables[axis].set(value)
                self.position_value_labels[axis].set(f"{value:.4f} m  ({value * 1_000.0:.1f} mm)")
            for axis, radians in enumerate(self.state.desired_ball_euler):
                degrees = float(np.rad2deg(radians))
                self.euler_variables[axis].set(degrees)
                self.euler_value_labels[axis].set(f"{degrees:.1f} deg")
            self._update_quaternion_labels()
        finally:
            self.state.updating_ui = False

    def set_running(self, running: bool) -> None:
        widget_state = self.tk.DISABLED if running else self.tk.NORMAL
        for scale in self.pose_scales:
            scale.configure(state=widget_state)
        self.mode_label.configure(text="Mode: SETTLING" if running else "Mode: EDIT")
        self.settle_button.configure(text="Pause Settling" if running else "Start Settling")

    def copy_state(self, text: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update_idletasks()


def _overlay_text(
    *,
    selected_index: int,
    joints: Sequence[HandJointMetadata],
    ball_qpos_address: int,
    data,
    running: bool,
    joint_step_index: int,
    ball_step_index: int,
) -> tuple[str, str]:
    selected = joints[selected_index]
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
            f"{selected.name}  [qpos {selected.qpos_address}]",
            f"{data.qpos[selected.qpos_address]: .6f} / {data.ctrl[selected.actuator_id]: .6f}",
            f"{JOINT_STEPS[joint_step_index]:.4f} rad",
            " ".join(
                f"{value: .6f}" for value in data.qpos[ball_qpos_address : ball_qpos_address + 3]
            ),
            f"{BALL_STEPS[ball_step_index] * 1_000.0:.1f} mm",
            "[ ] select | - = joint | , . joint step\n"
            "U/J X | I/K Y | O/L Z | N/M ball step\n"
            "Space settle | R freeze | P print state",
        )
    )
    return labels, values


def _print_state(prefix: str, payload: dict[str, object]) -> str:
    text = prefix + json.dumps(payload, separators=(",", ":"))
    print(text, flush=True)
    return text


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    import tkinter as tk

    import mujoco
    import mujoco.viewer

    qpos = normalize_pose_qpos(args.qpos) if args.qpos else _load_seed_qpos(args.seed_config)
    ctrl = np.asarray(args.ctrl, dtype=np.float64).copy() if args.ctrl else qpos[:HAND_DOF].copy()
    if ctrl.shape != (HAND_DOF,) or not np.isfinite(ctrl).all():
        raise ValueError(f"Expected finite ctrl shape ({HAND_DOF},), got {ctrl.shape}")

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    joints = find_hand_joint_metadata(model)
    ball_qpos_address = find_ball_freejoint_qpos_address(model)
    joint_names = [joint.name for joint in joints]
    try:
        selected_index = joint_names.index(str(args.selected_joint))
    except ValueError as exc:
        raise ValueError(
            f"Unknown hand joint {args.selected_joint!r}; available: {joint_names}"
        ) from exc

    data.qpos[:POSE_QPOS_SIZE] = qpos
    for index, joint in enumerate(joints):
        set_hand_joint_radians(data.qpos, data.ctrl, joint, ctrl[index])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    initial_qpos = np.asarray(data.qpos, dtype=np.float64).copy()
    initial_ctrl = np.asarray(data.ctrl, dtype=np.float64).copy()

    state = PoseEditorState(
        [data.qpos[joint.qpos_address] for joint in joints],
        data.qpos[ball_qpos_address : ball_qpos_address + 3],
        quat_wxyz_to_euler_zyx(data.qpos[ball_qpos_address + 3 : ball_qpos_address + 7]),
    )
    root = tk.Tk()
    panel = PoseControlPanel(
        root,
        state,
        joints,
        initial_qpos[ball_qpos_address : ball_qpos_address + 3],
        float(args.ball_position_span),
    )

    key_events: queue.SimpleQueue[int] = queue.SimpleQueue()
    joint_step_index = 1
    ball_step_index = 1
    changed = True

    def on_key(keycode: int) -> None:
        key_events.put(int(keycode))

    print("LEAP ball pose editor: slider and keyboard edit modes are active.")
    print("Press P to print state; closing either window prints final state.")

    try:
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

            while viewer.is_running() and not state.close_requested:
                frame_start = time.perf_counter()
                try:
                    root.update_idletasks()
                    root.update()
                except tk.TclError:
                    state.close_requested = True
                    break

                data_changed = False
                sync_panel = False
                copy_text: str | None = None
                with viewer.lock():
                    if state.toggle_settling_requested:
                        state.toggle_settling_requested = False
                        state.running = not state.running
                        if not state.running:
                            data.qvel[:] = 0.0
                            mujoco.mj_forward(model, data)
                            sync_editor_state_from_data(state, data, joints, ball_qpos_address)
                            sync_panel = True
                        data_changed = True

                    if state.freeze_requested:
                        state.freeze_requested = False
                        state.running = False
                        data.qvel[:] = 0.0
                        mujoco.mj_forward(model, data)
                        sync_editor_state_from_data(state, data, joints, ball_qpos_address)
                        data_changed = True
                        sync_panel = True

                    if state.reset_requested:
                        state.reset_requested = False
                        state.running = False
                        reset_pose_arrays(
                            data.qpos,
                            data.ctrl,
                            data.qvel,
                            initial_qpos,
                            initial_ctrl,
                        )
                        mujoco.mj_forward(model, data)
                        sync_editor_state_from_data(state, data, joints, ball_qpos_address)
                        data_changed = True
                        sync_panel = True

                    while not key_events.empty():
                        keycode = key_events.get()
                        if keycode == ord("["):
                            selected_index = (selected_index - 1) % HAND_DOF
                        elif keycode == ord("]"):
                            selected_index = (selected_index + 1) % HAND_DOF
                        elif keycode in (ord("-"), ord("=")):
                            selected = joints[selected_index]
                            delta = JOINT_STEPS[joint_step_index]
                            if keycode == ord("-"):
                                delta = -delta
                            set_hand_joint_radians(
                                data.qpos,
                                data.ctrl,
                                selected,
                                data.qpos[selected.qpos_address] + delta,
                            )
                            state.running = False
                            data.qvel[:] = 0.0
                            mujoco.mj_forward(model, data)
                            sync_panel = True
                        elif keycode == ord(","):
                            joint_step_index = max(0, joint_step_index - 1)
                        elif keycode == ord("."):
                            joint_step_index = min(len(JOINT_STEPS) - 1, joint_step_index + 1)
                        elif keycode in (ord("N"), ord("M")):
                            direction = -1 if keycode == ord("N") else 1
                            ball_step_index = int(
                                np.clip(
                                    ball_step_index + direction,
                                    0,
                                    len(BALL_STEPS) - 1,
                                )
                            )
                        elif keycode in (
                            ord("U"),
                            ord("J"),
                            ord("I"),
                            ord("K"),
                            ord("O"),
                            ord("L"),
                        ):
                            axis_and_sign = {
                                ord("U"): (0, 1.0),
                                ord("J"): (0, -1.0),
                                ord("I"): (1, 1.0),
                                ord("K"): (1, -1.0),
                                ord("O"): (2, 1.0),
                                ord("L"): (2, -1.0),
                            }
                            axis, sign = axis_and_sign[keycode]
                            apply_ball_delta(
                                data.qpos,
                                axis,
                                sign * BALL_STEPS[ball_step_index],
                                ball_qpos_address,
                            )
                            state.running = False
                            data.qvel[:] = 0.0
                            mujoco.mj_forward(model, data)
                            sync_panel = True
                        elif keycode == ord(" "):
                            state.running = not state.running
                            if not state.running:
                                data.qvel[:] = 0.0
                                mujoco.mj_forward(model, data)
                                sync_panel = True
                        elif keycode == ord("R"):
                            state.running = False
                            data.qvel[:] = 0.0
                            mujoco.mj_forward(model, data)
                            sync_panel = True
                        elif keycode == ord("P"):
                            _print_state("UNILAB_POSE_STATE=", state_payload(data))
                        data_changed = True

                    if sync_panel:
                        sync_editor_state_from_data(state, data, joints, ball_qpos_address)

                    if state.pose_dirty:
                        apply_desired_pose(data, joints, ball_qpos_address, state)
                        mujoco.mj_forward(model, data)
                        data_changed = True
                        sync_panel = True

                    if state.running:
                        for _ in range(simulation_steps_per_frame):
                            mujoco.mj_step(model, data)
                        sync_editor_state_from_data(state, data, joints, ball_qpos_address)
                        data_changed = True
                        sync_panel = True

                    if state.print_requested:
                        state.print_requested = False
                        _print_state("UNILAB_POSE_STATE=", state_payload(data))
                    if state.copy_requested:
                        state.copy_requested = False
                        copy_text = "UNILAB_POSE_STATE=" + json.dumps(
                            state_payload(data), separators=(",", ":")
                        )

                    if changed or data_changed:
                        labels, values = _overlay_text(
                            selected_index=selected_index,
                            joints=joints,
                            ball_qpos_address=ball_qpos_address,
                            data=data,
                            running=state.running,
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

                if sync_panel:
                    panel.sync_from_state()
                panel.set_running(state.running)
                if copy_text is not None:
                    panel.copy_state(copy_text)
                viewer.sync()
                remaining = target_frame_seconds - (time.perf_counter() - frame_start)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        _print_state("UNILAB_FINAL_STATE=", state_payload(data))
        try:
            root.destroy()
        except tk.TclError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
