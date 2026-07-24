"""Measure local ball-rotation authority around a LEAP canonical pose candidate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.scene import SceneCfg

HAND_DOF = 16
QPOS_SIZE = 23
CONTACT_SENSORS = (
    "leap_index_contact",
    "leap_middle_contact",
    "leap_ring_contact",
    "leap_thumb_contact",
)


def load_candidate(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Load and validate one canonical-pose candidate JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = payload.get("coordinate_contract", {})
    qpos = np.asarray(payload.get("qpos"), dtype=np.float64)
    ctrl = np.asarray(payload.get("ctrl"), dtype=np.float64)
    joint_names = tuple(str(name) for name in contract.get("qpos_joint_names", ()))

    if qpos.shape != (QPOS_SIZE,):
        raise ValueError(f"candidate qpos must have shape ({QPOS_SIZE},), got {qpos.shape}")
    if ctrl.shape != (HAND_DOF,):
        raise ValueError(f"candidate ctrl must have shape ({HAND_DOF},), got {ctrl.shape}")
    if len(joint_names) != HAND_DOF or len(set(joint_names)) != HAND_DOF:
        raise ValueError("coordinate_contract.qpos_joint_names must contain 16 unique names")
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(ctrl)):
        raise ValueError("candidate qpos and ctrl must be finite")

    quat_norm = float(np.linalg.norm(qpos[19:23]))
    if quat_norm <= 1e-8:
        raise ValueError("candidate object quaternion must have non-zero length")
    qpos = qpos.copy()
    qpos[19:23] /= quat_norm
    return qpos, ctrl.copy(), joint_names


def build_probe_targets(
    ctrl: np.ndarray,
    joint_names: Sequence[str],
    ctrl_range: np.ndarray,
    pulse_delta: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Create one baseline and signed single-joint position-target probes."""
    base = np.asarray(ctrl, dtype=np.float64)
    limits = np.asarray(ctrl_range, dtype=np.float64)
    if base.shape != (HAND_DOF,):
        raise ValueError(f"ctrl must have shape ({HAND_DOF},), got {base.shape}")
    if limits.shape != (HAND_DOF, 2):
        raise ValueError(f"ctrl_range must have shape ({HAND_DOF}, 2), got {limits.shape}")
    if len(joint_names) != HAND_DOF:
        raise ValueError(f"joint_names must contain {HAND_DOF} entries")
    if not np.isfinite(pulse_delta) or pulse_delta <= 0.0:
        raise ValueError("pulse_delta must be positive and finite")

    targets = [np.clip(base, limits[:, 0], limits[:, 1])]
    probes: list[dict[str, Any]] = [
        {
            "probe_index": 0,
            "joint_index": None,
            "joint_name": "baseline",
            "requested_delta_rad": 0.0,
            "applied_delta_rad": 0.0,
        }
    ]
    for joint_index, joint_name in enumerate(joint_names):
        for sign in (-1.0, 1.0):
            target = base.copy()
            requested_delta = sign * pulse_delta
            target[joint_index] = np.clip(
                target[joint_index] + requested_delta,
                limits[joint_index, 0],
                limits[joint_index, 1],
            )
            applied_delta = float(target[joint_index] - base[joint_index])
            targets.append(target)
            probes.append(
                {
                    "probe_index": len(probes),
                    "joint_index": joint_index,
                    "joint_name": str(joint_name),
                    "requested_delta_rad": requested_delta,
                    "applied_delta_rad": applied_delta,
                }
            )
    return np.stack(targets), probes


def classify_probe(
    *,
    axis_rotation: float,
    signed_peak_axis_speed: float,
    finite: bool,
    minimum_ball_height: float,
    maximum_ball_displacement: float,
    minimum_contact_count: int,
    thumb_contact_retained: bool,
    maximum_self_penetration: float,
    maximum_object_penetration: float,
    minimum_height: float,
    maximum_displacement: float,
    maximum_penetration: float,
    minimum_axis_rotation: float,
    minimum_axis_speed: float,
) -> dict[str, bool]:
    """Apply explicit safety and measurable-authority thresholds."""
    safe = bool(
        finite
        and minimum_ball_height >= minimum_height
        and maximum_ball_displacement <= maximum_displacement
        and minimum_contact_count >= 2
        and thumb_contact_retained
        and maximum_self_penetration <= maximum_penetration
        and maximum_object_penetration <= maximum_penetration
    )
    positive = bool(
        safe
        and axis_rotation >= minimum_axis_rotation
        and signed_peak_axis_speed >= minimum_axis_speed
    )
    negative = bool(
        safe
        and axis_rotation <= -minimum_axis_rotation
        and signed_peak_axis_speed <= -minimum_axis_speed
    )
    return {"safe": safe, "positive_authority": positive, "negative_authority": negative}


def select_safe_extreme(
    reports: Sequence[dict[str, Any]], *, direction: str
) -> dict[str, Any] | None:
    """Select the strongest axis rotation among probes that retain safety."""
    safe_reports = [report for report in reports if bool(report["safe"])]
    if not safe_reports:
        return None
    if direction == "positive":
        return max(safe_reports, key=lambda report: float(report["axis_rotation_rad"]))
    if direction == "negative":
        return min(safe_reports, key=lambda report: float(report["axis_rotation_rad"]))
    raise ValueError("direction must be 'positive' or 'negative'")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_candidate = (
        Path(ASSETS_ROOT_PATH)
        / "robots"
        / "leap_hand"
        / "canonical_poses"
        / "ball_candidate_01.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=default_candidate)
    parser.add_argument("--pulse-delta", type=float, default=0.04)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--pulse-seconds", type=float, default=0.25)
    parser.add_argument("--sample-seconds", type=float, default=0.05)
    parser.add_argument("--minimum-axis-rotation", type=float, default=0.0025)
    parser.add_argument("--minimum-axis-speed", type=float, default=0.05)
    return parser.parse_args(argv)


def _contact_flags(backend: MuJoCoBackend) -> np.ndarray:
    return np.stack(
        [
            np.asarray(backend.get_sensor_data(name)).reshape(backend.num_envs, -1)[:, 0] > 0.5
            for name in CONTACT_SENSORS
        ],
        axis=1,
    )


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    """Run all signed single-joint probes in one vectorized MuJoCo batch."""
    if args.settle_seconds <= 0.0 or args.pulse_seconds <= 0.0 or args.sample_seconds <= 0.0:
        raise ValueError("settle, pulse, and sample durations must be positive")

    qpos, ctrl, joint_names = load_candidate(args.candidate)
    scene_path = Path(ASSETS_ROOT_PATH) / "robots" / "leap_hand" / "scene_ball.xml"
    sim_dt = 1.0 / 120.0
    backend = MuJoCoBackend(
        SceneCfg(model_file=str(scene_path)),
        1 + 2 * HAND_DOF,
        sim_dt,
        base_name="palm_lower",
        add_body_sensors=True,
        position_actuator_gains={"kp": 3.0, "kd": 0.1, "actuator_ids": slice(0, 16)},
    )
    try:
        ctrl_range = backend.get_actuator_ctrl_range()[:HAND_DOF]
        probe_ctrl, probes = build_probe_targets(
            ctrl,
            joint_names,
            ctrl_range,
            float(args.pulse_delta),
        )
        num_envs = len(probes)
        env_ids = np.arange(num_envs, dtype=np.int32)
        backend.materialize()
        backend.set_state(
            env_ids,
            np.broadcast_to(qpos, (num_envs, QPOS_SIZE)).copy(),
            np.zeros((num_envs, backend.nv), dtype=np.float64),
        )

        settle_steps = max(1, int(round(float(args.settle_seconds) / sim_dt)))
        backend.step(np.broadcast_to(ctrl, (num_envs, HAND_DOF)), nsteps=settle_steps)

        object_body_ids = backend.get_body_ids(["leap_object"])
        object_body_id = int(object_body_ids[0])
        hand_body_ids = backend.get_body_subtree_ids(backend.get_body_ids(["palm_lower"])[0])
        anchor_pos = backend.get_body_pos_w(object_body_ids)[:, 0, :].copy()

        sample_sim_steps = max(1, int(round(float(args.sample_seconds) / sim_dt)))
        sample_dt = sample_sim_steps * sim_dt
        sample_count = max(1, int(np.ceil(float(args.pulse_seconds) / sample_dt)))

        axis_rotation = np.zeros(num_envs, dtype=np.float64)
        peak_positive = np.full(num_envs, -np.inf, dtype=np.float64)
        peak_negative = np.full(num_envs, np.inf, dtype=np.float64)
        max_orthogonal_speed = np.zeros(num_envs, dtype=np.float64)
        max_displacement = np.zeros(num_envs, dtype=np.float64)
        min_height = np.full(num_envs, np.inf, dtype=np.float64)
        min_contacts = np.full(num_envs, len(CONTACT_SENSORS), dtype=np.int32)
        thumb_retained = np.ones(num_envs, dtype=bool)
        finite = np.ones(num_envs, dtype=bool)
        max_self_penetration = np.zeros(num_envs, dtype=np.float64)
        max_object_penetration = np.zeros(num_envs, dtype=np.float64)

        for _ in range(sample_count):
            backend.step(probe_ctrl, nsteps=sample_sim_steps)
            ball_pos = backend.get_body_pos_w(object_body_ids)[:, 0, :]
            ball_linvel = backend.get_body_lin_vel_w(object_body_ids)[:, 0, :]
            ball_angvel = backend.get_body_ang_vel_w(object_body_ids)[:, 0, :]
            contacts = _contact_flags(backend)
            details = backend.get_contact_penetration_details(
                env_ids,
                self_collision_body_ids=hand_body_ids,
                object_body_id=object_body_id,
            )

            baseline_axis_speed = float(ball_angvel[0, 2])
            corrected_axis_speed = ball_angvel[:, 2] - baseline_axis_speed
            axis_rotation += corrected_axis_speed * sample_dt
            peak_positive = np.maximum(peak_positive, corrected_axis_speed)
            peak_negative = np.minimum(peak_negative, corrected_axis_speed)
            max_orthogonal_speed = np.maximum(
                max_orthogonal_speed, np.linalg.norm(ball_angvel[:, :2], axis=1)
            )
            max_displacement = np.maximum(
                max_displacement, np.linalg.norm(ball_pos - anchor_pos, axis=1)
            )
            min_height = np.minimum(min_height, ball_pos[:, 2])
            contact_count = np.count_nonzero(contacts, axis=1)
            min_contacts = np.minimum(min_contacts, contact_count)
            thumb_retained &= contacts[:, 3]
            finite &= np.all(
                np.isfinite(np.concatenate([ball_pos, ball_linvel, ball_angvel], axis=1)),
                axis=1,
            )
            max_self_penetration = np.maximum(
                max_self_penetration,
                np.asarray([detail.self_depth for detail in details]),
            )
            max_object_penetration = np.maximum(
                max_object_penetration,
                np.asarray([detail.object_depth for detail in details]),
            )

        reports: list[dict[str, Any]] = []
        for index, probe in enumerate(probes):
            signed_peak = (
                peak_positive[index]
                if axis_rotation[index] >= 0.0
                else peak_negative[index]
            )
            classification = classify_probe(
                axis_rotation=float(axis_rotation[index]),
                signed_peak_axis_speed=float(signed_peak),
                finite=bool(finite[index]),
                minimum_ball_height=float(min_height[index]),
                maximum_ball_displacement=float(max_displacement[index]),
                minimum_contact_count=int(min_contacts[index]),
                thumb_contact_retained=bool(thumb_retained[index]),
                maximum_self_penetration=float(max_self_penetration[index]),
                maximum_object_penetration=float(max_object_penetration[index]),
                minimum_height=0.4,
                maximum_displacement=0.005,
                maximum_penetration=0.001,
                minimum_axis_rotation=float(args.minimum_axis_rotation),
                minimum_axis_speed=float(args.minimum_axis_speed),
            )
            reports.append(
                {
                    **probe,
                    "axis_rotation_rad": float(axis_rotation[index]),
                    "signed_peak_axis_speed_rad_s": float(signed_peak),
                    "maximum_orthogonal_speed_rad_s": float(max_orthogonal_speed[index]),
                    "maximum_ball_displacement_m": float(max_displacement[index]),
                    "minimum_ball_height_m": float(min_height[index]),
                    "minimum_contact_count": int(min_contacts[index]),
                    "thumb_contact_retained": bool(thumb_retained[index]),
                    "maximum_self_penetration_m": float(max_self_penetration[index]),
                    "maximum_object_penetration_m": float(max_object_penetration[index]),
                    **classification,
                }
            )

        signed_reports = reports[1:]
        return {
            "candidate": str(args.candidate),
            "settings": {
                "pulse_delta_rad": float(args.pulse_delta),
                "settle_seconds": settle_steps * sim_dt,
                "pulse_seconds": sample_count * sample_dt,
                "sample_seconds": sample_dt,
                "minimum_axis_rotation_rad": float(args.minimum_axis_rotation),
                "minimum_axis_speed_rad_s": float(args.minimum_axis_speed),
                "maximum_ball_displacement_m": 0.005,
                "maximum_penetration_m": 0.001,
            },
            "summary": {
                "safe_probe_count": sum(bool(item["safe"]) for item in signed_reports),
                "total_probe_count": len(signed_reports),
                "has_positive_authority": any(
                    bool(item["positive_authority"]) for item in signed_reports
                ),
                "has_negative_authority": any(
                    bool(item["negative_authority"]) for item in signed_reports
                ),
                "best_safe_positive": select_safe_extreme(
                    signed_reports, direction="positive"
                ),
                "best_safe_negative": select_safe_extreme(
                    signed_reports, direction="negative"
                ),
            },
            "probes": reports,
        }
    finally:
        backend.cleanup_scene_assets()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_diagnostic(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
