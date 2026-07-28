"""Goal-conditioned LEAP ball rotation driven by a deterministic pose FSM."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import DomainRandomizationCapabilities, DomainRandomizationProvider
from unilab.dr.types import ResetPlan
from unilab.dtype_config import get_global_dtype
from unilab.envs.manipulation.allegro_inhand.base import ControlConfig, NoiseConfig
from unilab.envs.manipulation.allegro_inhand.rotation import (
    AllegroRotationPPO,
    AllegroRotationPPOCfg,
    compute_ball_angvel,
    normalize_rotation_axis,
)
from unilab.utils.rotation import (
    np_matrix_first_two_cols_from_quat,
    np_quat_apply_batched,
    np_quat_apply_inverse,
    np_quat_conjugate_batched,
    np_quat_inv,
    np_quat_mul,
)

from .base import LeapHandBaseEnv
from .state_cycle_pose_library import POSE_LIBRARY


class StateCyclePhase(IntEnum):
    READY_TO_A = 0
    A_TO_B = 1
    B_TO_READY = 2


NEXT_PHASE: dict[StateCyclePhase, StateCyclePhase] = {
    StateCyclePhase.READY_TO_A: StateCyclePhase.A_TO_B,
    StateCyclePhase.A_TO_B: StateCyclePhase.B_TO_READY,
    StateCyclePhase.B_TO_READY: StateCyclePhase.READY_TO_A,
}
TARGET_POSE: dict[StateCyclePhase, str] = {
    StateCyclePhase.READY_TO_A: "A",
    StateCyclePhase.A_TO_B: "B",
    StateCyclePhase.B_TO_READY: "ready",
}
RESET_POSE_NAMES: tuple[str, ...] = ("ready", "A", "B")
SOURCE_PHASE: dict[str, StateCyclePhase] = {
    "ready": StateCyclePhase.READY_TO_A,
    "A": StateCyclePhase.A_TO_B,
    "B": StateCyclePhase.B_TO_READY,
}


@dataclass(frozen=True)
class TransitionSpec:
    target_pose: str
    timeout_seconds: float
    pose_tolerance: float
    position_tolerance_m: float
    max_ball_speed_m_s: float
    minimum_contacts: int
    minimum_net_angle_rad: float
    hold_steps: int


def _transition_defaults(
    target_pose: str,
    timeout_seconds: float,
    minimum_net_angle_rad: float,
) -> dict[str, float | int | str]:
    return {
        "target_pose": target_pose,
        "timeout_seconds": timeout_seconds,
        "pose_tolerance": 0.08,
        "position_tolerance_m": 0.020,
        "max_ball_speed_m_s": 0.10,
        "minimum_contacts": 2,
        "minimum_net_angle_rad": minimum_net_angle_rad,
        "hold_steps": 4,
    }


@dataclass
class StateCycleConfig:
    cycle_target_net_angle_rad: float = 0.10
    ready_to_a: dict[str, float | int | str] = field(
        default_factory=lambda: _transition_defaults("A", 1.5, 0.0)
    )
    a_to_b: dict[str, float | int | str] = field(
        default_factory=lambda: _transition_defaults("B", 2.0, 0.08)
    )
    b_to_ready: dict[str, float | int | str] = field(
        default_factory=lambda: _transition_defaults("ready", 1.3, 0.0)
    )
    reset_pose_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    def transition_specs(self) -> tuple[TransitionSpec, ...]:
        return (
            TransitionSpec(**self.ready_to_a),
            TransitionSpec(**self.a_to_b),
            TransitionSpec(**self.b_to_ready),
        )


@dataclass
class StateCycleRewardConfig:
    pose_progress_scale: float = 4.0
    pose_tracking_scale: float = 0.50
    pose_sigma: float = 0.08
    rotation_progress_scale: float = 3.0
    reverse_rotation_scale: float = 5.0
    rotation_target_axis_speed_rad_s: float = 0.50
    rotation_overspeed_scale: float = 1.00
    rotation_workspace_full_radius_m: float = 0.015
    rotation_workspace_zero_radius_m: float = 0.035
    position_error_scale: float = 6.0
    object_linvel_scale: float = 0.3
    transition_success_bonus: float = 0.10
    cycle_success_bonus: float = 0.30
    invalid_cycle_penalty: float = 0.15
    timeout_penalty: float = 0.25
    failure_penalty: float = 3.0
    failure_rotation_clawback_cap: float = 5.0


@dataclass(frozen=True)
class StateCycleAdvance:
    phase: np.ndarray
    hold_steps: np.ndarray
    transition_event: np.ndarray
    cycle_event: np.ndarray


@dataclass(frozen=True)
class StateCycleRewardTerms:
    pose_progress: np.ndarray
    pose_tracking: np.ndarray
    rotation_progress: np.ndarray
    rotation_overspeed: np.ndarray
    reverse_rotation: np.ndarray
    position_error: np.ndarray
    obj_linvel: np.ndarray
    transition_event: np.ndarray
    cycle_event: np.ndarray
    invalid_cycle: np.ndarray
    timeout: np.ndarray
    failure: np.ndarray
    timeout_rotation_clawback: np.ndarray
    failure_rotation_clawback: np.ndarray
    phase_rotation_reward_earned_current: np.ndarray
    episode_rotation_reward_earned_current: np.ndarray
    total: np.ndarray


@dataclass(frozen=True)
class StateCycleRotationUpdate:
    cycle_net_angle: np.ndarray
    episode_net_angle: np.ndarray
    phase_positive_angle: np.ndarray
    phase_positive_angle_for_reward: np.ndarray
    completed_cycle_net_angle: np.ndarray
    rotation_cycle_success_event: np.ndarray
    invalid_rotation_cycle_event: np.ndarray
    last_completed_cycle_net_angle: np.ndarray
    completed_cycle_angle_sum: np.ndarray
    completed_cycle_count: np.ndarray
    valid_rotation_cycles_completed: np.ndarray


@dataclass(frozen=True)
class StateCycleRotationStabilityGates:
    workspace: np.ndarray
    contact: np.ndarray
    no_palm: np.ndarray
    linvel: np.ndarray
    stability: np.ndarray


@dataclass(frozen=True)
class StateCycleEarnedRewardUpdate:
    phase: np.ndarray
    episode: np.ndarray


def compute_pose_distance(
    current_hand_qpos: np.ndarray,
    target_hand_qpos: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
) -> np.ndarray:
    """Return joint-range normalized RMS pose distance."""
    current = np.asarray(current_hand_qpos)
    target = np.asarray(target_hand_qpos, dtype=current.dtype)
    lower = np.asarray(ctrl_lower, dtype=current.dtype)
    upper = np.asarray(ctrl_upper, dtype=current.dtype)
    normalized_error = (current - target) / (upper - lower + 1e-8)
    return np.sqrt(np.mean(np.square(normalized_error), axis=1)).astype(
        current.dtype, copy=False
    )


def compute_pose_progress(
    previous_pose_distance: np.ndarray,
    current_pose_distance: np.ndarray,
) -> np.ndarray:
    return np.asarray(previous_pose_distance) - np.asarray(current_pose_distance)


def rotation_condition(
    phases: np.ndarray,
    edge_net_angle: np.ndarray,
    minimum_angles: np.ndarray,
) -> np.ndarray:
    required = np.asarray(minimum_angles)[np.asarray(phases, dtype=np.intp)]
    return (required <= 0.0) | (np.asarray(edge_net_angle) >= required)


def update_state_cycle_rotation(
    *,
    axis_delta: np.ndarray,
    cycle_net_angle: np.ndarray,
    episode_net_angle: np.ndarray,
    phase_positive_angle: np.ndarray,
    last_completed_cycle_net_angle: np.ndarray,
    completed_cycle_angle_sum: np.ndarray,
    completed_cycle_count: np.ndarray,
    valid_rotation_cycles_completed: np.ndarray,
    transition_event: np.ndarray,
    cycle_event: np.ndarray,
    cycle_target_net_angle_rad: float,
) -> StateCycleRotationUpdate:
    """Accumulate signed rotation and finalize only completed FSM cycles."""
    delta = np.asarray(axis_delta)
    cycle_angle = np.asarray(cycle_net_angle, dtype=delta.dtype).copy()
    episode_angle = np.asarray(episode_net_angle, dtype=delta.dtype).copy()
    positive_angle = np.asarray(phase_positive_angle, dtype=delta.dtype).copy()
    cycle_angle += delta
    episode_angle += delta
    positive_angle += np.maximum(delta, 0.0)
    positive_angle_for_reward = positive_angle.copy()

    cycle_event_array = np.asarray(cycle_event, dtype=bool)
    transition_event_array = np.asarray(transition_event, dtype=bool)
    completed_angle = cycle_angle.copy()
    rotation_cycle_success = cycle_event_array & (
        completed_angle >= cycle_target_net_angle_rad
    )
    invalid_rotation_cycle = cycle_event_array & ~rotation_cycle_success

    last_completed = np.asarray(
        last_completed_cycle_net_angle, dtype=delta.dtype
    ).copy()
    completed_sum = np.asarray(completed_cycle_angle_sum, dtype=delta.dtype).copy()
    completed_count = np.asarray(completed_cycle_count, dtype=np.uint32).copy()
    valid_count = np.asarray(valid_rotation_cycles_completed, dtype=np.uint32).copy()
    last_completed[cycle_event_array] = completed_angle[cycle_event_array]
    completed_sum[cycle_event_array] += completed_angle[cycle_event_array]
    completed_count += cycle_event_array.astype(np.uint32)
    valid_count += rotation_cycle_success.astype(np.uint32)

    cycle_angle[cycle_event_array] = 0.0
    positive_angle[transition_event_array] = 0.0
    return StateCycleRotationUpdate(
        cycle_net_angle=cycle_angle,
        episode_net_angle=episode_angle,
        phase_positive_angle=positive_angle,
        phase_positive_angle_for_reward=positive_angle_for_reward,
        completed_cycle_net_angle=completed_angle,
        rotation_cycle_success_event=rotation_cycle_success,
        invalid_rotation_cycle_event=invalid_rotation_cycle,
        last_completed_cycle_net_angle=last_completed,
        completed_cycle_angle_sum=completed_sum,
        completed_cycle_count=completed_count,
        valid_rotation_cycles_completed=valid_count,
    )


def compute_rotation_stability_gates(
    *,
    reward_cfg: StateCycleRewardConfig,
    workspace_error: np.ndarray,
    ball_speed: np.ndarray,
    contact_count: np.ndarray,
    minimum_contacts: np.ndarray,
    palm_contact: np.ndarray,
    max_ball_speed: np.ndarray,
) -> StateCycleRotationStabilityGates:
    """Return smooth control-quality gates for positive rotation shaping."""
    workspace = np.asarray(workspace_error)
    dtype = workspace.dtype
    workspace_denominator = max(
        reward_cfg.rotation_workspace_zero_radius_m
        - reward_cfg.rotation_workspace_full_radius_m,
        1e-8,
    )
    workspace_gate = np.clip(
        (reward_cfg.rotation_workspace_zero_radius_m - workspace)
        / workspace_denominator,
        0.0,
        1.0,
    ).astype(dtype, copy=False)
    contact_gate = (
        np.asarray(contact_count) >= np.asarray(minimum_contacts)
    ).astype(dtype)
    no_palm_gate = (~np.asarray(palm_contact, dtype=bool)).astype(dtype)
    max_speed = np.asarray(max_ball_speed, dtype=dtype)
    full_speed = 0.5 * max_speed
    linvel_gate = np.clip(
        (max_speed - np.asarray(ball_speed, dtype=dtype))
        / np.maximum(max_speed - full_speed, 1e-8),
        0.0,
        1.0,
    ).astype(dtype, copy=False)
    stability_gate = workspace_gate * contact_gate * no_palm_gate * linvel_gate
    return StateCycleRotationStabilityGates(
        workspace=workspace_gate,
        contact=contact_gate,
        no_palm=no_palm_gate,
        linvel=linvel_gate,
        stability=stability_gate,
    )


def finalize_rotation_reward_buffers(
    *,
    phase_rotation_reward_earned_current: np.ndarray,
    episode_rotation_reward_earned_current: np.ndarray,
    transition_event: np.ndarray,
    timeout: np.ndarray,
    workspace_failure: np.ndarray,
) -> StateCycleEarnedRewardUpdate:
    """Clear earned-reward buffers only after terminal clawbacks are computed."""
    phase = np.asarray(phase_rotation_reward_earned_current).copy()
    episode = np.asarray(episode_rotation_reward_earned_current).copy()
    terminal = np.asarray(timeout, dtype=bool) | np.asarray(
        workspace_failure, dtype=bool
    )
    phase[np.asarray(transition_event, dtype=bool) | terminal] = 0.0
    episode[terminal] = 0.0
    return StateCycleEarnedRewardUpdate(phase=phase, episode=episode)


def compute_state_cycle_reward(
    *,
    reward_cfg: StateCycleRewardConfig,
    ctrl_dt: float,
    pose_progress: np.ndarray,
    pose_distance: np.ndarray,
    phase_start_pose_distance: np.ndarray,
    phase_rotation_reward_earned_before: np.ndarray,
    episode_rotation_reward_earned_before: np.ndarray,
    axis_delta: np.ndarray,
    rotation_stability_gate: np.ndarray,
    position_error: np.ndarray,
    ball_linvel: np.ndarray,
    transition_event: np.ndarray,
    rotation_cycle_success_event: np.ndarray,
    invalid_rotation_cycle_event: np.ndarray,
    timeout: np.ndarray,
    workspace_failure: np.ndarray,
) -> StateCycleRewardTerms:
    """Compute bounded per-step shaping and mutually exclusive failure events."""
    axis_delta_array = np.asarray(axis_delta)
    dtype = axis_delta_array.dtype

    def reward_array(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=dtype)

    pose_progress_reward = reward_cfg.pose_progress_scale * reward_array(pose_progress)
    pose_tracking_rate = reward_cfg.pose_tracking_scale * (
        np.exp(
            -np.square(
                reward_array(pose_distance) / max(reward_cfg.pose_sigma, 1e-8)
            )
        )
        - 1.0
    )

    positive_axis_delta = np.maximum(axis_delta_array, 0.0)
    reverse_axis_delta = np.maximum(-axis_delta_array, 0.0)
    target_axis_delta = reward_cfg.rotation_target_axis_speed_rad_s * ctrl_dt
    useful_positive_delta = np.minimum(positive_axis_delta, target_axis_delta)
    overspeed_delta = np.maximum(positive_axis_delta - target_axis_delta, 0.0)
    rotation_progress_reward = (
        reward_cfg.rotation_progress_scale
        * useful_positive_delta
        * reward_array(rotation_stability_gate)
    )
    rotation_overspeed_reward = (
        -reward_cfg.rotation_overspeed_scale * overspeed_delta
    )
    reverse_rotation_reward = -reward_cfg.reverse_rotation_scale * reverse_axis_delta
    phase_rotation_earned_current = (
        reward_array(phase_rotation_reward_earned_before) + rotation_progress_reward
    )
    episode_rotation_earned_current = (
        reward_array(episode_rotation_reward_earned_before) + rotation_progress_reward
    )
    position_error_rate = -reward_cfg.position_error_scale * reward_array(position_error)
    object_linvel_rate = -reward_cfg.object_linvel_scale * np.sum(
        np.abs(reward_array(ball_linvel)), axis=1
    )
    pose_tracking_reward = pose_tracking_rate * ctrl_dt
    position_error_reward = position_error_rate * ctrl_dt
    object_linvel_reward = object_linvel_rate * ctrl_dt

    transition_event_reward = (
        reward_cfg.transition_success_bonus * reward_array(transition_event)
    )
    cycle_event_reward = reward_cfg.cycle_success_bonus * reward_array(
        rotation_cycle_success_event
    )
    invalid_cycle_reward = -reward_cfg.invalid_cycle_penalty * reward_array(
        invalid_rotation_cycle_event
    )
    timeout_event = np.asarray(timeout) & ~np.asarray(workspace_failure)
    phase_pose_progress = np.maximum(
        reward_array(phase_start_pose_distance) - reward_array(pose_distance),
        0.0,
    )
    timeout_reward = -(
        reward_cfg.timeout_penalty
        + reward_cfg.pose_progress_scale * phase_pose_progress
        + phase_rotation_earned_current
    ) * timeout_event
    workspace_failure_array = np.asarray(workspace_failure)
    failure_rotation_clawback = np.minimum(
        episode_rotation_earned_current,
        reward_cfg.failure_rotation_clawback_cap,
    )
    failure_reward = -(
        reward_cfg.failure_penalty + failure_rotation_clawback
    ) * workspace_failure_array
    timeout_rotation_clawback_term = -phase_rotation_earned_current * timeout_event
    failure_rotation_clawback_term = (
        -failure_rotation_clawback * workspace_failure_array
    )
    total = (
        pose_progress_reward
        + pose_tracking_reward
        + rotation_progress_reward
        + rotation_overspeed_reward
        + reverse_rotation_reward
        + position_error_reward
        + object_linvel_reward
        + transition_event_reward
        + cycle_event_reward
        + invalid_cycle_reward
        + timeout_reward
        + failure_reward
    )
    return StateCycleRewardTerms(
        pose_progress=reward_array(pose_progress_reward),
        pose_tracking=reward_array(pose_tracking_reward),
        rotation_progress=reward_array(rotation_progress_reward),
        rotation_overspeed=reward_array(rotation_overspeed_reward),
        reverse_rotation=reward_array(reverse_rotation_reward),
        position_error=reward_array(position_error_reward),
        obj_linvel=reward_array(object_linvel_reward),
        transition_event=reward_array(transition_event_reward),
        cycle_event=reward_array(cycle_event_reward),
        invalid_cycle=reward_array(invalid_cycle_reward),
        timeout=reward_array(timeout_reward),
        failure=reward_array(failure_reward),
        timeout_rotation_clawback=reward_array(timeout_rotation_clawback_term),
        failure_rotation_clawback=reward_array(failure_rotation_clawback_term),
        phase_rotation_reward_earned_current=reward_array(
            phase_rotation_earned_current
        ),
        episode_rotation_reward_earned_current=reward_array(
            episode_rotation_earned_current
        ),
        total=reward_array(total),
    )


def compute_timeout_event(
    phase_steps: np.ndarray,
    timeout_steps: np.ndarray,
    transition_event: np.ndarray,
    workspace_failure: np.ndarray,
) -> np.ndarray:
    """Allow success on the final legal step and keep failures exclusive."""
    return (
        (np.asarray(phase_steps) >= np.asarray(timeout_steps))
        & ~np.asarray(transition_event)
        & ~np.asarray(workspace_failure)
    )


def compute_state_cycle_diagnostic_metrics(
    *,
    phases_before: np.ndarray,
    timeout: np.ndarray,
    pose_distance: np.ndarray,
    position_error: np.ndarray,
    ball_speed: np.ndarray,
    contact_count: np.ndarray,
    edge_net_angle: np.ndarray,
    remaining_angle: np.ndarray,
    hold_steps: np.ndarray,
    pose_ok: np.ndarray,
    position_ok: np.ndarray,
    speed_ok: np.ndarray,
    contact_ok: np.ndarray,
    rotation_ok: np.ndarray,
    no_palm_contact: np.ndarray,
    transition_event: np.ndarray,
    workspace_failure: np.ndarray,
) -> dict[str, float]:
    """Compute phase and terminal diagnostics using the pre-transition phase."""
    phases = np.asarray(phases_before)
    timeout_mask = np.asarray(timeout, dtype=bool)
    hold_steps_array = np.asarray(hold_steps)
    arrays = {
        "pose_distance": np.asarray(pose_distance),
        "position_error_m": np.asarray(position_error),
        "ball_speed_m_s": np.asarray(ball_speed),
        "contact_count": np.asarray(contact_count),
        "edge_net_angle": np.asarray(edge_net_angle),
        "remaining_angle": np.asarray(remaining_angle),
        "hold_steps": hold_steps_array,
    }
    gates = {
        "pose_ok": np.asarray(pose_ok, dtype=bool),
        "position_ok": np.asarray(position_ok, dtype=bool),
        "speed_ok": np.asarray(speed_ok, dtype=bool),
        "contact_ok": np.asarray(contact_ok, dtype=bool),
        "rotation_ok": np.asarray(rotation_ok, dtype=bool),
        "no_palm_contact": np.asarray(no_palm_contact, dtype=bool),
    }

    def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.mean(values[mask])) if np.any(mask) else 0.0

    log: dict[str, float] = {
        "timeout/final_phase_mean": masked_mean(phases, timeout_mask),
    }
    for name, values in arrays.items():
        log[f"timeout/final_{name}_mean"] = masked_mean(values, timeout_mask)
    for name, values in gates.items():
        log[f"timeout/final_{name}_rate"] = masked_mean(values, timeout_mask)

    all_gates_except_hold = (
        gates["pose_ok"]
        & gates["position_ok"]
        & gates["speed_ok"]
        & gates["contact_ok"]
        & gates["rotation_ok"]
        & gates["no_palm_contact"]
        & ~np.asarray(workspace_failure, dtype=bool)
    )
    near_transition = all_gates_except_hold & ~np.asarray(
        transition_event, dtype=bool
    )

    active_arrays = {
        "pose_distance": arrays["pose_distance"],
        "remaining_angle": arrays["remaining_angle"],
        "hold_steps": arrays["hold_steps"],
    }
    for phase in StateCyclePhase:
        phase_name = phase.name
        phase_mask = phases == int(phase)
        phase_timeout_mask = timeout_mask & phase_mask
        for name, values in active_arrays.items():
            log[f"state_cycle/{name}_{phase_name}_mean"] = masked_mean(
                values, phase_mask
            )
        for name, values in gates.items():
            log[f"state_cycle/{name}_{phase_name}_rate"] = masked_mean(
                values, phase_mask
            )
        log[f"state_cycle/near_transition_{phase_name}_rate"] = masked_mean(
            near_transition, phase_mask
        )
        log[f"state_cycle/timeout_{phase_name}_rate"] = float(
            np.mean(phase_timeout_mask)
        )
        log[f"state_cycle/timeout_{phase_name}_conditional_rate"] = masked_mean(
            timeout_mask, phase_mask
        )

        count = int(np.sum(phase_timeout_mask))
        log[f"timeout/{phase_name}_count"] = float(count)
        for name, values in arrays.items():
            log[f"timeout/{phase_name}_{name}_mean"] = masked_mean(
                values, phase_timeout_mask
            )
        for name, values in gates.items():
            log[f"timeout/{phase_name}_{name}_rate"] = masked_mean(
                values, phase_timeout_mask
            )
        blockers = {
            "pose_blocked": ~gates["pose_ok"],
            "position_blocked": ~gates["position_ok"],
            "speed_blocked": ~gates["speed_ok"],
            "contact_blocked": ~gates["contact_ok"],
            "rotation_blocked": ~gates["rotation_ok"],
            "palm_blocked": ~gates["no_palm_contact"],
        }
        for name, values in blockers.items():
            log[f"timeout/{phase_name}_{name}_rate"] = masked_mean(
                values, phase_timeout_mask
            )
    return log


def advance_state_cycle(
    phases: np.ndarray,
    hold_steps: np.ndarray,
    valid: np.ndarray,
    required_hold_steps: np.ndarray,
) -> StateCycleAdvance:
    """Advance only after the per-edge condition holds for consecutive steps."""
    phase = np.asarray(phases, dtype=np.int8).copy()
    hold = np.asarray(hold_steps, dtype=np.uint32).copy()
    valid_array = np.asarray(valid, dtype=bool)
    required = np.asarray(required_hold_steps, dtype=np.uint32)[phase.astype(np.intp)]
    hold[:] = np.where(valid_array, hold + 1, 0)
    transition_event = valid_array & (hold >= required)
    cycle_event = transition_event & (phase == int(StateCyclePhase.B_TO_READY))
    next_lookup = np.asarray(
        [int(NEXT_PHASE[StateCyclePhase(index)]) for index in range(len(StateCyclePhase))],
        dtype=np.int8,
    )
    phase[transition_event] = next_lookup[phase[transition_event]]
    hold[transition_event] = 0
    return StateCycleAdvance(phase, hold, transition_event, cycle_event)


def reset_phase_for_pose(source_pose_names: list[str] | tuple[str, ...]) -> np.ndarray:
    try:
        return np.asarray([int(SOURCE_PHASE[name]) for name in source_pose_names], dtype=np.int8)
    except KeyError as exc:
        raise ValueError(f"unsupported state-cycle reset pose: {exc.args[0]}") from exc


def build_state_cycle_reset_arrays(
    source_pose_names: list[str] | tuple[str, ...],
    *,
    nv: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build exact pose qpos/qvel/ctrl arrays for reset and unit tests."""
    phases = reset_phase_for_pose(source_pose_names)
    qpos = np.stack([POSE_LIBRARY[name].qpos for name in source_pose_names]).copy()
    qvel = np.zeros((len(source_pose_names), nv), dtype=np.float64)
    ctrl = np.stack([POSE_LIBRARY[name].ctrl for name in source_pose_names]).copy()
    return qpos, qvel, ctrl, phases


def _validate_transition_spec(phase: StateCyclePhase, spec: TransitionSpec) -> None:
    if spec.target_pose != TARGET_POSE[phase]:
        raise ValueError(f"{phase.name} target_pose must be {TARGET_POSE[phase]!r}")
    finite_positive = (
        spec.timeout_seconds,
        spec.pose_tolerance,
        spec.position_tolerance_m,
        spec.max_ball_speed_m_s,
    )
    if not all(np.isfinite(value) and value > 0.0 for value in finite_positive):
        raise ValueError(f"{phase.name} tolerances and timeout must be positive and finite")
    if not np.isfinite(spec.minimum_net_angle_rad) or spec.minimum_net_angle_rad < 0.0:
        raise ValueError(f"{phase.name} minimum_net_angle_rad must be finite and non-negative")
    if spec.minimum_contacts not in range(1, 5):
        raise ValueError(f"{phase.name} minimum_contacts must be between 1 and 4")
    if spec.hold_steps <= 0:
        raise ValueError(f"{phase.name} hold_steps must be positive")


@registry.envcfg("LeapInhandBallStateCycleRotation")
@dataclass
class LeapInhandBallStateCycleRotationCfg(AllegroRotationPPOCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "leap_hand" / "scene_ball.xml"),
            visual_model_file=str(
                ASSETS_ROOT_PATH
                / "robots"
                / "leap_hand"
                / "scene_ball_state_cycle_visual.xml"
            ),
        )
    )
    sim_dt: float = 0.005
    ctrl_dt: float = 0.05
    control_config: ControlConfig = field(
        default_factory=lambda: ControlConfig(action_scale=1.0 / 24.0, kp=3.0, kd=0.1)
    )
    noise_config: NoiseConfig = field(default_factory=lambda: NoiseConfig(level=0.0))
    reward_config: StateCycleRewardConfig | None = None
    state_cycle: StateCycleConfig = field(default_factory=StateCycleConfig)
    termination_workspace_radius: float = 0.05
    joint_velocity_scale: float = 0.20
    max_episode_seconds: float = 30.0

    def validate(self) -> None:
        super().validate()
        if self.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        specs = self.state_cycle.transition_specs()
        for phase, spec in zip(StateCyclePhase, specs):
            _validate_transition_spec(phase, spec)
        if not np.isfinite(self.state_cycle.cycle_target_net_angle_rad) or (
            self.state_cycle.cycle_target_net_angle_rad <= 0.0
        ):
            raise ValueError("cycle_target_net_angle_rad must be positive and finite")
        weights = np.asarray(self.state_cycle.reset_pose_weights, dtype=np.float64)
        if weights.shape != (len(RESET_POSE_NAMES),):
            raise ValueError("reset_pose_weights must contain Ready, A, and B weights")
        if not np.isfinite(weights).all() or np.any(weights < 0.0) or np.sum(weights) <= 0.0:
            raise ValueError("reset_pose_weights must be finite, non-negative, and non-zero")
        if not np.isfinite(self.termination_workspace_radius) or (
            self.termination_workspace_radius <= 0.0
        ):
            raise ValueError("termination_workspace_radius must be positive and finite")
        reward_values = vars(self.reward_config).values()
        if not all(np.isfinite(value) and value >= 0.0 for value in reward_values):
            raise ValueError("state-cycle reward values must be finite and non-negative")
        reward_cfg = self.reward_config
        if reward_cfg.rotation_target_axis_speed_rad_s <= 0.0:
            raise ValueError("rotation_target_axis_speed_rad_s must be positive")
        if not (
            reward_cfg.rotation_workspace_full_radius_m
            < reward_cfg.rotation_workspace_zero_radius_m
            < self.termination_workspace_radius
        ):
            raise ValueError(
                "rotation workspace radii must satisfy full < zero < termination"
            )


class LeapStateCycleResetProvider(DomainRandomizationProvider):
    """Reset directly to one of the recorded Ready/A/B source poses."""

    def validate(
        self, env: Any, capabilities: DomainRandomizationCapabilities
    ) -> None:
        del env, capabilities

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        source_names = env._sample_reset_pose_names(len(env_ids))
        qpos, qvel, ctrl, phases = build_state_cycle_reset_arrays(source_names, nv=env.nv)
        target_hand = env._target_hand_qpos(phases, dtype=get_global_dtype())
        previous_distance = compute_pose_distance(
            qpos[:, :16], target_hand, env._ctrl_lower, env._ctrl_upper
        )
        num_reset = len(env_ids)
        dtype = get_global_dtype()
        info_updates: dict[str, np.ndarray] = {
            "current_actions": np.zeros((num_reset, env._num_action), dtype=dtype),
            "last_actions": np.zeros((num_reset, env._num_action), dtype=dtype),
            "prev_ctrl": np.asarray(ctrl, dtype=dtype),
            "init_pose": np.asarray(qpos[:, :16], dtype=dtype).copy(),
            "prev_dof_pos": np.asarray(qpos[:, :16], dtype=dtype).copy(),
            "prev_ball_pos": np.asarray(qpos[:, 16:19], dtype=dtype).copy(),
            "prev_ball_quat": np.asarray(qpos[:, 19:23], dtype=dtype).copy(),
            "rotation_anchor_pos": np.asarray(qpos[:, 16:19], dtype=dtype).copy(),
            "state_cycle_phase": phases,
            "state_cycle_phase_steps": np.zeros(num_reset, dtype=np.uint32),
            "state_cycle_success_hold_steps": np.zeros(num_reset, dtype=np.uint32),
            "state_cycle_edge_start_quat": np.asarray(qpos[:, 19:23], dtype=dtype).copy(),
            "state_cycle_edge_net_angle": np.zeros(num_reset, dtype=dtype),
            "state_cycle_cycle_net_angle": np.zeros(num_reset, dtype=dtype),
            "state_cycle_episode_net_angle": np.zeros(num_reset, dtype=dtype),
            "state_cycle_phase_positive_angle": np.zeros(num_reset, dtype=dtype),
            "state_cycle_last_completed_cycle_net_angle": np.zeros(
                num_reset, dtype=dtype
            ),
            "state_cycle_completed_cycle_angle_sum": np.zeros(num_reset, dtype=dtype),
            "state_cycle_completed_cycle_count": np.zeros(num_reset, dtype=np.uint32),
            "state_cycle_valid_rotation_cycles_completed": np.zeros(
                num_reset, dtype=np.uint32
            ),
            "state_cycle_phase_rotation_reward_earned": np.zeros(
                num_reset, dtype=dtype
            ),
            "state_cycle_episode_rotation_reward_earned": np.zeros(
                num_reset, dtype=dtype
            ),
            "state_cycle_prev_pose_distance": previous_distance.astype(dtype),
            "state_cycle_phase_start_pose_distance": previous_distance.astype(dtype).copy(),
            "state_cycle_cycles_completed": np.zeros(num_reset, dtype=np.uint32),
            "state_cycle_transition_success": np.zeros(num_reset, dtype=bool),
            "state_cycle_timeout": np.zeros(num_reset, dtype=bool),
            "state_cycle_workspace_failure": np.zeros(num_reset, dtype=bool),
        }
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=None,
        )

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        ball_pos = env.get_ball_pos()[env_ids]
        ball_quat = env.get_ball_quat()[env_ids]
        info_updates["prev_ball_pos"] = ball_pos.copy()
        info_updates["prev_ball_quat"] = ball_quat.copy()
        info_updates["rotation_anchor_pos"] = ball_pos.copy()
        info_updates["state_cycle_edge_start_quat"] = ball_quat.copy()
        return cast(
            dict[str, np.ndarray],
            env._compute_state_cycle_obs(env_ids, info_updates),
        )


@registry.env("LeapInhandBallStateCycleRotation", sim_backend="motrix")
@registry.env("LeapInhandBallStateCycleRotation", sim_backend="mujoco")
class LeapInhandBallStateCycleRotationEnv(AllegroRotationPPO, LeapHandBaseEnv):
    """Learn adjacent Ready/A/B transitions selected by a deterministic FSM."""

    _cfg: LeapInhandBallStateCycleRotationCfg
    _reward_cfg: StateCycleRewardConfig
    _NUM_STATE_CYCLE_OBS = 142
    _CONTACT_SENSOR_NAMES = (
        "leap_index_contact",
        "leap_middle_contact",
        "leap_ring_contact",
        "leap_thumb_contact",
    )
    _PALM_CONTACT_SENSOR_NAME = "leap_palm_contact"

    def __init__(
        self,
        cfg: LeapInhandBallStateCycleRotationCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> None:
        self._transition_specs = cfg.state_cycle.transition_specs()
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._all_env_ids = np.arange(num_envs, dtype=np.int32)
        self._palm_body_ids = self._backend.get_body_ids([self._BASE_BODY_NAME])
        self._rotation_axis_w = normalize_rotation_axis(cfg.rotation_axis)
        self._timeout_steps = np.maximum(
            1,
            np.ceil(
                np.asarray([spec.timeout_seconds for spec in self._transition_specs])
                / cfg.ctrl_dt
            ),
        ).astype(np.uint32)
        self._hold_steps_required = np.asarray(
            [spec.hold_steps for spec in self._transition_specs], dtype=np.uint32
        )
        self._minimum_angles = np.asarray(
            [spec.minimum_net_angle_rad for spec in self._transition_specs],
            dtype=self._np_dtype,
        )
        self._max_required_angle = max(float(np.max(self._minimum_angles)), 1e-6)
        self._cycle_target_angle = float(cfg.state_cycle.cycle_target_net_angle_rad)
        self._diagnostic_log_interval_steps = 4

    def set_diagnostic_log_interval(self, interval_steps: int) -> None:
        """Set state-cycle diagnostic cadence for batch evaluation."""
        if interval_steps <= 0:
            raise ValueError("diagnostic log interval must be positive")
        self._diagnostic_log_interval_steps = int(interval_steps)

    def _make_domain_randomization_provider(self) -> DomainRandomizationProvider:
        return LeapStateCycleResetProvider()

    def _init_reward_functions(self) -> None:
        self._reward_fns = {}

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": self._NUM_STATE_CYCLE_OBS}

    def _sample_reset_pose_names(self, num_reset: int) -> list[str]:
        weights = np.asarray(self._cfg.state_cycle.reset_pose_weights, dtype=np.float64)
        weights /= np.sum(weights)
        indices = np.random.choice(len(RESET_POSE_NAMES), size=num_reset, p=weights)
        return [RESET_POSE_NAMES[index] for index in indices]

    def _target_hand_qpos(self, phases: np.ndarray, *, dtype: Any) -> np.ndarray:
        return np.stack(
            [POSE_LIBRARY[TARGET_POSE[StateCyclePhase(int(phase))]].hand_qpos for phase in phases]
        ).astype(dtype, copy=False)

    def _target_ball_pos(self, phases: np.ndarray, *, dtype: Any) -> np.ndarray:
        return np.stack(
            [POSE_LIBRARY[TARGET_POSE[StateCyclePhase(int(phase))]].ball_pos for phase in phases]
        ).astype(dtype, copy=False)

    @staticmethod
    def _info_rows(info: dict[str, Any], name: str, env_ids: np.ndarray) -> np.ndarray:
        values = np.asarray(info[name])
        return values if values.shape[0] == len(env_ids) else values[env_ids]

    def _contacts(self, env_ids: np.ndarray) -> np.ndarray:
        contacts = self._backend.get_sensor_data_batch(self._CONTACT_SENSOR_NAMES)
        return np.asarray(contacts[env_ids] > 0.5, dtype=bool)

    def _palm_contacts(self, env_ids: np.ndarray) -> np.ndarray:
        contacts = self._backend.get_sensor_data_batch((self._PALM_CONTACT_SENSOR_NAME,))
        return np.asarray(contacts[env_ids, 0] > 0.5, dtype=self._np_dtype)

    def _palm_pose(self, env_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        palm_pos = self._backend.get_body_pos_w(self._palm_body_ids)[env_ids, 0, :]
        palm_quat = self._backend.get_body_quat_w(self._palm_body_ids)[env_ids, 0, :]
        return palm_pos, palm_quat

    def _compute_state_cycle_obs(
        self, env_ids: np.ndarray, info: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        dtype = get_global_dtype()
        dof_pos = self.get_hand_dof_pos()[env_ids]
        dof_vel = self.get_hand_dof_vel()[env_ids]
        targets = self._info_rows(info, "prev_ctrl", env_ids).astype(dtype)
        last_actions = self._info_rows(info, "last_actions", env_ids).astype(dtype)
        ball_pos = self.get_ball_pos()[env_ids]
        ball_quat = self.get_ball_quat()[env_ids]
        ball_linvel = self.get_ball_linvel()[env_ids]
        ball_angvel = self.get_ball_angvel()[env_ids]
        fingertip_pos = self.get_fingertip_pos()[env_ids]
        palm_pos, palm_quat = self._palm_pose(env_ids)

        dof_pos_norm = 2.0 * (dof_pos - self._dof_mid) / (self._dof_range + 1e-8)
        dof_vel_scaled = dof_vel * self._cfg.joint_velocity_scale
        ball_rel_pos = np_quat_apply_inverse(palm_quat, ball_pos - palm_pos)
        ball_rel_quat = np_quat_mul(np_quat_inv(palm_quat), ball_quat)
        ball_rot_6d = np_matrix_first_two_cols_from_quat(ball_rel_quat)
        linvel_palm = np_quat_apply_inverse(palm_quat, ball_linvel)
        angvel_palm = np_quat_apply_inverse(palm_quat, ball_angvel)
        fingertip_rel_world = fingertip_pos - ball_pos[:, None, :]
        fingertip_rel_palm = np_quat_apply_batched(
            np_quat_conjugate_batched(palm_quat)[:, None, :], fingertip_rel_world
        ).reshape(len(env_ids), -1)
        target_axis = np.broadcast_to(self._rotation_axis_w, (len(env_ids), 3))
        target_axis_palm = np_quat_apply_inverse(palm_quat, target_axis)

        phases = self._info_rows(info, "state_cycle_phase", env_ids).astype(np.intp)
        target_hand = self._target_hand_qpos(phases, dtype=dtype)
        target_hand_norm = 2.0 * (target_hand - self._dof_mid) / (self._dof_range + 1e-8)
        target_hand_error = target_hand_norm - dof_pos_norm
        phase_one_hot = np.eye(len(StateCyclePhase), dtype=dtype)[phases]
        phase_steps = self._info_rows(info, "state_cycle_phase_steps", env_ids).astype(dtype)
        phase_progress = np.clip(
            phase_steps / self._timeout_steps[phases], 0.0, 1.0
        )[:, None]
        edge_net_angle = self._info_rows(
            info, "state_cycle_edge_net_angle", env_ids
        ).astype(dtype)
        required_angle = self._minimum_angles[phases]
        remaining_angle_raw = np.where(
            required_angle > 0.0,
            np.maximum(required_angle - edge_net_angle, 0.0),
            0.0,
        )
        remaining_angle = (remaining_angle_raw / self._max_required_angle)[:, None]
        target_ball_error = self._target_ball_pos(phases, dtype=dtype) - ball_pos
        rotation_required = (required_angle / self._max_required_angle)[:, None]
        cycle_net_angle = self._info_rows(
            info, "state_cycle_cycle_net_angle", env_ids
        ).astype(dtype)
        cycle_rotation_progress = np.clip(
            cycle_net_angle / self._cycle_target_angle,
            -2.0,
            2.0,
        )[:, None]
        cycle_rotation_remaining = np.clip(
            np.maximum(self._cycle_target_angle - cycle_net_angle, 0.0)
            / self._cycle_target_angle,
            0.0,
            1.0,
        )[:, None]

        obs = np.concatenate(
            [
                dof_pos_norm,
                dof_vel_scaled,
                targets,
                last_actions,
                ball_rel_pos,
                linvel_palm,
                angvel_palm,
                ball_rot_6d,
                target_axis_palm,
                fingertip_rel_palm,
                self._contacts(env_ids).astype(dtype),
                rotation_required,
                self._palm_contacts(env_ids)[:, None],
                target_hand_norm,
                target_hand_error,
                phase_one_hot,
                phase_progress,
                remaining_angle,
                target_ball_error,
                cycle_rotation_progress,
                cycle_rotation_remaining,
            ],
            axis=1,
            dtype=dtype,
        )
        return {"obs": np.asarray(obs, dtype=dtype)}

    def update_state(self, state: NpEnvState) -> NpEnvState:
        info = state.info
        dtype = get_global_dtype()
        phases = np.asarray(info["state_cycle_phase"], dtype=np.int8)
        phases_before = phases.copy()
        phase_steps = np.asarray(info["state_cycle_phase_steps"], dtype=np.uint32)
        phase_steps += 1

        dof_pos = self.get_hand_dof_pos()
        ball_pos = self.get_ball_pos()
        ball_quat = self.get_ball_quat()
        ball_linvel = self.get_ball_linvel()
        contacts = self._contacts(self._all_env_ids)
        target_hand = self._target_hand_qpos(phases_before, dtype=dtype)
        target_ball_pos = self._target_ball_pos(phases_before, dtype=dtype)
        pose_distance = compute_pose_distance(
            dof_pos, target_hand, self._ctrl_lower, self._ctrl_upper
        )
        previous_pose_distance = np.asarray(
            info["state_cycle_prev_pose_distance"], dtype=dtype
        )
        pose_progress = compute_pose_progress(previous_pose_distance, pose_distance)
        position_error = np.linalg.norm(ball_pos - target_ball_pos, axis=1)
        ball_speed = np.linalg.norm(ball_linvel, axis=1)
        contact_count = np.sum(contacts, axis=1)

        previous_ball_quat = np.asarray(info["prev_ball_quat"], dtype=dtype)
        ball_angvel = compute_ball_angvel(ball_quat, previous_ball_quat, self._cfg.ctrl_dt)
        axis_speed = ball_angvel @ self._rotation_axis_w
        axis_delta = axis_speed * self._cfg.ctrl_dt
        edge_net_angle = np.asarray(info["state_cycle_edge_net_angle"], dtype=dtype)
        edge_net_angle += axis_delta
        required_angle = self._minimum_angles[phases_before.astype(np.intp)]

        specs = self._transition_specs
        pose_tolerance = np.asarray([spec.pose_tolerance for spec in specs], dtype=dtype)[
            phases_before
        ]
        position_tolerance = np.asarray(
            [spec.position_tolerance_m for spec in specs], dtype=dtype
        )[phases_before]
        max_ball_speed = np.asarray(
            [spec.max_ball_speed_m_s for spec in specs], dtype=dtype
        )[phases_before]
        minimum_contacts = np.asarray(
            [spec.minimum_contacts for spec in specs], dtype=np.int8
        )[phases_before]
        workspace_error = np.linalg.norm(
            ball_pos - np.asarray(info["rotation_anchor_pos"], dtype=dtype), axis=1
        )
        workspace_failure = workspace_error > self._cfg.termination_workspace_radius
        rotation_ok = rotation_condition(
            phases_before,
            edge_net_angle,
            self._minimum_angles,
        )
        palm_contact = self._palm_contacts(self._all_env_ids) > 0.5
        rotation_gates = compute_rotation_stability_gates(
            reward_cfg=self._reward_cfg,
            workspace_error=workspace_error,
            ball_speed=ball_speed,
            contact_count=contact_count,
            minimum_contacts=minimum_contacts,
            palm_contact=palm_contact,
            max_ball_speed=max_ball_speed,
        )
        raw_transition_valid = (
            (pose_distance <= pose_tolerance)
            & (position_error <= position_tolerance)
            & (ball_speed <= max_ball_speed)
            & (contact_count >= minimum_contacts)
            & rotation_ok
            & ~palm_contact
            & ~workspace_failure
        )
        advance = advance_state_cycle(
            phases_before,
            np.asarray(info["state_cycle_success_hold_steps"], dtype=np.uint32),
            raw_transition_valid,
            self._hold_steps_required,
        )
        timeout = compute_timeout_event(
            phase_steps,
            self._timeout_steps[phases_before],
            advance.transition_event,
            workspace_failure,
        )
        terminated = workspace_failure | timeout
        phase_start_pose_distance = np.asarray(
            info["state_cycle_phase_start_pose_distance"], dtype=dtype
        )
        rotation_update = update_state_cycle_rotation(
            axis_delta=axis_delta,
            cycle_net_angle=info["state_cycle_cycle_net_angle"],
            episode_net_angle=info["state_cycle_episode_net_angle"],
            phase_positive_angle=info["state_cycle_phase_positive_angle"],
            last_completed_cycle_net_angle=info[
                "state_cycle_last_completed_cycle_net_angle"
            ],
            completed_cycle_angle_sum=info[
                "state_cycle_completed_cycle_angle_sum"
            ],
            completed_cycle_count=info["state_cycle_completed_cycle_count"],
            valid_rotation_cycles_completed=info[
                "state_cycle_valid_rotation_cycles_completed"
            ],
            transition_event=advance.transition_event,
            cycle_event=advance.cycle_event,
            cycle_target_net_angle_rad=self._cycle_target_angle,
        )

        reward_terms = compute_state_cycle_reward(
            reward_cfg=self._reward_cfg,
            ctrl_dt=self._cfg.ctrl_dt,
            pose_progress=pose_progress,
            pose_distance=pose_distance,
            phase_start_pose_distance=phase_start_pose_distance,
            phase_rotation_reward_earned_before=info[
                "state_cycle_phase_rotation_reward_earned"
            ],
            episode_rotation_reward_earned_before=info[
                "state_cycle_episode_rotation_reward_earned"
            ],
            axis_delta=axis_delta,
            rotation_stability_gate=rotation_gates.stability,
            position_error=position_error,
            ball_linvel=ball_linvel,
            transition_event=advance.transition_event,
            rotation_cycle_success_event=(
                rotation_update.rotation_cycle_success_event
            ),
            invalid_rotation_cycle_event=(
                rotation_update.invalid_rotation_cycle_event
            ),
            timeout=timeout,
            workspace_failure=workspace_failure,
        )
        reward = reward_terms.total
        earned_reward_update = finalize_rotation_reward_buffers(
            phase_rotation_reward_earned_current=(
                reward_terms.phase_rotation_reward_earned_current
            ),
            episode_rotation_reward_earned_current=(
                reward_terms.episode_rotation_reward_earned_current
            ),
            transition_event=advance.transition_event,
            timeout=timeout,
            workspace_failure=workspace_failure,
        )

        cycles_completed = np.asarray(
            info["state_cycle_cycles_completed"], dtype=np.uint32
        )
        cycles_completed += advance.cycle_event.astype(np.uint32)
        phases[:] = advance.phase
        phase_steps[advance.transition_event] = 0
        edge_net_angle_for_log = edge_net_angle.copy()
        edge_net_angle[advance.transition_event] = 0.0
        edge_start_quat = np.asarray(info["state_cycle_edge_start_quat"], dtype=dtype)
        # Debug snapshot only. Success uses integrated signed deltas, not an
        # endpoint start-to-current quaternion comparison.
        edge_start_quat[advance.transition_event] = ball_quat[advance.transition_event]
        next_target_hand = self._target_hand_qpos(phases, dtype=dtype)
        next_pose_distance = compute_pose_distance(
            dof_pos, next_target_hand, self._ctrl_lower, self._ctrl_upper
        )
        previous_pose_distance[:] = np.where(
            advance.transition_event, next_pose_distance, pose_distance
        )
        phase_start_pose_distance[advance.transition_event] = next_pose_distance[
            advance.transition_event
        ]

        info["state_cycle_phase"] = phases
        info["state_cycle_phase_steps"] = phase_steps
        info["state_cycle_success_hold_steps"] = advance.hold_steps
        info["state_cycle_edge_start_quat"] = edge_start_quat
        info["state_cycle_edge_net_angle"] = edge_net_angle
        info["state_cycle_cycle_net_angle"] = rotation_update.cycle_net_angle
        info["state_cycle_episode_net_angle"] = rotation_update.episode_net_angle
        info["state_cycle_phase_positive_angle"] = (
            rotation_update.phase_positive_angle
        )
        info["state_cycle_last_completed_cycle_net_angle"] = (
            rotation_update.last_completed_cycle_net_angle
        )
        info["state_cycle_completed_cycle_angle_sum"] = (
            rotation_update.completed_cycle_angle_sum
        )
        info["state_cycle_completed_cycle_count"] = (
            rotation_update.completed_cycle_count
        )
        info["state_cycle_valid_rotation_cycles_completed"] = (
            rotation_update.valid_rotation_cycles_completed
        )
        info["state_cycle_phase_rotation_reward_earned"] = (
            earned_reward_update.phase
        )
        info["state_cycle_episode_rotation_reward_earned"] = (
            earned_reward_update.episode
        )
        info["state_cycle_prev_pose_distance"] = previous_pose_distance
        info["state_cycle_phase_start_pose_distance"] = phase_start_pose_distance
        info["state_cycle_cycles_completed"] = cycles_completed
        info["state_cycle_transition_success"] = advance.transition_event
        info["state_cycle_timeout"] = timeout
        info["state_cycle_workspace_failure"] = workspace_failure
        info["curr_dof_pos"] = dof_pos.copy()
        info["curr_ball_pos"] = ball_pos.copy()
        info["curr_ball_quat"] = ball_quat.copy()
        info["prev_dof_pos"] = dof_pos.copy()
        info["prev_ball_pos"] = ball_pos.copy()
        info["prev_ball_quat"] = ball_quat.copy()

        self._log_state_cycle_metrics(
            info=info,
            reward=reward,
            phases=phases_before,
            pose_distance=pose_distance,
            position_error=position_error,
            remaining_angle=np.where(
                required_angle > 0.0,
                np.maximum(required_angle - edge_net_angle_for_log, 0.0),
                0.0,
            ),
            edge_net_angle=edge_net_angle_for_log,
            ball_speed=ball_speed,
            contact_count=contact_count,
            axis_delta=axis_delta,
            rotation_gates=rotation_gates,
            phase_rotation_reward_earned_current=(
                reward_terms.phase_rotation_reward_earned_current
            ),
            episode_rotation_reward_earned_current=(
                reward_terms.episode_rotation_reward_earned_current
            ),
            transition_event=advance.transition_event,
            rotation_cycle_success_event=(
                rotation_update.rotation_cycle_success_event
            ),
            timeout=timeout,
            workspace_failure=workspace_failure,
            pose_ok=pose_distance <= pose_tolerance,
            position_ok=position_error <= position_tolerance,
            speed_ok=ball_speed <= max_ball_speed,
            contact_ok=contact_count >= minimum_contacts,
            rotation_ok=rotation_ok,
            no_palm_contact=~palm_contact,
            reward_terms={
                "pose_progress": reward_terms.pose_progress,
                "pose_tracking": reward_terms.pose_tracking,
                "rotation_progress": reward_terms.rotation_progress,
                "rotation_overspeed": reward_terms.rotation_overspeed,
                "reverse_rotation": reward_terms.reverse_rotation,
                "position_error": reward_terms.position_error,
                "obj_linvel": reward_terms.obj_linvel,
                "transition_event": reward_terms.transition_event,
                "cycle_event": reward_terms.cycle_event,
                "invalid_cycle": reward_terms.invalid_cycle,
                "timeout": reward_terms.timeout,
                "failure": reward_terms.failure,
                "timeout_rotation_clawback": (
                    reward_terms.timeout_rotation_clawback
                ),
                "failure_rotation_clawback": (
                    reward_terms.failure_rotation_clawback
                ),
            },
        )
        obs = self._compute_state_cycle_obs(self._all_env_ids, info)
        return state.replace(obs=obs, reward=np.asarray(reward, dtype=dtype), terminated=terminated)

    def _log_state_cycle_metrics(
        self,
        *,
        info: dict[str, Any],
        reward: np.ndarray,
        phases: np.ndarray,
        pose_distance: np.ndarray,
        position_error: np.ndarray,
        remaining_angle: np.ndarray,
        edge_net_angle: np.ndarray,
        ball_speed: np.ndarray,
        contact_count: np.ndarray,
        axis_delta: np.ndarray,
        rotation_gates: StateCycleRotationStabilityGates,
        phase_rotation_reward_earned_current: np.ndarray,
        episode_rotation_reward_earned_current: np.ndarray,
        transition_event: np.ndarray,
        rotation_cycle_success_event: np.ndarray,
        timeout: np.ndarray,
        workspace_failure: np.ndarray,
        pose_ok: np.ndarray,
        position_ok: np.ndarray,
        speed_ok: np.ndarray,
        contact_ok: np.ndarray,
        rotation_ok: np.ndarray,
        no_palm_contact: np.ndarray,
        reward_terms: dict[str, np.ndarray],
    ) -> None:
        step_count = info.get("steps", np.zeros(self._num_envs, dtype=np.uint32))
        if (
            not self._enable_reward_log
            or int(step_count[0]) % self._diagnostic_log_interval_steps != 0
        ):
            return
        log = info.get("log", {})
        log["state_cycle/phase_mean"] = float(np.mean(phases))
        for phase in StateCyclePhase:
            log[f"state_cycle/phase_{phase.name}_fraction"] = float(
                np.mean(phases == int(phase))
            )
        log["state_cycle/pose_distance_mean"] = float(np.mean(pose_distance))
        log["state_cycle/pose_distance_p50"] = float(np.median(pose_distance))
        log["state_cycle/transition_success_rate"] = float(np.mean(transition_event))
        log["state_cycle/rotation_cycle_success_rate"] = float(
            np.mean(rotation_cycle_success_event)
        )
        log["state_cycle/cycles_completed_mean"] = float(
            np.mean(info["state_cycle_cycles_completed"])
        )
        log["state_cycle/timeout_rate"] = float(np.mean(timeout))
        log["state_cycle/edge_net_angle_mean"] = float(np.mean(edge_net_angle))
        a_to_b = phases == int(StateCyclePhase.A_TO_B)
        log["state_cycle/edge_net_angle_A_TO_B_mean"] = float(
            np.mean(edge_net_angle[a_to_b]) if np.any(a_to_b) else 0.0
        )
        log["state_cycle/remaining_angle_mean"] = float(np.mean(remaining_angle))
        log["state_cycle/pose_ok_rate"] = float(np.mean(pose_ok))
        log["state_cycle/position_ok_rate"] = float(np.mean(position_ok))
        log["state_cycle/speed_ok_rate"] = float(np.mean(speed_ok))
        log["state_cycle/contact_ok_rate"] = float(np.mean(contact_ok))
        log["state_cycle/rotation_ok_rate"] = float(np.mean(rotation_ok))
        log["state_cycle/no_palm_contact_rate"] = float(np.mean(no_palm_contact))
        log["state_cycle/hold_steps_mean"] = float(
            np.mean(info["state_cycle_success_hold_steps"])
        )
        log.update(
            compute_state_cycle_diagnostic_metrics(
                phases_before=phases,
                timeout=timeout,
                pose_distance=pose_distance,
                position_error=position_error,
                ball_speed=ball_speed,
                contact_count=contact_count,
                edge_net_angle=edge_net_angle,
                remaining_angle=remaining_angle,
                hold_steps=info["state_cycle_success_hold_steps"],
                pose_ok=pose_ok,
                position_ok=position_ok,
                speed_ok=speed_ok,
                contact_ok=contact_ok,
                rotation_ok=rotation_ok,
                no_palm_contact=no_palm_contact,
                transition_event=transition_event,
                workspace_failure=workspace_failure,
            )
        )

        two_pi = 2.0 * np.pi
        episode_net_angle = np.asarray(info["state_cycle_episode_net_angle"])
        cycle_net_angle = np.asarray(info["state_cycle_cycle_net_angle"])
        last_completed_angle = np.asarray(
            info["state_cycle_last_completed_cycle_net_angle"]
        )
        completed_angle_sum = np.asarray(
            info["state_cycle_completed_cycle_angle_sum"]
        )
        completed_count = np.asarray(info["state_cycle_completed_cycle_count"])
        valid_count = np.asarray(
            info["state_cycle_valid_rotation_cycles_completed"]
        )
        total_completed = int(np.sum(completed_count))
        log["rotation/episode_net_angle_mean"] = float(np.mean(episode_net_angle))
        log["rotation/episode_net_turns_mean"] = float(
            np.mean(episode_net_angle / two_pi)
        )
        log["rotation/cycle_net_angle_mean"] = float(np.mean(cycle_net_angle))
        log["rotation/cycle_net_turns_mean"] = float(
            np.mean(cycle_net_angle / two_pi)
        )
        log["rotation/last_completed_cycle_net_angle_mean"] = float(
            np.mean(last_completed_angle)
        )
        log["rotation/completed_cycle_net_angle_mean"] = float(
            np.sum(completed_angle_sum) / max(total_completed, 1)
        )
        log["rotation/valid_rotation_cycle_rate"] = float(
            np.sum(valid_count) / max(total_completed, 1)
        )
        log["rotation/valid_rotation_cycles_completed_mean"] = float(
            np.mean(valid_count)
        )
        axis_speed_rad_s = axis_delta / self._cfg.ctrl_dt
        target_axis_delta = (
            self._reward_cfg.rotation_target_axis_speed_rad_s * self._cfg.ctrl_dt
        )
        log["rotation/axis_speed_rad_s_mean"] = float(np.mean(axis_speed_rad_s))
        log["rotation/axis_speed_rad_s_p50"] = float(np.median(axis_speed_rad_s))
        log["rotation/axis_speed_rad_s_p90"] = float(
            np.percentile(axis_speed_rad_s, 90)
        )
        log["rotation/target_axis_speed_rad_s"] = float(
            self._reward_cfg.rotation_target_axis_speed_rad_s
        )
        log["rotation/overspeed_fraction"] = float(
            np.mean(np.maximum(axis_delta, 0.0) > target_axis_delta)
        )
        log["rotation/workspace_gate_mean"] = float(
            np.mean(rotation_gates.workspace)
        )
        log["rotation/linvel_gate_mean"] = float(np.mean(rotation_gates.linvel))
        log["rotation/contact_gate_mean"] = float(np.mean(rotation_gates.contact))
        log["rotation/stability_gate_mean"] = float(
            np.mean(rotation_gates.stability)
        )
        log["rotation/phase_reward_earned_mean"] = float(
            np.mean(phase_rotation_reward_earned_current)
        )
        log["rotation/episode_reward_earned_mean"] = float(
            np.mean(episode_rotation_reward_earned_current)
        )
        for phase in StateCyclePhase:
            phase_mask = phases == int(phase)
            phase_axis_delta = axis_delta[phase_mask]
            log[f"rotation/axis_delta_{phase.name}_mean"] = float(
                np.mean(phase_axis_delta) if np.any(phase_mask) else 0.0
            )
            log[f"rotation/reverse_fraction_{phase.name}"] = float(
                np.mean(phase_axis_delta < 0.0) if np.any(phase_mask) else 0.0
            )
        for name, values in reward_terms.items():
            log[f"reward/{name}"] = float(np.mean(values))
        log["reward/total"] = float(np.mean(reward))
        log["object/position_error_m"] = float(np.mean(position_error))
        log["termination/workspace_rate"] = float(np.mean(workspace_failure))
        log["termination/timeout_rate"] = float(np.mean(timeout))
        info["log"] = log


LeapInhandBallStateCycleRotation = LeapInhandBallStateCycleRotationEnv
