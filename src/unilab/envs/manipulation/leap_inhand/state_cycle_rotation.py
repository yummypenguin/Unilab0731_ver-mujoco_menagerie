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
    ready_to_a: dict[str, float | int | str] = field(
        default_factory=lambda: _transition_defaults("A", 1.5, 0.0)
    )
    a_to_b: dict[str, float | int | str] = field(
        default_factory=lambda: _transition_defaults("B", 2.0, 0.03)
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
    pose_tracking_scale: float = 0.25
    pose_sigma: float = 0.08
    rotation_progress_scale: float = 3.0
    reverse_rotation_scale: float = 4.0
    position_error_scale: float = 6.0
    object_linvel_scale: float = 0.3
    transition_success_bonus: float = 0.25
    cycle_success_bonus: float = 0.75
    timeout_penalty: float = 0.25
    failure_penalty: float = 1.0


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
    reverse_rotation: np.ndarray
    position_error: np.ndarray
    obj_linvel: np.ndarray
    transition_event: np.ndarray
    cycle_event: np.ndarray
    timeout: np.ndarray
    failure: np.ndarray
    total: np.ndarray


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


def compute_state_cycle_reward(
    *,
    reward_cfg: StateCycleRewardConfig,
    ctrl_dt: float,
    phases: np.ndarray,
    pose_progress: np.ndarray,
    pose_distance: np.ndarray,
    phase_start_pose_distance: np.ndarray,
    axis_delta: np.ndarray,
    edge_net_angle_before: np.ndarray,
    required_angle: np.ndarray,
    position_error: np.ndarray,
    ball_linvel: np.ndarray,
    transition_event: np.ndarray,
    cycle_event: np.ndarray,
    timeout: np.ndarray,
    workspace_failure: np.ndarray,
) -> StateCycleRewardTerms:
    """Compute bounded per-step shaping and mutually exclusive failure events."""
    pose_progress_reward = reward_cfg.pose_progress_scale * np.asarray(pose_progress)
    pose_tracking_rate = reward_cfg.pose_tracking_scale * (
        np.exp(
            -np.square(
                np.asarray(pose_distance) / max(reward_cfg.pose_sigma, 1e-8)
            )
        )
        - 1.0
    )

    positive_axis_delta = np.maximum(np.asarray(axis_delta), 0.0)
    reverse_axis_delta = np.maximum(-np.asarray(axis_delta), 0.0)
    remaining_before = np.where(
        np.asarray(required_angle) > 0.0,
        np.maximum(
            np.asarray(required_angle) - np.asarray(edge_net_angle_before),
            0.0,
        ),
        0.0,
    )
    useful_positive_delta = np.minimum(positive_axis_delta, remaining_before)
    rotation_edge = np.asarray(phases) == int(StateCyclePhase.A_TO_B)
    rotation_progress_reward = (
        reward_cfg.rotation_progress_scale * useful_positive_delta * rotation_edge
    )
    reverse_rotation_reward = (
        -reward_cfg.reverse_rotation_scale * reverse_axis_delta * rotation_edge
    )
    position_error_rate = -reward_cfg.position_error_scale * np.asarray(position_error)
    object_linvel_rate = -reward_cfg.object_linvel_scale * np.sum(
        np.abs(ball_linvel), axis=1
    )
    pose_tracking_reward = pose_tracking_rate * ctrl_dt
    position_error_reward = position_error_rate * ctrl_dt
    object_linvel_reward = object_linvel_rate * ctrl_dt

    transition_event_reward = (
        reward_cfg.transition_success_bonus * np.asarray(transition_event)
    )
    cycle_event_reward = reward_cfg.cycle_success_bonus * np.asarray(cycle_event)
    timeout_event = np.asarray(timeout) & ~np.asarray(workspace_failure)
    phase_pose_progress = np.maximum(
        np.asarray(phase_start_pose_distance) - np.asarray(pose_distance),
        0.0,
    )
    timeout_reward = -(
        reward_cfg.timeout_penalty
        + reward_cfg.pose_progress_scale * phase_pose_progress
    ) * timeout_event
    failure_reward = -reward_cfg.failure_penalty * np.asarray(workspace_failure)
    total = (
        pose_progress_reward
        + pose_tracking_reward
        + rotation_progress_reward
        + reverse_rotation_reward
        + position_error_reward
        + object_linvel_reward
        + transition_event_reward
        + cycle_event_reward
        + timeout_reward
        + failure_reward
    )
    return StateCycleRewardTerms(
        pose_progress=pose_progress_reward,
        pose_tracking=pose_tracking_reward,
        rotation_progress=rotation_progress_reward,
        reverse_rotation=reverse_rotation_reward,
        position_error=position_error_reward,
        obj_linvel=object_linvel_reward,
        transition_event=transition_event_reward,
        cycle_event=cycle_event_reward,
        timeout=timeout_reward,
        failure=failure_reward,
        total=total,
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
            model_file=str(ASSETS_ROOT_PATH / "robots" / "leap_hand" / "scene_ball.xml")
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
    _NUM_STATE_CYCLE_OBS = 140
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
        edge_net_angle_before = edge_net_angle.copy()
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

        reward_terms = compute_state_cycle_reward(
            reward_cfg=self._reward_cfg,
            ctrl_dt=self._cfg.ctrl_dt,
            phases=phases_before,
            pose_progress=pose_progress,
            pose_distance=pose_distance,
            phase_start_pose_distance=phase_start_pose_distance,
            axis_delta=axis_delta,
            edge_net_angle_before=edge_net_angle_before,
            required_angle=required_angle,
            position_error=position_error,
            ball_linvel=ball_linvel,
            transition_event=advance.transition_event,
            cycle_event=advance.cycle_event,
            timeout=timeout,
            workspace_failure=workspace_failure,
        )
        reward = reward_terms.total

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
            transition_event=advance.transition_event,
            timeout=timeout,
            workspace_failure=workspace_failure,
            reward_terms={
                "pose_progress": reward_terms.pose_progress,
                "pose_tracking": reward_terms.pose_tracking,
                "rotation_progress": reward_terms.rotation_progress,
                "reverse_rotation": reward_terms.reverse_rotation,
                "position_error": reward_terms.position_error,
                "obj_linvel": reward_terms.obj_linvel,
                "transition_event": reward_terms.transition_event,
                "cycle_event": reward_terms.cycle_event,
                "timeout": reward_terms.timeout,
                "failure": reward_terms.failure,
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
        transition_event: np.ndarray,
        timeout: np.ndarray,
        workspace_failure: np.ndarray,
        reward_terms: dict[str, np.ndarray],
    ) -> None:
        step_count = info.get("steps", np.zeros(self._num_envs, dtype=np.uint32))
        if not self._enable_reward_log or int(step_count[0]) % 4 != 0:
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
        for name, values in reward_terms.items():
            log[f"reward/{name}"] = float(np.mean(values))
        log["reward/total"] = float(np.mean(reward))
        log["object/position_error_m"] = float(np.mean(position_error))
        log["termination/workspace_rate"] = float(np.mean(workspace_failure))
        log["termination/timeout_rate"] = float(np.mean(timeout))
        info["log"] = log


LeapInhandBallStateCycleRotation = LeapInhandBallStateCycleRotationEnv
