"""Sustained LEAP ball rotation with staged, failure-resistant learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dr import DomainRandomizationProvider
from unilab.dtype_config import get_global_dtype
from unilab.envs.manipulation.allegro_inhand.rotation import compute_pd_torques

from .ball_rotation import LeapBallRotationResetProvider, LeapInhandBallRotationCfg
from .rotation_v2 import LeapInhandBallRotationV2Env


@dataclass
class SustainedRotationRewardConfig:
    # Mirrors the effective coefficients for generic config tooling; runtime uses
    # the typed fields below so each term retains its explicit unit/semantics.
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "spin_progress": 1.5,
            "spin_continuity": 0.0,
            "retention": 0.5,
            "anchor_proximity": 0.0,
            "fingertip_support": 0.25,
            "action_rate": -0.001,
            "torque": -0.005,
            "work": -0.05,
            "failure": -5.0,
        }
    )
    spin_progress_scale: float = 1.5
    spin_continuity_penalty_scale: float = 0.0
    direct_spin_reward: bool = False
    retention_scale: float = 0.5
    anchor_proximity_scale: float = 0.0
    fingertip_support_scale: float = 0.25
    action_rate_scale: float = 0.001
    torque_scale: float = 0.005
    work_scale: float = 0.05
    stage_bonuses: list[float] = field(
        default_factory=lambda: [0.10, 0.175, 0.25, 0.30, 0.35, 0.50, 1.0]
    )
    positive_spin_retention_floors: list[float] = field(default_factory=lambda: [0.5] * 8)
    final_success_bonus: float = 5.0
    failure_penalty: float = 5.0
    retention_sigma: float = 0.015
    failure_position_radius: float = 0.030
    reset_z_threshold: float = 0.66797318983078
    failure_debounce_steps: int = 3
    speed_tracking_tolerance_ratio: float = 0.25


@dataclass
class SustainedRotationCurriculumConfig:
    target_speeds: list[float] = field(
        default_factory=lambda: [0.0, 0.04, 0.07, 0.085, 0.10, 0.16, 0.25, 0.50]
    )
    stage_durations_seconds: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.5, 1.0, 2.0, 2.0, 4.0, 10.0]
    )
    orthogonal_speed_tolerances: list[float] = field(
        default_factory=lambda: [0.10, 0.08, 0.065, 0.058, 0.05, 0.06, 0.075, 0.10]
    )
    energy_level_scales: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 1.0]
    )
    sustain_ratio: float = 0.80
    gate_position_radius: float = 0.015
    minimum_fingertip_contacts: int = 2
    ema_time_constant: float = 0.1
    direct_target_mode: bool = False


@dataclass
class StageSkillUpdate:
    """Optional task-specific additions to sustained stage progression."""

    validity_mask: np.ndarray
    completion_ready: np.ndarray
    dense_reward: np.ndarray
    event_reward: np.ndarray
    log: dict[str, float] = field(default_factory=dict)


def compute_sustained_spin_terms(
    ball_angvel: np.ndarray,
    rotation_axis: np.ndarray,
    target_speed: np.ndarray,
    orthogonal_tolerance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute signed progress while keeping off-axis exploration observable."""
    axis_speed = np.sum(ball_angvel * rotation_axis, axis=1)
    orthogonal = ball_angvel - axis_speed[:, None] * rotation_axis
    orthogonal_speed = np.linalg.norm(orthogonal, axis=1)
    progress = np.zeros_like(axis_speed)
    rotating = target_speed > 1e-6
    progress[rotating] = np.clip(axis_speed[rotating] / target_speed[rotating], -1.0, 1.0)
    axis_purity = np.exp(-np.square(orthogonal_speed / np.maximum(orthogonal_tolerance, 1e-6)))
    visible_progress = progress * (0.5 + 0.5 * axis_purity)
    return axis_speed, orthogonal_speed, progress, visible_progress, axis_purity


def compute_speed_tracking_quality(
    axis_speed_ema: np.ndarray,
    target_speed: np.ndarray,
    tolerance_ratio: float,
) -> np.ndarray:
    """Score how closely the EMA follows the active positive target speed."""
    quality = np.zeros_like(axis_speed_ema)
    rotating = target_speed > 1e-6
    tolerance = np.maximum(tolerance_ratio * target_speed[rotating], 1e-6)
    quality[rotating] = np.exp(
        -np.square((axis_speed_ema[rotating] - target_speed[rotating]) / tolerance)
    )
    return quality


def compute_spin_continuity_penalty(
    axis_speed_ema: np.ndarray,
    target_speed: np.ndarray,
    penalty_scale: float,
) -> np.ndarray:
    """Penalize sustained target-speed shortfall using filtered axis speed."""
    penalty = np.zeros_like(axis_speed_ema)
    rotating = target_speed > 1e-6
    positive_progress = np.clip(
        axis_speed_ema[rotating] / target_speed[rotating],
        0.0,
        1.0,
    )
    penalty[rotating] = -penalty_scale * (1.0 - positive_progress)
    return penalty


def compute_anchor_proximity(
    position_error: np.ndarray,
    gate_position_radius: float,
    failure_position_radius: float,
) -> np.ndarray:
    """Provide a non-vanishing inward gradient across the retention grey zone."""
    grey_zone_width = max(failure_position_radius - gate_position_radius, 1e-6)
    return np.clip(
        (failure_position_radius - position_error) / grey_zone_width,
        0.0,
        1.0,
    )


def compute_reset_relative_drop(
    ball_pos: np.ndarray,
    anchor_pos: np.ndarray,
    termination_drop_distance: float,
) -> np.ndarray:
    """Return whether the ball fell below its sampled reset anchor."""
    vertical_drop = anchor_pos[:, 2] - ball_pos[:, 2]
    return np.asarray(vertical_drop >= termination_drop_distance, dtype=bool)


def compute_rotation_duration_valid(
    stage_valid: np.ndarray,
    target_speed: np.ndarray,
) -> np.ndarray:
    """Exclude stationary curriculum stages from rotation-duration metrics."""
    return np.asarray(stage_valid, dtype=bool) & (np.asarray(target_speed) > 1e-6)


def compute_stage_task_reward(
    *,
    hold_stage: np.ndarray,
    stage_valid: np.ndarray,
    axis_progress: np.ndarray,
    visible_progress: np.ndarray,
    retention: np.ndarray,
    fingertip_support: np.ndarray,
    speed_tracking_quality: np.ndarray,
    stage_duration_progress: np.ndarray,
    positive_spin_retention_floor: np.ndarray,
    spin_progress_scale: float,
    retention_scale: float,
    fingertip_support_scale: float,
    direct_spin_reward: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return hold reward plus either direct or conditioned rotation reward."""
    stability = retention_scale * retention + fingertip_support_scale * fingertip_support
    hold_reward = np.where(hold_stage & stage_valid, stability, 0.0)
    if direct_spin_reward:
        # Apply the same quadratic shaping in both directions so near-zero
        # exploration is unbiased. Since visible_progress is axis_progress
        # times the purity weight, this preserves the sign and applies purity
        # exactly once: sign(axis_progress) * axis_progress**2 * purity.
        direct_progress = np.abs(axis_progress) * visible_progress
        spin_reward = np.where(hold_stage, 0.0, spin_progress_scale * direct_progress)
        rotation_stability = np.zeros_like(spin_reward)
        return (
            hold_reward + spin_reward,
            hold_reward,
            spin_reward,
            rotation_stability,
        )

    retention_weight = (
        positive_spin_retention_floor + (1.0 - positive_spin_retention_floor) * retention
    )
    sustained_weight = (0.5 + 0.5 * speed_tracking_quality) * (0.5 + 0.5 * stage_duration_progress)
    retention_conditioned_progress = np.where(
        visible_progress > 0.0,
        visible_progress * retention_weight * sustained_weight,
        visible_progress,
    )
    spin_reward = np.where(hold_stage, 0.0, spin_progress_scale * retention_conditioned_progress)
    rotation_stability = np.where(
        hold_stage,
        0.0,
        np.maximum(axis_progress, 0.0) * stability * sustained_weight,
    )
    return (
        hold_reward + spin_reward + rotation_stability,
        hold_reward,
        spin_reward,
        rotation_stability,
    )


class LeapSustainedRotationResetProvider(LeapBallRotationResetProvider):
    """Reset from the configured source and initialize staged task state."""

    def build_reset_plan(self, env: Any, env_ids: np.ndarray):
        plan = super().build_reset_plan(env, env_ids)
        num_reset = len(env_ids)
        dtype = get_global_dtype()
        updates = dict(plan.info_updates or {})
        updates.pop("obs_lag_history", None)
        updates.update(
            {
                "rotation_anchor_pos": np.asarray(
                    plan.qpos[:, env._NUM_HAND_DOF : env._NUM_HAND_DOF + 3], dtype=dtype
                ).copy(),
                "rotation_level": np.zeros(num_reset, dtype=np.int8),
                "rotation_stage_steps": np.zeros(num_reset, dtype=np.uint32),
                "rotation_axis_speed_ema": np.zeros(num_reset, dtype=dtype),
                "rotation_orthogonal_speed_ema": np.zeros(num_reset, dtype=dtype),
                "rotation_pure_steps": np.zeros(num_reset, dtype=np.uint32),
                "rotation_success": np.zeros(num_reset, dtype=bool),
                "failure_drop_steps": np.zeros(num_reset, dtype=np.uint8),
                "failure_palm_steps": np.zeros(num_reset, dtype=np.uint8),
                "failure_workspace_steps": np.zeros(num_reset, dtype=np.uint8),
            }
        )
        plan.info_updates = updates
        return plan

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        return cast(dict[str, np.ndarray], env._compute_sustained_obs(env_ids, info_updates))


@registry.envcfg("LeapInhandBallSustainedRotation")
@dataclass
class LeapInhandBallSustainedRotationCfg(LeapInhandBallRotationCfg):
    reward_config: SustainedRotationRewardConfig | None = None
    curriculum: SustainedRotationCurriculumConfig = field(
        default_factory=SustainedRotationCurriculumConfig
    )
    max_episode_seconds: float = 25.0
    joint_velocity_scale: float = 0.20

    def validate(self) -> None:
        super().validate()
        if self.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        curriculum = self.curriculum
        count = len(curriculum.target_speeds)
        if curriculum.direct_target_mode:
            if count != 1 or curriculum.target_speeds[0] <= 0.0:
                raise ValueError("direct target mode requires exactly one positive target speed")
        else:
            if count < 2 or curriculum.target_speeds[0] != 0.0:
                raise ValueError("curriculum must define hold plus at least one rotation stage")
            if any(
                right <= left
                for left, right in zip(curriculum.target_speeds[1:], curriculum.target_speeds[2:])
            ):
                raise ValueError("positive target speeds must be strictly increasing")
        for values, name in (
            (curriculum.stage_durations_seconds, "stage_durations_seconds"),
            (curriculum.orthogonal_speed_tolerances, "orthogonal_speed_tolerances"),
            (curriculum.energy_level_scales, "energy_level_scales"),
        ):
            if len(values) != count:
                raise ValueError(f"{name} must match target_speeds")
        if len(self.reward_config.stage_bonuses) != count - 1:
            raise ValueError("stage_bonuses must match the number of stage promotions")
        if len(self.reward_config.positive_spin_retention_floors) != count:
            raise ValueError("positive_spin_retention_floors must match target_speeds")
        if any(
            value < 0.0 or value > 1.0
            for value in self.reward_config.positive_spin_retention_floors
        ):
            raise ValueError("positive_spin_retention_floors must be in [0, 1]")
        if any(value <= 0.0 for value in curriculum.stage_durations_seconds):
            raise ValueError("stage durations must be positive")
        if sum(curriculum.stage_durations_seconds) >= self.max_episode_seconds:
            raise ValueError("curriculum stages must leave time within the episode")
        if not 0.0 < curriculum.sustain_ratio <= 1.0:
            raise ValueError("sustain_ratio must be in (0, 1]")
        if curriculum.minimum_fingertip_contacts not in range(1, 5):
            raise ValueError("minimum_fingertip_contacts must be between 1 and 4")
        if self.reward_config.failure_debounce_steps <= 0:
            raise ValueError("failure_debounce_steps must be positive")
        if self.reward_config.speed_tracking_tolerance_ratio <= 0.0:
            raise ValueError("speed_tracking_tolerance_ratio must be positive")
        if self.reward_config.anchor_proximity_scale < 0.0:
            raise ValueError("anchor_proximity_scale must be non-negative")
        if self.reward_config.failure_position_radius <= curriculum.gate_position_radius:
            raise ValueError("failure_position_radius must exceed curriculum gate_position_radius")


@registry.env("LeapInhandBallSustainedRotation", sim_backend="motrix")
@registry.env("LeapInhandBallSustainedRotation", sim_backend="mujoco")
class LeapInhandBallSustainedRotationEnv(LeapInhandBallRotationV2Env):
    """Rotate around world +Z while retaining fingertip support off the palm."""

    _cfg: LeapInhandBallSustainedRotationCfg
    _reward_cfg: SustainedRotationRewardConfig
    _NUM_SUSTAINED_OBS = 100
    _PALM_CONTACT_SENSOR_NAME = "leap_palm_contact"

    def __init__(
        self,
        cfg: LeapInhandBallSustainedRotationCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> None:
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        curriculum = cfg.curriculum
        self._stage_steps_required = np.maximum(
            np.rint(np.asarray(curriculum.stage_durations_seconds) / cfg.ctrl_dt).astype(np.uint32),
            1,
        )
        self._orthogonal_tolerances = np.asarray(
            curriculum.orthogonal_speed_tolerances, dtype=self._np_dtype
        )
        self._positive_spin_retention_floors = np.asarray(
            cfg.reward_config.positive_spin_retention_floors,
            dtype=self._np_dtype,
        )

    def _make_domain_randomization_provider(self) -> DomainRandomizationProvider:
        return LeapSustainedRotationResetProvider()

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": self._NUM_SUSTAINED_OBS}

    def _palm_contacts(self, env_ids: np.ndarray) -> np.ndarray:
        contacts = self._backend.get_sensor_data_batch((self._PALM_CONTACT_SENSOR_NAME,))
        return np.asarray(contacts[env_ids, 0] > 0.5, dtype=self._np_dtype)

    def _compute_sustained_obs(
        self, env_ids: np.ndarray, info: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        base = self._compute_v2_obs(env_ids, info)["obs"]
        palm_contact = self._palm_contacts(env_ids)[:, None]
        return {"obs": np.concatenate([base, palm_contact], axis=1, dtype=get_global_dtype())}

    @staticmethod
    def _advance_failure_counter(counter: np.ndarray, condition: np.ndarray) -> np.ndarray:
        return np.where(condition, np.minimum(counter + 1, 255), 0).astype(np.uint8)

    def _compute_raw_drop(self, ball_pos: np.ndarray, anchor_pos: np.ndarray) -> np.ndarray:
        del anchor_pos
        return np.asarray(ball_pos[:, 2] < self._reward_cfg.reset_z_threshold, dtype=bool)

    def _update_stage_skill(
        self,
        *,
        info: dict[str, Any],
        fingertip_contacts: np.ndarray,
        palm_contact: np.ndarray,
        contact_count: np.ndarray,
        levels: np.ndarray,
        target_speed: np.ndarray,
        axis_speed_ema: np.ndarray,
        retention_ok: np.ndarray,
        no_failure_signal: np.ndarray,
        base_stage_valid: np.ndarray,
    ) -> StageSkillUpdate:
        """Return no-op skill terms unless a specialized task overrides them."""
        del (
            info,
            fingertip_contacts,
            palm_contact,
            contact_count,
            levels,
            target_speed,
            axis_speed_ema,
            retention_ok,
            no_failure_signal,
            base_stage_valid,
        )
        return StageSkillUpdate(
            validity_mask=np.ones(self._num_envs, dtype=bool),
            completion_ready=np.ones(self._num_envs, dtype=bool),
            dense_reward=np.zeros(self._num_envs, dtype=self._np_dtype),
            event_reward=np.zeros(self._num_envs, dtype=self._np_dtype),
        )

    def _on_stage_promotion(self, info: dict[str, Any], promoted: np.ndarray) -> None:
        """Allow specialized tasks to reset stage-local state after promotion."""
        del info, promoted

    def _compute_reward_adjustment(
        self,
        *,
        info: dict[str, Any],
        fingertip_contacts: np.ndarray,
        palm_contact: np.ndarray,
        contact_count: np.ndarray,
        target_speed: np.ndarray,
        tolerance: np.ndarray,
        axis_speed: np.ndarray,
        axis_speed_ema: np.ndarray,
        orthogonal_speed_ema: np.ndarray,
        position_error: np.ndarray,
        anchor_proximity: np.ndarray,
        retention_ok: np.ndarray,
        no_failure_signal: np.ndarray,
        stage_valid: np.ndarray,
        stage_duration_progress: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Allow a specialized task to add dense reward without replacing lifecycle logic."""
        del (
            info,
            fingertip_contacts,
            palm_contact,
            contact_count,
            target_speed,
            tolerance,
            axis_speed,
            axis_speed_ema,
            orthogonal_speed_ema,
            position_error,
            anchor_proximity,
            retention_ok,
            no_failure_signal,
            stage_valid,
            stage_duration_progress,
        )
        return np.zeros(self._num_envs, dtype=self._np_dtype), {}

    def update_state(self, state: NpEnvState) -> NpEnvState:
        info = state.info
        dtype = get_global_dtype()
        dof_pos = self.get_hand_dof_pos()
        dof_vel = self.get_hand_dof_vel()
        ball_pos = self.get_ball_pos()
        ball_quat = self.get_ball_quat()
        ball_angvel = self.get_ball_angvel()
        fingertip_contacts = self._contacts(self._all_env_ids)
        palm_contact = self._palm_contacts(self._all_env_ids).astype(bool)

        levels = np.asarray(info["rotation_level"], dtype=np.int8)
        levels_before = levels.copy()
        target_speed = self._target_speeds[levels_before]
        tolerance = self._orthogonal_tolerances[levels_before]
        rotation_axis = np.broadcast_to(self._rotation_axis_w, (self._num_envs, 3))
        axis_speed, orthogonal_speed, axis_progress, visible_progress, axis_purity = (
            compute_sustained_spin_terms(ball_angvel, rotation_axis, target_speed, tolerance)
        )

        anchor_pos = np.asarray(info["rotation_anchor_pos"], dtype=dtype)
        position_error = np.linalg.norm(ball_pos - anchor_pos, axis=1)
        retention = np.exp(-np.square(position_error / max(self._reward_cfg.retention_sigma, 1e-6)))
        anchor_proximity = compute_anchor_proximity(
            position_error,
            self._cfg.curriculum.gate_position_radius,
            self._reward_cfg.failure_position_radius,
        )
        contact_count = np.sum(fingertip_contacts, axis=1)
        fingertip_support = np.clip(
            contact_count / self._cfg.curriculum.minimum_fingertip_contacts, 0.0, 1.0
        ) * (~palm_contact)
        action_rate = np.sum(
            np.square(np.asarray(info["current_actions"]) - np.asarray(info["last_actions"])),
            axis=1,
        )
        torques = compute_pd_torques(
            targets=np.asarray(info["prev_ctrl"]),
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            kp=self._cfg.control_config.kp,
            kd=self._cfg.control_config.kd,
        )
        torque_cost = np.sum(np.square(torques), axis=1)
        work_cost = np.square(np.sum(torques * dof_vel, axis=1))
        energy_scale = self._energy_level_scales[levels_before]

        ema_tau = max(self._cfg.curriculum.ema_time_constant, self._cfg.ctrl_dt)
        ema_alpha = 1.0 - np.exp(-self._cfg.ctrl_dt / ema_tau)
        axis_ema = np.asarray(info["rotation_axis_speed_ema"], dtype=dtype)
        orthogonal_ema = np.asarray(info["rotation_orthogonal_speed_ema"], dtype=dtype)
        axis_ema += ema_alpha * (axis_speed - axis_ema)
        orthogonal_ema += ema_alpha * (orthogonal_speed - orthogonal_ema)

        raw_drop = self._compute_raw_drop(ball_pos, anchor_pos)
        raw_workspace = position_error > self._reward_cfg.failure_position_radius
        drop_steps = self._advance_failure_counter(info["failure_drop_steps"], raw_drop)
        palm_steps = self._advance_failure_counter(info["failure_palm_steps"], palm_contact)
        workspace_steps = self._advance_failure_counter(
            info["failure_workspace_steps"], raw_workspace
        )
        debounce = self._reward_cfg.failure_debounce_steps
        drop_failure = drop_steps >= debounce
        palm_failure = palm_steps >= debounce
        workspace_failure = workspace_steps >= debounce
        terminated = drop_failure | palm_failure | workspace_failure

        support_ok = contact_count >= self._cfg.curriculum.minimum_fingertip_contacts
        retention_ok = position_error <= self._cfg.curriculum.gate_position_radius
        no_failure_signal = ~raw_drop & ~raw_workspace & ~palm_contact
        hold_stage = (levels_before == 0) & (not self._cfg.curriculum.direct_target_mode)
        stationary_target = target_speed <= 1e-6
        speed_ok = (
            hold_stage
            | stationary_target
            | (axis_ema >= self._cfg.curriculum.sustain_ratio * target_speed)
        )
        orthogonal_ok = orthogonal_ema <= tolerance
        stage_valid = support_ok & retention_ok & no_failure_signal & speed_ok & orthogonal_ok
        skill_update = self._update_stage_skill(
            info=info,
            fingertip_contacts=fingertip_contacts.astype(bool),
            palm_contact=palm_contact,
            contact_count=contact_count,
            levels=levels_before,
            target_speed=target_speed,
            axis_speed_ema=axis_ema,
            retention_ok=retention_ok,
            no_failure_signal=no_failure_signal,
            base_stage_valid=stage_valid,
        )
        stage_valid &= skill_update.validity_mask
        stage_steps = np.asarray(info["rotation_stage_steps"], dtype=np.uint32)
        next_stage_steps = np.where(stage_valid, stage_steps + 1, 0).astype(np.uint32)
        stage_duration_progress = np.clip(
            next_stage_steps / self._stage_steps_required[levels_before], 0.0, 1.0
        ).astype(dtype)
        speed_tracking_quality = compute_speed_tracking_quality(
            axis_ema,
            target_speed,
            self._reward_cfg.speed_tracking_tolerance_ratio,
        )
        spin_continuity_penalty = compute_spin_continuity_penalty(
            axis_ema,
            target_speed,
            self._reward_cfg.spin_continuity_penalty_scale,
        )
        positive_spin_retention_floor = self._positive_spin_retention_floors[levels_before]
        positive_spin_retention_weight = (
            positive_spin_retention_floor + (1.0 - positive_spin_retention_floor) * retention
        )
        task_reward, hold_reward, spin_reward, rotation_stability_reward = (
            compute_stage_task_reward(
                hold_stage=hold_stage,
                stage_valid=stage_valid,
                axis_progress=axis_progress,
                visible_progress=visible_progress,
                retention=retention,
                fingertip_support=fingertip_support,
                speed_tracking_quality=speed_tracking_quality,
                stage_duration_progress=stage_duration_progress,
                positive_spin_retention_floor=positive_spin_retention_floor,
                spin_progress_scale=self._reward_cfg.spin_progress_scale,
                retention_scale=self._reward_cfg.retention_scale,
                fingertip_support_scale=self._reward_cfg.fingertip_support_scale,
                direct_spin_reward=self._reward_cfg.direct_spin_reward,
            )
        )
        reward_adjustment, reward_adjustment_log = self._compute_reward_adjustment(
            info=info,
            fingertip_contacts=fingertip_contacts.astype(bool),
            palm_contact=palm_contact,
            contact_count=contact_count,
            target_speed=target_speed,
            tolerance=tolerance,
            axis_speed=axis_speed,
            axis_speed_ema=axis_ema,
            orthogonal_speed_ema=orthogonal_ema,
            position_error=position_error,
            anchor_proximity=anchor_proximity,
            retention_ok=retention_ok,
            no_failure_signal=no_failure_signal,
            stage_valid=stage_valid,
            stage_duration_progress=stage_duration_progress,
        )
        dense = (
            task_reward
            + spin_continuity_penalty
            + self._reward_cfg.anchor_proximity_scale * anchor_proximity
            + skill_update.dense_reward
            + reward_adjustment
            - self._reward_cfg.action_rate_scale * action_rate
            - energy_scale * self._reward_cfg.torque_scale * torque_cost
            - energy_scale * self._reward_cfg.work_scale * work_cost
        )
        stage_steps[:] = next_stage_steps
        stage_complete = (
            stage_steps >= self._stage_steps_required[levels_before]
        ) & skill_update.completion_ready
        promote = stage_complete & (levels_before < len(self._target_speeds) - 1)
        levels[promote] += 1
        stage_steps[promote] = 0
        self._on_stage_promotion(info, promote)
        final_event = (
            stage_complete
            & (levels_before == len(self._target_speeds) - 1)
            & ~np.asarray(info["rotation_success"], dtype=bool)
        )
        success = np.asarray(info["rotation_success"], dtype=bool)
        success |= final_event

        pure_rotation = compute_rotation_duration_valid(stage_valid, target_speed)
        pure_steps = np.asarray(info["rotation_pure_steps"], dtype=np.uint32)
        pure_steps[:] = np.where(pure_rotation, pure_steps + 1, 0)
        stage_bonus_values = np.asarray(self._reward_cfg.stage_bonuses, dtype=dtype)
        stage_bonus = np.zeros(self._num_envs, dtype=dtype)
        promoted_rows = np.flatnonzero(promote)
        if promoted_rows.size:
            stage_bonus[promoted_rows] = stage_bonus_values[levels[promoted_rows] - 1]

        reward = dense * self._cfg.ctrl_dt
        reward += skill_update.event_reward
        reward += stage_bonus
        reward += self._reward_cfg.final_success_bonus * final_event
        reward -= self._reward_cfg.failure_penalty * terminated

        info["rotation_level"] = levels
        info["rotation_stage_steps"] = stage_steps
        info["rotation_axis_speed_ema"] = axis_ema
        info["rotation_orthogonal_speed_ema"] = orthogonal_ema
        info["rotation_pure_steps"] = pure_steps
        info["rotation_success"] = success
        info["failure_drop_steps"] = drop_steps
        info["failure_palm_steps"] = palm_steps
        info["failure_workspace_steps"] = workspace_steps
        info["curr_dof_pos"] = dof_pos.copy()
        info["curr_ball_pos"] = ball_pos.copy()
        info["curr_ball_quat"] = ball_quat.copy()

        step_count = info.get("steps", np.zeros(self._num_envs, dtype=np.uint32))
        if self._enable_reward_log and int(step_count[0]) % 4 == 0:
            elapsed = np.asarray(step_count, dtype=dtype) * self._cfg.ctrl_dt
            consecutive_valid_seconds = np.where(
                target_speed > 1e-6, next_stage_steps * self._cfg.ctrl_dt, 0.0
            )
            log = {
                "reward/hold": float(np.mean(hold_reward)),
                "reward/spin_progress": float(np.mean(spin_reward)),
                "reward/spin_continuity": float(np.mean(spin_continuity_penalty)),
                "reward/rotation_stability": float(np.mean(rotation_stability_reward)),
                "reward/anchor_proximity": float(
                    np.mean(self._reward_cfg.anchor_proximity_scale * anchor_proximity)
                ),
                "reward/action_rate": float(
                    np.mean(-self._reward_cfg.action_rate_scale * action_rate)
                ),
                "reward/energy": float(
                    np.mean(
                        -energy_scale
                        * (
                            self._reward_cfg.torque_scale * torque_cost
                            + self._reward_cfg.work_scale * work_cost
                        )
                    )
                ),
                "reward/failure": float(np.mean(-self._reward_cfg.failure_penalty * terminated)),
                "reward/total": float(np.mean(reward)),
                "rotation/target_speed_rad_s": float(np.mean(target_speed)),
                "rotation/axis_speed_rad_s": float(np.mean(axis_speed)),
                "rotation/orthogonal_speed_rad_s": float(np.mean(orthogonal_speed)),
                "rotation/axis_speed_ema": float(np.mean(axis_ema)),
                "rotation/orthogonal_speed_ema": float(np.mean(orthogonal_ema)),
                "rotation/axis_progress": float(np.mean(axis_progress)),
                "rotation/visible_progress": float(np.mean(visible_progress)),
                "rotation/axis_purity": float(np.mean(axis_purity)),
                "rotation/speed_tracking_quality": float(np.mean(speed_tracking_quality)),
                "rotation/stage_duration_progress": float(
                    np.mean(np.where(hold_stage, 0.0, stage_duration_progress))
                ),
                "rotation/positive_spin_retention_floor": float(
                    np.mean(positive_spin_retention_floor)
                ),
                "rotation/positive_spin_retention_weight": float(
                    np.mean(positive_spin_retention_weight)
                ),
                "rotation/consecutive_valid_seconds": float(np.mean(consecutive_valid_seconds)),
                "retention/position_error_m": float(np.mean(position_error)),
                "retention/anchor_proximity": float(np.mean(anchor_proximity)),
                "retention/fingertip_contact_count": float(np.mean(contact_count)),
                "retention/palm_contact_rate": float(np.mean(palm_contact)),
                "failure/drop_rate": float(np.mean(drop_failure)),
                "failure/palm_rate": float(np.mean(palm_failure)),
                "failure/workspace_rate": float(np.mean(workspace_failure)),
                "failure/termination_rate": float(np.mean(terminated)),
                "failure/raw_drop_rate": float(np.mean(raw_drop)),
                "failure/raw_palm_rate": float(np.mean(palm_contact)),
                "failure/raw_workspace_rate": float(np.mean(raw_workspace)),
                "curriculum/level": float(np.mean(levels)),
                "curriculum/stage_valid_fraction": float(np.mean(stage_valid)),
                "curriculum/promotion_rate": float(np.mean(promote)),
                "success/sustained_2s_fraction": float(
                    np.mean(pure_steps * self._cfg.ctrl_dt >= 2.0)
                ),
                "success/sustained_5s_fraction": float(
                    np.mean(pure_steps * self._cfg.ctrl_dt >= 5.0)
                ),
                "success/sustained_10s_fraction": float(
                    np.mean(pure_steps * self._cfg.ctrl_dt >= 10.0)
                ),
                "success/final_event_rate": float(np.mean(final_event)),
                "success/final_success_fraction": float(np.mean(success)),
                "episode/survival_seconds": float(np.mean(elapsed)),
            }
            for level_index in range(len(self._target_speeds)):
                level_mask = levels_before == level_index
                level_count = int(np.count_nonzero(level_mask))
                log[f"curriculum/level_{level_index}_fraction"] = float(
                    np.mean(levels == level_index)
                )
                log[f"diagnostic/level_{level_index}_sample_fraction"] = float(np.mean(level_mask))
                level_metrics = {
                    "axis_speed_ema": axis_ema,
                    "orthogonal_speed_ema": orthogonal_ema,
                    "support_ok_fraction": support_ok,
                    "retention_ok_fraction": retention_ok,
                    "no_failure_signal_fraction": no_failure_signal,
                    "speed_ok_fraction": speed_ok,
                    "orthogonal_ok_fraction": orthogonal_ok,
                    "base_valid_fraction": (
                        support_ok & retention_ok & no_failure_signal & speed_ok & orthogonal_ok
                    ),
                    "completion_ready_fraction": skill_update.completion_ready,
                    "stage_valid_fraction": stage_valid,
                }
                for metric_name, metric_values in level_metrics.items():
                    value = (
                        float(np.mean(np.asarray(metric_values)[level_mask]))
                        if level_count
                        else 0.0
                    )
                    log[f"diagnostic/level_{level_index}_{metric_name}"] = value
            log.update(skill_update.log)
            log.update(reward_adjustment_log)
            info["log"] = log

        obs = self._compute_sustained_obs(self._all_env_ids, info)
        return state.replace(
            obs=obs,
            reward=np.asarray(reward, dtype=dtype),
            terminated=np.asarray(terminated, dtype=bool),
        )


__all__ = [
    "LeapInhandBallSustainedRotationCfg",
    "LeapInhandBallSustainedRotationEnv",
    "SustainedRotationCurriculumConfig",
    "SustainedRotationRewardConfig",
    "StageSkillUpdate",
    "compute_anchor_proximity",
    "compute_reset_relative_drop",
    "compute_rotation_duration_valid",
    "compute_spin_continuity_penalty",
    "compute_speed_tracking_quality",
    "compute_stage_task_reward",
    "compute_sustained_spin_terms",
]
