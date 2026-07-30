"""LEAP ball cache rotation curriculum with active finger gaiting (release/recontact)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs.manipulation.allegro_inhand.rotation import compute_ball_angvel
from unilab.envs.manipulation.leap_inhand.finger_gaiting_rotation import (
    FingerGaitingConfig,
    LeapFingerGaitingResetProvider,
    LeapInhandBallFingerGaitingRotationCfg,
    LeapInhandBallFingerGaitingRotationEnv,
    advance_finger_gaiting,
    normalize_finger_gaiting_observation,
)
from unilab.envs.manipulation.leap_inhand.sustained_cache_rotation import (
    AllegroStyleRotationRewardConfig,
)
from unilab.envs.manipulation.leap_inhand.sustained_rotation import (
    StageSkillUpdate,
    SustainedRotationCurriculumConfig,
    compute_anchor_proximity,
)


@dataclass
class CacheGaitingRewardConfig(AllegroStyleRotationRewardConfig):
    """Reward config for V3-B cache finger-gaiting rotation task."""

    position_gate_outer_radius: float = 0.030
    failure_position_radius: float = 0.050

    scales: dict[str, float] = field(
        default_factory=lambda: {
            "rotate": 1.25,
            "obj_linvel": -0.3,
            "position_error": -1.5,
            "spin_progress": 0.0,
            "spin_continuity": -0.05,
            "retention": 0.0,
            "anchor_proximity": 0.0,
            "fingertip_support": 0.0,
            "action_rate": 0.0,
            "torque": 0.0,
            "work": 0.0,
            "failure": -1.0,
        }
    )
    stage_bonuses: list[float] = field(
        default_factory=lambda: [0.02, 0.03, 0.04, 0.05, 0.075, 0.10]
    )
    final_success_bonus: float = 0.25
    failure_penalty: float = 1.0
    positive_spin_retention_floors: list[float] = field(
        default_factory=lambda: [1.0] * 7
    )


class LeapCacheGaitingResetProvider(LeapFingerGaitingResetProvider):
    """Initialize cumulative angle and release-angle tracking in reset info."""

    def build_reset_plan(self, env: Any, env_ids: np.ndarray):
        plan = super().build_reset_plan(env, env_ids)
        num_reset = len(env_ids)
        dtype = get_global_dtype()
        updates = dict(plan.info_updates or {})
        updates.update(
            {
                "rotation_net_angle_rad": np.zeros(num_reset, dtype=dtype),
                "rotation_positive_angle_rad": np.zeros(num_reset, dtype=dtype),
                "rotation_absolute_angle_rad": np.zeros(num_reset, dtype=dtype),
            }
        )
        plan.info_updates = updates
        return plan


@registry.envcfg("LeapInhandBallCacheGaitingRotation")
@dataclass
class LeapInhandBallCacheGaitingRotationCfg(
    LeapInhandBallFingerGaitingRotationCfg
):
    """Low-speed finger gaiting curriculum with cache reset state."""

    reset_source: str = "cache"
    grasp_cache_path: str = (
        "robots/leap_hand/caches/ball_grasp_official_50k.npy"
    )
    termination_drop_distance: float = 0.05
    max_episode_seconds: float = 30.0

    finger_gaiting: FingerGaitingConfig = field(
        default_factory=lambda: FingerGaitingConfig(
            required_handoffs_by_stage=[0, 1, 1, 2, 2, 3, 4],
            minimum_contacts_by_stage=[3, 3, 3, 2, 2, 2, 2],
            minimum_other_contacts=2,
            minimum_release_steps=2,
            maximum_release_steps=10,
            handoff_cooldown_steps=4,
            minimum_speed_ratio=0.60,
            recovery_speed_ratio=0.80,
            stable_support_scale=0.0,
            release_progress_scale=0.0,
            qualified_handoff_bonus=0.05,
            minimum_handoff_angle_rad=0.03,
            release_allowed_fingers=[True, True, True, False],
        )
    )

    curriculum: SustainedRotationCurriculumConfig = field(
        default_factory=lambda: SustainedRotationCurriculumConfig(
            direct_target_mode=False,
            target_speeds=[0.00, 0.04, 0.07, 0.10, 0.15, 0.20, 0.30],
            stage_durations_seconds=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0],
            orthogonal_speed_tolerances=[0.10, 0.10, 0.08, 0.07, 0.06, 0.07, 0.10],
            energy_level_scales=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            sustain_ratio=0.80,
            gate_position_radius=0.015,
            minimum_fingertip_contacts=2,
            ema_time_constant=0.10,
        )
    )

    reward_config: CacheGaitingRewardConfig | None = field(
        default_factory=CacheGaitingRewardConfig
    )

    def validate(self) -> None:
        super().validate()
        reward_config = self.reward_config
        if reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        if isinstance(reward_config, CacheGaitingRewardConfig):
            if not (
                self.curriculum.gate_position_radius
                < reward_config.position_gate_outer_radius
                <= reward_config.failure_position_radius
                <= self.termination_drop_distance
            ):
                raise ValueError(
                    "radii must satisfy gate_position_radius < position_gate_outer_radius <= failure_position_radius <= termination_drop_distance"
                )


@registry.env("LeapInhandBallCacheGaitingRotation", sim_backend="motrix")
@registry.env("LeapInhandBallCacheGaitingRotation", sim_backend="mujoco")
class LeapInhandBallCacheGaitingRotationEnv(
    LeapInhandBallFingerGaitingRotationEnv
):
    """Learn sustained rotation with active finger gaiting transitions."""

    _cfg: LeapInhandBallCacheGaitingRotationCfg
    _reward_cfg: CacheGaitingRewardConfig
    _NUM_CACHE_GAITING_OBS = 118

    def __init__(
        self,
        cfg: LeapInhandBallCacheGaitingRotationCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> None:
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        reward_cfg = self._reward_cfg
        self._stage_bonuses = np.asarray(
            reward_cfg.stage_bonuses if reward_cfg else [],
            dtype=self._np_dtype,
        )

    def _make_domain_randomization_provider(self) -> LeapCacheGaitingResetProvider:
        return LeapCacheGaitingResetProvider()

    def _compute_terminated(self, position_error: np.ndarray) -> np.ndarray:
        """Terminate only when the ball leaves the wide workspace boundary."""
        return np.asarray(
            position_error > self._cfg.termination_drop_distance,
            dtype=bool,
        )

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": self._NUM_CACHE_GAITING_OBS}

    def _compute_sustained_obs(
        self, env_ids: np.ndarray, info: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        base_obs = super(LeapInhandBallFingerGaitingRotationEnv, self)._compute_sustained_obs(env_ids, info)["obs"]

        def rows(name: str) -> np.ndarray:
            values = np.asarray(info[name])
            return values if values.shape[0] == len(env_ids) else values[env_ids]

        levels = rows("rotation_level").astype(np.intp)
        gaiting_obs = normalize_finger_gaiting_observation(
            release_active=rows("gaiting_release_active"),
            release_steps=rows("gaiting_release_steps"),
            release_start_speed=rows("gaiting_release_start_speed"),
            cooldown_steps=rows("gaiting_cooldown_steps"),
            stage_handoffs=rows("gaiting_stage_handoffs"),
            required_handoffs=self._required_handoffs[levels],
            maximum_release_steps=self._cfg.finger_gaiting.maximum_release_steps,
            maximum_target_speed=float(self._target_speeds[-1]),
            maximum_cooldown_steps=self._cfg.finger_gaiting.handoff_cooldown_steps,
            release_start_angle=rows("gaiting_release_start_angle"),
            cumulative_angle=rows("rotation_net_angle_rad"),
            minimum_handoff_angle_rad=self._cfg.finger_gaiting.minimum_handoff_angle_rad,
        )
        return {"obs": np.concatenate([base_obs, gaiting_obs], axis=1, dtype=get_global_dtype())}

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
        release_start_angle: np.ndarray | None = None,
        cumulative_angle: np.ndarray | None = None,
    ) -> StageSkillUpdate:
        dtype = self._np_dtype
        cfg = self._cfg.finger_gaiting
        eligible = retention_ok & no_failure_signal & ~palm_contact
        required = self._required_handoffs[levels]

        start_angle = (
            release_start_angle
            if release_start_angle is not None
            else info.get("gaiting_release_start_angle", np.zeros((self._num_envs, 4), dtype=dtype))
        )
        curr_angle = (
            cumulative_angle
            if cumulative_angle is not None
            else info.get("rotation_net_angle_rad", np.zeros(self._num_envs, dtype=dtype))
        )

        stationary_allowed = np.zeros(self._num_envs, dtype=bool)
        if hasattr(self._reward_cfg, "stationary_handoff_stages"):
            for stage_idx in self._reward_cfg.stationary_handoff_stages:
                stationary_allowed |= levels == stage_idx

        gaiting_transition = advance_finger_gaiting(
            contacts=fingertip_contacts,
            previous_contacts=np.asarray(info.get("gaiting_previous_contacts", np.zeros((self._num_envs, 4), dtype=bool)), dtype=bool),
            active=np.asarray(info.get("gaiting_release_active", np.zeros((self._num_envs, 4), dtype=bool)), dtype=bool),
            release_steps=np.asarray(info.get("gaiting_release_steps", np.zeros((self._num_envs, 4), dtype=np.uint8)), dtype=np.uint8),
            release_start_speed=np.asarray(info.get("gaiting_release_start_speed", np.zeros((self._num_envs, 4), dtype=dtype)), dtype=dtype),
            cooldown_steps=np.asarray(info.get("gaiting_cooldown_steps", np.zeros(self._num_envs, dtype=np.uint8)), dtype=np.uint8),
            eligible=eligible,
            axis_speed_ema=axis_speed_ema,
            target_speed=target_speed,
            cfg=cfg,
            stationary_handoff_allowed=stationary_allowed,
            release_start_angle=start_angle,
            cumulative_angle=curr_angle,
        )

        info["gaiting_previous_contacts"] = fingertip_contacts.copy()
        info["gaiting_release_active"] = gaiting_transition.active.copy()
        info["gaiting_release_steps"] = gaiting_transition.release_steps.copy()
        info["gaiting_release_start_speed"] = gaiting_transition.release_start_speed.copy()
        info["gaiting_release_start_angle"] = gaiting_transition.release_start_angle.copy()
        info["gaiting_cooldown_steps"] = gaiting_transition.cooldown_steps.copy()

        stage_handoffs = np.asarray(info.get("gaiting_stage_handoffs", np.zeros(self._num_envs, dtype=np.uint8)), dtype=np.uint8).copy()
        total_handoffs = np.asarray(info.get("gaiting_total_handoffs", np.zeros(self._num_envs, dtype=np.uint16)), dtype=np.uint16).copy()

        useful_handoff = gaiting_transition.qualified_handoff & (stage_handoffs[:, None] < required[:, None]) & retention_ok[:, None]
        has_useful = np.any(useful_handoff, axis=1)

        stage_handoffs[has_useful] = np.minimum(stage_handoffs[has_useful] + 1, 255).astype(np.uint8)
        total_handoffs[has_useful] = np.minimum(total_handoffs[has_useful] + 1, 65535).astype(np.uint16)
        info["gaiting_stage_handoffs"] = stage_handoffs
        info["gaiting_total_handoffs"] = total_handoffs

        recovery_quality = (0.5 + 0.5 * np.clip(axis_speed_ema / np.maximum(target_speed, 1e-6), 0.0, 1.0)).astype(dtype, copy=False)
        event_reward = (cfg.qualified_handoff_bonus * has_useful.astype(dtype) * recovery_quality).astype(dtype, copy=False)

        minimum_contacts = self._minimum_stage_contacts[levels]
        validity_mask = contact_count >= minimum_contacts
        completion_ready = stage_handoffs >= required

        return StageSkillUpdate(
            validity_mask=validity_mask,
            completion_ready=completion_ready,
            dense_reward=np.zeros(self._num_envs, dtype=dtype),
            event_reward=event_reward,
            log={
                "gaiting/qualified_handoff_rate": float(np.mean(has_useful)),
                "gaiting/total_handoffs_mean": float(np.mean(total_handoffs)),
            },
        )

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
        dtype = self._np_dtype

        linear_gate = compute_anchor_proximity(
            position_error,
            self._cfg.curriculum.gate_position_radius,
            self._reward_cfg.position_gate_outer_radius,
        )
        position_gate = np.power(linear_gate, self._reward_cfg.position_gate_power).astype(dtype, copy=False)

        normalized_progress = np.zeros_like(axis_speed, dtype=dtype)
        rotating = target_speed > 1e-6
        normalized_progress[rotating] = np.clip(
            axis_speed[rotating] / target_speed[rotating],
            -1.0,
            1.0,
        )

        axis_purity = np.exp(
            -np.square(
                orthogonal_speed_ema / np.maximum(tolerance, 1e-6)
            )
        ).astype(dtype, copy=False)

        rotate_amplitude = self._reward_cfg.scales["rotate"] * target_speed
        positive_progress = np.maximum(normalized_progress, 0.0)
        positive_rotate_rate = rotate_amplitude * positive_progress * axis_purity * position_gate
        negative_rotate_rate = rotate_amplitude * np.minimum(normalized_progress, 0.0)
        rotate_rate = positive_rotate_rate + negative_rotate_rate

        stall_penalty_scale = self._reward_cfg.spin_continuity_penalty_scale
        stall_rate = np.zeros_like(axis_speed, dtype=dtype)
        stall_rate[rotating] = (
            -stall_penalty_scale
            * position_gate[rotating]
            * (1.0 - positive_progress[rotating])
        )

        position_error_scale = self._reward_cfg.scales["position_error"]
        position_rate = position_error_scale * position_error

        ball_pos = self.get_ball_pos()
        prev_ball_pos = np.asarray(info.get("prev_ball_pos", ball_pos), dtype=dtype)
        ball_linvel = (ball_pos - prev_ball_pos) / self._cfg.ctrl_dt
        obj_linvel_scale = self._reward_cfg.scales["obj_linvel"]
        linvel_rate = obj_linvel_scale * np.sum(np.abs(ball_linvel), axis=1)

        adjustment = rotate_rate + stall_rate + position_rate + linvel_rate

        log = {
            "reward/rotate": float(np.mean(rotate_rate)),
            "reward/stall": float(np.mean(stall_rate)),
            "reward/position_error": float(np.mean(position_rate)),
            "reward/obj_linvel": float(np.mean(linvel_rate)),
            "rotation/position_gate": float(np.mean(position_gate)),
            "rotation/axis_purity": float(np.mean(axis_purity)),
            "rotation/normalized_progress": float(np.mean(normalized_progress)),
            "rotation/positive_progress": float(np.mean(positive_progress)),
        }
        return adjustment.astype(dtype, copy=False), log

    def update_state(self, state: NpEnvState) -> NpEnvState:
        info = state.info
        dtype = self._np_dtype

        ball_pos = self.get_ball_pos()
        ball_quat = self.get_ball_quat()
        prev_ball_quat = np.asarray(info.get("prev_ball_quat", ball_quat), dtype=dtype)

        ball_angvel = compute_ball_angvel(ball_quat, prev_ball_quat, self._cfg.ctrl_dt)

        anchor_pos = np.asarray(info["rotation_anchor_pos"], dtype=dtype)
        position_error = np.linalg.norm(ball_pos - anchor_pos, axis=1).astype(dtype, copy=False)
        terminated = self._compute_terminated(position_error)

        levels = np.asarray(info.get("rotation_level", np.zeros(self._num_envs, dtype=np.int32)), dtype=np.intp).copy()
        levels_before = levels.copy()
        target_speed = self._target_speeds[levels_before].astype(dtype, copy=False)
        tolerance = self._orthogonal_tolerances[levels_before].astype(dtype, copy=False)
        required_contacts = self._minimum_stage_contacts[levels_before]

        axis_speed = (ball_angvel @ self._rotation_axis_w).astype(dtype, copy=False)
        orthogonal_vel = ball_angvel - axis_speed[:, None] * self._rotation_axis_w[None, :]
        orthogonal_speed = np.linalg.norm(orthogonal_vel, axis=1).astype(dtype, copy=False)

        ema_tau = float(self._cfg.curriculum.ema_time_constant)
        alpha = np.asarray(
            1.0 - np.exp(-self._cfg.ctrl_dt / max(ema_tau, 1e-6)),
            dtype=dtype,
        )
        axis_speed_ema = alpha * axis_speed + (1.0 - alpha) * np.asarray(
            info.get("axis_speed_ema", info.get("rotation_axis_speed_ema", axis_speed)), dtype=dtype
        )
        orthogonal_speed_ema = alpha * orthogonal_speed + (1.0 - alpha) * np.asarray(
            info.get("orthogonal_speed_ema", info.get("rotation_orthogonal_speed_ema", orthogonal_speed)), dtype=dtype
        )

        info["axis_speed_ema"] = axis_speed_ema
        info["rotation_axis_speed_ema"] = axis_speed_ema
        info["orthogonal_speed_ema"] = orthogonal_speed_ema
        info["rotation_orthogonal_speed_ema"] = orthogonal_speed_ema

        delta_axis_angle = axis_speed * self._cfg.ctrl_dt
        net_angle = np.asarray(info.get("rotation_net_angle_rad", np.zeros(self._num_envs, dtype=dtype)), dtype=dtype) + delta_axis_angle
        positive_angle = np.asarray(info.get("rotation_positive_angle_rad", np.zeros(self._num_envs, dtype=dtype)), dtype=dtype) + np.maximum(delta_axis_angle, 0.0)
        absolute_angle = np.asarray(info.get("rotation_absolute_angle_rad", np.zeros(self._num_envs, dtype=dtype)), dtype=dtype) + np.abs(delta_axis_angle)

        info["rotation_net_angle_rad"] = net_angle.copy()
        info["rotation_positive_angle_rad"] = positive_angle.copy()
        info["rotation_absolute_angle_rad"] = absolute_angle.copy()

        fingertip_contacts = self._contacts(self._all_env_ids)
        palm_contact_bool = self._palm_contacts(self._all_env_ids).astype(bool)
        fingertip_contact_count = np.sum(fingertip_contacts, axis=1)

        contact_steps = np.asarray(info.get("gaiting_contact_steps", np.zeros((self._num_envs, 4), dtype=np.uint32)), dtype=np.uint32)
        contact_steps += fingertip_contacts.astype(np.uint32)
        observed_steps = np.asarray(info.get("gaiting_observed_steps", np.zeros(self._num_envs, dtype=np.uint32)), dtype=np.uint32)
        observed_steps += 1
        info["gaiting_contact_steps"] = contact_steps
        info["gaiting_observed_steps"] = observed_steps

        linear_gate = compute_anchor_proximity(
            position_error,
            self._cfg.curriculum.gate_position_radius,
            self._reward_cfg.position_gate_outer_radius,
        )
        retention_ok = position_error <= self._cfg.curriculum.gate_position_radius
        support_ok = fingertip_contact_count >= required_contacts
        orthogonal_ok = orthogonal_speed_ema <= tolerance
        hold_stage = (levels_before == 0) & (not self._cfg.curriculum.direct_target_mode)
        stationary_target = target_speed <= 1e-6
        speed_ok = (
            hold_stage
            | stationary_target
            | (axis_speed_ema >= self._cfg.curriculum.sustain_ratio * target_speed)
        )
        base_stage_valid = support_ok & retention_ok & ~terminated & ~palm_contact_bool & speed_ok & orthogonal_ok

        stage_update = self._update_stage_skill(
            info=info,
            fingertip_contacts=fingertip_contacts,
            palm_contact=palm_contact_bool,
            contact_count=fingertip_contact_count,
            levels=levels_before,
            target_speed=target_speed,
            axis_speed_ema=axis_speed_ema,
            retention_ok=retention_ok,
            no_failure_signal=~terminated,
            base_stage_valid=base_stage_valid,
            release_start_angle=info.get("gaiting_release_start_angle", None),
            cumulative_angle=net_angle,
        )

        stage_valid = base_stage_valid & stage_update.validity_mask
        stage_steps = np.asarray(info.get("rotation_stage_steps", np.zeros(self._num_envs, dtype=np.uint32)), dtype=np.uint32)
        next_stage_steps = np.where(stage_valid, stage_steps + 1, 0).astype(np.uint32)
        stage_steps[:] = next_stage_steps

        stage_complete = (
            next_stage_steps >= self._stage_steps_required[levels_before]
        ) & stage_update.completion_ready

        promote = stage_complete & (levels_before < len(self._target_speeds) - 1)
        levels[promote] += 1
        next_stage_steps[promote] = 0

        self._on_stage_promotion(info, promote)

        info["rotation_level"] = levels
        info["rotation_stage_steps"] = next_stage_steps

        stage_duration_progress = np.clip(
            next_stage_steps / self._stage_steps_required[levels_before], 0.0, 1.0
        ).astype(dtype)

        reward_adjustment, reward_adjustment_log = self._compute_reward_adjustment(
            info=info,
            fingertip_contacts=fingertip_contacts.astype(bool),
            palm_contact=palm_contact_bool,
            contact_count=fingertip_contact_count,
            target_speed=target_speed,
            tolerance=tolerance,
            axis_speed=axis_speed,
            axis_speed_ema=axis_speed_ema,
            orthogonal_speed_ema=orthogonal_speed_ema,
            position_error=position_error,
            anchor_proximity=linear_gate,
            retention_ok=retention_ok,
            no_failure_signal=~terminated,
            stage_valid=stage_valid,
            stage_duration_progress=stage_duration_progress,
        )

        dense_reward = reward_adjustment + stage_update.dense_reward
        reward = np.asarray(dense_reward * self._cfg.ctrl_dt, dtype=dtype)
        reward += stage_update.event_reward

        stage_bonus = (
            promote.astype(dtype)
            * self._stage_bonuses[
                np.clip(levels_before, 0, len(self._stage_bonuses) - 1)
            ]
        )
        reward += stage_bonus

        rotation_success = np.asarray(
            info.get("rotation_success", np.zeros(self._num_envs, dtype=bool)),
            dtype=bool,
        )
        final_event = (
            stage_complete
            & (levels_before == len(self._target_speeds) - 1)
            & ~rotation_success
        )
        final_success_bonus = (
            final_event.astype(dtype) * self._reward_cfg.final_success_bonus
        )
        reward += final_success_bonus
        info["rotation_success"] = rotation_success | final_event

        failure_penalty = self._reward_cfg.failure_penalty
        failure_reward = (-failure_penalty * terminated.astype(dtype)).astype(dtype, copy=False)
        reward += failure_reward

        info["curr_ball_pos"] = ball_pos.copy()
        info["curr_ball_quat"] = ball_quat.copy()
        info["prev_ball_pos"] = ball_pos.copy()
        info["prev_ball_quat"] = ball_quat.copy()
        info["axis_speed_ema"] = axis_speed_ema.copy()
        info["orthogonal_speed_ema"] = orthogonal_speed_ema.copy()

        obs = self._compute_sustained_obs(self._all_env_ids, info)
        step_count = np.asarray(info.get("steps", np.zeros(self._num_envs, dtype=np.uint32)))

        if self._enable_reward_log and int(step_count[0]) % 4 == 0:
            efficiency = np.maximum(net_angle, 0.0) / np.maximum(absolute_angle, 1e-6)
            turns = net_angle / (2.0 * np.pi)
            rotating = target_speed > 1e-6
            above_target_frac = float(np.mean(axis_speed[rotating] >= target_speed[rotating])) if np.any(rotating) else 0.0

            contact_steps = np.asarray(info.get("gaiting_contact_steps", np.zeros((self._num_envs, 4), dtype=np.uint32)))
            observed_steps = np.maximum(np.asarray(info.get("gaiting_observed_steps", np.ones(self._num_envs, dtype=np.uint32))), 1)
            duty_per_finger = np.mean(contact_steps / observed_steps[:, None], axis=0)

            total_handoffs = np.asarray(info.get("gaiting_total_handoffs", np.zeros(self._num_envs, dtype=np.uint16)))

            log = {
                "reward/dense": float(np.mean(dense_reward * self._cfg.ctrl_dt)),
                "reward/event": float(np.mean(stage_update.event_reward)),
                "reward/stage_bonus": float(np.mean(stage_bonus)),
                "reward/final_success": float(np.mean(final_success_bonus)),
                "reward/failure": float(np.mean(failure_reward)),
                "reward/total": float(np.mean(reward)),
                "rotation/level_mean": float(np.mean(levels)),
                "rotation/level_max": int(np.max(levels)),
                "rotation/axis_speed_rad_s": float(np.mean(axis_speed)),
                "rotation/axis_speed_p10": float(np.percentile(axis_speed, 10)),
                "rotation/axis_speed_p25": float(np.percentile(axis_speed, 25)),
                "rotation/axis_speed_p50": float(np.percentile(axis_speed, 50)),
                "rotation/axis_speed_p75": float(np.percentile(axis_speed, 75)),
                "rotation/axis_speed_p90": float(np.percentile(axis_speed, 90)),
                "rotation/net_angle_rad_mean": float(np.mean(net_angle)),
                "rotation/net_angle_rad_median": float(np.median(net_angle)),
                "rotation/net_turns_mean": float(np.mean(turns)),
                "rotation/net_turns_median": float(np.median(turns)),
                "rotation/positive_angle_rad_mean": float(np.mean(positive_angle)),
                "rotation/absolute_angle_rad_mean": float(np.mean(absolute_angle)),
                "rotation/efficiency_mean": float(np.mean(efficiency)),
                "rotation/efficiency_median": float(np.median(efficiency)),
                "rotation/above_0_05_fraction": float(np.mean(axis_speed >= 0.05)),
                "rotation/above_0_10_fraction": float(np.mean(axis_speed >= 0.10)),
                "rotation/above_target_fraction": above_target_frac,
                "gaiting/qualified_handoff_rate": stage_update.log.get("gaiting/qualified_handoff_rate", 0.0),
                "gaiting/total_handoffs_mean": float(np.mean(total_handoffs)),
                "gaiting/total_handoffs_median": float(np.median(total_handoffs)),
                "gaiting/contact_duty_index": float(duty_per_finger[0]),
                "gaiting/contact_duty_middle": float(duty_per_finger[1]),
                "gaiting/contact_duty_ring": float(duty_per_finger[2]),
                "gaiting/contact_duty_thumb": float(duty_per_finger[3]),
                "object/position_error_m": float(np.mean(position_error)),
                "termination/task_rate": float(np.mean(terminated)),
            }
            log.update(reward_adjustment_log)
            info["log"] = log

        return state.replace(obs=obs, reward=reward, terminated=terminated)


__all__ = [
    "CacheGaitingRewardConfig",
    "LeapCacheGaitingResetProvider",
    "LeapInhandBallCacheGaitingRotationCfg",
    "LeapInhandBallCacheGaitingRotationEnv",
]
