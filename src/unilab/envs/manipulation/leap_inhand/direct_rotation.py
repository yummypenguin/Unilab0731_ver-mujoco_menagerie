"""Direct fixed-speed LEAP ball rotation without curriculum or handoff gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.dtype_config import get_global_dtype

from .sustained_cache_rotation import (
    LeapInhandBallSustainedCacheRotationCfg,
    LeapInhandBallSustainedCacheRotationEnv,
)
from .sustained_rotation import (
    LeapSustainedRotationResetProvider,
    SustainedRotationCurriculumConfig,
    SustainedRotationRewardConfig,
)


@dataclass
class DirectRotationRewardConfig(SustainedRotationRewardConfig):
    """Reward coefficients for direct positive rotation acquisition."""

    scales: dict[str, float] = field(
        default_factory=lambda: {
            "stable_rotation": 6.0,
            "rotation_streak": 0.5,
            "stall": -1.0,
            "reverse": -4.0,
            "thumb_contact": 0.1,
            "fingertip_support": 0.15,
            "object_center": -2.5,
            "center_recovery": 1.0,
            "object_linear_velocity": -0.4,
            "palm_contact": -1.0,
            "orthogonal_speed": -0.25,
            "action_rate": -0.008,
            "torque": -0.001,
            "work": -0.0001,
            "failure": -5.0,
        }
    )
    spin_progress_scale: float = 0.0
    retention_scale: float = 0.0
    anchor_proximity_scale: float = 0.0
    fingertip_support_scale: float = 0.0
    action_rate_scale: float = 0.008
    torque_scale: float = 0.001
    work_scale: float = 0.0001
    stage_bonuses: list[float] = field(default_factory=list)
    positive_spin_retention_floors: list[float] = field(default_factory=lambda: [1.0])
    final_success_bonus: float = 0.0
    stable_rotation_scale: float = 6.0
    rotation_streak_scale: float = 0.5
    stall_scale: float = 1.0
    reverse_scale: float = 4.0
    thumb_contact_scale: float = 0.1
    support_scale: float = 0.15
    object_center_scale: float = 2.5
    center_recovery_scale: float = 1.0
    object_linear_velocity_scale: float = 0.4
    palm_contact_scale: float = 1.0
    orthogonal_speed_scale: float = 0.25
    minimum_positive_speed: float = 0.05
    stall_grace_seconds: float = 1.0


def compute_direct_rotation_reward(
    *,
    axis_speed_ema: np.ndarray,
    target_speed: np.ndarray,
    orthogonal_speed_ema: np.ndarray,
    orthogonal_tolerance: np.ndarray,
    position_error: np.ndarray,
    previous_position_error: np.ndarray,
    failure_position_radius: float,
    fingertip_contacts: np.ndarray,
    palm_contact: np.ndarray,
    no_failure_signal: np.ndarray,
    stage_valid: np.ndarray,
    stage_duration_progress: np.ndarray,
    object_linear_velocity: np.ndarray,
    elapsed_seconds: np.ndarray,
    cfg: DirectRotationRewardConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute external-inspired direct rotation terms in reward-per-second units."""
    positive_progress = np.clip(axis_speed_ema / np.maximum(target_speed, 1e-6), 0.0, 1.0)
    support_quality = np.clip(np.sum(fingertip_contacts, axis=1) / 2.0, 0.0, 1.0)
    support_quality *= ~palm_contact
    stable_gate = no_failure_signal & (support_quality > 0.0)
    stable_rotation = cfg.stable_rotation_scale * positive_progress * support_quality * stable_gate
    rotation_streak = (
        cfg.rotation_streak_scale * stage_duration_progress * np.asarray(stage_valid, dtype=float)
    )
    stall = -cfg.stall_scale * (
        (elapsed_seconds >= cfg.stall_grace_seconds) & (axis_speed_ema < cfg.minimum_positive_speed)
    )
    reverse = -cfg.reverse_scale * np.clip(
        -axis_speed_ema / max(cfg.minimum_positive_speed, 1e-6),
        0.0,
        1.0,
    )
    thumb_contact = cfg.thumb_contact_scale * fingertip_contacts[:, 3]
    support = cfg.support_scale * support_quality
    center_ratio = np.clip(
        position_error / max(failure_position_radius, 1e-6),
        0.0,
        1.0,
    )
    object_center = -cfg.object_center_scale * center_ratio
    center_recovery = cfg.center_recovery_scale * np.clip(
        (previous_position_error - position_error) / max(failure_position_radius, 1e-6),
        0.0,
        1.0,
    )
    object_linear_velocity_term = -cfg.object_linear_velocity_scale * np.sum(
        np.square(object_linear_velocity),
        axis=1,
    )
    palm = -cfg.palm_contact_scale * palm_contact
    orthogonal = -cfg.orthogonal_speed_scale * np.clip(
        orthogonal_speed_ema / np.maximum(orthogonal_tolerance, 1e-6),
        0.0,
        1.0,
    )
    terms = {
        "stable_rotation": stable_rotation,
        "rotation_streak": rotation_streak,
        "stall": stall,
        "reverse": reverse,
        "thumb_contact": thumb_contact,
        "fingertip_support": support,
        "object_center": object_center,
        "center_recovery": center_recovery,
        "object_linear_velocity": object_linear_velocity_term,
        "palm_contact": palm,
        "orthogonal_speed": orthogonal,
    }
    reward = np.sum(np.stack(tuple(terms.values()), axis=0), axis=0)
    return reward, terms


class LeapDirectRotationResetProvider(LeapSustainedRotationResetProvider):
    """Initialize diagnostic-only natural contact-transition state."""

    def build_reset_plan(self, env: Any, env_ids: np.ndarray):
        plan = super().build_reset_plan(env, env_ids)
        num_reset = len(env_ids)
        dtype = get_global_dtype()
        updates = dict(plan.info_updates or {})
        updates.update(
            {
                "direct_previous_contacts": np.zeros((num_reset, 4), dtype=bool),
                "direct_natural_handoffs": np.zeros(num_reset, dtype=np.uint16),
                "direct_no_transition_steps": np.zeros(num_reset, dtype=np.uint32),
                "direct_longest_no_transition_steps": np.zeros(num_reset, dtype=np.uint32),
                "direct_previous_position_error": np.zeros(num_reset, dtype=dtype),
            }
        )
        plan.info_updates = updates
        return plan


@registry.envcfg("LeapInhandBallDirectRotation")
@dataclass
class LeapInhandBallDirectRotationCfg(LeapInhandBallSustainedCacheRotationCfg):
    """Acquire fixed-speed positive rotation directly from cache grasps."""

    reward_config: DirectRotationRewardConfig | None = None
    curriculum: SustainedRotationCurriculumConfig = field(
        default_factory=lambda: SustainedRotationCurriculumConfig(
            target_speeds=[0.30],
            stage_durations_seconds=[2.0],
            orthogonal_speed_tolerances=[0.30],
            energy_level_scales=[1.0],
            sustain_ratio=0.80,
            gate_position_radius=0.015,
            minimum_fingertip_contacts=2,
            ema_time_constant=0.1,
            direct_target_mode=True,
        )
    )

    def validate(self) -> None:
        super().validate()
        reward = self.reward_config
        if reward is None:
            raise ValueError("reward_config must be provided")
        if reward.minimum_positive_speed <= 0.0:
            raise ValueError("minimum_positive_speed must be positive")
        if reward.minimum_positive_speed >= self.curriculum.target_speeds[0]:
            raise ValueError("minimum_positive_speed must be below target speed")
        if reward.stall_grace_seconds < 0.0:
            raise ValueError("stall_grace_seconds must be non-negative")


@registry.env("LeapInhandBallDirectRotation", sim_backend="motrix")
@registry.env("LeapInhandBallDirectRotation", sim_backend="mujoco")
class LeapInhandBallDirectRotationEnv(LeapInhandBallSustainedCacheRotationEnv):
    """Learn positive rotation without curriculum promotion or handoff requirements."""

    _cfg: LeapInhandBallDirectRotationCfg
    _reward_cfg: DirectRotationRewardConfig

    def _make_domain_randomization_provider(self):
        return LeapDirectRotationResetProvider()

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
        del contact_count, axis_speed, anchor_proximity, retention_ok
        dtype = get_global_dtype()
        previous_position_error = np.asarray(
            info["direct_previous_position_error"],
            dtype=dtype,
        )
        elapsed_seconds = np.asarray(info["steps"], dtype=dtype) * self._cfg.ctrl_dt
        reward, terms = compute_direct_rotation_reward(
            axis_speed_ema=axis_speed_ema,
            target_speed=target_speed,
            orthogonal_speed_ema=orthogonal_speed_ema,
            orthogonal_tolerance=tolerance,
            position_error=position_error,
            previous_position_error=previous_position_error,
            failure_position_radius=self._reward_cfg.failure_position_radius,
            fingertip_contacts=fingertip_contacts,
            palm_contact=palm_contact,
            no_failure_signal=no_failure_signal,
            stage_valid=stage_valid,
            stage_duration_progress=stage_duration_progress,
            object_linear_velocity=self.get_ball_linvel(),
            elapsed_seconds=elapsed_seconds,
            cfg=self._reward_cfg,
        )

        previous_contacts = np.asarray(info["direct_previous_contacts"], dtype=bool)
        released = np.any(previous_contacts & ~fingertip_contacts, axis=1)
        acquired = np.any(~previous_contacts & fingertip_contacts, axis=1)
        contact_transition = np.any(previous_contacts != fingertip_contacts, axis=1)
        natural_handoff = released & acquired & (np.sum(fingertip_contacts, axis=1) >= 2)
        total_handoffs = np.asarray(info["direct_natural_handoffs"], dtype=np.uint16)
        total_handoffs[:] = np.minimum(
            total_handoffs.astype(np.uint32) + natural_handoff.astype(np.uint32),
            65535,
        ).astype(np.uint16)
        no_transition_steps = np.asarray(info["direct_no_transition_steps"], dtype=np.uint32)
        no_transition_steps[:] = np.where(contact_transition, 0, no_transition_steps + 1)
        longest_no_transition = np.asarray(
            info["direct_longest_no_transition_steps"],
            dtype=np.uint32,
        )
        longest_no_transition[:] = np.maximum(longest_no_transition, no_transition_steps)
        info["direct_previous_contacts"] = fingertip_contacts.copy()
        info["direct_natural_handoffs"] = total_handoffs
        info["direct_no_transition_steps"] = no_transition_steps
        info["direct_longest_no_transition_steps"] = longest_no_transition
        info["direct_previous_position_error"] = position_error.copy()

        log = {f"reward/direct_{name}": float(np.mean(value)) for name, value in terms.items()}
        log.update(
            {
                "gaiting/contact_transition_rate": float(np.mean(contact_transition)),
                "gaiting/natural_handoff_rate": float(np.mean(natural_handoff)),
                "gaiting/total_natural_handoffs": float(np.mean(total_handoffs)),
                "gaiting/longest_no_transition_seconds": float(
                    np.mean(longest_no_transition) * self._cfg.ctrl_dt
                ),
            }
        )
        return np.asarray(reward, dtype=dtype), log


__all__ = [
    "DirectRotationRewardConfig",
    "LeapInhandBallDirectRotationCfg",
    "LeapInhandBallDirectRotationEnv",
    "compute_direct_rotation_reward",
]
