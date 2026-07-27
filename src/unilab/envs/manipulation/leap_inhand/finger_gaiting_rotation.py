"""LEAP ball rotation curriculum that requires qualified finger handoffs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.dtype_config import get_global_dtype

from .sustained_rotation import (
    LeapInhandBallSustainedRotationCfg,
    LeapInhandBallSustainedRotationEnv,
    LeapSustainedRotationResetProvider,
    StageSkillUpdate,
    compute_reset_relative_drop,
    compute_rotation_duration_valid,
)


@dataclass
class FingerGaitingConfig:
    """Contact-handoff gates and rewards for sustained finger gaiting."""

    required_handoffs_by_stage: list[int] = field(default_factory=lambda: [0, 1, 1, 2, 2, 3, 4, 6])
    minimum_contacts_by_stage: list[int] = field(default_factory=lambda: [3, 2, 2, 2, 2, 2, 2, 2])
    minimum_other_contacts: int = 2
    minimum_release_steps: int = 2
    maximum_release_steps: int = 10
    handoff_cooldown_steps: int = 4
    minimum_speed_ratio: float = 0.60
    recovery_speed_ratio: float = 0.80
    stable_support_scale: float = 0.20
    release_progress_scale: float = 0.05
    qualified_handoff_bonus: float = 0.25
    minimum_handoff_angle_rad: float = 0.0
    release_allowed_fingers: list[bool] = field(
        default_factory=lambda: [
            True,
            True,
            True,
            True,
        ]
    )


@dataclass
class FingerGaitingTransition:
    active: np.ndarray
    release_steps: np.ndarray
    release_start_speed: np.ndarray
    cooldown_steps: np.ndarray
    qualified_handoff: np.ndarray
    release_progress: np.ndarray
    release_start_angle: np.ndarray


def normalize_finger_gaiting_observation(
    *,
    release_active: np.ndarray,
    release_steps: np.ndarray,
    release_start_speed: np.ndarray,
    cooldown_steps: np.ndarray,
    stage_handoffs: np.ndarray,
    required_handoffs: np.ndarray,
    maximum_release_steps: int,
    maximum_target_speed: float,
    maximum_cooldown_steps: int,
    release_start_angle: np.ndarray | None = None,
    cumulative_angle: np.ndarray | None = None,
    minimum_handoff_angle_rad: float = 0.0,
) -> np.ndarray:
    """Normalize the complete Markov state used by the handoff reward gate."""
    dtype = get_global_dtype()
    active = np.asarray(release_active, dtype=dtype)
    release_progress = np.clip(
        np.asarray(release_steps, dtype=dtype) / max(maximum_release_steps, 1),
        0.0,
        1.0,
    )
    start_speed = np.clip(
        np.asarray(release_start_speed, dtype=dtype) / max(maximum_target_speed, 1e-6),
        0.0,
        1.0,
    )
    cooldown = np.clip(
        np.asarray(cooldown_steps, dtype=dtype) / max(maximum_cooldown_steps, 1),
        0.0,
        1.0,
    )[:, None]
    required = np.asarray(required_handoffs, dtype=dtype)
    handoff_progress = np.zeros_like(required)
    has_requirement = required > 0.0
    handoff_progress[has_requirement] = np.clip(
        np.asarray(stage_handoffs, dtype=dtype)[has_requirement]
        / required[has_requirement],
        0.0,
        1.0,
    )

    obs_blocks = [active, release_progress, start_speed, cooldown, handoff_progress[:, None]]
    if release_start_angle is not None and cumulative_angle is not None:
        start_angle = np.asarray(release_start_angle, dtype=dtype)
        cum_angle = np.asarray(cumulative_angle, dtype=dtype)[:, None]
        denom = max(minimum_handoff_angle_rad, 1e-6)
        release_angle_progress = np.where(
            active > 0,
            np.clip((cum_angle - start_angle) / denom, 0.0, 1.0),
            0.0,
        )
        obs_blocks.append(release_angle_progress)

    return np.concatenate(
        obs_blocks,
        axis=1,
        dtype=dtype,
    )


def advance_finger_gaiting(
    *,
    contacts: np.ndarray,
    previous_contacts: np.ndarray,
    active: np.ndarray,
    release_steps: np.ndarray,
    release_start_speed: np.ndarray,
    cooldown_steps: np.ndarray,
    eligible: np.ndarray,
    axis_speed_ema: np.ndarray,
    target_speed: np.ndarray,
    cfg: FingerGaitingConfig,
    stationary_handoff_allowed: np.ndarray | None = None,
    release_start_angle: np.ndarray | None = None,
    cumulative_angle: np.ndarray | None = None,
) -> FingerGaitingTransition:
    """Advance debounced release/recontact state and emit at most one handoff."""
    contacts = np.asarray(contacts, dtype=bool)
    previous_contacts = np.asarray(previous_contacts, dtype=bool)
    active = np.asarray(active, dtype=bool).copy()
    release_steps = np.asarray(release_steps, dtype=np.uint8).copy()
    release_start_speed = np.asarray(release_start_speed).copy()
    cooldown_steps = np.maximum(
        np.asarray(cooldown_steps, dtype=np.uint8).astype(np.int16) - 1, 0
    ).astype(np.uint8)

    dtype = get_global_dtype()
    if release_start_angle is None:
        release_start_angle = np.zeros(contacts.shape, dtype=dtype)
    else:
        release_start_angle = np.asarray(release_start_angle, dtype=dtype).copy()

    release_allowed_mask = np.asarray(
        getattr(cfg, "release_allowed_fingers", [True, True, True, True]),
        dtype=bool,
    )
    minimum_handoff_angle_rad = getattr(cfg, "minimum_handoff_angle_rad", 0.0)

    contact_count = np.sum(contacts, axis=1)
    other_contacts = contact_count[:, None] - contacts.astype(np.int16)
    support_ok = other_contacts >= cfg.minimum_other_contacts
    rotating = target_speed > 1e-6
    if stationary_handoff_allowed is None:
        stationary_handoff_allowed = np.zeros_like(rotating)
    else:
        stationary_handoff_allowed = np.asarray(stationary_handoff_allowed, dtype=bool)
    speed_floor = cfg.minimum_speed_ratio * target_speed
    speed_ok = axis_speed_ema >= speed_floor
    speed_ok = ~rotating | speed_ok
    event_eligible = eligible & (rotating | stationary_handoff_allowed) & speed_ok

    active_before = active.copy()
    steps_before = release_steps.copy()
    start_speed_before = release_start_speed.copy()
    recontact = active_before & contacts
    recovery_ok = ~rotating[:, None] | (
        axis_speed_ema[:, None] >= cfg.recovery_speed_ratio * start_speed_before
    )

    if cumulative_angle is not None:
        cum_angle = np.asarray(cumulative_angle, dtype=dtype)[:, None]
        handoff_angle = cum_angle - release_start_angle
        angle_ok = handoff_angle >= minimum_handoff_angle_rad
    else:
        angle_ok = np.ones(contacts.shape, dtype=bool)

    qualified = (
        recontact
        & support_ok
        & event_eligible[:, None]
        & (cooldown_steps[:, None] == 0)
        & (steps_before >= cfg.minimum_release_steps)
        & (steps_before <= cfg.maximum_release_steps)
        & recovery_ok
        & angle_ok
        & release_allowed_mask[None, :]
    )
    has_qualified = np.any(qualified, axis=1)
    first_finger = np.argmax(qualified, axis=1)
    qualified &= np.arange(contacts.shape[1])[None, :] == first_finger[:, None]
    qualified &= has_qualified[:, None]

    active[recontact] = False
    release_steps[recontact] = 0
    release_start_speed[recontact] = 0.0
    release_start_angle[recontact] = 0.0

    continuing = active_before & ~contacts
    valid_release = continuing & support_ok & eligible[:, None]
    incremented = np.minimum(release_steps.astype(np.uint16) + 1, 255).astype(np.uint8)
    release_steps[valid_release] = incremented[valid_release]
    expired = release_steps > cfg.maximum_release_steps
    cancelled = continuing & (~support_ok | ~eligible[:, None] | expired)
    active[cancelled] = False
    release_steps[cancelled] = 0
    release_start_speed[cancelled] = 0.0
    release_start_angle[cancelled] = 0.0

    start = (
        previous_contacts
        & ~contacts
        & support_ok
        & event_eligible[:, None]
        & (cooldown_steps[:, None] == 0)
        & ~active
        & release_allowed_mask[None, :]
    )
    active[start] = True
    release_steps[start] = 1
    release_start_speed[start] = np.broadcast_to(axis_speed_ema[:, None], contacts.shape)[start]
    if cumulative_angle is not None:
        cum_angle = np.asarray(cumulative_angle, dtype=dtype)[:, None]
        release_start_angle[start] = np.broadcast_to(cum_angle, contacts.shape)[start]

    cooldown_steps[has_qualified] = cfg.handoff_cooldown_steps
    progress = np.max(
        np.where(
            active,
            np.clip(release_steps / max(cfg.minimum_release_steps, 1), 0.0, 1.0),
            0.0,
        ),
        axis=1,
    )
    return FingerGaitingTransition(
        active=active,
        release_steps=release_steps,
        release_start_speed=release_start_speed,
        cooldown_steps=cooldown_steps,
        qualified_handoff=qualified,
        release_progress=progress,
        release_start_angle=release_start_angle,
    )


class LeapFingerGaitingResetProvider(LeapSustainedRotationResetProvider):
    """Initialize contact-transition state without changing cache reset rows."""

    def build_reset_plan(self, env: Any, env_ids: np.ndarray):
        plan = super().build_reset_plan(env, env_ids)
        num_reset = len(env_ids)
        dtype = get_global_dtype()
        updates = dict(plan.info_updates or {})
        updates.update(
            {
                "gaiting_previous_contacts": np.zeros((num_reset, 4), dtype=bool),
                "gaiting_release_active": np.zeros((num_reset, 4), dtype=bool),
                "gaiting_release_steps": np.zeros((num_reset, 4), dtype=np.uint8),
                "gaiting_release_start_speed": np.zeros((num_reset, 4), dtype=dtype),
                "gaiting_release_start_angle": np.zeros((num_reset, 4), dtype=dtype),
                "gaiting_cooldown_steps": np.zeros(num_reset, dtype=np.uint8),
                "gaiting_stage_handoffs": np.zeros(num_reset, dtype=np.uint8),
                "gaiting_total_handoffs": np.zeros(num_reset, dtype=np.uint16),
                "gaiting_contact_steps": np.zeros((num_reset, 4), dtype=np.uint32),
                "gaiting_observed_steps": np.zeros(num_reset, dtype=np.uint32),
                "gaiting_no_transition_steps": np.zeros(num_reset, dtype=np.uint32),
                "gaiting_longest_no_transition_steps": np.zeros(num_reset, dtype=np.uint32),
                "gaiting_rotation_steps": np.zeros(num_reset, dtype=np.uint32),
                "gaiting_longest_rotation_steps": np.zeros(num_reset, dtype=np.uint32),
            }
        )
        plan.info_updates = updates
        return plan


@registry.envcfg("LeapInhandBallFingerGaitingRotation")
@dataclass
class LeapInhandBallFingerGaitingRotationCfg(LeapInhandBallSustainedRotationCfg):
    """Require safe release/recontact cycles before each stage promotion."""

    reset_source: str = "cache"
    grasp_cache_path: str = (
        "robots/leap_hand/caches/ball_grasp_official_50k.npy"
    )
    termination_drop_distance: float = 0.007
    finger_gaiting: FingerGaitingConfig = field(default_factory=FingerGaitingConfig)
    max_episode_seconds: float = 35.0

    def validate(self) -> None:
        super().validate()
        if not np.isfinite(self.termination_drop_distance) or (
            self.termination_drop_distance <= 0.0
        ):
            raise ValueError("termination_drop_distance must be positive and finite")
        cfg = self.finger_gaiting
        stage_count = len(self.curriculum.target_speeds)
        if len(cfg.required_handoffs_by_stage) != stage_count:
            raise ValueError("required_handoffs_by_stage must match target_speeds")
        if len(cfg.minimum_contacts_by_stage) != stage_count:
            raise ValueError("minimum_contacts_by_stage must match target_speeds")
        if any(value < 0 for value in cfg.required_handoffs_by_stage):
            raise ValueError("required handoff counts must be non-negative")
        if any(value not in range(1, 5) for value in cfg.minimum_contacts_by_stage):
            raise ValueError("minimum stage contacts must be between 1 and 4")
        if cfg.minimum_other_contacts not in range(1, 4):
            raise ValueError("minimum_other_contacts must be between 1 and 3")
        if not 0 < cfg.minimum_release_steps <= cfg.maximum_release_steps:
            raise ValueError("release step bounds must be positive and ordered")
        if cfg.handoff_cooldown_steps < 0:
            raise ValueError("handoff_cooldown_steps must be non-negative")
        if not 0.0 < cfg.minimum_speed_ratio <= 1.0:
            raise ValueError("minimum_speed_ratio must be in (0, 1]")
        if not 0.0 < cfg.recovery_speed_ratio <= 1.0:
            raise ValueError("recovery_speed_ratio must be in (0, 1]")
        if not (np.isfinite(cfg.minimum_handoff_angle_rad) and cfg.minimum_handoff_angle_rad >= 0.0):
            raise ValueError("minimum_handoff_angle_rad must be finite and non-negative")
        if len(cfg.release_allowed_fingers) != 4:
            raise ValueError("release_allowed_fingers must have length 4")
        if not all(isinstance(value, (bool, np.bool_)) for value in cfg.release_allowed_fingers):
            raise ValueError("release_allowed_fingers elements must be boolean")
        for value, name in (
            (cfg.stable_support_scale, "stable_support_scale"),
            (cfg.release_progress_scale, "release_progress_scale"),
            (cfg.qualified_handoff_bonus, "qualified_handoff_bonus"),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")


@registry.env("LeapInhandBallFingerGaitingRotation", sim_backend="motrix")
@registry.env("LeapInhandBallFingerGaitingRotation", sim_backend="mujoco")
class LeapInhandBallFingerGaitingRotationEnv(LeapInhandBallSustainedRotationEnv):
    """Teach cyclic finger handoffs while maintaining retained +Z rotation."""

    _cfg: LeapInhandBallFingerGaitingRotationCfg
    _FINGER_NAMES = ("index", "middle", "ring", "thumb")
    _NUM_GAITING_OBS = 114

    def _compute_raw_drop(
        self, ball_pos: np.ndarray, anchor_pos: np.ndarray
    ) -> np.ndarray:
        return np.linalg.norm(ball_pos - anchor_pos, axis=1) > self._cfg.termination_drop_distance

    def __init__(
        self,
        cfg: LeapInhandBallFingerGaitingRotationCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> None:
        self._required_handoffs = np.asarray(
            cfg.finger_gaiting.required_handoffs_by_stage, dtype=np.uint8
        )
        self._minimum_stage_contacts = np.asarray(
            cfg.finger_gaiting.minimum_contacts_by_stage, dtype=np.uint8
        )
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)

    def _make_domain_randomization_provider(self):
        return LeapFingerGaitingResetProvider()

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": self._NUM_GAITING_OBS}

    def _compute_sustained_obs(
        self, env_ids: np.ndarray, info: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        base_obs = super()._compute_sustained_obs(env_ids, info)["obs"]

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
        cfg = self._cfg.finger_gaiting
        eligible = retention_ok & no_failure_signal & ~palm_contact
        required = self._required_handoffs[levels]
        start_angle = (
            release_start_angle
            if release_start_angle is not None
            else info.get("gaiting_release_start_angle", None)
        )
        curr_angle = (
            cumulative_angle
            if cumulative_angle is not None
            else info.get("rotation_net_angle_rad", None)
        )
        transition = advance_finger_gaiting(
            contacts=fingertip_contacts,
            previous_contacts=info["gaiting_previous_contacts"],
            active=info["gaiting_release_active"],
            release_steps=info["gaiting_release_steps"],
            release_start_speed=info["gaiting_release_start_speed"],
            cooldown_steps=info["gaiting_cooldown_steps"],
            eligible=eligible,
            axis_speed_ema=axis_speed_ema,
            target_speed=target_speed,
            cfg=cfg,
            stationary_handoff_allowed=(target_speed <= 1e-6) & (required > 0),
            release_start_angle=start_angle,
            cumulative_angle=curr_angle,
        )
        if start_angle is not None and "gaiting_release_start_angle" in info:
            info["gaiting_release_start_angle"] = transition.release_start_angle.copy()
        handoff_event = np.any(transition.qualified_handoff, axis=1)
        stage_handoffs = np.asarray(info["gaiting_stage_handoffs"], dtype=np.uint8)
        useful_handoff = handoff_event & (stage_handoffs < required)
        stage_handoffs[:] = np.minimum(
            stage_handoffs.astype(np.uint16) + useful_handoff.astype(np.uint16), 255
        ).astype(np.uint8)
        total_handoffs = np.asarray(info["gaiting_total_handoffs"], dtype=np.uint16)
        total_handoffs[:] = np.minimum(
            total_handoffs.astype(np.uint32) + handoff_event.astype(np.uint32), 65535
        ).astype(np.uint16)

        minimum_contacts = self._minimum_stage_contacts[levels]
        validity_mask = contact_count >= minimum_contacts
        completion_ready = stage_handoffs >= required
        hold_stage = levels == 0
        stable_support = hold_stage & validity_mask & eligible
        dense_reward = (
            cfg.stable_support_scale * stable_support
            + cfg.release_progress_scale * transition.release_progress * (~hold_stage)
        )
        recovery_quality = np.zeros(self._num_envs, dtype=self._np_dtype)
        rotating = target_speed > 1e-6
        recovery_quality[rotating] = np.clip(
            axis_speed_ema[rotating] / target_speed[rotating], 0.0, 1.0
        )
        event_reward = cfg.qualified_handoff_bonus * useful_handoff * (0.5 + 0.5 * recovery_quality)

        previous_contacts = np.asarray(info["gaiting_previous_contacts"], dtype=bool)
        contact_transition = np.any(fingertip_contacts != previous_contacts, axis=1)
        no_transition_steps = np.asarray(info["gaiting_no_transition_steps"], dtype=np.uint32)
        no_transition_steps[:] = np.where(contact_transition, 0, no_transition_steps + 1)
        longest_no_transition = np.asarray(
            info["gaiting_longest_no_transition_steps"], dtype=np.uint32
        )
        longest_no_transition[:] = np.maximum(longest_no_transition, no_transition_steps)
        rotation_steps = np.asarray(info["gaiting_rotation_steps"], dtype=np.uint32)
        rotation_valid = compute_rotation_duration_valid(
            base_stage_valid & validity_mask,
            target_speed,
        )
        rotation_steps[:] = np.where(rotation_valid, rotation_steps + 1, 0)
        longest_rotation = np.asarray(info["gaiting_longest_rotation_steps"], dtype=np.uint32)
        longest_rotation[:] = np.maximum(longest_rotation, rotation_steps)
        contact_steps = np.asarray(info["gaiting_contact_steps"], dtype=np.uint32)
        contact_steps[:] += fingertip_contacts.astype(np.uint32)
        observed_steps = np.asarray(info["gaiting_observed_steps"], dtype=np.uint32)
        observed_steps[:] += 1

        info["gaiting_previous_contacts"] = fingertip_contacts.copy()
        info["gaiting_release_active"] = transition.active
        info["gaiting_release_steps"] = transition.release_steps
        info["gaiting_release_start_speed"] = transition.release_start_speed
        info["gaiting_cooldown_steps"] = transition.cooldown_steps
        info["gaiting_stage_handoffs"] = stage_handoffs
        info["gaiting_total_handoffs"] = total_handoffs
        info["gaiting_contact_steps"] = contact_steps
        info["gaiting_observed_steps"] = observed_steps
        info["gaiting_no_transition_steps"] = no_transition_steps
        info["gaiting_longest_no_transition_steps"] = longest_no_transition
        info["gaiting_rotation_steps"] = rotation_steps
        info["gaiting_longest_rotation_steps"] = longest_rotation

        denominator = np.maximum(observed_steps[:, None], 1)
        duty = contact_steps / denominator
        required_denominator = np.maximum(required, 1)
        log = {
            "reward/gaiting_stable_support": float(
                np.mean(cfg.stable_support_scale * stable_support)
            ),
            "reward/gaiting_release_progress": float(
                np.mean(cfg.release_progress_scale * transition.release_progress)
            ),
            "reward/gaiting_handoff_event": float(np.mean(event_reward)),
            "gaiting/qualified_handoff_rate": float(np.mean(handoff_event)),
            "gaiting/useful_handoff_rate": float(np.mean(useful_handoff)),
            "gaiting/stage_handoff_progress": float(
                np.mean(np.clip(stage_handoffs / required_denominator, 0.0, 1.0))
            ),
            "gaiting/total_handoffs": float(np.mean(total_handoffs)),
            "gaiting/release_active_fraction": float(np.mean(transition.active)),
            "gaiting/longest_rotation_seconds": float(
                np.mean(longest_rotation) * self._cfg.ctrl_dt
            ),
            "gaiting/longest_no_transition_seconds": float(
                np.mean(longest_no_transition) * self._cfg.ctrl_dt
            ),
            "gaiting/inactive_finger_fraction": float(np.mean(contact_steps == 0)),
        }
        for index, name in enumerate(self._FINGER_NAMES):
            log[f"gaiting/contact_duty_{name}"] = float(np.mean(duty[:, index]))
            log[f"gaiting/handoff_rate_{name}"] = float(
                np.mean(transition.qualified_handoff[:, index])
            )
        return StageSkillUpdate(
            validity_mask=validity_mask,
            completion_ready=completion_ready,
            dense_reward=np.asarray(dense_reward, dtype=self._np_dtype),
            event_reward=np.asarray(event_reward, dtype=self._np_dtype),
            log=log,
        )

    def _on_stage_promotion(self, info: dict[str, Any], promoted: np.ndarray) -> None:
        stage_handoffs = np.asarray(info["gaiting_stage_handoffs"], dtype=np.uint8)
        stage_handoffs[promoted] = 0
        info["gaiting_stage_handoffs"] = stage_handoffs


__all__ = [
    "FingerGaitingConfig",
    "FingerGaitingTransition",
    "LeapInhandBallFingerGaitingRotationCfg",
    "LeapInhandBallFingerGaitingRotationEnv",
    "advance_finger_gaiting",
    "normalize_finger_gaiting_observation",
]
