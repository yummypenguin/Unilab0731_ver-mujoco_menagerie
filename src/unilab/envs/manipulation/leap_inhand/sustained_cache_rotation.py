"""Sustained LEAP ball rotation initialized from the validated grasp cache."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs.manipulation.allegro_inhand.rotation import compute_ball_angvel
from unilab.utils.rotation import np_quat_apply_batched

from .sustained_rotation import (
    LeapInhandBallSustainedRotationCfg,
    LeapInhandBallSustainedRotationEnv,
    SustainedRotationRewardConfig,
    compute_anchor_proximity,
    compute_sustained_spin_terms,
)

SUPPORT_JOINT_INDICES = np.asarray(
    [0, 1, 2, 3, 8, 9, 10, 11],
    dtype=np.int64,
)

STATE_A_SUPPORT_QPOS = np.asarray(
    [
        # Index
        1.4835214808958397,
        -0.1765436837919707,
        0.3156926825231613,
        0.28408987301132416,

        # Ring
        1.4150975323947736,
        0.040541514116881845,
        0.07763925092635474,
        -0.008650506225424903,
    ],
    dtype=np.float64,
)

LEAP_TIP_COLLISION_LOCAL_POS = np.asarray(
    [
        # Index (index_tip_col)
        [-0.001296903161143803, -0.033552395765232425, 0.014520657667431451],
        # Middle (middle_tip_col)
        [-0.001296903161143803, -0.033552395765232425, 0.014520657667431451],
        # Ring (ring_tip_col)
        [-0.001296903161143803, -0.033552395765232425, 0.014520657667431451],
        # Thumb (thumb_tip_col)
        [-0.00127744377325541, -0.04556193662786432, -0.014477408474461434],
    ],
    dtype=np.float64,
)


@dataclass
class AllegroStyleRotationRewardConfig(SustainedRotationRewardConfig):
    """Signed-axis reward with index-ring support progress shaping."""

    scales: dict[str, float] = field(
        default_factory=lambda: {
            "rotate": 1.25,
            "obj_linvel": -0.3,
            "position_error": -6.0,
            "spin_progress": 0.0,
            # Metadata mirror only; runtime uses spin_continuity_penalty_scale.
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
    angvel_clip_min: float = -0.5
    angvel_clip_max: float = 0.5
    positive_spin_base_contact_scale: float = 1.0
    positive_spin_index_contact_scale: float = 0.0
    positive_spin_middle_contact_scale: float = 0.0
    support_pose_progress_scale: float = 0.0
    support_pose_progress_clip: float = 0.04
    opposition_progress_scale: float = 0.0
    opposition_progress_clip: float = 0.05
    position_gate_power: float = 2.0
    failure_penalty: float = 1.0
    failure_position_radius: float = 0.030
    spin_continuity_penalty_scale: float = 0.05


def compute_position_safety_gate(
    position_error: np.ndarray,
    *,
    gate_position_radius: float,
    failure_position_radius: float,
    power: float,
) -> np.ndarray:
    """Return [0, 1] safety gate for positive spin reward."""
    linear_gate = compute_anchor_proximity(
        position_error,
        gate_position_radius,
        failure_position_radius,
    )
    return np.power(linear_gate, power).astype(position_error.dtype, copy=False)


def compute_position_gated_stall_penalty(
    normalized_progress: np.ndarray,
    position_gate: np.ndarray,
    *,
    penalty_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Penalize failure to rotate while the object remains in the safe region."""
    positive_progress = np.clip(
        normalized_progress,
        0.0,
        1.0,
    )
    stall_penalty_rate = (
        -penalty_scale
        * position_gate
        * (1.0 - positive_progress)
    )
    return (
        positive_progress.astype(
            normalized_progress.dtype,
            copy=False,
        ),
        stall_penalty_rate.astype(
            normalized_progress.dtype,
            copy=False,
        ),
    )


def compute_target_speed_gated_rotate_reward(
    ball_angvel: np.ndarray,
    rotation_axis: np.ndarray,
    target_speed: np.ndarray,
    orthogonal_tolerance: np.ndarray,
    position_gate: np.ndarray,
    *,
    scale: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return target-speed normalized and position-gated rotation rates."""
    rotation_axis_batch = np.broadcast_to(
        rotation_axis,
        ball_angvel.shape,
    )
    (
        axis_speed,
        orthogonal_speed,
        normalized_progress,
        _visible_progress,
        axis_purity,
    ) = compute_sustained_spin_terms(
        ball_angvel,
        rotation_axis_batch,
        target_speed,
        orthogonal_tolerance,
    )

    rotation_amplitude = scale * target_speed

    positive_progress = np.maximum(normalized_progress, 0.0)
    positive_rotate_rate = (
        rotation_amplitude
        * positive_progress
        * axis_purity
        * position_gate
    )

    negative_progress = np.minimum(normalized_progress, 0.0)
    negative_rotate_rate = rotation_amplitude * negative_progress

    base_rotate_rate = rotation_amplitude * normalized_progress
    gated_rotate_rate = positive_rotate_rate + negative_rotate_rate

    return (
        axis_speed,
        orthogonal_speed,
        normalized_progress,
        axis_purity,
        base_rotate_rate,
        gated_rotate_rate,
    )


def compute_support_pose_distance(
    dof_pos: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
    reference_qpos: np.ndarray = STATE_A_SUPPORT_QPOS,
) -> np.ndarray:
    """Return normalized support-pose RMS distance to State A reference."""
    dof_pos = np.asarray(dof_pos)
    dtype = dof_pos.dtype

    lower = np.asarray(ctrl_lower, dtype=dtype)[SUPPORT_JOINT_INDICES]
    upper = np.asarray(ctrl_upper, dtype=dtype)[SUPPORT_JOINT_INDICES]
    reference = np.asarray(reference_qpos, dtype=dtype)

    qpos = dof_pos[:, SUPPORT_JOINT_INDICES]
    epsilon = np.asarray(1e-8, dtype=dtype)
    joint_range = upper - lower + epsilon

    normalized = (qpos - lower) / joint_range
    reference_normalized = (reference - lower) / joint_range
    error = normalized - reference_normalized

    return np.sqrt(np.mean(np.square(error), axis=1)).astype(dtype, copy=False)


def compute_tip_collision_reference_positions(
    fingertip_body_pos: np.ndarray,
    fingertip_body_quat: np.ndarray,
) -> np.ndarray:
    """Return world coordinates of tip collision geom origins for each fingertip."""
    dtype = fingertip_body_pos.dtype
    local_offset = np.asarray(
        LEAP_TIP_COLLISION_LOCAL_POS,
        dtype=dtype,
    )
    local_offsets = np.broadcast_to(
        local_offset,
        fingertip_body_pos.shape,
    )
    return (
        fingertip_body_pos
        + np_quat_apply_batched(
            fingertip_body_quat,
            local_offsets,
        )
    ).astype(dtype, copy=False)


def compute_index_ring_opposition_quality(
    index_tip_pos: np.ndarray,
    ring_tip_pos: np.ndarray,
    ball_pos: np.ndarray,
) -> np.ndarray:
    """Return opposition quality in [0, 1] between index and ring fingertips."""
    dtype = index_tip_pos.dtype
    epsilon = np.asarray(1e-8, dtype=dtype)
    half = np.asarray(0.5, dtype=dtype)
    zero = np.asarray(0.0, dtype=dtype)
    one = np.asarray(1.0, dtype=dtype)

    index_vector = index_tip_pos - ball_pos
    ring_vector = ring_tip_pos - ball_pos

    index_unit = index_vector / (
        np.linalg.norm(index_vector, axis=1, keepdims=True) + epsilon
    )
    ring_unit = ring_vector / (
        np.linalg.norm(ring_vector, axis=1, keepdims=True) + epsilon
    )

    dot = np.sum(index_unit * ring_unit, axis=1)
    quality = np.clip(half * (one - dot), zero, one)
    return quality.astype(dtype, copy=False)


def compute_reset_safe_opposition_progress(
    current_potential: np.ndarray,
    previous_potential: np.ndarray,
    step_count: np.ndarray,
) -> np.ndarray:
    """Return opposition progress zeroed out on reset step."""
    dtype = current_potential.dtype
    first_step_after_reset = np.asarray(step_count == 0, dtype=bool)
    effective_previous = np.where(
        first_step_after_reset,
        current_potential,
        previous_potential,
    )
    return (current_potential - effective_previous).astype(dtype, copy=False)


def compute_allegro_style_rotate_reward(
    ball_angvel: np.ndarray,
    rotation_axis: np.ndarray,
    *,
    scale: float,
    clip_min: float,
    clip_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Allegro's clipped signed-axis speed and its weighted reward."""
    axis_speed = ball_angvel @ rotation_axis
    clipped_axis_speed = np.clip(axis_speed, clip_min, clip_max)
    return clipped_axis_speed, scale * clipped_axis_speed


def apply_positive_spin_finger_participation(
    base_rotate_reward: np.ndarray,
    fingertip_contacts: np.ndarray,
    *,
    base_contact_scale: float,
    index_contact_scale: float,
    middle_contact_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale only positive spin reward by index and middle fingertip contact."""
    participation_scale = (
        base_contact_scale
        + index_contact_scale * fingertip_contacts[:, 0]
        + middle_contact_scale * fingertip_contacts[:, 1]
    )
    rotate_reward = np.where(
        base_rotate_reward > 0.0,
        base_rotate_reward * participation_scale,
        base_rotate_reward,
    )
    return participation_scale, rotate_reward


def compute_allegro_style_obj_linvel_reward(
    ball_linvel: np.ndarray,
    *,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Allegro's object L1 linear-speed cost and its weighted reward."""
    linear_speed_l1 = np.sum(np.abs(ball_linvel), axis=1)
    return linear_speed_l1, scale * linear_speed_l1


def compute_position_error_reward(
    ball_pos: np.ndarray,
    anchor_pos: np.ndarray,
    *,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Penalize persistent displacement from the episode's cache reset."""
    position_error = np.linalg.norm(ball_pos - anchor_pos, axis=1)
    return position_error, scale * position_error


@registry.envcfg("LeapInhandBallSustainedCacheRotation")
@dataclass
class LeapInhandBallSustainedCacheRotationCfg(
    LeapInhandBallSustainedRotationCfg
):
    """Use cache grasps with an Allegro-style rotation-only objective."""

    reset_source: str = "cache"
    grasp_cache_path: str = (
        "robots/leap_hand/caches/ball_grasp_official_50k.npy"
    )
    termination_workspace_radius: float = 0.05
    reward_config: AllegroStyleRotationRewardConfig | None = None

    def validate(self) -> None:
        super().validate()
        reward_config = self.reward_config
        if reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        if isinstance(reward_config, AllegroStyleRotationRewardConfig):
            if not (
                np.isfinite(reward_config.angvel_clip_min)
                and np.isfinite(reward_config.angvel_clip_max)
                and reward_config.angvel_clip_min < reward_config.angvel_clip_max
            ):
                raise ValueError("angvel_clip_min must be less than angvel_clip_max")
            for name in ("rotate", "obj_linvel", "position_error"):
                if (
                    name not in reward_config.scales
                    or not np.isfinite(reward_config.scales[name])
                ):
                    raise ValueError(f"reward scales must define finite {name}")
            contact_scales = (
                reward_config.positive_spin_base_contact_scale,
                reward_config.positive_spin_index_contact_scale,
                reward_config.positive_spin_middle_contact_scale,
            )
            if not all(np.isfinite(scale) and scale >= 0.0 for scale in contact_scales):
                raise ValueError("positive spin contact scales must be finite and non-negative")
            if not (
                np.isfinite(reward_config.support_pose_progress_scale)
                and reward_config.support_pose_progress_scale >= 0.0
                and np.isfinite(reward_config.opposition_progress_scale)
                and reward_config.opposition_progress_scale >= 0.0
            ):
                raise ValueError("progress scales must be finite and non-negative")
            if not (
                np.isfinite(reward_config.support_pose_progress_clip)
                and reward_config.support_pose_progress_clip > 0.0
                and np.isfinite(reward_config.opposition_progress_clip)
                and reward_config.opposition_progress_clip > 0.0
            ):
                raise ValueError("progress clips must be finite and positive")
            if not (
                np.isfinite(reward_config.position_gate_power)
                and reward_config.position_gate_power > 0.0
            ):
                raise ValueError("position_gate_power must be positive and finite")
            if not (
                np.isfinite(reward_config.failure_penalty)
                and reward_config.failure_penalty >= 0.0
            ):
                raise ValueError("failure_penalty must be finite and non-negative")
            if not (
                np.isfinite(reward_config.spin_continuity_penalty_scale)
                and reward_config.spin_continuity_penalty_scale >= 0.0
            ):
                raise ValueError(
                    "spin_continuity_penalty_scale must be finite and non-negative"
                )
            if not (
                self.curriculum.gate_position_radius
                < reward_config.failure_position_radius
                < self.termination_workspace_radius
            ):
                raise ValueError(
                    "reward position radii must satisfy gate_position_radius < failure_position_radius < termination_workspace_radius"
                )
        if not np.isfinite(self.termination_workspace_radius) or (
            self.termination_workspace_radius <= 0.0
        ):
            raise ValueError("termination_workspace_radius must be positive and finite")


@registry.env("LeapInhandBallSustainedCacheRotation", sim_backend="motrix")
@registry.env("LeapInhandBallSustainedCacheRotation", sim_backend="mujoco")
class LeapInhandBallSustainedCacheRotationEnv(
    LeapInhandBallSustainedRotationEnv
):
    """Learn signed-axis rotation with index-ring support progress shaping."""

    _cfg: LeapInhandBallSustainedCacheRotationCfg
    _reward_cfg: AllegroStyleRotationRewardConfig

    def _compute_terminated(self, position_error: np.ndarray) -> np.ndarray:
        """Terminate only when the ball leaves the wide reset-relative workspace."""
        return np.asarray(
            position_error > self._cfg.termination_workspace_radius,
            dtype=bool,
        )

    def update_state(self, state: NpEnvState) -> NpEnvState:
        info = state.info
        dtype = get_global_dtype()
        dof_pos = self.get_hand_dof_pos()
        ball_pos = self.get_ball_pos()
        ball_quat = self.get_ball_quat()
        prev_ball_pos = np.asarray(info.get("prev_ball_pos", ball_pos), dtype=dtype)
        prev_ball_quat = np.asarray(info.get("prev_ball_quat", ball_quat), dtype=dtype)
        ball_linvel = (ball_pos - prev_ball_pos) / self._cfg.ctrl_dt
        ball_angvel = compute_ball_angvel(ball_quat, prev_ball_quat, self._cfg.ctrl_dt)

        support_shaping_enabled = isinstance(
            self._reward_cfg,
            AllegroStyleRotationRewardConfig,
        )

        anchor_pos = np.asarray(info["rotation_anchor_pos"], dtype=dtype)
        position_error, position_error_reward = compute_position_error_reward(
            ball_pos,
            anchor_pos,
            scale=(
                self._reward_cfg.scales["position_error"]
                if support_shaping_enabled
                else self._reward_cfg.scales.get("position_error", -6.0)
            ),
        )
        terminated = self._compute_terminated(position_error)

        if support_shaping_enabled:
            rotate_scale = self._reward_cfg.scales["rotate"]
            base_contact_scale = self._reward_cfg.positive_spin_base_contact_scale
            index_contact_scale = self._reward_cfg.positive_spin_index_contact_scale
            middle_contact_scale = self._reward_cfg.positive_spin_middle_contact_scale
            obj_linvel_scale = self._reward_cfg.scales["obj_linvel"]
            position_gate_power = self._reward_cfg.position_gate_power
            failure_position_radius = self._reward_cfg.failure_position_radius
            failure_penalty = self._reward_cfg.failure_penalty
            spin_continuity_penalty_scale = self._reward_cfg.spin_continuity_penalty_scale
        else:
            rotate_scale = self._reward_cfg.scales.get("rotate", 1.25)
            base_contact_scale = getattr(self._reward_cfg, "positive_spin_base_contact_scale", 1.0)
            index_contact_scale = getattr(self._reward_cfg, "positive_spin_index_contact_scale", 0.0)
            middle_contact_scale = getattr(self._reward_cfg, "positive_spin_middle_contact_scale", 0.0)
            obj_linvel_scale = self._reward_cfg.scales.get("obj_linvel", -0.3)
            position_gate_power = getattr(self._reward_cfg, "position_gate_power", 2.0)
            failure_position_radius = getattr(self._reward_cfg, "failure_position_radius", 0.030)
            failure_penalty = getattr(self._reward_cfg, "failure_penalty", 1.0)
            spin_continuity_penalty_scale = getattr(self._reward_cfg, "spin_continuity_penalty_scale", 0.05)

        position_gate = compute_position_safety_gate(
            position_error,
            gate_position_radius=self._cfg.curriculum.gate_position_radius,
            failure_position_radius=failure_position_radius,
            power=position_gate_power,
        )

        target_speed = np.full(
            self._num_envs,
            self._cfg.curriculum.target_speeds[0],
            dtype=dtype,
        )
        orthogonal_tolerance = np.full(
            self._num_envs,
            self._cfg.curriculum.orthogonal_speed_tolerances[0],
            dtype=dtype,
        )

        (
            axis_speed,
            orthogonal_speed,
            normalized_progress,
            axis_purity,
            base_rotate_reward,
            gated_rotate_reward,
        ) = compute_target_speed_gated_rotate_reward(
            ball_angvel,
            self._rotation_axis_w,
            target_speed,
            orthogonal_tolerance,
            position_gate,
            scale=rotate_scale,
        )

        (
            positive_progress,
            stall_penalty_rate,
        ) = compute_position_gated_stall_penalty(
            normalized_progress,
            position_gate,
            penalty_scale=spin_continuity_penalty_scale,
        )

        fingertip_contacts = self._contacts(self._all_env_ids)
        participation_scale, rotate_reward = apply_positive_spin_finger_participation(
            gated_rotate_reward,
            fingertip_contacts,
            base_contact_scale=base_contact_scale,
            index_contact_scale=index_contact_scale,
            middle_contact_scale=middle_contact_scale,
        )
        linear_speed_l1, obj_linvel_reward = compute_allegro_style_obj_linvel_reward(
            ball_linvel,
            scale=obj_linvel_scale,
        )

        palm_contact_bool = self._palm_contacts(self._all_env_ids).astype(bool)
        fingertip_contact_count = np.sum(fingertip_contacts, axis=1)
        reward_adjustment, reward_adjustment_log = self._compute_reward_adjustment(
            info=info,
            fingertip_contacts=fingertip_contacts.astype(bool),
            palm_contact=palm_contact_bool,
            contact_count=fingertip_contact_count,
            target_speed=target_speed,
            tolerance=orthogonal_tolerance,
            axis_speed=axis_speed,
            axis_speed_ema=axis_speed,
            orthogonal_speed_ema=orthogonal_speed,
            position_error=position_error,
            anchor_proximity=position_gate,
            retention_ok=np.ones(self._num_envs, dtype=bool),
            no_failure_signal=~terminated,
            stage_valid=np.ones(self._num_envs, dtype=bool),
            stage_duration_progress=np.ones(self._num_envs, dtype=dtype),
        )
        dense_reward = (
            rotate_reward
            + obj_linvel_reward
            + position_error_reward
            + stall_penalty_rate
            + reward_adjustment
        )
        reward = np.asarray(dense_reward * self._cfg.ctrl_dt, dtype=dtype)

        if support_shaping_enabled:
            current_support_distance = compute_support_pose_distance(
                dof_pos,
                self._ctrl_lower,
                self._ctrl_upper,
            )
            previous_support_distance = compute_support_pose_distance(
                np.asarray(info.get("prev_dof_pos", dof_pos), dtype=dtype),
                self._ctrl_lower,
                self._ctrl_upper,
            )
            raw_support_progress = (
                previous_support_distance - current_support_distance
            ).astype(dtype, copy=False)
            support_pose_progress_reward = (
                self._reward_cfg.support_pose_progress_scale
                * np.clip(
                    raw_support_progress,
                    -self._reward_cfg.support_pose_progress_clip,
                    self._reward_cfg.support_pose_progress_clip,
                )
            ).astype(dtype, copy=False)

            fingertip_body_pos, fingertip_body_quat = self._backend.get_body_pose_w(
                self._fingertip_body_ids
            )
            tip_collision_pos = compute_tip_collision_reference_positions(
                fingertip_body_pos,
                fingertip_body_quat,
            )
            index_tip_pos = tip_collision_pos[:, 0, :]
            ring_tip_pos = tip_collision_pos[:, 2, :]
            opposition_quality = compute_index_ring_opposition_quality(
                index_tip_pos,
                ring_tip_pos,
                ball_pos,
            )

            index_contact = fingertip_contacts[:, 0].astype(bool)
            ring_contact = fingertip_contacts[:, 2].astype(bool)
            index_ring_contact = index_contact & ring_contact
            current_opposition_potential = (
                index_ring_contact.astype(dtype) * opposition_quality
            ).astype(dtype, copy=False)

            step_count = np.asarray(
                info.get("steps", np.zeros(self._num_envs, dtype=np.uint32))
            )
            previous_opposition_potential = np.asarray(
                info.get("prev_opposition_potential", current_opposition_potential),
                dtype=dtype,
            )
            raw_opposition_progress = compute_reset_safe_opposition_progress(
                current_opposition_potential,
                previous_opposition_potential,
                step_count,
            )
            opposition_progress_reward = (
                self._reward_cfg.opposition_progress_scale
                * np.clip(
                    raw_opposition_progress,
                    -self._reward_cfg.opposition_progress_clip,
                    self._reward_cfg.opposition_progress_clip,
                )
            ).astype(dtype, copy=False)

            reward += support_pose_progress_reward
            reward += opposition_progress_reward
            info["prev_opposition_potential"] = current_opposition_potential.copy()

        failure_reward = (-failure_penalty * terminated.astype(dtype)).astype(
            dtype, copy=False
        )
        reward += failure_reward

        info["curr_dof_pos"] = dof_pos.copy()
        info["curr_ball_pos"] = ball_pos.copy()
        info["curr_ball_quat"] = ball_quat.copy()
        info["prev_dof_pos"] = dof_pos.copy()
        info["prev_ball_pos"] = ball_pos.copy()
        info["prev_ball_quat"] = ball_quat.copy()

        obs = self._compute_sustained_obs(self._all_env_ids, info)
        step_count = info.get("steps", np.zeros(self._num_envs, dtype=np.uint32))
        if self._enable_reward_log and int(step_count[0]) % 4 == 0:
            displacement = ball_pos - anchor_pos
            palm_contact = self._palm_contacts(self._all_env_ids)
            log = {
                "reward/rotate_base": float(np.mean(base_rotate_reward)),
                "reward/rotate_ungated": float(np.mean(base_rotate_reward)),
                "reward/rotate": float(np.mean(rotate_reward)),
                "reward/finger_participation": float(
                    np.mean(rotate_reward - gated_rotate_reward)
                ),
                "reward/rotation_gating_reduction": float(
                    np.mean(gated_rotate_reward - base_rotate_reward)
                ),
                "reward/obj_linvel": float(np.mean(obj_linvel_reward)),
                "reward/position_error": float(np.mean(position_error_reward)),
                "reward/stall": float(np.mean(stall_penalty_rate)),
                "reward/failure": float(np.mean(failure_reward)),
                "reward/total": float(np.mean(reward)),
                "rotation/axis_speed_rad_s": float(np.mean(axis_speed)),
                "rotation/clipped_axis_speed_rad_s": float(np.mean(axis_speed)),
                "rotation/target_speed_rad_s": float(np.mean(target_speed)),
                "rotation/orthogonal_speed_rad_s": float(np.mean(orthogonal_speed)),
                "rotation/normalized_progress": float(np.mean(normalized_progress)),
                "rotation/positive_progress": float(np.mean(positive_progress)),
                "rotation/axis_purity": float(np.mean(axis_purity)),
                "rotation/position_gate": float(np.mean(position_gate)),
                "rotation/positive_spin_participation_scale": float(
                    np.mean(participation_scale)
                ),
                "object/linvel_l1_m_s": float(np.mean(linear_speed_l1)),
                "object/linvel_horizontal_m_s": float(
                    np.mean(np.linalg.norm(ball_linvel[:, :2], axis=1))
                ),
                "object/linvel_vertical_m_s": float(np.mean(ball_linvel[:, 2])),
                "object/position_error_m": float(np.mean(position_error)),
                "object/horizontal_error_m": float(
                    np.mean(np.linalg.norm(displacement[:, :2], axis=1))
                ),
                "object/vertical_displacement_m": float(np.mean(displacement[:, 2])),
                "contact/fingertip_count": float(np.mean(fingertip_contact_count)),
                "contact/index_rate": float(np.mean(fingertip_contacts[:, 0])),
                "contact/middle_rate": float(np.mean(fingertip_contacts[:, 1])),
                "contact/ring_rate": float(np.mean(fingertip_contacts[:, 2])),
                "contact/thumb_rate": float(np.mean(fingertip_contacts[:, 3])),
                "contact/index_middle_count": float(
                    np.mean(np.sum(fingertip_contacts[:, :2], axis=1))
                ),
                "contact/palm_contact_rate": float(np.mean(palm_contact)),
                "termination/workspace_rate": float(np.mean(terminated)),
                "termination/task_rate": float(np.mean(terminated)),
            }
            if support_shaping_enabled:
                log.update(
                    {
                        "reward/support_pose_progress": float(
                            np.mean(support_pose_progress_reward)
                        ),
                        "reward/opposition_progress": float(
                            np.mean(opposition_progress_reward)
                        ),
                        "support/state_a_pose_distance": float(
                            np.mean(current_support_distance)
                        ),
                        "support/state_a_pose_raw_progress": float(
                            np.mean(raw_support_progress)
                        ),
                        "support/index_ring_contact_rate": float(
                            np.mean(index_ring_contact)
                        ),
                        "support/opposition_quality": float(
                            np.mean(opposition_quality)
                        ),
                        "support/opposition_potential": float(
                            np.mean(current_opposition_potential)
                        ),
                        "support/opposition_raw_progress": float(
                            np.mean(raw_opposition_progress)
                        ),
                    }
                )
            log.update(reward_adjustment_log)
            info["log"] = log

        return state.replace(obs=obs, reward=reward, terminated=terminated)


__all__ = [
    "LEAP_TIP_COLLISION_LOCAL_POS",
    "STATE_A_SUPPORT_QPOS",
    "SUPPORT_JOINT_INDICES",
    "AllegroStyleRotationRewardConfig",
    "LeapInhandBallSustainedCacheRotationCfg",
    "LeapInhandBallSustainedCacheRotationEnv",
    "apply_positive_spin_finger_participation",
    "compute_allegro_style_obj_linvel_reward",
    "compute_allegro_style_rotate_reward",
    "compute_index_ring_opposition_quality",
    "compute_position_error_reward",
    "compute_position_gated_stall_penalty",
    "compute_position_safety_gate",
    "compute_reset_safe_opposition_progress",
    "compute_support_pose_distance",
    "compute_target_speed_gated_rotate_reward",
    "compute_tip_collision_reference_positions",
]
