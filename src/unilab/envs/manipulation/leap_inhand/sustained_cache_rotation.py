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
        0.0132864241085335,
        -0.0061142383865420,
        0.0145,
    ],
    dtype=np.float64,
)


@dataclass
class AllegroStyleRotationRewardConfig(SustainedRotationRewardConfig):
    """Signed-axis reward with index/middle participation on positive spin."""

    scales: dict[str, float] = field(
        default_factory=lambda: {
            "rotate": 1.25,
            "obj_linvel": -0.3,
            "position_error": -5.0,
            "spin_progress": 0.0,
            "spin_continuity": 0.0,
            "retention": 0.0,
            "anchor_proximity": 0.0,
            "fingertip_support": 0.0,
            "action_rate": 0.0,
            "torque": 0.0,
            "work": 0.0,
            "failure": 0.0,
        }
    )
    angvel_clip_min: float = -0.5
    angvel_clip_max: float = 0.5
    positive_spin_base_contact_scale: float = 1.0
    positive_spin_index_contact_scale: float = 0.0
    positive_spin_middle_contact_scale: float = 0.0
    support_pose_progress_scale: float = 0.25
    support_pose_progress_clip: float = 0.04
    opposition_progress_scale: float = 0.20
    opposition_progress_clip: float = 0.05


def compute_support_pose_distance(
    dof_pos: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
    reference_qpos: np.ndarray = STATE_A_SUPPORT_QPOS,
) -> np.ndarray:
    """Return normalized support-pose RMS distance to State A reference."""
    qpos = dof_pos[:, SUPPORT_JOINT_INDICES]
    lower = ctrl_lower[SUPPORT_JOINT_INDICES]
    upper = ctrl_upper[SUPPORT_JOINT_INDICES]
    normalized = (qpos - lower) / (upper - lower + 1e-8)
    ref_normalized = (reference_qpos - lower) / (upper - lower + 1e-8)
    error = normalized - ref_normalized
    return np.sqrt(np.mean(np.square(error), axis=1))


def compute_tip_collision_reference_positions(
    fingertip_body_pos: np.ndarray,
    fingertip_body_quat: np.ndarray,
) -> np.ndarray:
    """Return world coordinates of tip collision geom origins for each fingertip."""
    local_offset = np.asarray(
        LEAP_TIP_COLLISION_LOCAL_POS,
        dtype=fingertip_body_pos.dtype,
    )
    local_offsets = np.broadcast_to(
        local_offset,
        fingertip_body_pos.shape,
    )
    return fingertip_body_pos + np_quat_apply_batched(
        fingertip_body_quat,
        local_offsets,
    )


def compute_index_ring_opposition_quality(
    index_tip_pos: np.ndarray,
    ring_tip_pos: np.ndarray,
    ball_pos: np.ndarray,
) -> np.ndarray:
    """Return opposition quality in [0, 1] between index and ring fingertips."""
    index_vector = index_tip_pos - ball_pos
    ring_vector = ring_tip_pos - ball_pos

    index_unit = index_vector / (
        np.linalg.norm(index_vector, axis=1, keepdims=True) + 1e-8
    )
    ring_unit = ring_vector / (
        np.linalg.norm(ring_vector, axis=1, keepdims=True) + 1e-8
    )

    dot = np.sum(index_unit * ring_unit, axis=1)

    return np.clip(
        0.5 * (1.0 - dot),
        0.0,
        1.0,
    )


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
        angvel_clip_min = getattr(reward_config, "angvel_clip_min", -0.5)
        angvel_clip_max = getattr(reward_config, "angvel_clip_max", 0.5)
        if not (
            np.isfinite(angvel_clip_min)
            and np.isfinite(angvel_clip_max)
            and angvel_clip_min < angvel_clip_max
        ):
            raise ValueError("angvel_clip_min must be less than angvel_clip_max")
        for name in ("rotate", "obj_linvel", "position_error"):
            if name in reward_config.scales and not np.isfinite(
                reward_config.scales[name]
            ):
                raise ValueError(f"reward scales must define finite {name}")
        contact_scales = (
            getattr(reward_config, "positive_spin_base_contact_scale", 1.0),
            getattr(reward_config, "positive_spin_index_contact_scale", 0.0),
            getattr(reward_config, "positive_spin_middle_contact_scale", 0.0),
        )
        if not all(np.isfinite(scale) and scale >= 0.0 for scale in contact_scales):
            raise ValueError("positive spin contact scales must be finite and non-negative")
        support_scale = getattr(reward_config, "support_pose_progress_scale", 0.25)
        opposition_scale = getattr(reward_config, "opposition_progress_scale", 0.20)
        if not (
            np.isfinite(support_scale)
            and support_scale >= 0.0
            and np.isfinite(opposition_scale)
            and opposition_scale >= 0.0
        ):
            raise ValueError("progress scales must be finite and non-negative")
        support_clip = getattr(reward_config, "support_pose_progress_clip", 0.04)
        opposition_clip = getattr(reward_config, "opposition_progress_clip", 0.05)
        if not (
            np.isfinite(support_clip)
            and support_clip > 0.0
            and np.isfinite(opposition_clip)
            and opposition_clip > 0.0
        ):
            raise ValueError("progress clips must be finite and positive")
        if not np.isfinite(self.termination_workspace_radius) or (
            self.termination_workspace_radius <= 0.0
        ):
            raise ValueError("termination_workspace_radius must be positive and finite")


@registry.env("LeapInhandBallSustainedCacheRotation", sim_backend="motrix")
@registry.env("LeapInhandBallSustainedCacheRotation", sim_backend="mujoco")
class LeapInhandBallSustainedCacheRotationEnv(
    LeapInhandBallSustainedRotationEnv
):
    """Learn signed-axis rotation with no reward shaping or auxiliary failure gates."""

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

        clipped_axis_speed, base_rotate_reward = compute_allegro_style_rotate_reward(
            ball_angvel,
            self._rotation_axis_w,
            scale=self._reward_cfg.scales.get("rotate", 1.25),
            clip_min=getattr(self._reward_cfg, "angvel_clip_min", -0.5),
            clip_max=getattr(self._reward_cfg, "angvel_clip_max", 0.5),
        )
        fingertip_contacts = self._contacts(self._all_env_ids)
        participation_scale, rotate_reward = apply_positive_spin_finger_participation(
            base_rotate_reward,
            fingertip_contacts,
            base_contact_scale=getattr(self._reward_cfg, "positive_spin_base_contact_scale", 1.0),
            index_contact_scale=getattr(self._reward_cfg, "positive_spin_index_contact_scale", 0.0),
            middle_contact_scale=getattr(self._reward_cfg, "positive_spin_middle_contact_scale", 0.0),
        )
        axis_speed = ball_angvel @ self._rotation_axis_w
        linear_speed_l1, obj_linvel_reward = compute_allegro_style_obj_linvel_reward(
            ball_linvel,
            scale=self._reward_cfg.scales.get("obj_linvel", -0.3),
        )

        anchor_pos = np.asarray(info["rotation_anchor_pos"], dtype=dtype)
        position_error, position_error_reward = compute_position_error_reward(
            ball_pos,
            anchor_pos,
            scale=self._reward_cfg.scales.get("position_error", -5.0),
        )
        terminated = self._compute_terminated(position_error)
        palm_contact = self._palm_contacts(self._all_env_ids).astype(bool)
        fingertip_contact_count = np.sum(fingertip_contacts, axis=1)
        reward_adjustment, reward_adjustment_log = self._compute_reward_adjustment(
            info=info,
            fingertip_contacts=fingertip_contacts.astype(bool),
            palm_contact=palm_contact,
            contact_count=fingertip_contact_count,
            target_speed=np.full(self._num_envs, 0.30, dtype=dtype),
            tolerance=np.full(self._num_envs, 0.10, dtype=dtype),
            axis_speed=axis_speed,
            axis_speed_ema=axis_speed,
            orthogonal_speed_ema=np.zeros(self._num_envs, dtype=dtype),
            position_error=position_error,
            anchor_proximity=np.zeros(self._num_envs, dtype=dtype),
            retention_ok=np.ones(self._num_envs, dtype=bool),
            no_failure_signal=~terminated,
            stage_valid=np.ones(self._num_envs, dtype=bool),
            stage_duration_progress=np.ones(self._num_envs, dtype=dtype),
        )
        dense_reward = rotate_reward + obj_linvel_reward + position_error_reward + reward_adjustment
        reward = np.asarray(dense_reward * self._cfg.ctrl_dt, dtype=dtype)

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
        )
        support_pose_progress_scale = getattr(
            self._reward_cfg, "support_pose_progress_scale", 0.25
        )
        support_pose_progress_clip = getattr(
            self._reward_cfg, "support_pose_progress_clip", 0.04
        )
        support_pose_progress_reward = (
            support_pose_progress_scale
            * np.clip(
                raw_support_progress,
                -support_pose_progress_clip,
                support_pose_progress_clip,
            )
        )

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
        )

        step_count = np.asarray(
            info.get("steps", np.zeros(self._num_envs, dtype=np.uint32))
        )
        previous_opposition_potential = np.asarray(
            info.get("prev_opposition_potential", current_opposition_potential),
            dtype=dtype,
        )
        first_step_after_reset = step_count == 0
        previous_opposition_potential = np.where(
            first_step_after_reset,
            current_opposition_potential,
            previous_opposition_potential,
        )
        raw_opposition_progress = (
            current_opposition_potential - previous_opposition_potential
        )
        opposition_progress_scale = getattr(
            self._reward_cfg, "opposition_progress_scale", 0.20
        )
        opposition_progress_clip = getattr(
            self._reward_cfg, "opposition_progress_clip", 0.05
        )
        opposition_progress_reward = (
            opposition_progress_scale
            * np.clip(
                raw_opposition_progress,
                -opposition_progress_clip,
                opposition_progress_clip,
            )
        )

        reward += np.asarray(support_pose_progress_reward, dtype=dtype)
        reward += np.asarray(opposition_progress_reward, dtype=dtype)

        info["curr_dof_pos"] = dof_pos.copy()
        info["curr_ball_pos"] = ball_pos.copy()
        info["curr_ball_quat"] = ball_quat.copy()
        info["prev_dof_pos"] = dof_pos.copy()
        info["prev_ball_pos"] = ball_pos.copy()
        info["prev_ball_quat"] = ball_quat.copy()
        info["prev_opposition_potential"] = current_opposition_potential.copy()

        obs = self._compute_sustained_obs(self._all_env_ids, info)
        if self._enable_reward_log and int(step_count[0]) % 4 == 0:
            displacement = ball_pos - anchor_pos
            fingertip_contact_count = np.sum(fingertip_contacts, axis=1)
            palm_contact = self._palm_contacts(self._all_env_ids)
            info["log"] = {
                "reward/rotate_base": float(np.mean(base_rotate_reward)),
                "reward/rotate": float(np.mean(rotate_reward)),
                "reward/finger_participation": float(
                    np.mean(rotate_reward - base_rotate_reward)
                ),
                "reward/obj_linvel": float(np.mean(obj_linvel_reward)),
                "reward/position_error": float(np.mean(position_error_reward)),
                "reward/support_pose_progress": float(
                    np.mean(support_pose_progress_reward)
                ),
                "reward/opposition_progress": float(
                    np.mean(opposition_progress_reward)
                ),
                "reward/total": float(np.mean(reward)),
                "rotation/axis_speed_rad_s": float(np.mean(axis_speed)),
                "rotation/clipped_axis_speed_rad_s": float(np.mean(clipped_axis_speed)),
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
                "termination/workspace_rate": float(np.mean(terminated)),
                "termination/task_rate": float(np.mean(terminated)),
            }
            info["log"].update(reward_adjustment_log)

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
    "compute_support_pose_distance",
    "compute_tip_collision_reference_positions",
]

