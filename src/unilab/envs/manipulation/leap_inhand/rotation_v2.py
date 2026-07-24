"""LEAP ball rotation V2 with observable spin and performance-gated targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dr import DomainRandomizationProvider
from unilab.dtype_config import get_global_dtype
from unilab.envs.manipulation.allegro_inhand.rotation import (
    AllegroRotationDomainRandomizationProvider,
    compute_pd_torques,
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

from .ball_rotation import LeapInhandBallRotationCfg, LeapInhandBallRotationEnv


@dataclass
class RotationV2RewardConfig:
    axis_progress_scale: float = 2.0
    directional_quality_scale: float = 0.5
    retention_scale: float = 0.25
    action_rate_scale: float = 0.001
    torque_scale: float = 0.01
    work_scale: float = 0.10
    sustained_rotation_bonus: float = 0.20
    quarter_turn_bonus: float = 0.25
    drop_penalty: float = 5.0
    retention_sigma: float = 0.03
    orthogonal_speed_tolerance: float = 0.50
    reset_z_threshold: float = 0.40
    workspace_radius: float = 0.12


@dataclass
class RotationV2CurriculumConfig:
    target_speeds: list[float] = field(default_factory=lambda: [0.10, 0.25, 0.50])
    sustain_ratio: float = 0.60
    sustain_steps: int = 5
    energy_level_scales: list[float] = field(default_factory=lambda: [0.0, 0.25, 1.0])


def compute_rotation_terms(
    ball_angvel: np.ndarray,
    rotation_axis: np.ndarray,
    target_speed: np.ndarray,
    orthogonal_speed_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return signed axis progress and direction-aware spin quality."""
    axis_speed = np.sum(ball_angvel * rotation_axis, axis=1)
    orthogonal = ball_angvel - axis_speed[:, None] * rotation_axis
    orthogonal_speed = np.linalg.norm(orthogonal, axis=1)
    axis_progress = np.clip(axis_speed / np.maximum(target_speed, 1e-6), -1.0, 1.0)
    directional_quality = np.maximum(axis_progress, 0.0) * np.exp(
        -np.square(orthogonal_speed / max(orthogonal_speed_tolerance, 1e-6))
    )
    return axis_speed, orthogonal_speed, axis_progress, directional_quality


class LeapBallRotationV2ResetProvider(AllegroRotationDomainRandomizationProvider):
    """Initialize V2 episode progress without changing the shared ball cache."""

    def build_reset_plan(self, env: Any, env_ids: np.ndarray):
        plan = super().build_reset_plan(env, env_ids)
        num_reset = len(env_ids)
        num_levels = len(env.cfg.curriculum.target_speeds)
        updates = dict(plan.info_updates or {})
        updates.pop("obs_lag_history", None)
        updates.update(
            {
                "rotation_anchor_pos": np.asarray(
                    plan.qpos[:, env._NUM_HAND_DOF : env._NUM_HAND_DOF + 3],
                    dtype=get_global_dtype(),
                ).copy(),
                "rotation_level": np.zeros(num_reset, dtype=np.int8),
                "rotation_sustain_steps": np.zeros(num_reset, dtype=np.uint32),
                "rotation_sustain_awarded": np.zeros((num_reset, num_levels), dtype=bool),
                "rotation_signed_progress": np.zeros(num_reset, dtype=get_global_dtype()),
                "rotation_highwater": np.zeros(num_reset, dtype=get_global_dtype()),
                "rotation_milestone_index": np.zeros(num_reset, dtype=np.int32),
                "rotation_axis_speed_ema": np.zeros(num_reset, dtype=get_global_dtype()),
            }
        )
        plan.info_updates = updates
        return plan

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        return cast(dict[str, np.ndarray], env._compute_v2_obs(env_ids, info_updates))


@registry.envcfg("LeapInhandBallRotationV2")
@dataclass
class LeapInhandBallRotationV2Cfg(LeapInhandBallRotationCfg):
    reward_config: RotationV2RewardConfig | None = None
    curriculum: RotationV2CurriculumConfig = field(default_factory=RotationV2CurriculumConfig)
    joint_velocity_scale: float = 0.20

    def validate(self) -> None:
        super().validate()
        if self.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        speeds = self.curriculum.target_speeds
        if not speeds or any(speed <= 0.0 for speed in speeds):
            raise ValueError("curriculum target_speeds must contain positive values")
        if any(right <= left for left, right in zip(speeds, speeds[1:])):
            raise ValueError("curriculum target_speeds must be strictly increasing")
        if len(self.curriculum.energy_level_scales) != len(speeds):
            raise ValueError("energy_level_scales must match target_speeds")
        if self.curriculum.sustain_steps <= 0:
            raise ValueError("curriculum sustain_steps must be positive")


@registry.env("LeapInhandBallRotationV2", sim_backend="motrix")
@registry.env("LeapInhandBallRotationV2", sim_backend="mujoco")
class LeapInhandBallRotationV2Env(LeapInhandBallRotationEnv):
    """LEAP-specific rotation learner that observes and rewards actual ball spin."""

    _cfg: LeapInhandBallRotationV2Cfg
    _reward_cfg: RotationV2RewardConfig
    _NUM_V2_OBS = 99
    _CONTACT_SENSOR_NAMES = (
        "leap_index_contact",
        "leap_middle_contact",
        "leap_ring_contact",
        "leap_thumb_contact",
    )

    def __init__(
        self, cfg: LeapInhandBallRotationV2Cfg, num_envs: int = 1, backend_type: str = "mujoco"
    ) -> None:
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._all_env_ids = np.arange(num_envs, dtype=np.int32)
        self._palm_body_ids = self._backend.get_body_ids([self._BASE_BODY_NAME])
        self._rotation_axis_w = normalize_rotation_axis(cfg.rotation_axis)
        self._target_speeds = np.asarray(cfg.curriculum.target_speeds, dtype=self._np_dtype)
        self._energy_level_scales = np.asarray(
            cfg.curriculum.energy_level_scales, dtype=self._np_dtype
        )

    def _make_domain_randomization_provider(self) -> DomainRandomizationProvider:
        return LeapBallRotationV2ResetProvider()

    def _init_reward_functions(self) -> None:
        self._reward_fns = {}

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": self._NUM_V2_OBS}

    def _palm_pose(self, env_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        palm_pos = self._backend.get_body_pos_w(self._palm_body_ids)[env_ids, 0, :]
        palm_quat = self._backend.get_body_quat_w(self._palm_body_ids)[env_ids, 0, :]
        return palm_pos, palm_quat

    def _contacts(self, env_ids: np.ndarray) -> np.ndarray:
        contacts = self._backend.get_sensor_data_batch(self._CONTACT_SENSOR_NAMES)
        return np.asarray(contacts[env_ids] > 0.5, dtype=self._np_dtype)

    def _target_speed(self, info: dict[str, Any]) -> np.ndarray:
        levels = np.asarray(info["rotation_level"], dtype=np.intp)
        return self._target_speeds[levels]

    def _compute_v2_obs(
        self, env_ids: np.ndarray, info: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        dtype = get_global_dtype()
        dof_pos = self.get_hand_dof_pos()[env_ids]
        dof_vel = self.get_hand_dof_vel()[env_ids]
        targets = np.asarray(info["prev_ctrl"], dtype=dtype)
        last_actions = np.asarray(info["last_actions"], dtype=dtype)
        ball_pos = self.get_ball_pos()[env_ids]
        ball_quat = self.get_ball_quat()[env_ids]
        ball_linvel = self.get_ball_linvel()[env_ids]
        ball_angvel = self.get_ball_angvel()[env_ids]
        fingertip_pos = self.get_fingertip_pos()[env_ids]
        palm_pos, palm_quat = self._palm_pose(env_ids)

        dof_pos_norm = 2.0 * (dof_pos - self._dof_mid) / (self._dof_range + 1e-8)
        noise_cfg = self._cfg.noise_config
        if noise_cfg.level > 0.0:
            dof_pos_norm += (
                np.random.uniform(-1.0, 1.0, dof_pos_norm.shape).astype(dtype)
                * noise_cfg.level
                * noise_cfg.scale_joint_angle
            )
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
        target_axis_w = np.broadcast_to(self._rotation_axis_w, (len(env_ids), 3))
        target_axis_palm = np_quat_apply_inverse(
            palm_quat,
            target_axis_w,
        )
        target_speed = self._target_speed(info)[:, None] / self._target_speeds[-1]

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
                self._contacts(env_ids),
                target_speed,
            ],
            axis=1,
            dtype=dtype,
        )
        return {"obs": np.asarray(obs, dtype=dtype)}

    def update_state(self, state: NpEnvState) -> NpEnvState:
        info = state.info
        dtype = get_global_dtype()
        dof_pos = self.get_hand_dof_pos()
        dof_vel = self.get_hand_dof_vel()
        ball_pos = self.get_ball_pos()
        ball_quat = self.get_ball_quat()
        ball_angvel = self.get_ball_angvel()
        rotation_axis = np.broadcast_to(
            self._rotation_axis_w, (self._num_envs, 3)
        )

        levels = np.asarray(info["rotation_level"], dtype=np.int8)
        levels_before = levels.copy()
        target_speed = self._target_speeds[levels_before]
        axis_speed, orthogonal_speed, axis_progress, directional_quality = (
            compute_rotation_terms(
                ball_angvel,
                rotation_axis,
                target_speed,
                self._reward_cfg.orthogonal_speed_tolerance,
            )
        )

        anchor_pos = np.asarray(info["rotation_anchor_pos"], dtype=dtype)
        position_error = np.linalg.norm(ball_pos - anchor_pos, axis=1)
        retention = np.exp(
            -np.square(position_error / max(self._reward_cfg.retention_sigma, 1e-6))
        )
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

        dense = (
            self._reward_cfg.axis_progress_scale * axis_progress
            + self._reward_cfg.directional_quality_scale * directional_quality
            + self._reward_cfg.retention_scale * retention
            - self._reward_cfg.action_rate_scale * action_rate
            - energy_scale * self._reward_cfg.torque_scale * torque_cost
            - energy_scale * self._reward_cfg.work_scale * work_cost
        )

        signed_progress = np.asarray(info["rotation_signed_progress"], dtype=dtype)
        signed_progress += axis_speed * self._cfg.ctrl_dt
        highwater = np.maximum(
            np.asarray(info["rotation_highwater"], dtype=dtype), signed_progress
        )
        axis_speed_ema = np.asarray(info["rotation_axis_speed_ema"], dtype=dtype)
        ema_alpha = 1.0 - np.exp(-self._cfg.ctrl_dt)
        axis_speed_ema += ema_alpha * (axis_speed - axis_speed_ema)
        milestone_index = np.asarray(info["rotation_milestone_index"], dtype=np.int32)
        next_milestone_index = np.floor(
            np.maximum(highwater, 0.0) / (0.5 * np.pi)
        ).astype(np.int32)
        milestone_events = np.maximum(next_milestone_index - milestone_index, 0)

        sustain_steps = np.asarray(info["rotation_sustain_steps"], dtype=np.uint32)
        sustained = (
            (axis_speed >= self._cfg.curriculum.sustain_ratio * target_speed)
            & (orthogonal_speed <= self._reward_cfg.orthogonal_speed_tolerance)
        )
        sustain_steps[:] = np.where(sustained, sustain_steps + 1, 0)
        awarded = np.asarray(info["rotation_sustain_awarded"], dtype=bool)
        rows = np.arange(self._num_envs)
        sustain_event = (
            sustain_steps >= self._cfg.curriculum.sustain_steps
        ) & ~awarded[rows, levels_before]
        event_rows = np.flatnonzero(sustain_event)
        awarded[event_rows, levels_before[event_rows]] = True
        promote = sustain_event & (levels_before < len(self._target_speeds) - 1)
        levels[promote] += 1
        sustain_steps[promote] = 0

        dropped = (ball_pos[:, 2] < self._reward_cfg.reset_z_threshold) | (
            position_error > self._reward_cfg.workspace_radius
        )
        reward = dense * self._cfg.ctrl_dt
        reward += self._reward_cfg.sustained_rotation_bonus * sustain_event
        reward += self._reward_cfg.quarter_turn_bonus * milestone_events
        reward -= self._reward_cfg.drop_penalty * dropped

        info["rotation_level"] = levels
        info["rotation_sustain_steps"] = sustain_steps
        info["rotation_sustain_awarded"] = awarded
        info["rotation_signed_progress"] = signed_progress
        info["rotation_highwater"] = highwater
        info["rotation_milestone_index"] = next_milestone_index
        info["rotation_axis_speed_ema"] = axis_speed_ema
        info["curr_dof_pos"] = dof_pos.copy()
        info["curr_ball_pos"] = ball_pos.copy()
        info["curr_ball_quat"] = ball_quat.copy()

        step_count = info.get("steps", np.zeros(self._num_envs, dtype=np.uint32))
        if self._enable_reward_log and int(step_count[0]) % 4 == 0:
            info["log"] = {
                "reward/axis_progress": float(
                    np.mean(self._reward_cfg.axis_progress_scale * axis_progress)
                ),
                "reward/directional_quality": float(
                    np.mean(self._reward_cfg.directional_quality_scale * directional_quality)
                ),
                "reward/retention": float(
                    np.mean(self._reward_cfg.retention_scale * retention)
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
                "reward/total": float(np.mean(reward)),
                "rotation/axis_speed_rad_s": float(np.mean(axis_speed)),
                "rotation/axis_speed_1s_mean": float(np.mean(axis_speed_ema)),
                "rotation/orthogonal_speed_rad_s": float(np.mean(orthogonal_speed)),
                "rotation/positive_progress_rad": float(np.mean(np.maximum(highwater, 0.0))),
                "rotation/completed_turns": float(np.mean(highwater / (2.0 * np.pi))),
                "rotation/sustained_fraction": float(np.mean(sustained)),
                "rotation/quarter_turn_events": float(np.mean(milestone_events)),
                "rotation/drop_rate": float(np.mean(dropped)),
                "rotation/ball_position_error": float(np.mean(position_error)),
                "curriculum/level": float(np.mean(levels)),
                "curriculum/target_speed_rad_s": float(
                    np.mean(self._target_speeds[levels])
                ),
            }

        obs = self._compute_v2_obs(self._all_env_ids, info)
        return state.replace(
            obs=obs,
            reward=np.asarray(reward, dtype=dtype),
            terminated=np.asarray(dropped, dtype=bool),
        )


__all__ = [
    "LeapInhandBallRotationV2Cfg",
    "LeapInhandBallRotationV2Env",
    "RotationV2CurriculumConfig",
    "RotationV2RewardConfig",
    "compute_rotation_terms",
]
