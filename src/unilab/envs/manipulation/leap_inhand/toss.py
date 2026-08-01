"""Curriculum task for thumb-launched assisted cube rebound and capture."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import DomainRandomizationProvider
from unilab.dtype_config import get_global_dtype
from unilab.envs.manipulation.allegro_inhand.base import ControlConfig, NoiseConfig
from unilab.envs.manipulation.allegro_inhand.rotation import (
    AllegroRotationDomainRandomizationProvider,
    AllegroRotationPPO,
    AllegroRotationPPOCfg,
)
from unilab.utils.rotation import (
    np_matrix_first_two_cols_from_quat,
    np_quat_apply,
    np_quat_apply_batched,
    np_quat_apply_inverse,
    np_quat_conjugate_batched,
    np_quat_inv,
    np_quat_mul,
)

from .base import LeapHandBaseEnv


class TossPhase(IntEnum):
    SUPPORT = 0
    FLIGHT = 1
    IMPACT = 2
    RETURN = 3
    CAPTURE = 4


RESET_PHASES = np.asarray(
    [TossPhase.SUPPORT, TossPhase.FLIGHT, TossPhase.RETURN, TossPhase.CAPTURE],
    dtype=np.int8,
)


def curriculum_level(step_counter: int, level_steps: list[int]) -> int:
    """Resolve the active curriculum level from vectorized control steps."""
    if not level_steps:
        return 0
    thresholds = np.asarray(level_steps, dtype=np.int64)
    resolved = np.searchsorted(thresholds, step_counter, side="right") - 1
    return int(np.clip(resolved, 0, len(thresholds) - 1))


def assisted_rebound_candidate(
    nonthumb_contacts: np.ndarray,
    outward_speed: np.ndarray,
    *,
    min_impact_speed: float,
) -> np.ndarray:
    """Identify non-thumb impacts while allowing active finger assistance."""
    return np.any(nonthumb_contacts, axis=1) & (outward_speed >= min_impact_speed)


@dataclass
class TossRewardConfig:
    support_scale: float = 0.5
    guard_scale: float = 0.25
    trajectory_scale: float = 1.0
    orientation_scale: float = 0.1
    return_progress_scale: float = 1.0
    capture_scale: float = 1.0
    action_rate_scale: float = 0.001
    ready_bonus: float = 0.2
    valid_launch_bonus: float = 0.5
    assisted_rebound_bonus: float = 2.0
    stable_capture_bonus: float = 10.0
    failure_penalty: float = 10.0
    trajectory_pos_sigma: float = 0.035
    trajectory_vel_sigma: float = 0.20
    orientation_sigma: float = 1.5
    return_progress_distance: float = 0.02
    capture_pos_sigma: float = 0.025
    capture_vel_sigma: float = 0.08
    guard_distance_sigma: float = 0.08
    thumb_lateral_sigma: float = 0.04
    reset_z_threshold: float = 0.4


@dataclass
class TossCurriculumConfig:
    enabled: bool = True
    level_steps: list[int] = field(default_factory=lambda: [0, 1000, 2500, 4000])
    # Rows are levels; columns are support, flight, return, and capture resets.
    reset_phase_weights: list[list[float]] = field(
        default_factory=lambda: [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.5, 0.35, 0.15],
            [0.7, 0.15, 0.10, 0.05],
        ]
    )
    flight_reset_offset: float = 0.025
    flight_reset_height: float = 0.010
    flight_reset_speed: float = 0.16
    return_reset_offset: float = 0.035
    return_reset_height: float = 0.010
    return_reset_speed: float = 0.12
    capture_pos_jitter: float = 0.004
    capture_vel_jitter: float = 0.02


@registry.envcfg("LeapInhandToss")
@dataclass
class LeapInhandTossCfg(AllegroRotationPPOCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "leap_hand" / "scene_toss.xml")
        )
    )
    reward_config: TossRewardConfig | None = None
    curriculum: TossCurriculumConfig = field(default_factory=TossCurriculumConfig)
    max_episode_seconds: float = 15.0
    control_config: ControlConfig = field(
        default_factory=lambda: ControlConfig(action_scale=1.0 / 24.0, kp=3.0, kd=0.01)
    )
    noise_config: NoiseConfig = field(default_factory=lambda: NoiseConfig(level=0.0))
    grasp_cache_path: str = "robots/leap_hand/caches/cube_grasp_s10_1k.npy"
    palm_normal_local: tuple[float, float, float] = (0.0, 0.0, -1.0)
    fallback_toss_axis_local: tuple[float, float, float] = (0.0, -1.0, 0.0)
    target_forward_speed: float = 0.20
    target_apex_fraction: float = 0.20
    gravity_magnitude: float = 9.81
    min_launch_forward_speed: float = 0.08
    min_launch_up_speed: float = 0.10
    min_impact_speed: float = 0.05
    min_rebound_speed: float = 0.03
    ready_cube_speed: float = 0.05
    ready_guard_distance: float = 0.09
    capture_radius: float = 0.040
    capture_max_speed: float = 0.08
    capture_max_angvel: float = 1.0
    capture_dwell_seconds: float = 0.40
    workspace_radius: float = 0.15
    support_timeout_seconds: float = 5.0
    flight_timeout_seconds: float = 1.2
    impact_timeout_seconds: float = 0.5
    return_timeout_seconds: float = 2.5
    capture_timeout_seconds: float = 3.0

    def validate(self) -> None:
        super().validate()
        if self.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        if len(self.curriculum.level_steps) != len(self.curriculum.reset_phase_weights):
            raise ValueError("curriculum level_steps and reset_phase_weights must have equal length")
        for weights in self.curriculum.reset_phase_weights:
            if len(weights) != len(RESET_PHASES) or any(weight < 0.0 for weight in weights):
                raise ValueError("each curriculum weight row must contain four non-negative values")
            if sum(weights) <= 0.0:
                raise ValueError("each curriculum weight row must have positive total weight")
        if not 0.0 < self.target_apex_fraction <= 1.0:
            raise ValueError("target_apex_fraction must be in (0, 1]")


class LeapTossResetProvider(AllegroRotationDomainRandomizationProvider):
    def build_reset_plan(self, env: Any, env_ids: np.ndarray):
        plan = super().build_reset_plan(env, env_ids)
        phases = env._sample_reset_phases(len(env_ids))
        capture_pos = plan.qpos[:, 16:19].copy()
        ball_pos = plan.qpos[:, 16:19]
        ball_vel = plan.qvel[:, 16:19]
        toss_axis = env._toss_axis_w[env_ids]
        palm_normal = env._palm_normal_w[env_ids]
        curriculum = env.cfg.curriculum

        flight = phases == int(TossPhase.FLIGHT)
        ball_pos[flight] += (
            toss_axis[flight] * curriculum.flight_reset_offset
            + palm_normal[flight] * curriculum.flight_reset_height
        )
        ball_vel[flight] = toss_axis[flight] * curriculum.flight_reset_speed

        returning = phases == int(TossPhase.RETURN)
        ball_pos[returning] += (
            toss_axis[returning] * curriculum.return_reset_offset
            + palm_normal[returning] * curriculum.return_reset_height
        )
        ball_vel[returning] = -toss_axis[returning] * curriculum.return_reset_speed

        capture = phases == int(TossPhase.CAPTURE)
        if np.any(capture):
            ball_pos[capture] += np.random.uniform(
                -curriculum.capture_pos_jitter,
                curriculum.capture_pos_jitter,
                (int(np.sum(capture)), 3),
            )
            ball_vel[capture] = np.random.uniform(
                -curriculum.capture_vel_jitter,
                curriculum.capture_vel_jitter,
                (int(np.sum(capture)), 3),
            )

        info_updates = self._build_info_updates(
            env,
            plan.qpos[:, :16],
            plan.qpos[:, 16:19],
            plan.qpos[:, 19:23],
        )
        info_updates.update(
            env._build_toss_reset_info(
                phases=phases,
                capture_pos=capture_pos,
                ball_pos=plan.qpos[:, 16:19],
            )
        )
        plan.info_updates = info_updates
        return plan

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        env._initialize_reset_kinematics(env_ids, info_updates)
        return cast(dict[str, np.ndarray], env._compute_toss_obs(env_ids, info_updates))


@registry.env("LeapInhandToss", sim_backend="motrix")
@registry.env("LeapInhandToss", sim_backend="mujoco")
class LeapInhandTossEnv(AllegroRotationPPO, LeapHandBaseEnv):
    """LEAP Hand task for a thumb toss, assisted finger rebound, and recapture."""

    _cfg: LeapInhandTossCfg
    _reward_cfg: TossRewardConfig
    _NUM_TOSS_OBS = 75
    _CONTACT_SENSOR_NAMES = (
        "leap_index_contact",
        "leap_middle_contact",
        "leap_ring_contact",
        "leap_thumb_contact",
    )

    def __init__(
        self, cfg: LeapInhandTossCfg, num_envs: int = 1, backend_type: str = "mujoco"
    ) -> None:
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._all_env_ids = np.arange(num_envs, dtype=np.int32)
        self._palm_body_ids = self._backend.get_body_ids([self._BASE_BODY_NAME])
        self._initialize_task_geometry()

    def _make_domain_randomization_provider(self) -> DomainRandomizationProvider:
        return LeapTossResetProvider()

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": self._NUM_TOSS_OBS}

    def _initialize_task_geometry(self) -> None:
        palm_pos = self._backend.get_body_pos_w(self._palm_body_ids)[:, 0, :]
        palm_quat = self._backend.get_body_quat_w(self._palm_body_ids)[:, 0, :]
        fingertip_pos = self.get_fingertip_pos()

        local_normal = np.asarray(self._cfg.palm_normal_local, dtype=self._np_dtype)
        local_normal /= np.linalg.norm(local_normal)
        palm_normal = np_quat_apply(palm_quat, local_normal)
        palm_normal /= np.linalg.norm(palm_normal, axis=1, keepdims=True)

        finger_direction = np.mean(fingertip_pos[:, :3, :], axis=1) - palm_pos
        finger_direction -= (
            np.sum(finger_direction * palm_normal, axis=1, keepdims=True) * palm_normal
        )
        direction_norm = np.linalg.norm(finger_direction, axis=1, keepdims=True)
        fallback_local = np.asarray(self._cfg.fallback_toss_axis_local, dtype=self._np_dtype)
        fallback_local /= np.linalg.norm(fallback_local)
        fallback_world = np_quat_apply(palm_quat, fallback_local)
        toss_axis = np.where(
            direction_norm > 1e-6,
            finger_direction / np.maximum(direction_norm, 1e-6),
            fallback_world,
        )
        toss_axis /= np.linalg.norm(toss_axis, axis=1, keepdims=True)

        hand_lengths = np.linalg.norm(fingertip_pos[:, :3, :] - palm_pos[:, None, :], axis=2)
        self._nominal_hand_length = float(np.min(hand_lengths[0]))
        self._palm_normal_w = np.asarray(palm_normal, dtype=self._np_dtype)
        self._toss_axis_w = np.asarray(toss_axis, dtype=self._np_dtype)
        target_apex = self._cfg.target_apex_fraction * self._nominal_hand_length
        self._target_up_speed = float(np.sqrt(2.0 * self._cfg.gravity_magnitude * target_apex))

    def _sample_reset_phases(self, num_reset: int) -> np.ndarray:
        curriculum = self._cfg.curriculum
        if not curriculum.enabled:
            return np.full(num_reset, int(TossPhase.SUPPORT), dtype=np.int8)
        level = curriculum_level(self.step_counter, curriculum.level_steps)
        weights = np.asarray(curriculum.reset_phase_weights[level], dtype=np.float64)
        weights /= np.sum(weights)
        return np.random.choice(RESET_PHASES, size=num_reset, p=weights).astype(np.int8)

    def _build_toss_reset_info(
        self, *, phases: np.ndarray, capture_pos: np.ndarray, ball_pos: np.ndarray
    ) -> dict[str, np.ndarray]:
        num_reset = len(phases)
        distance = np.linalg.norm(ball_pos - capture_pos, axis=1)
        return {
            "toss_phase": phases.copy(),
            "toss_phase_steps": np.zeros(num_reset, dtype=np.uint32),
            "toss_capture_pos": capture_pos.copy(),
            "toss_launch_pos": ball_pos.copy(),
            "toss_max_rise": np.zeros(num_reset, dtype=self._np_dtype),
            "toss_prev_capture_distance": distance.astype(self._np_dtype),
            "toss_ready_awarded": phases != int(TossPhase.SUPPORT),
            "toss_launch_awarded": phases != int(TossPhase.SUPPORT),
            "toss_rebound_awarded": phases >= int(TossPhase.RETURN),
            "toss_impact_candidate": np.zeros(num_reset, dtype=bool),
            "toss_had_thumb_contact": np.zeros(num_reset, dtype=bool),
            "toss_stable_steps": np.zeros(num_reset, dtype=np.uint32),
            "toss_success": np.zeros(num_reset, dtype=bool),
            "toss_failure": np.zeros(num_reset, dtype=bool),
        }

    def _initialize_reset_kinematics(
        self, env_ids: np.ndarray, info: dict[str, Any]
    ) -> None:
        info["prev_ball_pos"] = self.get_ball_pos()[env_ids].copy()
        info["prev_ball_quat"] = self.get_ball_quat()[env_ids].copy()

    def _contacts(self, env_ids: np.ndarray) -> np.ndarray:
        contacts = self._backend.get_sensor_data_batch(self._CONTACT_SENSOR_NAMES)
        return np.asarray(contacts[env_ids] > 0.5, dtype=bool)

    def _palm_pose(self, env_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos = self._backend.get_body_pos_w(self._palm_body_ids)[env_ids, 0, :]
        quat = self._backend.get_body_quat_w(self._palm_body_ids)[env_ids, 0, :]
        return pos, quat

    def _compute_toss_obs(
        self, env_ids: np.ndarray, info: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        dtype = get_global_dtype()
        dof_pos = self.get_hand_dof_pos()[env_ids]
        targets = np.asarray(info["prev_ctrl"], dtype=dtype)
        ball_pos = self.get_ball_pos()[env_ids]
        ball_quat = self.get_ball_quat()[env_ids]
        ball_linvel = self.get_ball_linvel()[env_ids]
        ball_angvel = self.get_ball_angvel()[env_ids]
        fingertip_pos = self.get_fingertip_pos()[env_ids]
        palm_pos, palm_quat = self._palm_pose(env_ids)

        dof_pos_norm = 2.0 * (dof_pos - self._dof_mid) / (self._dof_range + 1e-8)
        ball_rel_pos = np_quat_apply_inverse(palm_quat, ball_pos - palm_pos)
        ball_rel_quat = np_quat_mul(np_quat_inv(palm_quat), ball_quat)
        ball_rot_6d = np_matrix_first_two_cols_from_quat(ball_rel_quat)
        linvel_palm = np_quat_apply_inverse(palm_quat, ball_linvel)
        angvel_palm = np_quat_apply_inverse(palm_quat, ball_angvel)
        fingertip_rel_world = fingertip_pos - ball_pos[:, None, :]
        fingertip_rel_palm = np_quat_apply_batched(
            np_quat_conjugate_batched(palm_quat)[:, None, :], fingertip_rel_world
        ).reshape(len(env_ids), -1)
        contacts = self._contacts(env_ids).astype(dtype)
        phases = np.asarray(info["toss_phase"], dtype=np.intp)
        phase_one_hot = np.eye(len(TossPhase), dtype=dtype)[phases]
        toss_axis_palm = np_quat_apply_inverse(palm_quat, self._toss_axis_w[env_ids])
        palm_normal_palm = np_quat_apply_inverse(palm_quat, self._palm_normal_w[env_ids])
        phase_elapsed = (
            np.asarray(info["toss_phase_steps"], dtype=dtype)[:, None]
            * self._cfg.ctrl_dt
            / max(self._cfg.support_timeout_seconds, 1e-6)
        )

        obs = np.concatenate(
            [
                dof_pos_norm,
                targets,
                ball_rel_pos,
                ball_rot_6d,
                linvel_palm,
                angvel_palm,
                fingertip_rel_palm,
                contacts,
                phase_one_hot,
                toss_axis_palm,
                palm_normal_palm,
                phase_elapsed,
            ],
            axis=1,
            dtype=dtype,
        )
        return {"obs": np.asarray(obs, dtype=dtype)}

    def _phase_timeout_steps(self) -> np.ndarray:
        seconds = np.asarray(
            [
                self._cfg.support_timeout_seconds,
                self._cfg.flight_timeout_seconds,
                self._cfg.impact_timeout_seconds,
                self._cfg.return_timeout_seconds,
                self._cfg.capture_timeout_seconds,
            ],
            dtype=np.float64,
        )
        return np.maximum(1, np.ceil(seconds / self._cfg.ctrl_dt)).astype(np.uint32)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        info = state.info
        phase = np.asarray(info["toss_phase"], dtype=np.int8)
        phase_before = phase.copy()
        phase_steps = np.asarray(info["toss_phase_steps"], dtype=np.uint32)
        phase_steps += 1
        reward_phase_steps = phase_steps.copy()

        ball_pos = self.get_ball_pos()
        ball_quat = self.get_ball_quat()
        ball_linvel = self.get_ball_linvel()
        ball_angvel = self.get_ball_angvel()
        fingertip_pos = self.get_fingertip_pos()
        contacts = self._contacts(self._all_env_ids)
        nonthumb_contacts = contacts[:, :3]
        thumb_contact = contacts[:, 3]
        toss_axis = self._toss_axis_w
        palm_normal = self._palm_normal_w
        outward_speed = np.sum(ball_linvel * toss_axis, axis=1)
        up_speed = np.sum(ball_linvel * palm_normal, axis=1)
        cube_speed = np.linalg.norm(ball_linvel, axis=1)
        capture_pos = np.asarray(info["toss_capture_pos"])
        capture_distance = np.linalg.norm(ball_pos - capture_pos, axis=1)

        thumb_to_cube = ball_pos - fingertip_pos[:, 3, :]
        thumb_height = np.sum(thumb_to_cube * palm_normal, axis=1)
        thumb_lateral = thumb_to_cube - thumb_height[:, None] * palm_normal
        thumb_lateral_error = np.sum(np.square(thumb_lateral), axis=1)
        guard_distances = np.linalg.norm(fingertip_pos[:, :3, :] - ball_pos[:, None, :], axis=2)

        ready = (
            thumb_contact
            & (thumb_height > 0.0)
            & (cube_speed <= self._cfg.ready_cube_speed)
            & (np.min(guard_distances, axis=1) <= self._cfg.ready_guard_distance)
        )
        ready_event = (
            (phase_before == int(TossPhase.SUPPORT))
            & ready
            & ~np.asarray(info["toss_ready_awarded"])
        )
        info["toss_ready_awarded"] |= ready_event

        launch_event = (
            (phase_before == int(TossPhase.SUPPORT))
            & np.asarray(info["toss_had_thumb_contact"])
            & ~thumb_contact
            & (outward_speed >= self._cfg.min_launch_forward_speed)
            & (up_speed >= self._cfg.min_launch_up_speed)
        )
        if np.any(launch_event):
            phase[launch_event] = int(TossPhase.FLIGHT)
            phase_steps[launch_event] = 0
            info["toss_launch_pos"][launch_event] = ball_pos[launch_event]
            info["toss_launch_awarded"][launch_event] = True

        impact_event = (phase_before == int(TossPhase.FLIGHT)) & np.any(
            nonthumb_contacts, axis=1
        )
        impact_candidate = assisted_rebound_candidate(
            nonthumb_contacts,
            outward_speed,
            min_impact_speed=self._cfg.min_impact_speed,
        )
        if np.any(impact_event):
            phase[impact_event] = int(TossPhase.IMPACT)
            phase_steps[impact_event] = 0
            info["toss_impact_candidate"][impact_event] = impact_candidate[impact_event]

        rebound_event = (
            (phase_before == int(TossPhase.IMPACT))
            & np.asarray(info["toss_impact_candidate"])
            & ~np.asarray(info["toss_rebound_awarded"])
            & (outward_speed <= -self._cfg.min_rebound_speed)
        )
        if np.any(rebound_event):
            phase[rebound_event] = int(TossPhase.RETURN)
            phase_steps[rebound_event] = 0
            info["toss_rebound_awarded"][rebound_event] = True

        capture_entry = (
            (phase_before == int(TossPhase.RETURN))
            & (capture_distance <= self._cfg.capture_radius)
            & (cube_speed <= self._cfg.capture_max_speed)
        )
        if np.any(capture_entry):
            phase[capture_entry] = int(TossPhase.CAPTURE)
            phase_steps[capture_entry] = 0

        capture_stable = (
            (phase == int(TossPhase.CAPTURE))
            & (capture_distance <= self._cfg.capture_radius)
            & (cube_speed <= self._cfg.capture_max_speed)
            & (np.linalg.norm(ball_angvel, axis=1) <= self._cfg.capture_max_angvel)
            & thumb_contact
            & np.any(nonthumb_contacts, axis=1)
        )
        stable_steps = np.asarray(info["toss_stable_steps"], dtype=np.uint32)
        stable_steps[capture_stable] += 1
        stable_steps[~capture_stable] = 0
        dwell_steps = max(1, int(np.ceil(self._cfg.capture_dwell_seconds / self._cfg.ctrl_dt)))
        success_event = capture_stable & (stable_steps >= dwell_steps)

        launch_pos = np.asarray(info["toss_launch_pos"])
        rise = np.sum((ball_pos - launch_pos) * palm_normal, axis=1)
        max_rise = np.asarray(info["toss_max_rise"])
        np.maximum(max_rise, rise, out=max_rise)
        workspace_escape = capture_distance > self._cfg.workspace_radius
        apex_escape = max_rise > self._nominal_hand_length
        dropped = ball_pos[:, 2] < self._reward_cfg.reset_z_threshold
        timeout_limit = self._phase_timeout_steps()[phase.astype(np.intp)]
        phase_timeout = phase_steps >= timeout_limit
        failure = (workspace_escape | apex_escape | dropped | phase_timeout) & ~success_event

        reward = self._compute_toss_reward(
            info=info,
            phase_before=phase_before,
            phase_steps=reward_phase_steps,
            ball_pos=ball_pos,
            ball_linvel=ball_linvel,
            ball_angvel=ball_angvel,
            contacts=contacts,
            thumb_height=thumb_height,
            thumb_lateral_error=thumb_lateral_error,
            guard_distances=guard_distances,
            capture_distance=capture_distance,
            ready_event=ready_event,
            launch_event=launch_event,
            rebound_event=rebound_event,
            success_event=success_event,
            failure=failure,
        )

        info["toss_phase"] = phase
        info["toss_phase_steps"] = phase_steps
        info["toss_prev_capture_distance"] = capture_distance.astype(self._np_dtype)
        info["toss_had_thumb_contact"] |= thumb_contact
        info["toss_success"] = success_event
        info["toss_failure"] = failure
        info["curr_dof_pos"] = self.get_hand_dof_pos().copy()
        info["curr_ball_pos"] = ball_pos.copy()
        info["curr_ball_quat"] = ball_quat.copy()
        info["prev_ball_pos"] = ball_pos.copy()
        info["prev_ball_quat"] = ball_quat.copy()

        obs = self._compute_toss_obs(self._all_env_ids, info)
        terminated = np.asarray(success_event | failure, dtype=bool)
        self._log_toss_metrics(
            info,
            reward,
            phase,
            ready_event,
            launch_event,
            rebound_event,
            success_event,
            failure,
        )
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_toss_reward(
        self,
        *,
        info: dict[str, Any],
        phase_before: np.ndarray,
        phase_steps: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        contacts: np.ndarray,
        thumb_height: np.ndarray,
        thumb_lateral_error: np.ndarray,
        guard_distances: np.ndarray,
        capture_distance: np.ndarray,
        ready_event: np.ndarray,
        launch_event: np.ndarray,
        rebound_event: np.ndarray,
        success_event: np.ndarray,
        failure: np.ndarray,
    ) -> np.ndarray:
        cfg = self._reward_cfg
        dtype = get_global_dtype()
        dense = np.zeros(self._num_envs, dtype=dtype)
        support = phase_before == int(TossPhase.SUPPORT)
        flight = phase_before == int(TossPhase.FLIGHT)
        returning = phase_before == int(TossPhase.RETURN)
        capture = phase_before == int(TossPhase.CAPTURE)

        thumb_score = (
            contacts[:, 3].astype(dtype)
            * (thumb_height > 0.0).astype(dtype)
            * np.exp(-thumb_lateral_error / max(cfg.thumb_lateral_sigma**2, 1e-8))
            * np.exp(-np.sum(np.square(ball_linvel), axis=1) / max(cfg.capture_vel_sigma**2, 1e-8))
        )
        guard_score = np.mean(
            np.exp(-np.square(guard_distances) / max(cfg.guard_distance_sigma**2, 1e-8)),
            axis=1,
        )
        dense += support * (cfg.support_scale * thumb_score + cfg.guard_scale * guard_score)

        flight_time = np.asarray(phase_steps, dtype=dtype) * self._cfg.ctrl_dt
        launch_pos = np.asarray(info["toss_launch_pos"])
        reference_pos = (
            launch_pos
            + self._toss_axis_w * (self._cfg.target_forward_speed * flight_time[:, None])
            + self._palm_normal_w
            * (
                self._target_up_speed * flight_time[:, None]
                - 0.5 * self._cfg.gravity_magnitude * np.square(flight_time[:, None])
            )
        )
        reference_vel = (
            self._toss_axis_w * self._cfg.target_forward_speed
            + self._palm_normal_w
            * (self._target_up_speed - self._cfg.gravity_magnitude * flight_time)[:, None]
        )
        pos_error = np.sum(np.square(ball_pos - reference_pos), axis=1)
        vel_error = np.sum(np.square(ball_linvel - reference_vel), axis=1)
        trajectory_score = np.exp(-pos_error / max(cfg.trajectory_pos_sigma**2, 1e-8)) * np.exp(
            -vel_error / max(cfg.trajectory_vel_sigma**2, 1e-8)
        )
        orientation_score = np.exp(
            -np.sum(np.square(ball_angvel), axis=1) / max(cfg.orientation_sigma**2, 1e-8)
        )
        dense += flight * (
            cfg.trajectory_scale * trajectory_score + cfg.orientation_scale * orientation_score
        )

        previous_distance = np.asarray(info["toss_prev_capture_distance"])
        return_progress = np.clip(
            (previous_distance - capture_distance) / max(cfg.return_progress_distance, 1e-8),
            0.0,
            1.0,
        )
        dense += returning * (
            cfg.return_progress_scale * return_progress
            + cfg.orientation_scale * orientation_score
        )

        capture_score = (
            np.exp(-np.square(capture_distance) / max(cfg.capture_pos_sigma**2, 1e-8))
            * np.exp(
                -np.sum(np.square(ball_linvel), axis=1) / max(cfg.capture_vel_sigma**2, 1e-8)
            )
            * orientation_score
            * (0.5 + 0.25 * contacts[:, 3] + 0.25 * np.any(contacts[:, :3], axis=1))
        )
        dense += capture * cfg.capture_scale * capture_score

        current_actions = np.asarray(info["current_actions"])
        last_actions = np.asarray(info["last_actions"])
        dense -= cfg.action_rate_scale * np.sum(np.square(current_actions - last_actions), axis=1)

        reward = dense * self._cfg.ctrl_dt
        reward += cfg.ready_bonus * ready_event
        reward += cfg.valid_launch_bonus * launch_event
        reward += cfg.assisted_rebound_bonus * rebound_event
        reward += cfg.stable_capture_bonus * success_event
        reward -= cfg.failure_penalty * failure
        return np.asarray(reward, dtype=dtype)

    def _log_toss_metrics(
        self,
        info: dict[str, Any],
        reward: np.ndarray,
        phase: np.ndarray,
        ready: np.ndarray,
        launch: np.ndarray,
        rebound: np.ndarray,
        success: np.ndarray,
        failure: np.ndarray,
    ) -> None:
        step_count = info.get("steps", np.zeros(self._num_envs, dtype=np.uint32))
        if not self._enable_reward_log or int(step_count[0]) % 4 != 0:
            return
        log = info.get("log", {})
        log["reward/total"] = float(np.mean(reward))
        log["toss/curriculum_level"] = float(
            curriculum_level(self.step_counter, self._cfg.curriculum.level_steps)
        )
        for task_phase in TossPhase:
            log[f"toss/phase_{task_phase.name.lower()}"] = float(
                np.mean(phase == int(task_phase))
            )
        log["toss/ready_event"] = float(np.mean(ready))
        log["toss/launch_event"] = float(np.mean(launch))
        log["toss/rebound_event"] = float(np.mean(rebound))
        log["toss/success_event"] = float(np.mean(success))
        log["toss/failure_event"] = float(np.mean(failure))
        log["toss/max_rise"] = float(np.mean(info["toss_max_rise"]))
        log["toss/hand_length_limit"] = self._nominal_hand_length
        info["log"] = log


LeapInhandToss = LeapInhandTossEnv
