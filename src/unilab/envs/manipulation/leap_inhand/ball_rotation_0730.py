"""Independent LEAP ball-rotation task with Allegro reward semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.manipulation.allegro_inhand.base import ControlConfig, NoiseConfig
from unilab.envs.manipulation.allegro_inhand.rotation import (
    AllegroRotationDomainRandomizationProvider,
    AllegroRotationPPO,
    AllegroRotationPPOCfg,
    RewardConfigPPO,
)

from .base import LeapHandBaseEnv

_CACHE_GENERATION_NOMINAL_HAND_QPOS = (
    1.5152045040427635,
    0.11430147259750476,
    0.2876406730815961,
    0.19280835997306603,
    1.4188457206477074,
    0.025681830807677088,
    -0.26717932336688344,
    0.5369823550831088,
    1.5294890485315962,
    -0.01798386011739139,
    0.27558019211759954,
    0.19821762108233876,
    1.9245445859343515,
    0.04788276935232176,
    -0.021885380331691334,
    0.19524630120127295,
)

_MIDDLE_CONTACT_SENSOR_NAMES = ("leap_middle_contact",)


def apply_middle_contact_rotation_share(
    weighted_rotate_reward: np.ndarray,
    middle_contact: np.ndarray,
    *,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Require middle-finger contact for a fraction of positive spin reward.

    Negative rotation is intentionally unchanged: missing middle-finger contact
    must not make rotation in the wrong direction less costly.
    """
    missing_contact = 1.0 - np.asarray(middle_contact, dtype=weighted_rotate_reward.dtype)
    adjustment = -fraction * np.maximum(weighted_rotate_reward, 0.0) * missing_contact
    return weighted_rotate_reward + adjustment, adjustment


@dataclass
class Leap0730RewardConfig(RewardConfigPPO):
    """Allegro reward with a configurable middle-finger rotation share."""

    middle_contact_rotation_fraction: float = 0.2


@registry.envcfg("LeapInhandBall0730Rotation")
@dataclass
class LeapInhandBall0730RotationCfg(AllegroRotationPPOCfg):
    """LEAP embodiment with cache-relative drop termination."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "leap_hand" / "scene_ball.xml")
        )
    )
    sim_dt: float = 0.005
    ctrl_dt: float = 0.05
    max_episode_seconds: float = 20.0
    control_config: ControlConfig = field(
        default_factory=lambda: ControlConfig(action_scale=1.0 / 24.0, kp=3.0, kd=0.01)
    )
    noise_config: NoiseConfig = field(default_factory=lambda: NoiseConfig(level=0.0))
    gen_grasp: bool = False
    grasp_cache_path: str = (
        "robots/leap_hand/caches/ball_grasp_allegro_new_physics_0731_50k.npy"
    )
    pose_diff_target_qpos: list[float] = field(
        default_factory=lambda: list(_CACHE_GENERATION_NOMINAL_HAND_QPOS)
    )
    termination_drop_distance: float = 0.03
    reward_config: Leap0730RewardConfig | None = None

    def validate(self) -> None:
        super().validate()
        pose_diff_target = np.asarray(self.pose_diff_target_qpos, dtype=np.float64)
        if pose_diff_target.shape != (16,):
            raise ValueError(
                "pose_diff_target_qpos must contain exactly 16 LEAP hand joint angles"
            )
        if not np.isfinite(pose_diff_target).all():
            raise ValueError("pose_diff_target_qpos must contain only finite values")
        if (
            not np.isfinite(self.termination_drop_distance)
            or self.termination_drop_distance <= 0.0
        ):
            raise ValueError("termination_drop_distance must be positive and finite")
        if self.reward_config is not None:
            fraction = self.reward_config.middle_contact_rotation_fraction
            if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    "middle_contact_rotation_fraction must be finite and within [0, 1]"
                )


class LeapBall0730ResetProvider(AllegroRotationDomainRandomizationProvider):
    """Retain the selected cache row's ball height as the episode drop anchor."""

    def _build_info_updates(
        self,
        env: Any,
        hand_qpos: np.ndarray,
        ball_pos: np.ndarray,
        ball_quat: np.ndarray,
    ) -> dict[str, np.ndarray]:
        updates = super()._build_info_updates(env, hand_qpos, ball_pos, ball_quat)
        pose_diff_target = np.asarray(
            env.cfg.pose_diff_target_qpos,
            dtype=get_global_dtype(),
        )
        updates["init_pose"] = np.broadcast_to(
            pose_diff_target,
            hand_qpos.shape,
        ).copy()
        updates["initial_ball_z"] = np.asarray(
            ball_pos[:, 2],
            dtype=get_global_dtype(),
        ).copy()
        return updates


@registry.env("LeapInhandBall0730Rotation", sim_backend="mujoco")
class LeapInhandBall0730RotationEnv(AllegroRotationPPO, LeapHandBaseEnv):
    """Allegro reward logic with a per-reset 30 mm LEAP drop condition."""

    enable_training_episode_diagnostics = True
    _cfg: LeapInhandBall0730RotationCfg
    _reward_cfg: Leap0730RewardConfig

    def _make_domain_randomization_provider(self) -> LeapBall0730ResetProvider:
        return LeapBall0730ResetProvider()

    def _compute_terminated(self, ball_pos: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("environment state is unavailable during termination")
        initial_ball_z = np.asarray(
            self.state.info.get("initial_ball_z"),
            dtype=get_global_dtype(),
        )
        if initial_ball_z.shape != (self._num_envs,):
            raise RuntimeError(
                "initial_ball_z must be initialized for every environment at reset"
            )
        threshold = initial_ball_z - float(self._cfg.termination_drop_distance)
        return np.asarray(ball_pos[:, 2] <= threshold, dtype=bool)

    def _compute_reward(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        reward = super()._compute_reward(
            info,
            dof_pos,
            dof_vel,
            ball_pos,
            ball_linvel,
            ball_angvel,
            torques,
            terminated,
        )
        sensor_data = self._backend.get_sensor_data_batch(_MIDDLE_CONTACT_SENSOR_NAMES)
        middle_contact = np.asarray(sensor_data[:, 0] > 0.5, dtype=get_global_dtype())
        clipped_axis_speed = np.clip(
            ball_angvel @ self._rot_axis,
            self._reward_cfg.angvel_clip_min,
            self._reward_cfg.angvel_clip_max,
        )
        weighted_rotate = clipped_axis_speed * self._reward_cfg.scales.get("rotate", 0.0)
        _, adjustment = apply_middle_contact_rotation_share(
            weighted_rotate,
            middle_contact,
            fraction=self._reward_cfg.middle_contact_rotation_fraction,
        )
        reward = reward + adjustment * self._cfg.ctrl_dt

        step_count = info.get("steps", np.zeros(self._num_envs, dtype=np.uint32))
        if self._enable_reward_log and int(step_count[0]) % 4 == 0:
            log = info.setdefault("log", {})
            log["contact/middle_rate"] = float(np.mean(middle_contact))
            log["reward/middle_contact_rotation_adjustment"] = float(np.mean(adjustment))
            log["reward/total"] = float(np.mean(reward / self._cfg.ctrl_dt))

        return np.asarray(reward, dtype=get_global_dtype())


__all__ = [
    "Leap0730RewardConfig",
    "LeapBall0730ResetProvider",
    "LeapInhandBall0730RotationCfg",
    "LeapInhandBall0730RotationEnv",
    "apply_middle_contact_rotation_share",
]
