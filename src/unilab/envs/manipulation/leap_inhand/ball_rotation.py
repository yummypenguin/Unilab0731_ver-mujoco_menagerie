"""LEAP Hand in-hand rotation task using the LEAP-owned ball."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.manipulation.allegro_inhand.base import ControlConfig
from unilab.envs.manipulation.allegro_inhand.rotation import (
    AllegroRotationDomainRandomizationProvider,
    AllegroRotationPPO,
    AllegroRotationPPOCfg,
)

from .base import LeapHandBaseEnv


@registry.envcfg("LeapInhandBallRotation")
@dataclass
class LeapInhandBallRotationCfg(AllegroRotationPPOCfg):
    """Use the established rotation behavior with LEAP model and ball assets."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "leap_hand" / "scene_ball.xml")
        )
    )
    sim_dt: float = 1.0 / 120.0
    control_config: ControlConfig = field(
        default_factory=lambda: ControlConfig(action_scale=1.0 / 24.0, kp=3.0, kd=0.1)
    )
    grasp_cache_path: str = "robots/leap_hand/caches/ball_grasp_s10_5k.npy"
    reset_source: str = "cache"

    def validate(self) -> None:
        super().validate()
        if self.reset_source not in {"home", "cache"}:
            raise ValueError("reset_source must be 'home' or 'cache'")


class LeapBallRotationResetProvider(AllegroRotationDomainRandomizationProvider):
    """Select the fixed scene home or the retained LEAP ball cache."""

    def _load_grasp_cache(self, env: Any) -> np.ndarray | None:
        if env.cfg.reset_source == "cache":
            return super()._load_grasp_cache(env)
        if env.cfg.reset_source != "home":
            raise ValueError("reset_source must be 'home' or 'cache'")
        env._grasp_cache = None
        env._grasp_cache_loaded = True
        return None

    def _sample_reset_state(
        self, env: Any, num_reset: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        hand_qpos, ball_pos, ball_quat, qvel = super()._sample_reset_state(env, num_reset)
        if env.cfg.reset_source == "home":
            home_quat = env._init_qpos[env._NUM_HAND_DOF + 3 : env._NUM_HAND_DOF + 7]
            ball_quat = np.broadcast_to(home_quat, (num_reset, 4)).copy()
        return hand_qpos, ball_pos, ball_quat, qvel


@registry.env("LeapInhandBallRotation", sim_backend="motrix")
@registry.env("LeapInhandBallRotation", sim_backend="mujoco")
class LeapInhandBallRotationEnv(AllegroRotationPPO, LeapHandBaseEnv):
    """LEAP model binding for the in-hand ball-rotation behavior."""

    _cfg: LeapInhandBallRotationCfg

    def _make_domain_randomization_provider(self) -> LeapBallRotationResetProvider:
        return LeapBallRotationResetProvider()
