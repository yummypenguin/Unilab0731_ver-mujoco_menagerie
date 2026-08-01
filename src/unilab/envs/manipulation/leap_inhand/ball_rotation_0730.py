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
)

from .base import LeapHandBaseEnv


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
    termination_drop_distance: float = 0.005

    def validate(self) -> None:
        super().validate()
        if (
            not np.isfinite(self.termination_drop_distance)
            or self.termination_drop_distance <= 0.0
        ):
            raise ValueError("termination_drop_distance must be positive and finite")


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
        updates["initial_ball_z"] = np.asarray(
            ball_pos[:, 2],
            dtype=get_global_dtype(),
        ).copy()
        return updates


@registry.env("LeapInhandBall0730Rotation", sim_backend="mujoco")
class LeapInhandBall0730RotationEnv(AllegroRotationPPO, LeapHandBaseEnv):
    """Allegro reward logic with a per-reset 5 mm LEAP drop condition."""

    _cfg: LeapInhandBall0730RotationCfg

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


__all__ = [
    "LeapBall0730ResetProvider",
    "LeapInhandBall0730RotationCfg",
    "LeapInhandBall0730RotationEnv",
]
