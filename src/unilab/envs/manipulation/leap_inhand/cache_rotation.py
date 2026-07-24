"""Independent LEAP ball-rotation task backed by the official grasp cache."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unilab.base import registry

from .ball_rotation import (
    LeapBallRotationResetProvider,
    LeapInhandBallRotationCfg,
    LeapInhandBallRotationEnv,
)


@registry.envcfg("LeapInhandBallCacheRotation")
@dataclass
class LeapInhandBallCacheRotationCfg(LeapInhandBallRotationCfg):
    """Independent cache-reset rotation configuration."""

    reset_source: str = "cache"
    grasp_cache_path: str = "robots/leap_hand/caches/ball_grasp_official_50k.npy"
    termination_drop_distance: float = 0.007

    def validate(self) -> None:
        super().validate()
        if not np.isfinite(self.termination_drop_distance) or (
            self.termination_drop_distance <= 0.0
        ):
            raise ValueError("termination_drop_distance must be positive and finite")


class LeapInhandBallCacheResetProvider(LeapBallRotationResetProvider):
    """Record each cache sample's ball center as its termination reference."""

    def build_reset_plan(self, env, env_ids):
        plan = super().build_reset_plan(env, env_ids)
        env._termination_initial_ball_pos[env_ids] = plan.qpos[:, 16:19]
        return plan


@registry.env("LeapInhandBallCacheRotation", sim_backend="motrix")
@registry.env("LeapInhandBallCacheRotation", sim_backend="mujoco")
class LeapInhandBallCacheRotationEnv(LeapInhandBallRotationEnv):
    """Rotate the LEAP ball until its center drops 7 mm from reset."""

    _cfg: LeapInhandBallCacheRotationCfg

    def __init__(
        self,
        cfg: LeapInhandBallCacheRotationCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> None:
        self._termination_initial_ball_pos = np.zeros((num_envs, 3), dtype=np.float64)
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)

    def _make_domain_randomization_provider(self) -> LeapInhandBallCacheResetProvider:
        return LeapInhandBallCacheResetProvider()

    def _compute_terminated(self, ball_pos: np.ndarray) -> np.ndarray:
        vertical_drop = self._termination_initial_ball_pos[:, 2] - ball_pos[:, 2]
        return np.asarray(vertical_drop >= self._cfg.termination_drop_distance, dtype=bool)


__all__ = [
    "LeapInhandBallCacheRotationCfg",
    "LeapInhandBallCacheRotationEnv",
    "LeapInhandBallCacheResetProvider",
]
