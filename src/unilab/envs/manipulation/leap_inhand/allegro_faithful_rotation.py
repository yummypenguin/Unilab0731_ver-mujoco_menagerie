"""Allegro-faithful ball rotation adapted to the LEAP Hand embodiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.dtype_config import get_global_dtype

from .ball_rotation import LeapInhandBallRotationCfg, LeapInhandBallRotationEnv


@registry.envcfg("LeapInhandBallAllegroRotation")
@dataclass
class LeapInhandBallAllegroRotationCfg(LeapInhandBallRotationCfg):
    """LEAP parameters that preserve the successful Allegro reward semantics."""

    sim_dt: float = 0.005
    reset_source: str = "home"


@registry.env("LeapInhandBallAllegroRotation", sim_backend="motrix")
@registry.env("LeapInhandBallAllegroRotation", sim_backend="mujoco")
class LeapInhandBallAllegroRotationEnv(LeapInhandBallRotationEnv):
    """LEAP ball rotation with raw pose cost and palm-contact termination."""

    _cfg: LeapInhandBallAllegroRotationCfg
    def _compute_drop_event(self, ball_pos: np.ndarray) -> np.ndarray:
        return np.asarray(
            ball_pos[:, 2] < self._reward_cfg.reset_z_threshold,
            dtype=bool,
        )

    def _compute_terminated(self, ball_pos: np.ndarray) -> np.ndarray:
        del ball_pos
        return self.get_palm_contact_flags()

    def _reward_drop(
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
        del info, dof_pos, dof_vel, ball_linvel, ball_angvel, torques, terminated
        return np.asarray(
            self._compute_drop_event(ball_pos),
            dtype=get_global_dtype(),
        )

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
        step_count = info.get("steps", np.zeros(self._num_envs, dtype=np.uint32))
        if self._enable_reward_log and int(step_count[0]) % 4 == 0:
            pose_error = dof_pos - np.asarray(info["init_pose"])
            pose_cost = np.sum(np.square(pose_error), axis=1)
            drop_event = self._compute_drop_event(ball_pos)
            log = info.get("log", {})
            threshold = self._reward_cfg.reset_z_threshold
            log["diagnostic/ball_z_mean"] = float(np.mean(ball_pos[:, 2]))
            log["diagnostic/ball_z_min"] = float(np.min(ball_pos[:, 2]))
            log["diagnostic/drop_margin_mean"] = float(np.mean(ball_pos[:, 2] - threshold))
            log["diagnostic/drop_rate"] = float(np.mean(drop_event))
            log["diagnostic/palm_contact_rate"] = float(np.mean(terminated))
            log["diagnostic/termination_rate"] = float(np.mean(terminated))
            log["diagnostic/torque_saturation_fraction"] = float(
                np.mean(np.abs(torques) >= 0.5 - 1e-6)
            )
            log["diagnostic/raw_pose_l2_rms"] = float(np.sqrt(np.mean(pose_cost)))
            info["log"] = log
        return reward


__all__ = [
    "LeapInhandBallAllegroRotationCfg",
    "LeapInhandBallAllegroRotationEnv",
]
