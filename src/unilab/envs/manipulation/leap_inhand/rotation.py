"""PPO task for rotating a dedicated LEAP Hand cube."""

from __future__ import annotations

from dataclasses import dataclass, field

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.manipulation.allegro_inhand.base import ControlConfig, NoiseConfig
from unilab.envs.manipulation.allegro_inhand.rotation import (
    AllegroRotationPPO,
    AllegroRotationPPOCfg,
)

from .base import LeapHandBaseEnv


@registry.envcfg("LeapInhandRotation")
@dataclass
class LeapInhandRotationCfg(AllegroRotationPPOCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "leap_hand" / "scene.xml")
        )
    )
    control_config: ControlConfig = field(
        default_factory=lambda: ControlConfig(action_scale=1.0 / 24.0, kp=3.0, kd=0.1)
    )
    noise_config: NoiseConfig = field(default_factory=lambda: NoiseConfig(level=0.0))
    grasp_cache_path: str = "robots/leap_hand/caches/cube_grasp_s10_1k.npy"


@registry.env("LeapInhandRotation", sim_backend="motrix")
@registry.env("LeapInhandRotation", sim_backend="mujoco")
class LeapInhandRotationEnv(AllegroRotationPPO, LeapHandBaseEnv):
    """LEAP model binding for the shared in-hand rotation implementation."""

    _cfg: LeapInhandRotationCfg
