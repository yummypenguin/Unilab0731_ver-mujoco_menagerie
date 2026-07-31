from __future__ import annotations

from dataclasses import dataclass, field

from unilab.terrains.terrain_generator import TerrainGeneratorCfg


@dataclass
class JointDynamicsCfg:
    """Cold-path joint dynamics overrides selected by joint name."""

    joint_names: list[str] = field(default_factory=list)
    damping: float | None = None
    frictionloss: float | None = None
    armature: float | None = None


@dataclass
class TerrainSceneCfg:
    """Backend-agnostic terrain slot declaration for a scene."""

    generator: TerrainGeneratorCfg | None = None
    hfield_name: str = "terrain_hfield"
    geom_name: str | None = None


@dataclass
class SceneCfg:
    """Scene source and optional cold-path composition configuration."""

    model_file: str
    fragment_files: list[str] = field(default_factory=list)
    terrain: TerrainSceneCfg | None = None
    joint_dynamics: JointDynamicsCfg | None = None
    # Optional render-only model override. When set, offline playback/video
    # export renders this XML instead of ``model_file`` while physics keeps
    # using ``model_file``. Used to give the renderer a visual twin of the
    # scene (e.g. a per-env replicable obstacle) without touching the trained
    # collision model. ``None`` => render with ``model_file`` (unchanged).
    visual_model_file: str | None = None
