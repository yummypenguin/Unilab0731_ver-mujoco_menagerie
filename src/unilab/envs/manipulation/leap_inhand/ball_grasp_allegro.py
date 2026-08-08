"""Allegro-equivalent LEAP ball-grasp generation with final-row deduplication."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.dr import (
    DomainRandomizationProvider,
    GeomSizeOverride,
    InitRandomizationPlan,
    ModelVariantSpec,
)
from unilab.dtype_config import get_global_dtype
from unilab.envs.manipulation.allegro_inhand.grasp_gen import AllegroRotationGrasp
from unilab.envs.manipulation.allegro_inhand.rotation import (
    AllegroRotationDomainRandomizationProvider,
)

from .ball_rotation import LeapInhandBallRotationCfg
from .base import LeapHandBaseEnv


def quantized_grasp_key(
    row: np.ndarray,
    *,
    joint_resolution: float,
    ball_position_resolution: float,
) -> tuple[int, ...]:
    """Quantize settled float32 hand qpos and ball XYZ into a 19D key."""
    candidate = np.asarray(row, dtype=np.float32)
    if candidate.shape != (23,):
        raise ValueError(f"Expected one settled grasp row with shape (23,), got {candidate.shape}")
    if not np.isfinite(candidate).all():
        raise ValueError("Settled grasp row contains non-finite values")
    if not np.isfinite(joint_resolution) or joint_resolution <= 0.0:
        raise ValueError("joint_resolution must be positive and finite")
    if not np.isfinite(ball_position_resolution) or ball_position_resolution <= 0.0:
        raise ValueError("ball_position_resolution must be positive and finite")

    hand_key = np.rint(candidate[:16] / joint_resolution).astype(np.int64)
    ball_key = np.rint(candidate[16:19] / ball_position_resolution).astype(np.int64)
    return tuple(int(value) for value in np.concatenate([hand_key, ball_key]))


def fingertip_surface_gap_mask(
    signed_distances: np.ndarray,
    *,
    max_gap: float,
) -> np.ndarray:
    """Require all four fingertip collision surfaces strictly within ``max_gap``."""
    raw_distances = np.asarray(signed_distances)
    distance_dtype = (
        raw_distances.dtype
        if np.issubdtype(raw_distances.dtype, np.floating)
        else np.dtype(np.float64)
    )
    distances = np.asarray(raw_distances, dtype=distance_dtype)
    if distances.ndim != 2 or distances.shape[1] != 4:
        raise ValueError(f"signed_distances must have shape (?, 4), got {distances.shape}")
    if not np.isfinite(max_gap) or max_gap <= 0.0:
        raise ValueError("max_gap must be positive and finite")
    typed_max_gap = np.asarray(max_gap, dtype=distance_dtype).item()
    surface_gaps = np.maximum(distances, np.asarray(0.0, dtype=distance_dtype))
    return np.asarray(
        np.all(np.isfinite(distances), axis=1) & (np.max(surface_gaps, axis=1) < typed_max_gap),
        dtype=bool,
    )


@registry.envcfg("LeapInhandBallGraspAllegro")
@dataclass
class LeapInhandBallGraspAllegroCfg(LeapInhandBallRotationCfg):
    """Configuration for Allegro-lifecycle LEAP grasp collection."""

    gen_grasp: bool = True
    grasp_cache_path: str = "robots/leap_hand/caches/ball_grasp_allegro_new_physics_0731_50k.npy"
    grasp_collection_target: int = 50_000
    grasp_auto_save: bool = True
    grasp_auto_save_interval: int = 1_000
    grasp_quality_check: bool = True
    grasp_min_contacts: int = 2
    grasp_seed_qpos: list[float] = field(default_factory=list)
    grasp_max_fingertip_distance: float = 0.1
    grasp_max_fingertip_surface_gap: float = 0.005
    termination_drop_distance: float = 0.005
    grasp_dedup_enabled: bool = True
    grasp_dedup_joint_resolution: float = 0.001
    grasp_dedup_ball_position_resolution: float = 0.0005
    # One collection run owns exactly one physical object scale. The output
    # cache path must be selected explicitly by the launch command so scale
    # buckets can never overwrite or silently share one cache.
    object_scale: float = 1.0

    def validate(self) -> None:
        super().validate()
        seed = np.asarray(self.grasp_seed_qpos, dtype=np.float64)
        if seed.shape != (23,):
            raise ValueError(f"grasp_seed_qpos must have shape (23,), got {seed.shape}")
        if not np.isfinite(seed).all():
            raise ValueError("grasp_seed_qpos must contain only finite values")
        if float(np.linalg.norm(seed[19:23])) <= 1e-8:
            raise ValueError("grasp_seed_qpos quaternion must have non-zero length")
        if (
            not np.isfinite(self.grasp_max_fingertip_distance)
            or self.grasp_max_fingertip_distance <= 0.0
        ):
            raise ValueError("grasp_max_fingertip_distance must be positive and finite")
        if self.grasp_collection_target <= 0:
            raise ValueError("grasp_collection_target must be positive")
        if (
            not np.isfinite(self.grasp_max_fingertip_surface_gap)
            or self.grasp_max_fingertip_surface_gap <= 0.0
        ):
            raise ValueError("grasp_max_fingertip_surface_gap must be positive and finite")
        if not np.isfinite(self.termination_drop_distance) or self.termination_drop_distance <= 0.0:
            raise ValueError("termination_drop_distance must be positive and finite")
        if not 0 <= self.grasp_min_contacts <= 4:
            raise ValueError("grasp_min_contacts must be within [0, 4]")
        if (
            not np.isfinite(self.grasp_dedup_joint_resolution)
            or self.grasp_dedup_joint_resolution <= 0.0
        ):
            raise ValueError("grasp_dedup_joint_resolution must be positive and finite")
        if (
            not np.isfinite(self.grasp_dedup_ball_position_resolution)
            or self.grasp_dedup_ball_position_resolution <= 0.0
        ):
            raise ValueError("grasp_dedup_ball_position_resolution must be positive and finite")
        if not np.isfinite(self.object_scale) or self.object_scale <= 0.0:
            raise ValueError("object_scale must be positive and finite")


class LeapAllegroGraspResetProvider(AllegroRotationDomainRandomizationProvider):
    """Sample LEAP hand proposals around the task-owned 23D nominal seed."""

    def build_init_randomization_plan(self, env: Any) -> InitRandomizationPlan:
        """Compile one ball-size model variant for this collection run."""

        scale = float(env.cfg.object_scale)
        base_size = np.asarray(env._backend.get_geom_size("leap_object_col"), dtype=np.float64)
        return InitRandomizationPlan(
            model_assignments=np.zeros(env._num_envs, dtype=np.int32),
            model_variants=(
                ModelVariantSpec(
                    geom_size_overrides=(
                        GeomSizeOverride(
                            geom_name="leap_object_col",
                            size=tuple(np.asarray(base_size * scale, dtype=np.float64)),
                        ),
                    )
                ),
            ),
        )

    def _sample_reset_state(
        self, env: Any, num_reset: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seed = np.asarray(env.cfg.grasp_seed_qpos, dtype=np.float64)
        hand_qpos = np.broadcast_to(seed[:16], (num_reset, 16)).copy()
        joint_noise = float(env.cfg.domain_rand.joint_noise)
        hand_qpos += np.random.uniform(
            -joint_noise,
            joint_noise,
            size=(num_reset, env._NUM_HAND_DOF),
        )
        hand_qpos = np.clip(
            hand_qpos,
            np.asarray(env._ctrl_lower, dtype=np.float64),
            np.asarray(env._ctrl_upper, dtype=np.float64),
        )
        ball_pos = np.broadcast_to(seed[16:19], (num_reset, 3)).copy()
        ball_quat = np.broadcast_to(seed[19:23], (num_reset, 4)).copy()
        qvel = np.zeros((num_reset, env.nv), dtype=np.float64)
        return hand_qpos, ball_pos, ball_quat, qvel

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


@registry.env("LeapInhandBallGraspAllegro", sim_backend="mujoco")
class LeapInhandBallGraspAllegroEnv(AllegroRotationGrasp, LeapHandBaseEnv):
    """Allegro-equivalent physical acceptance plus final-row deduplication."""

    _cfg: LeapInhandBallGraspAllegroCfg
    _CONTACT_SENSORS = (
        "leap_index_contact",
        "leap_middle_contact",
        "leap_ring_contact",
        "leap_thumb_contact",
    )
    _FINGERTIP_GEOMS = (
        "index_tip_col",
        "middle_tip_col",
        "ring_tip_col",
        "thumb_tip_col",
    )

    def __init__(
        self,
        cfg: LeapInhandBallGraspAllegroCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> None:
        self._tip_object_geom_pairs: np.ndarray | None = None
        self._saved_grasp_keys: set[tuple[int, ...]] = set()
        self._dedup_candidates = 0
        self._dedup_rejected = 0
        self._dedup_accepted = 0
        self._serialized_surface_candidates = 0
        self._serialized_surface_rejected = 0
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        object_geom_id = self._backend.get_geom_id("leap_object_col")
        self._tip_object_geom_pairs = np.asarray(
            [(self._backend.get_geom_id(name), object_geom_id) for name in self._FINGERTIP_GEOMS],
            dtype=np.int32,
        )
        self._restore_partial_grasp_cache()

    def _restore_partial_grasp_cache(self) -> None:
        """Resume an interrupted scale-cache collection from its atomic autosave."""

        if not bool(self._cfg.grasp_auto_save):
            return
        cache_file = Path(self._cfg.grasp_cache_path)
        if not cache_file.is_absolute():
            cache_file = ASSETS_ROOT_PATH / cache_file
        if not cache_file.exists():
            return

        rows = np.asarray(np.load(cache_file), dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != self._NUM_HAND_DOF + 7:
            raise ValueError(f"Cannot resume invalid LEAP grasp cache shape {rows.shape}")
        target = int(self._cfg.grasp_collection_target)
        if target > 0:
            rows = rows[:target]
        self._saved_grasping_states = [rows.copy()]
        self._last_grasp_auto_save_total = int(rows.shape[0])
        self._grasp_cache_saved = True
        for row in rows:
            self._saved_grasp_keys.add(
                quantized_grasp_key(
                    row,
                    joint_resolution=float(self._cfg.grasp_dedup_joint_resolution),
                    ball_position_resolution=float(self._cfg.grasp_dedup_ball_position_resolution),
                )
            )
        print(f"[Leap grasp cache] Resumed {rows.shape[0]} rows from {cache_file}")

    def _make_domain_randomization_provider(self) -> DomainRandomizationProvider:
        return LeapAllegroGraspResetProvider()

    def _compute_grasp_conditions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ball_pos = self.get_ball_pos()
        fingertip_pos = self.get_fingertip_pos()

        if self.state is None:
            raise RuntimeError("environment state is unavailable during grasp validation")
        initial_ball_z = np.asarray(
            self.state.info.get("initial_ball_z"),
            dtype=get_global_dtype(),
        )
        if initial_ball_z.shape != (self._num_envs,):
            raise RuntimeError("initial_ball_z must be initialized for every environment at reset")

        cond1 = np.all(
            np.linalg.norm(fingertip_pos - ball_pos[:, None, :], axis=-1)
            < float(self._cfg.grasp_max_fingertip_distance),
            axis=1,
        )
        cond2 = self._contact_count() >= int(self._cfg.grasp_min_contacts)
        drop_threshold = initial_ball_z - float(self._cfg.termination_drop_distance)
        cond3 = ball_pos[:, 2] > drop_threshold
        return (
            np.asarray(cond1, dtype=bool),
            np.asarray(cond2, dtype=bool),
            np.asarray(cond3, dtype=bool),
        )

    def _surface_gap_quality(
        self,
        env_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._tip_object_geom_pairs is None:
            raise RuntimeError("fingertip surface geom pairs were not initialized")
        limit = float(self._cfg.grasp_max_fingertip_surface_gap)
        signed_distances = self._backend.get_geom_pair_distances(
            env_ids,
            self._tip_object_geom_pairs,
            max_distance=max(0.2, 2.0 * limit),
        )
        surface_gaps = np.maximum(np.asarray(signed_distances, dtype=np.float64), 0.0)
        valid = fingertip_surface_gap_mask(signed_distances, max_gap=limit)
        if self.state is not None:
            log = self.state.info.get("log", {})
            log["grasp/max_fingertip_surface_gap"] = float(np.max(surface_gaps, initial=0.0))
            log["grasp/fingertip_surface_valid"] = float(np.mean(valid.astype(np.float32)))
            self.state.info["log"] = log
        return valid, surface_gaps

    def _check_grasp_quality(self, env_ids: np.ndarray) -> np.ndarray:
        physical_valid = super()._check_grasp_quality(env_ids)
        surface_valid, _ = self._surface_gap_quality(env_ids)
        return np.asarray(physical_valid & surface_valid, dtype=bool)

    def _filter_grasp_rows(self, states: np.ndarray) -> np.ndarray:
        rows = np.asarray(states, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != 23:
            raise ValueError(f"Expected settled grasp rows with shape (?, 23), got {rows.shape}")
        if not np.isfinite(rows).all():
            raise ValueError("Settled grasp rows contain non-finite values")

        if self._tip_object_geom_pairs is None:
            raise RuntimeError("fingertip surface geom pairs were not initialized")
        limit = float(self._cfg.grasp_max_fingertip_surface_gap)
        serialized_distances = self._backend.get_geom_pair_distances_for_qpos(
            rows,
            self._tip_object_geom_pairs,
            max_distance=max(0.2, 2.0 * limit),
        )
        serialized_surface_valid = fingertip_surface_gap_mask(
            serialized_distances,
            max_gap=limit,
        )
        self._serialized_surface_candidates += int(rows.shape[0])
        self._serialized_surface_rejected += int(np.count_nonzero(~serialized_surface_valid))
        if not np.all(serialized_surface_valid):
            rows = rows[np.flatnonzero(serialized_surface_valid)]

        self._dedup_candidates += int(rows.shape[0])
        if not self._cfg.grasp_dedup_enabled:
            self._dedup_accepted += int(rows.shape[0])
            filtered = rows
        else:
            keep: list[int] = []
            for index, row in enumerate(rows):
                key = quantized_grasp_key(
                    row,
                    joint_resolution=self._cfg.grasp_dedup_joint_resolution,
                    ball_position_resolution=(self._cfg.grasp_dedup_ball_position_resolution),
                )
                if key in self._saved_grasp_keys:
                    self._dedup_rejected += 1
                    continue
                self._saved_grasp_keys.add(key)
                self._dedup_accepted += 1
                keep.append(index)
            filtered = rows[np.asarray(keep, dtype=np.int64)]

        if self.state is not None:
            log = self.state.info.get("log", {})
            log["grasp/dedup_candidates"] = float(self._dedup_candidates)
            log["grasp/dedup_rejected"] = float(self._dedup_rejected)
            log["grasp/dedup_accepted"] = float(self._dedup_accepted)
            log["grasp/dedup_rejection_rate"] = float(
                self._dedup_rejected / max(self._dedup_candidates, 1)
            )
            log["grasp/serialized_surface_candidates"] = float(self._serialized_surface_candidates)
            log["grasp/serialized_surface_rejected"] = float(self._serialized_surface_rejected)
            log["grasp/serialized_surface_rejection_rate"] = float(
                self._serialized_surface_rejected / max(self._serialized_surface_candidates, 1)
            )
            log["grasp/cache_size"] = float(self._total_saved_grasps())
            self.state.info["log"] = log
        return filtered


LeapInhandBallGraspAllegro = LeapInhandBallGraspAllegroEnv
