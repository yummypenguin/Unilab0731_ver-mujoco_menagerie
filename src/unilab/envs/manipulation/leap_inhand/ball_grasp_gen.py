"""LEAP ball-grasp cache generation with backend-neutral stability gates."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.base.backend.base import ContactPenetrationDetail
from unilab.base.np_env import NpEnvState
from unilab.dr import DomainRandomizationProvider
from unilab.envs.manipulation.allegro_inhand.grasp_gen import AllegroRotationGrasp
from unilab.envs.manipulation.allegro_inhand.rotation import (
    AllegroRotationDomainRandomizationProvider,
    AllegroRotationPPO,
    compute_pd_torques,
)

from .ball_rotation import LeapInhandBallRotationCfg
from .base import LeapHandBaseEnv


def normalize_grasp_cache_rows(states: np.ndarray) -> np.ndarray:
    """Validate and normalize LEAP cache rows without changing their layout."""
    rows = np.asarray(states, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 23:
        raise ValueError(f"Expected LEAP grasp cache shape (?, 23), got {rows.shape}")
    if not np.isfinite(rows).all():
        raise ValueError("LEAP grasp cache contains non-finite values")

    rows = rows.copy()
    quat_norm = np.linalg.norm(rows[:, 19:23], axis=1, keepdims=True)
    if np.any(quat_norm <= 1e-8):
        raise ValueError("LEAP grasp cache contains a zero-length quaternion")
    rows[:, 19:23] /= quat_norm
    return rows


def grasp_cache_row_key(row: np.ndarray) -> tuple[int, ...]:
    """Return the training-reset identity key for one LEAP ball grasp row."""
    candidate = normalize_grasp_cache_rows(np.asarray(row).reshape(1, -1))[0]
    # Quaternion is irrelevant for a sphere; key on hand pose and ball position.
    quantized = np.concatenate(
        [np.rint(candidate[:16] / 1e-3), np.rint(candidate[16:19] / 5e-4)]
    ).astype(np.int64)
    return tuple(int(value) for value in quantized)


def deduplicate_grasp_cache_rows(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first row for each reset-equivalent quantized grasp."""
    rows = normalize_grasp_cache_rows(states)
    keep: list[int] = []
    seen: set[tuple[int, ...]] = set()
    for index, row in enumerate(rows):
        key = grasp_cache_row_key(row)
        if key in seen:
            continue
        seen.add(key)
        keep.append(index)
    indices = np.asarray(keep, dtype=np.int64)
    return rows[indices], indices


def save_grasp_cache_atomic(
    path: str | Path,
    states: np.ndarray,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically publish a normalized cache after validation has completed."""
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing grasp cache: {output}")
    rows = normalize_grasp_cache_rows(states)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


@dataclass(frozen=True)
class GraspCacheReplayResult:
    """Per-row evidence from one full cache reload and settling pass."""

    settle_steps: int
    settle_seconds: float
    accepted: np.ndarray
    terminated_during_settle: np.ndarray
    final_quality: np.ndarray
    penetration_valid: np.ndarray
    conditions: dict[str, np.ndarray]
    contacts: np.ndarray
    measurements: dict[str, np.ndarray]
    self_penetration: np.ndarray
    object_penetration: np.ndarray


def resolve_grasp_proposal_center(
    canonical_qpos: np.ndarray,
    configured_qpos: list[float],
) -> np.ndarray:
    """Resolve an optional generator-owned seed without reading another cache."""
    canonical = np.asarray(canonical_qpos, dtype=np.float64)
    if canonical.shape != (23,):
        raise ValueError(f"Expected canonical qpos shape (23,), got {canonical.shape}")
    if not configured_qpos:
        return canonical.copy()
    return normalize_grasp_cache_rows(
        np.asarray(configured_qpos, dtype=np.float64).reshape(1, -1)
    )[0].astype(np.float64)


def sample_bounded_joint_offsets(
    num_samples: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Sample per-joint asymmetric proposal offsets."""
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if low.shape != (16,) or high.shape != (16,):
        raise ValueError("joint offset bounds must both have shape (16,)")
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("joint offset bounds must be finite")
    if np.any(low > high):
        raise ValueError("joint offset lower bounds must not exceed upper bounds")
    return np.random.uniform(low, high, size=(num_samples, 16))


def penetration_quality_mask(
    self_depths: np.ndarray,
    object_depths: np.ndarray,
    *,
    max_self_depth: float,
    max_object_depth: float,
) -> np.ndarray:
    """Accept cache candidates only when both penetration limits are met."""
    self_values = np.asarray(self_depths, dtype=np.float64)
    object_values = np.asarray(object_depths, dtype=np.float64)
    if self_values.shape != object_values.shape:
        raise ValueError(
            "self_depths and object_depths must have the same shape, "
            f"got {self_values.shape} and {object_values.shape}"
        )
    if max_self_depth < 0.0 or max_object_depth < 0.0:
        raise ValueError("Penetration limits must be non-negative")
    return np.asarray(
        np.isfinite(self_values)
        & np.isfinite(object_values)
        & (self_values <= max_self_depth)
        & (object_values <= max_object_depth),
        dtype=bool,
    )


def fingertip_surface_gap_quality_mask(
    signed_distances: np.ndarray,
    *,
    max_gap: float,
) -> np.ndarray:
    """Accept rows whose four fingertip collision surfaces stay near the ball."""
    distances = np.asarray(signed_distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[1] != 4:
        raise ValueError(
            "signed_distances must have shape (?, 4), "
            f"got {distances.shape}"
        )
    if not np.isfinite(max_gap) or max_gap < 0.0:
        raise ValueError("max_gap must be non-negative and finite")
    surface_gaps = np.maximum(distances, 0.0)
    return np.asarray(
        np.all(np.isfinite(distances), axis=1)
        & (np.max(surface_gaps, axis=1) <= max_gap),
        dtype=bool,
    )


def build_canonical_grasp_proposals(
    canonical_qpos: np.ndarray,
    joint_offsets: np.ndarray,
    ball_offsets: np.ndarray,
    *,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build grasp proposals around the task-owned canonical keyframe."""
    canonical = np.asarray(canonical_qpos, dtype=np.float64)
    joint_delta = np.asarray(joint_offsets, dtype=np.float64)
    ball_delta = np.asarray(ball_offsets, dtype=np.float64)
    lower = np.asarray(joint_lower, dtype=np.float64)
    upper = np.asarray(joint_upper, dtype=np.float64)

    if canonical.shape != (23,):
        raise ValueError(f"Expected canonical qpos shape (23,), got {canonical.shape}")
    if joint_delta.ndim != 2 or joint_delta.shape[1] != 16:
        raise ValueError(f"Expected joint_offsets shape (?, 16), got {joint_delta.shape}")
    if ball_delta.shape != (joint_delta.shape[0], 3):
        raise ValueError(
            "ball_offsets must match the proposal count with shape (?, 3), "
            f"got {ball_delta.shape}"
        )
    if lower.shape != (16,) or upper.shape != (16,):
        raise ValueError("joint_lower and joint_upper must both have shape (16,)")

    hand_qpos = np.clip(canonical[None, :16] + joint_delta, lower, upper)
    ball_pos = canonical[None, 16:19] + ball_delta
    ball_quat = np.broadcast_to(canonical[None, 19:23], (joint_delta.shape[0], 4)).copy()
    return hand_qpos, ball_pos, ball_quat


def select_frontier_rows(
    rows: np.ndarray, scores: np.ndarray, *, capacity: int
) -> tuple[np.ndarray, np.ndarray]:
    """Retain the lowest-penetration unique candidates for local refinement."""
    candidates = normalize_grasp_cache_rows(rows)
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (candidates.shape[0],):
        raise ValueError(f"Expected scores shape ({candidates.shape[0]},), got {values.shape}")
    if capacity <= 0:
        raise ValueError("frontier capacity must be positive")

    order = np.argsort(values, kind="stable")
    selected_rows: list[np.ndarray] = []
    selected_scores: list[float] = []
    seen: set[tuple[int, ...]] = set()
    for index in order:
        row = candidates[index]
        key = tuple(np.rint(row[:19] / 1e-4).astype(np.int64))
        if key in seen:
            continue
        seen.add(key)
        selected_rows.append(row)
        selected_scores.append(float(values[index]))
        if len(selected_rows) >= capacity:
            break
    if not selected_rows:
        return np.empty((0, 23), dtype=np.float32), np.empty((0,), dtype=np.float64)
    return np.stack(selected_rows).astype(np.float32), np.asarray(selected_scores)


def build_joint_coordinate_probes(
    hand_qpos: np.ndarray,
    coordinate_indices: np.ndarray,
    delta_magnitudes: np.ndarray,
    *,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build baseline and signed finite-difference probes within joint limits."""
    base = np.asarray(hand_qpos, dtype=np.float64)
    indices = np.asarray(coordinate_indices, dtype=np.intp).reshape(-1)
    magnitudes = np.asarray(delta_magnitudes, dtype=np.float64).reshape(-1)
    lower = np.asarray(joint_lower, dtype=np.float64)
    upper = np.asarray(joint_upper, dtype=np.float64)
    if base.shape != (16,) or lower.shape != (16,) or upper.shape != (16,):
        raise ValueError("hand_qpos and joint limits must have shape (16,)")
    if np.any(indices < 0) or np.any(indices >= 16):
        raise ValueError("coordinate_indices must be within [0, 16)")
    if magnitudes.size == 0 or np.any(~np.isfinite(magnitudes)) or np.any(magnitudes <= 0.0):
        raise ValueError("delta_magnitudes must contain positive finite values")

    probes = [np.clip(base, lower, upper)]
    probe_indices = [-1]
    applied_deltas = [0.0]
    for index in indices:
        for magnitude in magnitudes:
            for sign in (-1.0, 1.0):
                probe = probes[0].copy()
                requested = sign * float(magnitude)
                probe[index] = np.clip(probe[index] + requested, lower[index], upper[index])
                probes.append(probe)
                probe_indices.append(int(index))
                applied_deltas.append(float(probe[index] - probes[0][index]))
    return (
        np.stack(probes),
        np.asarray(probe_indices, dtype=np.int32),
        np.asarray(applied_deltas, dtype=np.float64),
    )


@registry.envcfg("LeapInhandBallGrasp")
@dataclass
class LeapInhandBallGraspCfg(LeapInhandBallRotationCfg):
    """Configuration for randomized search of stable LEAP ball grasps."""

    gen_grasp: bool = True
    grasp_cache_path: str = "robots/leap_hand/caches/ball_grasp_official_50k.npy"
    grasp_collection_target: int = 50_000
    grasp_auto_save: bool = True
    grasp_quality_check: bool = True
    grasp_min_contacts: int = 2
    grasp_require_thumb_contact: bool = True
    grasp_seed_qpos: list[float] = field(default_factory=list)
    grasp_joint_offset_lower: list[float] = field(default_factory=list)
    grasp_joint_offset_upper: list[float] = field(default_factory=list)
    grasp_joint_noise: float = 0.10
    grasp_ball_position_noise: float = 0.015
    grasp_frontier_capacity: int = 256
    grasp_frontier_fraction: float = 0.75
    grasp_frontier_joint_noise: float = 0.03
    grasp_frontier_ball_position_noise: float = 0.001
    grasp_warmup_seconds: float = 0.5
    grasp_max_fingertip_distance: float = 0.1
    # A 0.05 mm serialization margin keeps float32 cache rows within the
    # externally audited 10 mm surface-gap contract.
    grasp_max_fingertip_surface_gap: float | None = 0.00995
    grasp_max_ball_drift: float = 0.005
    grasp_max_ball_linear_speed: float = 0.05
    grasp_max_ball_angular_speed: float = 0.5
    grasp_max_joint_speed: float = 0.5
    grasp_max_abs_work: float = 0.25
    grasp_max_self_penetration: float | None = None
    grasp_max_object_penetration: float | None = None

    def validate(self) -> None:
        super().validate()
        if self.grasp_seed_qpos:
            resolve_grasp_proposal_center(np.zeros(23), self.grasp_seed_qpos)
        lower = np.asarray(self.grasp_joint_offset_lower, dtype=np.float64)
        upper = np.asarray(self.grasp_joint_offset_upper, dtype=np.float64)
        if (lower.size == 0) != (upper.size == 0):
            raise ValueError(
                "grasp_joint_offset_lower and grasp_joint_offset_upper must both be set or empty"
            )
        if lower.size:
            sample_bounded_joint_offsets(0, lower, upper)
        if self.grasp_joint_noise < 0.0 or self.grasp_ball_position_noise < 0.0:
            raise ValueError("grasp proposal noise values must be non-negative")
        surface_gap = self.grasp_max_fingertip_surface_gap
        if surface_gap is not None and (
            not np.isfinite(surface_gap) or surface_gap < 0.0
        ):
            raise ValueError(
                "grasp_max_fingertip_surface_gap must be non-negative and finite or null"
            )
        if self.grasp_frontier_capacity <= 0:
            raise ValueError("grasp_frontier_capacity must be positive")
        if not 0.0 <= self.grasp_frontier_fraction <= 1.0:
            raise ValueError("grasp_frontier_fraction must be in [0, 1]")
        if self.grasp_frontier_joint_noise < 0.0:
            raise ValueError("grasp_frontier_joint_noise must be non-negative")
        if self.grasp_frontier_ball_position_noise < 0.0:
            raise ValueError("grasp_frontier_ball_position_noise must be non-negative")


class LeapBallGraspDomainRandomizationProvider(AllegroRotationDomainRandomizationProvider):
    """Sample perturbed proposals from the LEAP ball task's canonical keyframe."""

    def _sample_reset_state(
        self, env: LeapInhandBallGraspEnv, num_reset: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = env.cfg
        canonical_qpos = resolve_grasp_proposal_center(
            np.asarray(env._init_qpos[:23], dtype=np.float64),
            cfg.grasp_seed_qpos,
        )
        if cfg.grasp_joint_offset_lower:
            joint_offsets = sample_bounded_joint_offsets(
                num_reset,
                np.asarray(cfg.grasp_joint_offset_lower, dtype=np.float64),
                np.asarray(cfg.grasp_joint_offset_upper, dtype=np.float64),
            )
        else:
            joint_offsets = np.random.uniform(
                -cfg.grasp_joint_noise,
                cfg.grasp_joint_noise,
                (num_reset, env._NUM_HAND_DOF),
            )
        ball_offsets = np.random.uniform(
            -cfg.grasp_ball_position_noise,
            cfg.grasp_ball_position_noise,
            (num_reset, 3),
        )
        hand_qpos, ball_pos, ball_quat = build_canonical_grasp_proposals(
            canonical_qpos,
            joint_offsets,
            ball_offsets,
            joint_lower=env._ctrl_lower,
            joint_upper=env._ctrl_upper,
        )

        frontier_count = min(
            len(env._grasp_frontier_rows),
            int(round(num_reset * cfg.grasp_frontier_fraction)),
        )
        if frontier_count > 0:
            indices = np.random.randint(0, len(env._grasp_frontier_rows), size=frontier_count)
            seeds = np.asarray(env._grasp_frontier_rows[indices], dtype=np.float64)
            hand_qpos[:frontier_count] = np.clip(
                seeds[:, :16]
                + np.random.uniform(
                    -cfg.grasp_frontier_joint_noise,
                    cfg.grasp_frontier_joint_noise,
                    (frontier_count, env._NUM_HAND_DOF),
                ),
                env._ctrl_lower,
                env._ctrl_upper,
            )
            ball_pos[:frontier_count] = seeds[:, 16:19] + np.random.uniform(
                -cfg.grasp_frontier_ball_position_noise,
                cfg.grasp_frontier_ball_position_noise,
                (frontier_count, 3),
            )
            ball_quat[:frontier_count] = seeds[:, 19:23]
        qvel = np.zeros((num_reset, env.nv), dtype=np.float64)
        return hand_qpos, ball_pos, ball_quat, qvel

    def _build_info_updates(
        self,
        env: LeapInhandBallGraspEnv,
        hand_qpos: np.ndarray,
        ball_pos: np.ndarray,
        ball_quat: np.ndarray,
    ) -> dict[str, np.ndarray]:
        updates = super()._build_info_updates(env, hand_qpos, ball_pos, ball_quat)
        updates["grasp_anchor_pos"] = ball_pos.copy()
        return updates


@registry.env("LeapInhandBallGrasp", sim_backend="motrix")
@registry.env("LeapInhandBallGrasp", sim_backend="mujoco")
class LeapInhandBallGraspEnv(AllegroRotationGrasp, LeapHandBaseEnv):
    """Collect stable LEAP-owned ball states into a separate cache."""

    _cfg: LeapInhandBallGraspCfg
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
        cfg: LeapInhandBallGraspCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> None:
        self._saved_grasp_keys: set[tuple[int, ...]] = set()
        self._tip_object_geom_pairs: np.ndarray | None = None
        self._penetration_candidates = 0
        self._penetration_rejected = 0
        self._grasp_frontier_rows = np.empty((0, 23), dtype=np.float32)
        self._grasp_frontier_scores = np.empty((0,), dtype=np.float64)
        self._best_penetration_score = float("inf")
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._hand_body_ids = np.empty((0,), dtype=np.int32)
        self._object_body_id = int(self._ball_body_ids[0])
        if self._cfg.grasp_max_fingertip_surface_gap is not None:
            object_geom_id = self._backend.get_geom_id("leap_object_col")
            self._tip_object_geom_pairs = np.asarray(
                [
                    (self._backend.get_geom_id(name), object_geom_id)
                    for name in self._FINGERTIP_GEOMS
                ],
                dtype=np.int32,
            )
        if self._penetration_filter_enabled():
            base_body_id = self._backend.get_body_id(self._BASE_BODY_NAME)
            self._hand_body_ids = self._backend.get_body_subtree_ids(base_body_id)

    def _make_domain_randomization_provider(self) -> DomainRandomizationProvider:
        return LeapBallGraspDomainRandomizationProvider()

    def _penetration_filter_enabled(self) -> bool:
        self_limit = self._cfg.grasp_max_self_penetration
        object_limit = self._cfg.grasp_max_object_penetration
        if (self_limit is None) != (object_limit is None):
            raise ValueError(
                "grasp_max_self_penetration and grasp_max_object_penetration "
                "must both be set or both be null"
            )
        if self_limit is None:
            return False
        if self_limit < 0.0 or object_limit is None or object_limit < 0.0:
            raise ValueError("Penetration limits must be non-negative")
        return True

    def _penetration_quality(
        self, env_ids: np.ndarray
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[ContactPenetrationDetail, ...],
    ]:
        if not self._penetration_filter_enabled():
            zeros = np.zeros(env_ids.shape, dtype=np.float64)
            return np.ones(env_ids.shape, dtype=bool), zeros, zeros, ()

        details = self._backend.get_contact_penetration_details(
            env_ids,
            self_collision_body_ids=self._hand_body_ids,
            object_body_id=self._object_body_id,
        )
        self_depths = np.asarray([detail.self_depth for detail in details], dtype=np.float64)
        object_depths = np.asarray([detail.object_depth for detail in details], dtype=np.float64)
        max_self_depth = self._cfg.grasp_max_self_penetration
        max_object_depth = self._cfg.grasp_max_object_penetration
        assert max_self_depth is not None and max_object_depth is not None
        valid = penetration_quality_mask(
            self_depths,
            object_depths,
            max_self_depth=max_self_depth,
            max_object_depth=max_object_depth,
        )
        self._penetration_candidates += int(valid.size)
        self._penetration_rejected += int(np.count_nonzero(~valid))

        if self.state is not None:
            log = self.state.info.get("log", {})
            log["grasp/max_self_penetration"] = float(np.max(self_depths, initial=0.0))
            log["grasp/max_object_penetration"] = float(np.max(object_depths, initial=0.0))
            log["grasp/penetration_valid"] = float(np.mean(valid.astype(np.float32)))
            log["grasp/penetration_rejected"] = float(self._penetration_rejected)
            self.state.info["log"] = log
        return valid, self_depths, object_depths, details

    def _fingertip_surface_quality(
        self, env_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        limit = self._cfg.grasp_max_fingertip_surface_gap
        if limit is None:
            return (
                np.ones(env_ids.shape, dtype=bool),
                np.zeros((len(env_ids), len(self._FINGERTIP_GEOMS)), dtype=np.float64),
            )
        if self._tip_object_geom_pairs is None:
            raise RuntimeError("fingertip surface geom pairs were not initialized")
        signed_distances = self._backend.get_geom_pair_distances(
            env_ids,
            self._tip_object_geom_pairs,
            max_distance=max(0.2, 2.0 * limit),
        )
        valid = fingertip_surface_gap_quality_mask(
            signed_distances,
            max_gap=limit,
        )
        surface_gaps = np.maximum(np.asarray(signed_distances, dtype=np.float64), 0.0)
        if self.state is not None:
            log = self.state.info.get("log", {})
            log["grasp/max_fingertip_surface_gap"] = float(
                np.max(surface_gaps, initial=0.0)
            )
            log["grasp/fingertip_surface_valid"] = float(
                np.mean(valid.astype(np.float32))
            )
            self.state.info["log"] = log
        return valid, surface_gaps

    def _update_grasp_frontier(
        self,
        rows: np.ndarray,
        self_depths: np.ndarray,
        object_depths: np.ndarray,
        details: tuple[ContactPenetrationDetail, ...],
    ) -> None:
        if rows.shape[0] == 0:
            return
        self_limit = self._cfg.grasp_max_self_penetration
        object_limit = self._cfg.grasp_max_object_penetration
        if self_limit is None or object_limit is None:
            return
        scores = np.maximum(
            np.asarray(self_depths, dtype=np.float64) / max(self_limit, 1e-12),
            np.asarray(object_depths, dtype=np.float64) / max(object_limit, 1e-12),
        )
        best_index = int(np.argmin(scores))
        best_score = float(scores[best_index])
        if best_score < self._best_penetration_score:
            self._best_penetration_score = best_score
            detail = details[best_index]
            state_qpos = " ".join(f"{value:.6f}" for value in rows[best_index])
            print(
                "Ball-grasp frontier best: "
                f"score={best_score:.4f}, "
                f"self={detail.self_depth * 1_000.0:.3f} mm "
                f"bodies={detail.self_body_pair} geoms={detail.self_geom_pair}, "
                f"object={detail.object_depth * 1_000.0:.3f} mm "
                f"bodies={detail.object_body_pair} geoms={detail.object_geom_pair}, "
                f"qpos=[{state_qpos}]"
            )
        combined_rows = np.concatenate([self._grasp_frontier_rows, rows], axis=0)
        combined_scores = np.concatenate([self._grasp_frontier_scores, scores], axis=0)
        self._grasp_frontier_rows, self._grasp_frontier_scores = select_frontier_rows(
            combined_rows,
            combined_scores,
            capacity=self._cfg.grasp_frontier_capacity,
        )
        if self.state is not None:
            log = self.state.info.get("log", {})
            log["grasp/frontier_size"] = float(len(self._grasp_frontier_rows))
            log["grasp/frontier_best_score"] = float(np.min(self._grasp_frontier_scores))
            self.state.info["log"] = log

    def _contact_flags(self) -> np.ndarray:
        return np.stack(
            [
                self._sensor_scalar(self.get_sensor_data(name)) > 0.5
                for name in self._CONTACT_SENSORS
            ],
            axis=1,
        )

    def diagnose_grasp_state(
        self,
        qpos: np.ndarray,
        *,
        settle_seconds: float,
    ) -> dict[str, Any]:
        """Settle one candidate through the production control and quality paths."""
        if self._num_envs != 1:
            raise ValueError("grasp-state diagnostics require num_envs=1")
        if settle_seconds <= 0.0:
            raise ValueError("settle_seconds must be positive")
        candidate = normalize_grasp_cache_rows(
            np.asarray(qpos, dtype=np.float64).reshape(1, -1)
        )[0]

        self.init_state()
        self.set_autoreset(False)
        assert self.state is not None
        env_ids = np.asarray([0], dtype=np.int32)
        self._backend.set_state(env_ids, candidate[None, :], np.zeros((1, self.nv)))
        self.state.info["prev_ctrl"] = candidate[None, :16].copy()
        self.state.info["grasp_anchor_pos"] = candidate[None, 16:19].copy()
        self.state.info["steps"][:] = 0
        self.state.terminated[:] = False
        self.state.truncated[:] = False

        object_geom_id = self._backend.get_geom_id("leap_object_col")
        tip_geom_ids = [self._backend.get_geom_id(name) for name in self._FINGERTIP_GEOMS]
        geom_pairs = np.asarray(
            [(tip_geom_id, object_geom_id) for tip_geom_id in tip_geom_ids],
            dtype=np.int32,
        )
        initial_tip_distances = self._backend.get_geom_pair_distances(
            env_ids,
            geom_pairs,
            max_distance=0.2,
        )[0]
        initial_contacts = self._contact_flags()[0]

        settle_steps = max(1, int(np.ceil(settle_seconds / self._cfg.ctrl_dt)))
        actions = np.zeros((1, self._NUM_HAND_DOF), dtype=self._np_dtype)
        terminated_during_settle = False
        for _ in range(settle_steps):
            state = self.step(actions)
            terminated_during_settle |= bool(state.terminated[0])

        conditions = self._quality_conditions()
        contacts = self._contact_flags()[0]
        settled_tip_distances = self._backend.get_geom_pair_distances(
            env_ids,
            geom_pairs,
            max_distance=0.2,
        )[0]
        dof_pos = self.get_hand_dof_pos()
        dof_vel = self.get_hand_dof_vel()
        ball_pos = self.get_ball_pos()
        ball_linvel = self.get_ball_linvel()
        ball_angvel = self.get_ball_angvel()
        fingertip_pos = self.get_fingertip_pos()
        targets = np.asarray(self.state.info["prev_ctrl"])
        torques = compute_pd_torques(
            targets,
            dof_pos,
            dof_vel,
            self._cfg.control_config.kp,
            self._cfg.control_config.kd,
        )
        work = np.abs(np.sum(torques * dof_vel, axis=1))
        details = self._backend.get_contact_penetration_details(
            env_ids,
            self_collision_body_ids=self._hand_body_ids,
            object_body_id=self._object_body_id,
        )
        detail = details[0]
        max_self = self._cfg.grasp_max_self_penetration
        max_object = self._cfg.grasp_max_object_penetration
        if max_self is None or max_object is None:
            raise ValueError("grasp-state diagnostics require penetration limits")
        penetration_valid = bool(
            penetration_quality_mask(
                np.asarray([detail.self_depth]),
                np.asarray([detail.object_depth]),
                max_self_depth=max_self,
                max_object_depth=max_object,
            )[0]
        )
        condition_values = {name: bool(values[0]) for name, values in conditions.items()}
        surface_gap_limit = self._cfg.grasp_max_fingertip_surface_gap
        surface_gap_valid = bool(
            surface_gap_limit is None
            or fingertip_surface_gap_quality_mask(
                settled_tip_distances[None, :],
                max_gap=surface_gap_limit,
            )[0]
        )
        condition_values["surface_gap"] = surface_gap_valid
        return {
            "settle_steps": settle_steps,
            "settle_seconds": settle_steps * self._cfg.ctrl_dt,
            "terminated_during_settle": terminated_during_settle,
            "conditions": condition_values,
            "contacts": {
                name: bool(value)
                for name, value in zip(self._CONTACT_SENSORS, contacts, strict=True)
            },
            "initial_contacts": {
                name: bool(value)
                for name, value in zip(self._CONTACT_SENSORS, initial_contacts, strict=True)
            },
            "tip_surface_distance": {
                geom_name: {
                    "initial": float(initial_distance),
                    "settled": float(settled_distance),
                }
                for geom_name, initial_distance, settled_distance in zip(
                    self._FINGERTIP_GEOMS,
                    initial_tip_distances,
                    settled_tip_distances,
                    strict=True,
                )
            },
            "measurements": {
                "ball_height": float(ball_pos[0, 2]),
                "ball_drift": float(np.linalg.norm(ball_pos[0] - candidate[16:19])),
                "max_fingertip_distance": float(
                    np.max(np.linalg.norm(fingertip_pos[0] - ball_pos[0], axis=-1))
                ),
                "max_fingertip_surface_gap": float(
                    np.max(np.maximum(settled_tip_distances, 0.0))
                ),
                "contact_count": int(np.count_nonzero(contacts)),
                "ball_linear_speed": float(np.linalg.norm(ball_linvel[0])),
                "ball_angular_speed": float(np.linalg.norm(ball_angvel[0])),
                "max_joint_speed": float(np.max(np.abs(dof_vel[0]))),
                "abs_work": float(work[0]),
            },
            "penetration": {
                "valid": penetration_valid,
                "self_depth": detail.self_depth,
                "self_body_pair": detail.self_body_pair,
                "self_geom_pair": detail.self_geom_pair,
                "object_depth": detail.object_depth,
                "object_body_pair": detail.object_body_pair,
                "object_geom_pair": detail.object_geom_pair,
            },
            "quality_valid": bool(all(condition_values.values()) and penetration_valid),
        }

    def replay_validate_grasp_cache_rows(
        self,
        rows: np.ndarray,
        *,
        settle_seconds: float,
    ) -> GraspCacheReplayResult:
        """Reload and settle cache rows through the production grasp contract."""
        candidates = normalize_grasp_cache_rows(rows)
        num_rows = candidates.shape[0]
        if num_rows == 0:
            raise ValueError("grasp replay validation requires at least one row")
        if num_rows > self._num_envs:
            raise ValueError(
                f"Cannot validate {num_rows} rows with only {self._num_envs} environments"
            )
        if settle_seconds <= 0.0:
            raise ValueError("settle_seconds must be positive")
        if not self._penetration_filter_enabled():
            raise ValueError("grasp replay validation requires penetration limits")

        self.init_state()
        self.set_autoreset(False)
        assert self.state is not None
        env_ids = np.arange(num_rows, dtype=np.int32)
        self._backend.set_state(
            env_ids,
            candidates,
            np.zeros((num_rows, self.nv), dtype=np.float64),
        )

        prev_ctrl = np.asarray(self.state.info["prev_ctrl"]).copy()
        prev_ctrl[env_ids] = candidates[:, :16]
        self.state.info["prev_ctrl"] = prev_ctrl
        anchors = np.asarray(self.state.info.get("grasp_anchor_pos", self.get_ball_pos())).copy()
        anchors[env_ids] = candidates[:, 16:19]
        self.state.info["grasp_anchor_pos"] = anchors
        self.state.info["steps"][env_ids] = 0
        self.state.terminated[env_ids] = False
        self.state.truncated[env_ids] = False

        settle_steps = max(1, int(np.ceil(settle_seconds / self._cfg.ctrl_dt)))
        actions = np.zeros((self._num_envs, self._NUM_HAND_DOF), dtype=self._np_dtype)
        terminated = np.zeros(num_rows, dtype=bool)
        for _ in range(settle_steps):
            state = self.step(actions)
            terminated |= np.asarray(state.terminated[env_ids], dtype=bool)

        all_final_quality, all_conditions = self._strict_quality_mask()
        final_quality = np.asarray(all_final_quality[env_ids], dtype=bool)
        conditions = {
            name: np.asarray(values[env_ids], dtype=bool).copy()
            for name, values in all_conditions.items()
        }
        penetration_valid, self_depths, object_depths, _ = self._penetration_quality(env_ids)
        penetration_valid = np.asarray(penetration_valid, dtype=bool)
        surface_valid, surface_gaps = self._fingertip_surface_quality(env_ids)
        surface_valid = np.asarray(surface_valid, dtype=bool)
        conditions["surface_gap"] = surface_valid.copy()
        final_quality &= surface_valid
        contacts = np.asarray(self._contact_flags()[env_ids], dtype=bool)

        dof_pos = self.get_hand_dof_pos()[env_ids]
        dof_vel = self.get_hand_dof_vel()[env_ids]
        ball_pos = self.get_ball_pos()[env_ids]
        ball_linvel = self.get_ball_linvel()[env_ids]
        ball_angvel = self.get_ball_angvel()[env_ids]
        fingertip_pos = self.get_fingertip_pos()[env_ids]
        targets = np.asarray(self.state.info["prev_ctrl"])[env_ids]
        torques = compute_pd_torques(
            targets,
            dof_pos,
            dof_vel,
            self._cfg.control_config.kp,
            self._cfg.control_config.kd,
        )
        work = np.abs(np.sum(torques * dof_vel, axis=1))
        measurements = {
            "ball_height": ball_pos[:, 2].copy(),
            "ball_drift": np.linalg.norm(ball_pos - candidates[:, 16:19], axis=1),
            "max_fingertip_distance": np.max(
                np.linalg.norm(fingertip_pos - ball_pos[:, None, :], axis=-1),
                axis=1,
            ),
            "max_fingertip_surface_gap": np.max(surface_gaps, axis=1),
            "contact_count": np.sum(contacts, axis=1, dtype=np.int32),
            "ball_linear_speed": np.linalg.norm(ball_linvel, axis=1),
            "ball_angular_speed": np.linalg.norm(ball_angvel, axis=1),
            "max_joint_speed": np.max(np.abs(dof_vel), axis=1),
            "abs_work": work,
        }
        accepted = final_quality & penetration_valid & ~terminated
        return GraspCacheReplayResult(
            settle_steps=settle_steps,
            settle_seconds=settle_steps * self._cfg.ctrl_dt,
            accepted=accepted,
            terminated_during_settle=terminated,
            final_quality=final_quality,
            penetration_valid=penetration_valid,
            conditions=conditions,
            contacts=contacts,
            measurements=measurements,
            self_penetration=np.asarray(self_depths, dtype=np.float64),
            object_penetration=np.asarray(object_depths, dtype=np.float64),
        )

    def diagnose_joint_coordinate_probes(
        self,
        qpos: np.ndarray,
        *,
        joint_names: list[str],
        delta_magnitudes: np.ndarray,
        settle_seconds: float,
    ) -> list[dict[str, Any]]:
        """Settle signed coordinate probes through the full grasp-quality path."""
        base_qpos = normalize_grasp_cache_rows(
            np.asarray(qpos, dtype=np.float64).reshape(1, -1)
        )[0]
        coordinate_indices = self._backend.get_joint_dof_pos_indices(joint_names)
        hand_probes, probe_indices, applied_deltas = build_joint_coordinate_probes(
            base_qpos[:16],
            coordinate_indices,
            np.asarray(delta_magnitudes, dtype=np.float64),
            joint_lower=self._ctrl_lower,
            joint_upper=self._ctrl_upper,
        )
        joint_by_index = dict(zip(coordinate_indices.tolist(), joint_names, strict=True))
        reports: list[dict[str, Any]] = []
        for hand_qpos, coordinate_index, delta in zip(
            hand_probes,
            probe_indices,
            applied_deltas,
            strict=True,
        ):
            probe_qpos = base_qpos.copy()
            probe_qpos[:16] = hand_qpos
            report = self.diagnose_grasp_state(probe_qpos, settle_seconds=settle_seconds)
            if coordinate_index < 0:
                joint_name = "baseline"
                joint_value = None
            else:
                joint_name = joint_by_index[int(coordinate_index)]
                joint_value = float(hand_qpos[coordinate_index])
            report["probe"] = {
                "joint": joint_name,
                "delta": float(delta),
                "joint_value": joint_value,
            }
            reports.append(report)
        return reports

    def _quality_conditions(self) -> dict[str, np.ndarray]:
        cfg = self._cfg
        ball_pos = self.get_ball_pos()
        ball_linvel = self.get_ball_linvel()
        ball_angvel = self.get_ball_angvel()
        dof_pos = self.get_hand_dof_pos()
        dof_vel = self.get_hand_dof_vel()
        fingertip_pos = self.get_fingertip_pos()
        contacts = self._contact_flags()

        anchor = ball_pos
        if self.state is not None:
            anchor = np.asarray(self.state.info.get("grasp_anchor_pos", ball_pos))

        targets = dof_pos
        if self.state is not None:
            targets = np.asarray(self.state.info.get("prev_ctrl", dof_pos))
        torques = compute_pd_torques(
            targets,
            dof_pos,
            dof_vel,
            cfg.control_config.kp,
            cfg.control_config.kd,
        )
        work = np.abs(np.sum(torques * dof_vel, axis=1))

        finite = np.all(
            np.isfinite(
                np.concatenate(
                    [dof_pos, dof_vel, ball_pos, ball_linvel, ball_angvel], axis=1
                )
            ),
            axis=1,
        )
        finite = np.asarray(finite & np.isfinite(work), dtype=bool)
        joint_limits = np.all(
            (dof_pos >= self._ctrl_lower[None, :] - 1e-5)
            & (dof_pos <= self._ctrl_upper[None, :] + 1e-5),
            axis=1,
        )
        close_fingertips = np.all(
            np.linalg.norm(fingertip_pos - ball_pos[:, None, :], axis=-1)
            <= cfg.grasp_max_fingertip_distance,
            axis=1,
        )
        enough_contacts = np.sum(contacts, axis=1) >= cfg.grasp_min_contacts
        thumb_contact = contacts[:, 3] | (not cfg.grasp_require_thumb_contact)

        return {
            "finite": finite,
            "joint_limits": joint_limits,
            "height": ball_pos[:, 2] > self._reward_cfg.reset_z_threshold,
            "drift": np.linalg.norm(ball_pos - anchor, axis=1) <= cfg.grasp_max_ball_drift,
            "fingertips": close_fingertips,
            "contacts": enough_contacts,
            "thumb": thumb_contact,
            "ball_linvel": np.linalg.norm(ball_linvel, axis=1)
            <= cfg.grasp_max_ball_linear_speed,
            "ball_angvel": np.linalg.norm(ball_angvel, axis=1)
            <= cfg.grasp_max_ball_angular_speed,
            "joint_speed": np.max(np.abs(dof_vel), axis=1) <= cfg.grasp_max_joint_speed,
            "work": work <= cfg.grasp_max_abs_work,
        }

    def _strict_quality_mask(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        conditions = self._quality_conditions()
        valid = np.logical_and.reduce(tuple(conditions.values()))
        return np.asarray(valid, dtype=bool), conditions

    def _check_grasp_quality(self, env_ids: np.ndarray) -> np.ndarray:
        valid, _ = self._strict_quality_mask()
        return valid[env_ids]

    def update_state(self, state: NpEnvState) -> NpEnvState:
        # Bypass Allegro's weaker grasp gate while retaining the rotation env contract.
        next_state = AllegroRotationPPO.update_state(self, state)
        valid, conditions = self._strict_quality_mask()

        step_count = np.asarray(
            next_state.info.get("steps", np.zeros(self._num_envs, dtype=np.uint32))
        )
        warmup_steps = max(0, int(np.ceil(self._cfg.grasp_warmup_seconds / self._cfg.ctrl_dt)))
        validation_active = step_count >= warmup_steps
        basic_valid = conditions["finite"] & conditions["joint_limits"] & conditions["height"]
        terminated = np.asarray(
            next_state.terminated | (~basic_valid) | (validation_active & (~valid)),
            dtype=bool,
        )

        should_log = self._enable_reward_log and (int(step_count[0]) % 4 == 0)
        if should_log:
            log = next_state.info.get("log", {})
            for name, condition in conditions.items():
                log[f"grasp/{name}"] = float(np.mean(condition.astype(np.float32)))
            log["grasp/validation_active"] = float(np.mean(validation_active.astype(np.float32)))
            log["grasp/valid"] = float(np.mean(valid.astype(np.float32)))
            log["grasp/cache_size"] = float(self._total_saved_grasps())
            next_state.info["log"] = log

        return next_state.replace(
            reward=np.zeros(self._num_envs, dtype=self._np_dtype),
            terminated=terminated,
        )

    @staticmethod
    def _cache_key(row: np.ndarray) -> tuple[int, ...]:
        return grasp_cache_row_key(row)

    def _collect_successful_grasps(self, env_ids: np.ndarray) -> None:
        if self.state is None or env_ids.size == 0:
            return

        success = self.state.truncated[env_ids] & ~self.state.terminated[env_ids]
        success_env_ids = env_ids[np.flatnonzero(success)]
        if success_env_ids.size == 0:
            return
        quality = self._check_grasp_quality(success_env_ids)
        success_env_ids = success_env_ids[np.flatnonzero(quality)]
        if success_env_ids.size == 0:
            return
        candidate_rows = np.concatenate(
            [
                self.get_hand_dof_pos()[success_env_ids],
                self.get_ball_pos()[success_env_ids],
                self.get_ball_quat()[success_env_ids],
            ],
            axis=1,
        )
        candidate_rows = normalize_grasp_cache_rows(candidate_rows)

        # Reconstruct the exact float32 rows that will be written to disk before
        # applying cache-only geometry gates. These environments are already
        # done and will be reset immediately after collection, so replacing
        # their backend state here cannot affect a live rollout. This is a
        # serialization round-trip check, not a physics replay.
        self._backend.set_state(
            success_env_ids,
            candidate_rows.astype(np.float64),
            np.zeros((len(success_env_ids), self.nv), dtype=np.float64),
        )
        serialized_contacts = self._contact_flags()[success_env_ids]
        serialized_contact_quality = (
            np.sum(serialized_contacts, axis=1) >= self._cfg.grasp_min_contacts
        ) & (
            serialized_contacts[:, 3] | (not self._cfg.grasp_require_thumb_contact)
        )
        serialized_indices = np.flatnonzero(serialized_contact_quality)
        success_env_ids = success_env_ids[serialized_indices]
        candidate_rows = candidate_rows[serialized_indices]
        if self.state is not None:
            log = self.state.info.get("log", {})
            log["grasp/serialized_contact_valid"] = float(
                np.mean(serialized_contact_quality.astype(np.float32))
            )
            self.state.info["log"] = log
        if success_env_ids.size == 0:
            return

        surface_quality, _ = self._fingertip_surface_quality(success_env_ids)
        surface_indices = np.flatnonzero(surface_quality)
        success_env_ids = success_env_ids[surface_indices]
        candidate_rows = candidate_rows[surface_indices]
        if success_env_ids.size == 0:
            return
        penetration_quality, self_depths, object_depths, details = self._penetration_quality(
            success_env_ids
        )
        rejected = ~penetration_quality
        rejected_indices = np.flatnonzero(rejected)
        self._update_grasp_frontier(
            candidate_rows[rejected_indices],
            self_depths[rejected_indices],
            object_depths[rejected_indices],
            tuple(details[index] for index in rejected_indices),
        )
        success_env_ids = success_env_ids[np.flatnonzero(penetration_quality)]
        if success_env_ids.size == 0:
            return

        rows = candidate_rows[np.flatnonzero(penetration_quality)]
        unique_rows: list[np.ndarray] = []
        for row in rows:
            key = self._cache_key(row)
            if key in self._saved_grasp_keys:
                continue
            self._saved_grasp_keys.add(key)
            unique_rows.append(row)

        if not unique_rows:
            return
        self._saved_grasping_states.append(np.stack(unique_rows).astype(np.float32))
        self._save_grasp_cache()
        self._stop_collection()

        log = self.state.info.get("log", {})
        log["grasp/cache_size"] = float(self._total_saved_grasps())
        self.state.info["log"] = log
