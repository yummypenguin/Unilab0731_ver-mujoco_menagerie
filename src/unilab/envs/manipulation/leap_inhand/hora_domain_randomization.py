"""HORA-specific sim-to-real domain-randomization contracts for LEAP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unilab.dr import DomainRandomizationCapabilities, ResetRandomizationPayload
from unilab.dr.types import (
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_GRAVITY,
)

PRIV_MASS_RATIO = 0
PRIV_FRICTION_SCALE = 1
PRIV_COM_SLICE = slice(2, 5)
PRIV_GRAVITY_SLICE = slice(5, 8)
PRIV_ACTION_DELAY = 8


def _validate_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"hora_domain_rand.{name} must be bool")


def _validate_positive_range(name: str, lower: float, upper: float) -> None:
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"hora_domain_rand.{name} bounds must be finite")
    if lower <= 0.0 or lower > upper:
        raise ValueError(
            f"hora_domain_rand.{name} must satisfy 0 < lower <= upper"
        )


@dataclass
class LeapHoraDomainRandomizationConfig:
    """Narrow reset and observation randomization used by the LEAP HORA task."""

    enabled: bool = True
    randomize_object_mass: bool = True
    object_mass_ratio_lower: float = 0.90
    object_mass_ratio_upper: float = 1.10
    randomize_object_friction: bool = True
    object_friction_scale_lower: float = 0.80
    object_friction_scale_upper: float = 1.20
    randomize_object_com: bool = True
    object_com_offset_lower: tuple[float, float, float] = (-0.001, -0.001, -0.001)
    object_com_offset_upper: tuple[float, float, float] = (0.001, 0.001, 0.001)
    randomize_gravity_direction: bool = True
    gravity_tilt_max_deg: float = 3.0
    joint_measurement_noise_rad: float = 0.003
    action_delay_min_steps: int = 0
    action_delay_max_steps: int = 1

    def validate(self) -> None:
        for name in (
            "enabled",
            "randomize_object_mass",
            "randomize_object_friction",
            "randomize_object_com",
            "randomize_gravity_direction",
        ):
            _validate_bool(name, getattr(self, name))

        _validate_positive_range(
            "object_mass_ratio",
            self.object_mass_ratio_lower,
            self.object_mass_ratio_upper,
        )
        _validate_positive_range(
            "object_friction_scale",
            self.object_friction_scale_lower,
            self.object_friction_scale_upper,
        )

        com_lower = np.asarray(self.object_com_offset_lower, dtype=np.float64)
        com_upper = np.asarray(self.object_com_offset_upper, dtype=np.float64)
        if com_lower.shape != (3,) or com_upper.shape != (3,):
            raise ValueError("hora_domain_rand object COM bounds must have shape (3,)")
        if not np.isfinite(com_lower).all() or not np.isfinite(com_upper).all():
            raise ValueError("hora_domain_rand object COM bounds must be finite")
        if np.any(com_lower > com_upper):
            raise ValueError(
                "hora_domain_rand object COM lower bounds must be <= upper bounds"
            )

        if (
            not np.isfinite(self.gravity_tilt_max_deg)
            or not 0.0 <= self.gravity_tilt_max_deg <= 15.0
        ):
            raise ValueError(
                "hora_domain_rand.gravity_tilt_max_deg must be finite and in [0, 15]"
            )
        if (
            not np.isfinite(self.joint_measurement_noise_rad)
            or self.joint_measurement_noise_rad < 0.0
        ):
            raise ValueError(
                "hora_domain_rand.joint_measurement_noise_rad must be finite and non-negative"
            )

        for name in ("action_delay_min_steps", "action_delay_max_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"hora_domain_rand.{name} must be an integer")
        if not 0 <= self.action_delay_min_steps <= self.action_delay_max_steps <= 1:
            raise ValueError(
                "hora_domain_rand action delay must satisfy 0 <= min <= max <= 1"
            )


@dataclass(frozen=True)
class LeapHoraResetSamples:
    mass_ratio: np.ndarray
    friction_scale: np.ndarray
    com_offset: np.ndarray
    gravity: np.ndarray
    gravity_direction: np.ndarray
    action_delay_steps: np.ndarray


def sample_hora_reset_values(
    cfg: LeapHoraDomainRandomizationConfig,
    num_reset: int,
    nominal_gravity: np.ndarray,
) -> LeapHoraResetSamples:
    """Sample one internally consistent set of physics and privileged values."""

    cfg.validate()
    gravity_nominal = np.asarray(nominal_gravity, dtype=np.float64)
    if gravity_nominal.shape != (3,) or not np.isfinite(gravity_nominal).all():
        raise ValueError("nominal_gravity must be a finite vector with shape (3,)")
    gravity_magnitude = float(np.linalg.norm(gravity_nominal))
    if gravity_magnitude <= 1.0e-8:
        raise ValueError("nominal_gravity magnitude must be positive")
    nominal_direction = gravity_nominal / gravity_magnitude

    mass_ratio = np.ones(num_reset, dtype=np.float64)
    friction_scale = np.ones(num_reset, dtype=np.float64)
    com_offset = np.zeros((num_reset, 3), dtype=np.float64)
    gravity_direction = np.broadcast_to(
        nominal_direction, (num_reset, 3)
    ).copy()
    action_delay_steps = np.zeros(num_reset, dtype=np.int32)

    if cfg.enabled:
        if cfg.randomize_object_mass:
            mass_ratio = np.random.uniform(
                cfg.object_mass_ratio_lower,
                cfg.object_mass_ratio_upper,
                size=num_reset,
            )
        if cfg.randomize_object_friction:
            friction_scale = np.random.uniform(
                cfg.object_friction_scale_lower,
                cfg.object_friction_scale_upper,
                size=num_reset,
            )
        if cfg.randomize_object_com:
            com_offset = np.random.uniform(
                np.asarray(cfg.object_com_offset_lower, dtype=np.float64),
                np.asarray(cfg.object_com_offset_upper, dtype=np.float64),
                size=(num_reset, 3),
            )
        if cfg.randomize_gravity_direction:
            max_tilt = np.deg2rad(cfg.gravity_tilt_max_deg)
            tilt = max_tilt * np.sqrt(np.random.uniform(0.0, 1.0, size=num_reset))
            azimuth = np.random.uniform(-np.pi, np.pi, size=num_reset)
            gravity_direction = np.stack(
                [
                    np.sin(tilt) * np.cos(azimuth),
                    np.sin(tilt) * np.sin(azimuth),
                    -np.cos(tilt),
                ],
                axis=1,
            )
        action_delay_steps = np.random.randint(
            cfg.action_delay_min_steps,
            cfg.action_delay_max_steps + 1,
            size=num_reset,
            dtype=np.int32,
        )

    gravity = gravity_direction * gravity_magnitude
    return LeapHoraResetSamples(
        mass_ratio=np.asarray(mass_ratio, dtype=np.float64),
        friction_scale=np.asarray(friction_scale, dtype=np.float64),
        com_offset=np.asarray(com_offset, dtype=np.float64),
        gravity=np.asarray(gravity, dtype=np.float64),
        gravity_direction=np.asarray(gravity_direction, dtype=np.float64),
        action_delay_steps=action_delay_steps,
    )


def build_hora_critic_info(
    samples: LeapHoraResetSamples,
    action_delay_max_steps: int,
) -> np.ndarray:
    """Build the fixed nine-channel privileged vector from applied samples."""

    num_reset = samples.mass_ratio.shape[0]
    critic_info = np.zeros((num_reset, 9), dtype=np.float64)
    critic_info[:, PRIV_MASS_RATIO] = samples.mass_ratio
    critic_info[:, PRIV_FRICTION_SCALE] = samples.friction_scale
    critic_info[:, PRIV_COM_SLICE] = samples.com_offset
    critic_info[:, PRIV_GRAVITY_SLICE] = samples.gravity_direction
    critic_info[:, PRIV_ACTION_DELAY] = samples.action_delay_steps / max(
        action_delay_max_steps, 1
    )
    return critic_info


def build_hora_reset_payload(
    samples: LeapHoraResetSamples,
    *,
    cfg: LeapHoraDomainRandomizationConfig,
    object_body_id: int,
    object_geom_id: int,
    nominal_body_mass: np.ndarray,
    nominal_body_ipos: np.ndarray,
    nominal_geom_friction: np.ndarray,
) -> ResetRandomizationPayload | None:
    """Build full backend tables while changing only the configured object entries."""

    if not cfg.enabled:
        return None
    num_reset = samples.mass_ratio.shape[0]
    payload = ResetRandomizationPayload()

    if cfg.randomize_object_mass:
        payload.body_mass = np.broadcast_to(
            nominal_body_mass, (num_reset, nominal_body_mass.size)
        ).copy()
        payload.body_mass[:, object_body_id] = (
            nominal_body_mass[object_body_id] * samples.mass_ratio
        )
    if cfg.randomize_object_com:
        payload.body_ipos = np.broadcast_to(
            nominal_body_ipos, (num_reset, *nominal_body_ipos.shape)
        ).copy()
        payload.body_ipos[:, object_body_id, :] += samples.com_offset
    if cfg.randomize_object_friction:
        payload.geom_friction = np.broadcast_to(
            nominal_geom_friction, (num_reset, *nominal_geom_friction.shape)
        ).copy()
        payload.geom_friction[:, object_geom_id, :] = (
            nominal_geom_friction[object_geom_id] * samples.friction_scale[:, None]
        )
    if cfg.randomize_gravity_direction:
        payload.gravity = samples.gravity.copy()
    return None if payload.is_empty() else payload


def validate_hora_backend_capabilities(
    cfg: LeapHoraDomainRandomizationConfig,
    capabilities: DomainRandomizationCapabilities,
    backend_type: str,
) -> None:
    """Fail closed for every enabled backend-owned HORA randomization term."""

    if not cfg.enabled:
        return
    requested = (
        (cfg.randomize_object_mass, RESET_TERM_BODY_MASS, "randomize_object_mass"),
        (cfg.randomize_object_com, RESET_TERM_BODY_IPOS, "randomize_object_com"),
        (
            cfg.randomize_object_friction,
            RESET_TERM_GEOM_FRICTION,
            "randomize_object_friction",
        ),
        (
            cfg.randomize_gravity_direction,
            RESET_TERM_GRAVITY,
            "randomize_gravity_direction",
        ),
    )
    for enabled, term, field in requested:
        if enabled and not capabilities.supports_reset_term(term):
            raise NotImplementedError(
                f"{backend_type} backend does not support randomization term {term!r} "
                f"required by env.hora_domain_rand.{field}"
            )


__all__ = [
    "PRIV_ACTION_DELAY",
    "PRIV_COM_SLICE",
    "PRIV_FRICTION_SCALE",
    "PRIV_GRAVITY_SLICE",
    "PRIV_MASS_RATIO",
    "LeapHoraDomainRandomizationConfig",
    "LeapHoraResetSamples",
    "build_hora_critic_info",
    "build_hora_reset_payload",
    "sample_hora_reset_values",
    "validate_hora_backend_capabilities",
]
