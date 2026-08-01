"""LEAP Hand model-specific in-hand environment contract."""

from __future__ import annotations

import numpy as np

from unilab.envs.manipulation.allegro_inhand.base import AllegroBaseEnv

# The source Isaac Gym ordering is retained for future deployment adapters.
SOURCE_SIM_JOINT_ORDER: tuple[int, ...] = (
    1,
    0,
    2,
    3,
    12,
    13,
    14,
    15,
    5,
    4,
    6,
    7,
    9,
    8,
    10,
    11,
)
SOURCE_SIM_TO_REAL_INDICES: tuple[int, ...] = (
    1,
    0,
    2,
    3,
    9,
    8,
    10,
    11,
    13,
    12,
    14,
    15,
    4,
    5,
    6,
    7,
)
SOURCE_REAL_TO_SIM_INDICES: tuple[int, ...] = (
    1,
    0,
    2,
    3,
    12,
    13,
    14,
    15,
    5,
    4,
    6,
    7,
    9,
    8,
    10,
    11,
)

# Legacy numeric joint identifiers in the pre-Menagerie UniLab model's qpos
# traversal order. Keep this mapping for cache/deployment provenance only; the
# active MuJoCo model uses MENAGERIE_SIM_JOINT_NAMES.
UNILAB_SIM_JOINT_ORDER: tuple[int, ...] = (
    1,
    0,
    2,
    3,
    5,
    4,
    6,
    7,
    9,
    8,
    10,
    11,
    12,
    13,
    14,
    15,
)

MENAGERIE_SIM_JOINT_NAMES: tuple[str, ...] = (
    "if_mcp",
    "if_rot",
    "if_pip",
    "if_dip",
    "mf_mcp",
    "mf_rot",
    "mf_pip",
    "mf_dip",
    "rf_mcp",
    "rf_rot",
    "rf_pip",
    "rf_dip",
    "th_cmc",
    "th_axl",
    "th_mcp",
    "th_ipl",
)

LEAP_PALM_CONTACT_SENSOR_NAMES: tuple[str, ...] = tuple(
    f"leap_palm_contact_{index}" for index in range(1, 11)
)


class LeapHandBaseEnv(AllegroBaseEnv):
    """Bind generic 16-DoF in-hand behavior to LEAP model names."""

    _BASE_BODY_NAME = "palm_lower"
    _OBJECT_BODY_NAME = "leap_object"
    _LOG_PREFIX = "leap_inhand"
    _GRASP_GENERATION_TASK = None
    _MODEL_PD_KP = 3.0
    _MODEL_PD_KD = 0.01
    # Menagerie position actuators intentionally have no force limit. Reward
    # diagnostics must therefore use the unclipped PD estimate as well.
    _PD_TORQUE_LIMIT: float | None = None
    _FINGERTIP_BODY_NAMES = (
        "fingertip",
        "fingertip_2",
        "fingertip_3",
        "thumb_fingertip",
    )

    def _backend_position_actuator_gains(
        self,
        cfg,
        backend_type: str,
    ) -> dict[str, object] | None:
        if backend_type != "mujoco":
            self._uses_model_owned_pd_gains = False
            return super()._backend_position_actuator_gains(cfg, backend_type)

        self._uses_model_owned_pd_gains = True
        if cfg.scene.joint_dynamics is not None:
            raise ValueError(
                "LEAP MuJoCo damping/frictionloss/armature are model-owned "
                "constants in leap_hand.xml; env.scene.joint_dynamics overrides "
                "are not allowed"
            )
        configured = (float(cfg.control_config.kp), float(cfg.control_config.kd))
        expected = (self._MODEL_PD_KP, self._MODEL_PD_KD)
        if configured != expected:
            raise ValueError(
                "LEAP MuJoCo kp/kd are model-owned constants in leap_hand.xml; "
                f"expected kp={expected[0]}, kd={expected[1]}, got "
                f"kp={configured[0]}, kd={configured[1]}"
            )
        return None

    def get_pd_gains(self) -> tuple[float, float]:
        if getattr(self, "_uses_model_owned_pd_gains", False):
            return self._MODEL_PD_KP, self._MODEL_PD_KD
        return super().get_pd_gains()

    def get_palm_contact_flags(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        """Aggregate all Menagerie palm collision sensors into one flag per env."""
        sensor_values = np.asarray(
            self._backend.get_sensor_data_batch(LEAP_PALM_CONTACT_SENSOR_NAMES)
        )
        flags = np.any(sensor_values > 0.5, axis=1)
        if env_ids is None:
            return np.asarray(flags, dtype=bool)
        return np.asarray(flags[np.asarray(env_ids, dtype=np.intp)], dtype=bool)
