"""LEAP Hand model-specific in-hand environment contract."""

from __future__ import annotations

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

# This order is shared by qpos and position actuators in leap_hand.xml.
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


class LeapHandBaseEnv(AllegroBaseEnv):
    """Bind generic 16-DoF in-hand behavior to LEAP model names."""

    _BASE_BODY_NAME = "palm_lower"
    _OBJECT_BODY_NAME = "leap_object"
    _LOG_PREFIX = "leap_inhand"
    _GRASP_GENERATION_TASK = None
    _FINGERTIP_BODY_NAMES = (
        "fingertip",
        "fingertip_2",
        "fingertip_3",
        "thumb_fingertip",
    )
