"""Deployable HORA observation contract for the LEAP 0730 rotation task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dr import DomainRandomizationCapabilities, ResetPlan, ResetRandomizationPayload
from unilab.dtype_config import get_global_dtype

from .ball_rotation_0730 import (
    LeapBall0730ResetProvider,
    LeapInhandBall0730RotationCfg,
    LeapInhandBall0730RotationEnv,
)
from .deploy_contract import (
    ACTOR_FRAME_DIM,
    ACTOR_HISTORY_LEN,
    ACTOR_OBS_DIM,
    PRIV_INFO_DIM,
    PROPRIO_FRAME_DIM,
    PROPRIO_HISTORY_LEN,
    build_actor_frame,
    build_proprio_frame,
    validate_axis,
)
from .hora_domain_randomization import (
    LeapHoraDomainRandomizationConfig,
    LeapHoraResetSamples,
    build_hora_critic_info,
    build_hora_reset_payload,
    sample_hora_reset_values,
    validate_hora_backend_capabilities,
)


@registry.envcfg("LeapInhandBall0730HoraRotation")
@dataclass
class LeapInhandBall0730HoraRotationCfg(LeapInhandBall0730RotationCfg):
    """LEAP 0730 task with an explicit deployable HORA observation layout."""

    actor_history_len: int = ACTOR_HISTORY_LEN
    proprio_history_len: int = PROPRIO_HISTORY_LEN
    critic_info_dim: int = PRIV_INFO_DIM
    rotation_axis_command: tuple[float, float, float] = (0.0, 0.0, 1.0)
    hora_domain_rand: LeapHoraDomainRandomizationConfig = field(
        default_factory=LeapHoraDomainRandomizationConfig
    )

    def validate(self) -> None:
        super().validate()
        if self.actor_history_len != ACTOR_HISTORY_LEN:
            raise ValueError(f"actor_history_len must be exactly {ACTOR_HISTORY_LEN}")
        if self.proprio_history_len != PROPRIO_HISTORY_LEN:
            raise ValueError(
                f"proprio_history_len must be exactly {PROPRIO_HISTORY_LEN}"
            )
        if self.critic_info_dim != PRIV_INFO_DIM:
            raise ValueError(f"critic_info_dim must be exactly {PRIV_INFO_DIM}")
        validate_axis(self.rotation_axis_command)
        self.hora_domain_rand.validate()


class LeapBall0730HoraResetProvider(LeapBall0730ResetProvider):
    """Initialize HORA histories from the selected cache row without shifting them."""

    def __init__(self) -> None:
        self._pending_samples: LeapHoraResetSamples | None = None

    def validate(
        self, env: Any, capabilities: DomainRandomizationCapabilities
    ) -> None:
        super().validate(env, capabilities)
        env._initialize_hora_domain_randomization_assets()
        validate_hora_backend_capabilities(
            env.cfg.hora_domain_rand,
            capabilities,
            env._backend.backend_type,
        )

    def _build_info_updates(
        self,
        env: Any,
        hand_qpos: np.ndarray,
        ball_pos: np.ndarray,
        ball_quat: np.ndarray,
    ) -> dict[str, np.ndarray]:
        updates = super()._build_info_updates(env, hand_qpos, ball_pos, ball_quat)
        previous_target = np.asarray(updates["prev_ctrl"], dtype=get_global_dtype())
        actor_frame, proprio_frame = env._build_observation_frames(
            hand_qpos, previous_target
        )

        actor_history = env._repeat_history(actor_frame, ACTOR_HISTORY_LEN)
        proprio_history = env._repeat_history(proprio_frame, PROPRIO_HISTORY_LEN)
        samples = self._pending_samples
        if samples is None:
            samples = env._sample_hora_reset_values(hand_qpos.shape[0])
        critic_info = env._build_reset_critic_info(samples)
        action_queue = np.zeros(
            (hand_qpos.shape[0], 2, env._num_action), dtype=get_global_dtype()
        )

        updates.pop("obs_lag_history", None)
        updates.update(
            {
                "observation_previous_target": previous_target.copy(),
                "hora_actor_history": actor_history,
                "hora_proprio_history": proprio_history,
                "critic_info": critic_info,
                "proprio_hist": proprio_history.copy(),
                "hora_action_delay_steps": samples.action_delay_steps.copy(),
                "hora_action_queue": action_queue,
            }
        )
        return updates

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        samples = env._sample_hora_reset_values(len(env_ids))
        self._pending_samples = samples
        try:
            plan = super().build_reset_plan(env, env_ids)
        finally:
            self._pending_samples = None
        hora_payload = env._build_hora_reset_payload(samples)
        plan.randomization = self._merge_payloads(plan.randomization, hora_payload)
        return plan

    @staticmethod
    def _merge_payloads(
        existing: ResetRandomizationPayload | None,
        hora: ResetRandomizationPayload | None,
    ) -> ResetRandomizationPayload | None:
        if hora is None:
            return existing
        if existing is None:
            return hora
        for field_name in ("body_mass", "body_ipos", "geom_friction", "gravity"):
            hora_value = getattr(hora, field_name)
            if hora_value is None:
                continue
            if getattr(existing, field_name) is not None:
                raise ValueError(
                    f"duplicate reset randomization owner for {field_name}"
                )
            setattr(existing, field_name, hora_value)
        return existing

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        del env, env_ids
        actor_history = np.asarray(
            info_updates["hora_actor_history"], dtype=get_global_dtype()
        )
        return {"obs": actor_history.reshape(actor_history.shape[0], ACTOR_OBS_DIM)}


@registry.env("LeapInhandBall0730HoraRotation", sim_backend="mujoco")
class LeapInhandBall0730HoraRotationEnv(LeapInhandBall0730RotationEnv):
    """LEAP rotation task whose policy inputs are directly deployable on hardware."""

    episode_static_critic_info = True
    _cfg: LeapInhandBall0730HoraRotationCfg

    def __init__(
        self,
        cfg: LeapInhandBall0730HoraRotationCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> None:
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._hora_rotation_axis = np.asarray(
            validate_axis(cfg.rotation_axis_command), dtype=get_global_dtype()
        )

    def _make_domain_randomization_provider(self) -> LeapBall0730HoraResetProvider:
        return LeapBall0730HoraResetProvider()

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": ACTOR_OBS_DIM}

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        previous_target = state.info.get(
            "prev_ctrl",
            np.broadcast_to(
                self.default_angles, (actions.shape[0], self._num_action)
            ).copy(),
        )
        state.info["observation_previous_target"] = np.asarray(
            previous_target, dtype=get_global_dtype()
        ).copy()
        proposed_action = np.asarray(
            np.clip(actions, -1.0, 1.0), dtype=get_global_dtype()
        )
        expected_queue_shape = (actions.shape[0], 2, self._num_action)
        queue = np.asarray(
            state.info.get(
                "hora_action_queue",
                np.zeros(expected_queue_shape, dtype=get_global_dtype()),
            ),
            dtype=get_global_dtype(),
        )
        if queue.shape != expected_queue_shape:
            raise RuntimeError(
                f"hora_action_queue must have shape {expected_queue_shape}, got {queue.shape}"
            )
        delay_steps = np.asarray(
            state.info.get(
                "hora_action_delay_steps",
                np.zeros(actions.shape[0], dtype=np.int32),
            ),
            dtype=np.int32,
        )
        if delay_steps.shape != (actions.shape[0],) or np.any(
            (delay_steps < 0) | (delay_steps > 1)
        ):
            raise RuntimeError("hora_action_delay_steps must contain only 0 or 1")

        queue[:, 0] = queue[:, 1]
        queue[:, 1] = proposed_action
        delayed_action = np.where(delay_steps[:, None] == 0, queue[:, 1], queue[:, 0])
        state.info["hora_action_queue"] = queue
        return super().apply_action(delayed_action, state)

    def _initialize_hora_domain_randomization_assets(self) -> None:
        """Resolve all model metadata once on the cold initialization path."""

        self._hora_object_body_id = self._backend.get_body_id(self._OBJECT_BODY_NAME)
        self._hora_object_geom_id = self._backend.get_geom_id("leap_object_col")
        self._hora_nominal_body_mass = np.asarray(
            self._backend.get_body_mass(), dtype=np.float64
        ).copy()
        self._hora_nominal_body_ipos = np.asarray(
            self._backend.get_body_ipos(), dtype=np.float64
        ).copy()
        self._hora_nominal_geom_friction = np.asarray(
            self._backend.get_geom_friction(), dtype=np.float64
        ).copy()
        self._hora_nominal_gravity = np.asarray(
            self._backend.get_gravity(), dtype=np.float64
        ).copy()
        if self._hora_nominal_gravity.shape != (3,):
            raise ValueError(
                "backend gravity must have shape (3,), got "
                f"{self._hora_nominal_gravity.shape}"
            )
        self._nominal_gravity_direction = np.asarray(
            validate_axis(self._hora_nominal_gravity), dtype=get_global_dtype()
        )

    def _sample_hora_reset_values(self, num_reset: int) -> LeapHoraResetSamples:
        return sample_hora_reset_values(
            self._cfg.hora_domain_rand,
            num_reset,
            self._hora_nominal_gravity,
        )

    def _build_reset_critic_info(
        self, samples: LeapHoraResetSamples
    ) -> np.ndarray:
        return np.asarray(
            build_hora_critic_info(
                samples,
                self._cfg.hora_domain_rand.action_delay_max_steps,
            ),
            dtype=get_global_dtype(),
        )

    def _build_hora_reset_payload(
        self, samples: LeapHoraResetSamples
    ) -> ResetRandomizationPayload | None:
        return build_hora_reset_payload(
            samples,
            cfg=self._cfg.hora_domain_rand,
            object_body_id=self._hora_object_body_id,
            object_geom_id=self._hora_object_geom_id,
            nominal_body_mass=self._hora_nominal_body_mass,
            nominal_body_ipos=self._hora_nominal_body_ipos,
            nominal_geom_friction=self._hora_nominal_geom_friction,
        )

    @staticmethod
    def _repeat_history(frame: np.ndarray, history_len: int) -> np.ndarray:
        return np.repeat(frame[:, None, :], history_len, axis=1)

    def _build_actor_frame(
        self, measured_q: np.ndarray, previous_target: np.ndarray
    ) -> np.ndarray:
        return np.asarray(
            build_actor_frame(
                measured_q,
                previous_target,
                self._hora_rotation_axis,
                self._ctrl_lower,
                self._ctrl_upper,
            ),
            dtype=get_global_dtype(),
        )

    def _build_proprio_frame(
        self, measured_q: np.ndarray, previous_target: np.ndarray
    ) -> np.ndarray:
        return np.asarray(
            build_proprio_frame(
                measured_q,
                previous_target,
                self._ctrl_lower,
                self._ctrl_upper,
            ),
            dtype=get_global_dtype(),
        )

    def _build_observation_frames(
        self, measured_q: np.ndarray, previous_target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        observed_q = np.asarray(measured_q, dtype=get_global_dtype()).copy()
        dr_cfg = self._cfg.hora_domain_rand
        if dr_cfg.enabled and dr_cfg.joint_measurement_noise_rad > 0.0:
            observed_q += np.random.uniform(
                -dr_cfg.joint_measurement_noise_rad,
                dr_cfg.joint_measurement_noise_rad,
                size=observed_q.shape,
            ).astype(get_global_dtype())
        return (
            self._build_actor_frame(observed_q, previous_target),
            self._build_proprio_frame(observed_q, previous_target),
        )

    def _build_nominal_critic_info(self, num_envs: int) -> np.ndarray:
        template = np.concatenate(
            [
                np.asarray([1.0, 1.0, 0.0, 0.0, 0.0], dtype=get_global_dtype()),
                self._nominal_gravity_direction,
                np.asarray([0.0], dtype=get_global_dtype()),
            ]
        )
        if template.shape != (PRIV_INFO_DIM,):
            raise RuntimeError(f"critic info width must be {PRIV_INFO_DIM}")
        return np.broadcast_to(template, (num_envs, PRIV_INFO_DIM)).copy()

    @staticmethod
    def _push_history(
        info: dict[str, Any], key: str, frame: np.ndarray, history_len: int, frame_dim: int
    ) -> np.ndarray:
        expected_shape = (frame.shape[0], history_len, frame_dim)
        history = info.get(key)
        if history is None:
            history_array = np.repeat(frame[:, None, :], history_len, axis=1)
        else:
            history_array = np.asarray(history, dtype=get_global_dtype())
            if history_array.shape != expected_shape:
                raise RuntimeError(
                    f"{key} must have shape {expected_shape}, got {history_array.shape}"
                )
            history_array[:, :-1] = history_array[:, 1:]
            history_array[:, -1] = frame
        info[key] = history_array
        return history_array

    def _compute_obs(
        self, info: dict[str, Any], dof_pos: np.ndarray, ball_pos: np.ndarray
    ) -> dict[str, np.ndarray]:
        del ball_pos
        previous_target = np.asarray(
            info.get("observation_previous_target", info["prev_ctrl"]),
            dtype=get_global_dtype(),
        )
        actor_frame, proprio_frame = self._build_observation_frames(
            dof_pos, previous_target
        )

        actor_history = self._push_history(
            info,
            "hora_actor_history",
            actor_frame,
            ACTOR_HISTORY_LEN,
            ACTOR_FRAME_DIM,
        )
        proprio_history = self._push_history(
            info,
            "hora_proprio_history",
            proprio_frame,
            PROPRIO_HISTORY_LEN,
            PROPRIO_FRAME_DIM,
        )
        critic_info = np.asarray(
            info.get("critic_info", self._build_nominal_critic_info(dof_pos.shape[0])),
            dtype=get_global_dtype(),
        )
        if critic_info.shape != (dof_pos.shape[0], PRIV_INFO_DIM):
            raise RuntimeError(
                "critic_info must have shape "
                f"{(dof_pos.shape[0], PRIV_INFO_DIM)}, got {critic_info.shape}"
            )
        info["critic_info"] = critic_info
        info["proprio_hist"] = proprio_history.copy()
        return {
            "obs": np.asarray(
                actor_history.reshape(dof_pos.shape[0], ACTOR_OBS_DIM),
                dtype=get_global_dtype(),
            )
        }


__all__ = [
    "LeapBall0730HoraResetProvider",
    "LeapInhandBall0730HoraRotationCfg",
    "LeapInhandBall0730HoraRotationEnv",
]
