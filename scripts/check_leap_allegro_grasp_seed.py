"""Run a single-environment, non-saving preflight of the LEAP nominal seed."""

from __future__ import annotations

import json

import numpy as np

from unilab.base.np_env import NpEnv
from unilab.envs.manipulation.allegro_inhand.rotation import (
    DomainRandConfig,
    RewardConfigPPO,
)
from unilab.envs.manipulation.leap_inhand.ball_grasp_allegro import (
    LeapAllegroGraspResetProvider,
    LeapInhandBallGraspAllegroCfg,
    LeapInhandBallGraspAllegroEnv,
)

NOMINAL_QPOS = np.asarray(
    [
        1.5152045040427635,
        0.11430147259750476,
        0.2876406730815961,
        0.19280835997306603,
        1.4188457206477074,
        0.025681830807677088,
        -0.26717932336688344,
        0.5369823550831088,
        1.5294890485315962,
        -0.01798386011739139,
        0.27558019211759954,
        0.19821762108233876,
        1.9245445859343515,
        0.04788276935232176,
        -0.021885380331691334,
        0.19524630120127295,
        -0.032440416893199604,
        0.041151239943936,
        0.664301098275159,
        0.9300906819767993,
        0.07052047191574277,
        -0.04548098804911446,
        0.3576166467976395,
    ],
    dtype=np.float64,
)


def _contact_flags(env: LeapInhandBallGraspAllegroEnv) -> np.ndarray:
    return np.asarray(
        [
            bool(env._sensor_scalar(env.get_sensor_data(name))[0] > 0.5)
            for name in env._CONTACT_SENSORS
        ],
        dtype=bool,
    )


def _measure(env: LeapInhandBallGraspAllegroEnv) -> dict[str, object]:
    ball_pos = env.get_ball_pos()[0]
    distances = np.linalg.norm(env.get_fingertip_pos()[0] - ball_pos[None, :], axis=-1)
    flags = _contact_flags(env)
    cond1, cond2, cond3 = env._compute_grasp_conditions()
    surface_valid, surface_gaps = env._surface_gap_quality(np.asarray([0], dtype=np.int32))
    initial_ball_z = float(np.asarray(env.state.info["initial_ball_z"])[0])
    drop_threshold = initial_ball_z - float(env.cfg.termination_drop_distance)
    return {
        "fingertip_body_origin_distances": distances.tolist(),
        "max_fingertip_distance": float(np.max(distances)),
        "fingertip_distance_threshold": float(env.cfg.grasp_max_fingertip_distance),
        "contact_flags": {
            name: bool(value) for name, value in zip(env._CONTACT_SENSORS, flags, strict=True)
        },
        "contact_count": int(np.count_nonzero(flags)),
        "ball_center_z": float(ball_pos[2]),
        "initial_ball_z": initial_ball_z,
        "drop_distance": float(env.cfg.termination_drop_distance),
        "height_threshold": drop_threshold,
        "fingertip_surface_gaps": surface_gaps[0].tolist(),
        "max_fingertip_surface_gap": float(np.max(surface_gaps[0])),
        "fingertip_surface_gap_threshold": float(
            env.cfg.grasp_max_fingertip_surface_gap
        ),
        "conditions": {
            "fingertip_distance": bool(cond1[0]),
            "contact_count": bool(cond2[0]),
            "height": bool(cond3[0]),
            "fingertip_surface_gap": bool(surface_valid[0]),
        },
    }


def _build_cfg() -> LeapInhandBallGraspAllegroCfg:
    cfg = LeapInhandBallGraspAllegroCfg(
        max_episode_seconds=2.5,
        reset_source="home",
        grasp_seed_qpos=NOMINAL_QPOS.tolist(),
        grasp_auto_save=False,
        grasp_collection_target=50_000,
        grasp_quality_check=True,
        grasp_min_contacts=2,
        grasp_max_fingertip_distance=0.1,
        grasp_max_fingertip_surface_gap=0.005,
        termination_drop_distance=0.005,
        domain_rand=DomainRandConfig(
            randomize_base_mass=False,
            random_com=False,
            randomize_gravity=False,
            push_robots=False,
            joint_noise=0.25,
            ball_vel_noise=0.0,
            ball_z_offset=0.0,
        ),
        reward_config=RewardConfigPPO(
            scales={
                "rotate": 0.0,
                "obj_linvel": 0.0,
                "pose_diff": 0.0,
                "torque": 0.0,
                "work": 0.0,
                "drop": 0.0,
            },
            angvel_clip_min=-0.5,
            angvel_clip_max=0.5,
            reset_z_threshold=0.0,
        ),
    )
    cfg.validate()
    return cfg


def run_preflight() -> dict[str, object]:
    """Settle the exact seed without autoreset, collector calls, or cache writes."""
    cfg = _build_cfg()
    env = LeapInhandBallGraspAllegroEnv(cfg, num_envs=1, backend_type="mujoco")
    try:
        env.init_state()
        env.set_autoreset(False)
        assert env.state is not None
        env_ids = np.asarray([0], dtype=np.int32)
        env._backend.set_state(
            env_ids,
            NOMINAL_QPOS[None, :],
            np.zeros((1, env.nv), dtype=np.float64),
        )
        provider = LeapAllegroGraspResetProvider()
        info_updates = provider._build_info_updates(
            env,
            NOMINAL_QPOS[None, :16],
            NOMINAL_QPOS[None, 16:19],
            NOMINAL_QPOS[None, 19:23],
        )
        env.state.info.update(info_updates)
        env.state.info["steps"][:] = 0
        env.state.terminated[:] = False
        env.state.truncated[:] = False

        initial = _measure(env)
        zero_action = np.zeros((1, env._NUM_HAND_DOF), dtype=env._np_dtype)
        requested_steps = int(np.ceil(cfg.max_episode_seconds / cfg.ctrl_dt))
        ever_terminated = False
        executed_steps = 0
        for _ in range(requested_steps):
            state = env.step(zero_action)
            executed_steps += 1
            ever_terminated |= bool(state.terminated[0])
            if state.terminated[0] or state.truncated[0]:
                break

        assert env.state is not None
        final = _measure(env)
        final_truncated = bool(env.state.truncated[0])
        final_terminated = bool(env.state.terminated[0])
        timeout_success = bool(final_truncated and not final_terminated)
        return {
            "nominal_qpos": NOMINAL_QPOS.tolist(),
            "nominal_quaternion_norm": float(np.linalg.norm(NOMINAL_QPOS[19:23])),
            "initial": initial,
            "requested_seconds": float(cfg.max_episode_seconds),
            "requested_steps": requested_steps,
            "executed_steps": executed_steps,
            "simulated_seconds": executed_steps * cfg.ctrl_dt,
            "ever_terminated": ever_terminated,
            "final_truncated": final_truncated,
            "final_terminated": final_terminated,
            "final": final,
            "timeout_success": timeout_success,
            "cache_save_called": False,
        }
    finally:
        # Deliberately bypass AllegroRotationGrasp.close(), whose responsibility
        # includes cache publication. This diagnostic owns no cache lifecycle.
        NpEnv.close(env)


def main() -> int:
    report = run_preflight()
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["timeout_success"]:
        return 0
    failed = [name for name, passed in report["final"]["conditions"].items() if not passed]
    requested_seconds = float(report["requested_seconds"])
    print(
        f"Nominal seed failed the {requested_seconds:g} s timeout-success contract; "
        f"failed conditions: {failed or ['terminated_before_timeout']}."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
