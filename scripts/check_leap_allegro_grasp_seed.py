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
        1.615086902652708,
        0.05592890833161862,
        0.287868545519634,
        0.05789584343082383,
        1.385416217870236,
        0.019181783056566676,
        -0.020953303695846966,
        0.16266530072328944,
        1.6940339396072988,
        -0.042944887708793206,
        0.08608101675297872,
        0.04204787407706335,
        1.6353669902981327,
        0.5618974997807215,
        -0.1469763717278566,
        0.520989074906656,
        -0.03218510819218568,
        0.03676290825215784,
        0.6626576150346267,
        0.92598793755222,
        -0.010678814768592742,
        -0.06800823346343168,
        0.37122389821253027,
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
    return {
        "fingertip_body_origin_distances": distances.tolist(),
        "max_fingertip_distance": float(np.max(distances)),
        "fingertip_distance_threshold": float(env.cfg.grasp_max_fingertip_distance),
        "contact_flags": {
            name: bool(value) for name, value in zip(env._CONTACT_SENSORS, flags, strict=True)
        },
        "contact_count": int(np.count_nonzero(flags)),
        "ball_center_z": float(ball_pos[2]),
        "height_threshold": float(env._reward_cfg.reset_z_threshold),
        "conditions": {
            "fingertip_distance": bool(cond1[0]),
            "contact_count": bool(cond2[0]),
            "height": bool(cond3[0]),
        },
    }


def _build_cfg() -> LeapInhandBallGraspAllegroCfg:
    cfg = LeapInhandBallGraspAllegroCfg(
        max_episode_seconds=3.0,
        reset_source="home",
        grasp_seed_qpos=NOMINAL_QPOS.tolist(),
        grasp_auto_save=False,
        grasp_collection_target=50_000,
        grasp_quality_check=True,
        grasp_min_contacts=2,
        grasp_max_fingertip_distance=0.1061,
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
            reset_z_threshold=0.6576576150346267,
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
        requested_steps = int(np.ceil(3.0 / cfg.ctrl_dt))
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
    print(
        "Nominal seed failed the 3.0 s timeout-success contract; "
        f"failed conditions: {failed or ['terminated_before_timeout']}."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
