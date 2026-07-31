"""Config system verification tests.

These tests enforce that:
1. Base Hydra configs compose without legacy config groups.
2. Every supported runtime variant resolves through exactly one task owner file.
3. Final reward/env/algo sections are present on the composed config, not mounted by Python glue.
4. Backend-specific hyperparameters preserve the intended pre-refactor behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

CONF_DIR = Path(__file__).parent.parent.parent / "conf"
_PPO_MLX_TASKS = {"go1_joystick_flat", "go2_joystick_flat", "g1_walk_flat"}
_BACKENDS = ("mujoco", "motrix")
_NUMBA_ACCEL_SUPPORTED_TASK_NAMES = {
    "G1MotionTracking",
    "G1MotionTrackingSAC",
    "G1WalkFlat",
    "G1WalkRough",
}
_LEAP_MUJOCO_DYNAMICS_TASKS = (
    "leap_inhand",
    "leap_inhand_ball",
    "leap_inhand_ball_allegro",
    "leap_inhand_ball_cache",
    "leap_inhand_ball_cache_gaiting",
    "leap_inhand_ball_direct",
    "leap_inhand_ball_finger_gaiting",
    "leap_inhand_ball_grasp",
    "leap_inhand_ball_grasp_allegro",
    "leap_inhand_ball_rotation_v2",
    "leap_inhand_ball_state_cycle",
    "leap_inhand_ball_sustained",
    "leap_inhand_ball_sustained_cache",
    "leap_inhand_toss",
)
_LEAP_MOTRIX_DYNAMICS_TASKS = tuple(
    task for task in _LEAP_MUJOCO_DYNAMICS_TASKS if task != "leap_inhand_ball_grasp_allegro"
)


def _expected_backend_from_variant(name: str) -> str | None:
    for backend in _BACKENDS:
        if name == backend or name.startswith(f"{backend}_"):
            return backend
    return None


def _compose(algo_dir: str, config_name: str = "config", overrides: list[str] | None = None):
    normalized_overrides = _normalize_overrides(algo_dir, overrides)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / algo_dir), version_base="1.3"):
        return compose(config_name, overrides=normalized_overrides)


def _normalize_overrides(algo_dir: str, overrides: list[str] | None) -> list[str]:
    algo = "sac"
    normalized: list[str] = []
    task_selected = False

    for override in overrides or []:
        if override.startswith("algo="):
            algo = override.split("=", 1)[1]
            normalized.append(override)
            continue
        if override.startswith("task="):
            task_selected = True
            normalized.append(override)
            continue
        normalized.append(override)

    if not task_selected:
        if algo_dir == "offpolicy":
            normalized.append(f"task={algo}/g1_walk_flat/mujoco")
        else:
            normalized.append("task=go1_joystick_flat/mujoco")

    return normalized


def _assert_reward_populated(cfg, label: str):
    assert hasattr(cfg, "reward"), f"{label} missing cfg.reward"
    reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
    assert isinstance(reward_dict, dict), f"{label} reward must resolve to mapping"
    if "scales" in reward_dict:
        assert isinstance(reward_dict["scales"], dict) and len(reward_dict["scales"]) > 0, (
            f"{label} reward.scales must be non-empty mapping"
        )
    else:
        assert len(reward_dict) > 0, f"{label} reward config must be non-empty"


@pytest.mark.parametrize("task", _LEAP_MUJOCO_DYNAMICS_TASKS)
def test_ppo_leap_mujoco_owners_use_menagerie_dynamics(task: str) -> None:
    cfg = _compose("ppo", overrides=[f"task={task}/mujoco"])

    assert list(cfg.env.scene.joint_dynamics.joint_names) == [
        str(index) for index in range(16)
    ]
    assert cfg.env.scene.joint_dynamics.damping == pytest.approx(0.03)
    assert cfg.env.scene.joint_dynamics.frictionloss == pytest.approx(0.001)
    assert cfg.env.scene.joint_dynamics.armature == pytest.approx(0.0)
    assert cfg.env.control_config.kp == pytest.approx(3.0)
    assert cfg.env.control_config.kd == pytest.approx(0.01)


@pytest.mark.parametrize("task", _LEAP_MOTRIX_DYNAMICS_TASKS)
def test_ppo_leap_motrix_owners_keep_isaac_style_dynamics(task: str) -> None:
    cfg = _compose("ppo", overrides=[f"task={task}/motrix"])

    assert cfg.env.scene.joint_dynamics is None
    assert cfg.env.control_config.kp == pytest.approx(3.0)
    assert cfg.env.control_config.kd == pytest.approx(0.1)


def _supported_task_cases() -> list[tuple[str, str, str, str, str, list[str]]]:
    cases: list[tuple[str, str, str, str, str, list[str]]] = []

    for algo_dir in ["ppo", "appo"]:
        root = CONF_DIR / algo_dir / "task"
        for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for backend_file in sorted(task_dir.glob("*.yaml")):
                expected_backend = _expected_backend_from_variant(backend_file.stem)
                if expected_backend is None:
                    continue
                cases.append(
                    (
                        algo_dir,
                        "config",
                        task_dir.name,
                        expected_backend,
                        str(backend_file.relative_to(CONF_DIR)),
                        [f"task={task_dir.name}/{backend_file.stem}"],
                    )
                )
                if algo_dir == "ppo" and task_dir.name in _PPO_MLX_TASKS:
                    cases.append(
                        (
                            algo_dir,
                            "config_mlx",
                            task_dir.name,
                            expected_backend,
                            str(backend_file.relative_to(CONF_DIR)),
                            [f"task={task_dir.name}/{backend_file.stem}"],
                        )
                    )

    offpolicy_root = CONF_DIR / "offpolicy" / "task"
    for algo_root in sorted(path for path in offpolicy_root.iterdir() if path.is_dir()):
        for task_dir in sorted(path for path in algo_root.iterdir() if path.is_dir()):
            for backend_file in sorted(task_dir.glob("*.yaml")):
                expected_backend = _expected_backend_from_variant(backend_file.stem)
                if expected_backend is None:
                    continue
                cases.append(
                    (
                        "offpolicy",
                        "config",
                        task_dir.name,
                        expected_backend,
                        str(backend_file.relative_to(CONF_DIR)),
                        [
                            f"algo={algo_root.name}",
                            f"task={algo_root.name}/{task_dir.name}/{backend_file.stem}",
                        ],
                    )
                )

    return cases


@pytest.mark.parametrize(
    "algo_dir,config_name",
    [
        ("offpolicy", "config"),
        ("appo", "config"),
        ("ppo", "config"),
        ("ppo", "config_mlx"),
    ],
)
def test_algo_config_composes(algo_dir: str, config_name: str):
    cfg = _compose(algo_dir, config_name)
    assert cfg.training.task_name
    assert cfg.training.sim_backend == "mujoco"


def test_legacy_config_groups_removed():
    for path in [
        CONF_DIR / "ppo" / "reward",
        CONF_DIR / "ppo" / "backend_task_preset",
        CONF_DIR / "ppo" / "algo_preset",
        CONF_DIR / "ppo" / "sim_backend",
        CONF_DIR / "appo" / "reward",
        CONF_DIR / "appo" / "backend_task_preset",
        CONF_DIR / "appo" / "sim_backend",
        CONF_DIR / "offpolicy" / "reward",
        CONF_DIR / "offpolicy" / "backend_task_preset",
        CONF_DIR / "offpolicy" / "algo_preset",
        CONF_DIR / "offpolicy" / "sim_backend",
    ]:
        assert not path.exists(), f"legacy config group should be removed: {path}"


def test_task_files_keep_full_identity_without_hidden_backend_marker():
    for path in sorted(CONF_DIR.glob("*/task/**/*.yaml")):
        cfg = OmegaConf.load(path)
        cfg_dict_raw = OmegaConf.to_container(cfg, resolve=True) or {}
        assert isinstance(cfg_dict_raw, dict)
        assert "_selected_sim_backend" not in cfg_dict_raw, (
            f"task has hidden backend marker: {path}"
        )
        if path.stem not in _BACKENDS:
            continue
        training_raw = cfg_dict_raw.get("training", {})
        assert isinstance(training_raw, dict)
        assert "task_name" in training_raw, f"task missing task_name: {path}"
        assert "sim_backend" in training_raw, f"task missing sim_backend: {path}"


def test_motrix_task_files_do_not_declare_post_step_forward_sensor():
    for path in sorted(CONF_DIR.glob("*/task/**/*motrix*.yaml")):
        cfg = OmegaConf.load(path)

        assert OmegaConf.select(cfg, "env.post_step_forward_sensor") is None, (
            "post_step_forward_sensor is routed only to MuJoCo backends: "
            f"{path.relative_to(CONF_DIR)}"
        )


@pytest.mark.parametrize(
    "algo_dir,config_name,task,backend,task_file,overrides",
    _supported_task_cases(),
)
def test_supported_task_composes(
    algo_dir: str,
    config_name: str,
    task: str,
    backend: str,
    task_file: str,
    overrides: list[str],
):
    cfg = _compose(algo_dir, config_name, overrides=overrides)

    assert cfg.training.task_name, f"{task_file} should resolve task_name"
    assert cfg.training.sim_backend == backend, f"{task_file} should set backend"
    _assert_reward_populated(cfg, task_file)


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_ppo_leap_finger_gaiting_starts_with_stationary_handoff_stage(backend: str):
    cfg = _compose(
        "ppo",
        overrides=[f"task=leap_inhand_ball_finger_gaiting/{backend}"],
    )

    assert cfg.training.task_name == "LeapInhandBallFingerGaitingRotation"
    assert cfg.env.curriculum.target_speeds == [
        0.0,
        0.0,
        0.04,
        0.07,
        0.085,
        0.10,
        0.16,
        0.25,
        0.50,
    ]
    assert cfg.env.finger_gaiting.required_handoffs_by_stage == [0, 1, 1, 1, 2, 2, 3, 4, 6]
    assert len(cfg.env.curriculum.stage_durations_seconds) == 9
    assert sum(cfg.env.curriculum.stage_durations_seconds) == pytest.approx(30.0)
    assert len(cfg.reward.stage_bonuses) == 8
    assert len(cfg.reward.positive_spin_retention_floors) == 9


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_ppo_leap_sustained_cache_uses_selected_direct_rewards(backend: str):
    cfg = _compose(
        "ppo",
        overrides=[f"task=leap_inhand_ball_sustained_cache/{backend}"],
    )

    assert cfg.training.task_name == "LeapInhandBallSustainedCacheRotation"
    assert cfg.training.sim_backend == backend
    assert cfg.reward.scales.rotate == pytest.approx(1.25)
    assert cfg.reward.scales.obj_linvel == pytest.approx(-0.3)
    assert cfg.reward.scales.position_error == pytest.approx(-6.0)
    assert cfg.reward.spin_continuity_penalty_scale == pytest.approx(0.05)


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_ppo_leap_cache_gaiting_composes(backend: str):
    cfg = _compose(
        "ppo",
        overrides=[f"task=leap_inhand_ball_cache_gaiting/{backend}"],
    )

    assert cfg.training.task_name == "LeapInhandBallCacheGaitingRotation"
    assert cfg.training.sim_backend == backend
    assert cfg.reward.scales.rotate == pytest.approx(1.25)
    assert cfg.reward.scales.obj_linvel == pytest.approx(-0.3)
    assert cfg.reward.scales.position_error == pytest.approx(-6.0)
    assert cfg.reward.spin_continuity_penalty_scale == pytest.approx(0.05)
    assert cfg.env.finger_gaiting.qualified_handoff_bonus == pytest.approx(0.05)
    assert cfg.env.finger_gaiting.minimum_handoff_angle_rad == pytest.approx(0.03)
    assert list(cfg.env.finger_gaiting.release_allowed_fingers) == [True, True, True, False]
    assert list(cfg.env.finger_gaiting.required_handoffs_by_stage) == [0, 1, 1, 2, 2, 3, 4]
    assert len(cfg.env.finger_gaiting.required_handoffs_by_stage) == len(cfg.env.curriculum.target_speeds)
    assert len(cfg.env.finger_gaiting.minimum_contacts_by_stage) == len(cfg.env.curriculum.target_speeds)
    assert OmegaConf.select(cfg, "finger_gaiting") is None
    assert cfg.env.curriculum.target_speeds == [0.00, 0.04, 0.07, 0.10, 0.15, 0.20, 0.30]
    assert cfg.reward.final_success_bonus == pytest.approx(0.25)
    assert cfg.reward.failure_penalty == pytest.approx(1.0)
    assert cfg.env.termination_drop_distance == pytest.approx(0.05)
    assert cfg.env.max_episode_seconds == pytest.approx(30.0)
    assert int(cfg.env.max_episode_seconds / cfg.env.ctrl_dt) == 600


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_v3b_cache_gaiting_config_values(backend: str):
    cfg = _compose(
        "ppo",
        overrides=[f"task=leap_inhand_ball_cache_gaiting/{backend}"],
    )

    assert cfg.reward.scales.rotate == pytest.approx(1.25)
    assert cfg.reward.scales.obj_linvel == pytest.approx(-0.3)
    assert cfg.reward.scales.position_error == pytest.approx(-6.0)
    assert cfg.reward.spin_continuity_penalty_scale == pytest.approx(0.05)


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_ppo_leap_state_cycle_composes_as_independent_owner(backend: str):
    cfg = _compose(
        "ppo",
        overrides=[f"task=leap_inhand_ball_state_cycle/{backend}"],
    )

    assert cfg.training.task_name == "LeapInhandBallStateCycleRotation"
    assert cfg.training.sim_backend == backend
    assert cfg.training.eval_only is False
    assert cfg.evaluation.num_envs == 4096
    assert cfg.evaluation.num_steps == 1800
    assert cfg.evaluation.deterministic is True
    assert cfg.evaluation.write_tensorboard is True
    assert cfg.evaluation.output_dir is None
    assert cfg.reward.rotation_progress_scale == pytest.approx(3.0)
    assert cfg.reward.reverse_rotation_scale == pytest.approx(5.0)
    assert cfg.reward.rotation_target_axis_speed_rad_s == pytest.approx(0.50)
    assert cfg.reward.rotation_overspeed_scale == pytest.approx(1.0)
    assert cfg.reward.rotation_workspace_full_radius_m == pytest.approx(0.015)
    assert cfg.reward.rotation_workspace_zero_radius_m == pytest.approx(0.035)
    assert cfg.reward.pose_progress_scale == pytest.approx(4.0)
    assert cfg.reward.pose_tracking_scale == pytest.approx(0.65)
    assert cfg.reward.pose_sigma == pytest.approx(0.08)
    assert cfg.reward.transition_success_bonus == pytest.approx(0.10)
    assert cfg.reward.cycle_success_bonus == pytest.approx(0.30)
    assert cfg.reward.invalid_cycle_penalty == pytest.approx(0.15)
    assert cfg.reward.failure_penalty == pytest.approx(3.0)
    assert cfg.reward.failure_rotation_clawback_cap == pytest.approx(5.0)
    assert cfg.env.state_cycle.cycle_target_net_angle_rad == pytest.approx(0.10)
    assert cfg.env.state_cycle.a_to_b.minimum_net_angle_rad == pytest.approx(0.08)
    assert cfg.env.state_cycle.ready_to_a.minimum_net_angle_rad == pytest.approx(0.0)
    assert cfg.env.state_cycle.b_to_ready.minimum_net_angle_rad == pytest.approx(0.0)
    assert cfg.env.state_cycle.ready_to_a.hold_steps == 4
    assert cfg.env.state_cycle.ready_to_a.timeout_seconds == pytest.approx(1.5)
    assert cfg.env.state_cycle.a_to_b.timeout_seconds == pytest.approx(2.0)
    assert cfg.env.state_cycle.b_to_ready.timeout_seconds == pytest.approx(1.3)
    ready_to_a_timeout_steps = round(
        cfg.env.state_cycle.ready_to_a.timeout_seconds / cfg.env.ctrl_dt
    )
    a_to_b_timeout_steps = round(
        cfg.env.state_cycle.a_to_b.timeout_seconds / cfg.env.ctrl_dt
    )
    b_to_ready_timeout_steps = round(
        cfg.env.state_cycle.b_to_ready.timeout_seconds / cfg.env.ctrl_dt
    )
    assert ready_to_a_timeout_steps == 30
    assert a_to_b_timeout_steps == 40
    assert b_to_ready_timeout_steps == 26
    assert cfg.env.state_cycle.b_to_ready.target_pose == "ready"
    assert OmegaConf.select(cfg, "state_cycle") is None


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_v4_state_cycle_keeps_v3b_training_and_environment_background(backend: str):
    v3b = _compose(
        "ppo",
        overrides=[f"task=leap_inhand_ball_cache_gaiting/{backend}"],
    )
    v4 = _compose(
        "ppo",
        overrides=[f"task=leap_inhand_ball_state_cycle/{backend}"],
    )

    for path in (
        "algo.num_envs",
        "algo.num_steps_per_env",
        "algo.max_iterations",
        "algo.save_interval",
        "algo.actor.hidden_dims",
        "algo.actor.activation",
        "algo.actor.obs_normalization",
        "algo.actor.distribution_cfg.init_std",
        "algo.critic.hidden_dims",
        "algo.critic.activation",
        "algo.critic.obs_normalization",
        "algo.algorithm.value_loss_coef",
        "algo.algorithm.entropy_coef",
        "algo.algorithm.desired_kl",
        "env.sim_dt",
        "env.ctrl_dt",
        "env.rotation_axis",
        "env.joint_velocity_scale",
        "env.noise_config.level",
        "env.control_config.action_scale",
        "env.control_config.kp",
        "env.control_config.kd",
        "env.domain_rand.randomize_base_mass",
        "env.domain_rand.random_com",
        "env.domain_rand.randomize_gravity",
        "env.domain_rand.push_robots",
    ):
        assert OmegaConf.select(v4, path) == OmegaConf.select(v3b, path), path


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_ppo_leap_direct_rotation_uses_one_fixed_positive_target(backend: str):
    cfg = _compose(
        "ppo",
        overrides=[f"task=leap_inhand_ball_direct/{backend}"],
    )

    assert cfg.training.task_name == "LeapInhandBallDirectRotation"
    assert cfg.training.sim_backend == backend
    assert cfg.env.curriculum.direct_target_mode is True
    assert cfg.env.curriculum.target_speeds == [0.30]
    assert cfg.env.curriculum.stage_durations_seconds == [2.0]
    assert cfg.reward.stable_rotation_scale == pytest.approx(6.0)
    assert cfg.reward.stall_scale == pytest.approx(1.0)
    assert cfg.reward.reverse_scale == pytest.approx(4.0)
    assert cfg.env.control_config.action_scale == pytest.approx(1.0 / 24.0)


def test_ppo_go2_arm_manip_loco_motrix_preserves_backend_overrides():
    cfg = _compose("ppo", overrides=["task=go2_arm_manip_loco/motrix"])

    assert cfg.training.task_name == "Go2ArmManipLoco"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.num_envs == 4096
    assert cfg.algo.max_iterations == 3000
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(2.0)
    assert cfg.env.domain_rand.randomize_dof_armature is False
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False


def test_offpolicy_g1_walk_flat_motrix_sac_preserves_backend_overrides():
    cfg = _compose("offpolicy", overrides=["algo=sac", "task=sac/g1_walk_flat/motrix"])

    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 5000
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(2.2)
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False


def test_numba_accelerated_task_owners_declare_default_disabled_only_for_supported_tasks():
    matched: list[Path] = []

    for path in sorted(CONF_DIR.glob("*/task/**/*.yaml")):
        cfg = OmegaConf.load(path)
        task_name = OmegaConf.select(cfg, "training.task_name")
        declared = OmegaConf.select(cfg, "env.numba_acceleration")
        if declared is not None:
            assert task_name in _NUMBA_ACCEL_SUPPORTED_TASK_NAMES, (
                "only task owners with direct numba acceleration support may declare "
                f"env.numba_acceleration: {path.relative_to(CONF_DIR)}"
            )
        if task_name not in _NUMBA_ACCEL_SUPPORTED_TASK_NAMES:
            continue
        matched.append(path)
        assert declared is False, (
            "numba-supported task owners must expose env.numba_acceleration "
            f"with default false: {path.relative_to(CONF_DIR)}"
        )

    assert matched, "expected at least one numba-supported owner config"


def test_offpolicy_g1_walk_flat_mujoco_td3_uses_td3_task_owner():
    cfg = _compose("offpolicy", overrides=["algo=td3", "task=td3/g1_walk_flat/mujoco"])

    assert cfg.training.task_name == "G1WalkFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.max_iterations == 100000
    assert cfg.algo.tau == pytest.approx(0.1)
    assert cfg.algo.actor_hidden_dim == 512
    assert cfg.algo.critic_hidden_dim == 1024
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(2.0)
    assert cfg.env.control_config.action_scale == pytest.approx(1.0)


def test_offpolicy_td3_go2_joystick_flat_motrix_composes():
    cfg = _compose(
        "offpolicy",
        overrides=["algo=td3", "task=td3/go2_joystick_flat/motrix"],
    )

    assert cfg.training.task_name == "Go2JoystickFlat"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.algo == "td3"
    assert cfg.algo.tau == pytest.approx(0.1)
    assert cfg.algo.algo_params.weight_decay == pytest.approx(0.1)
    assert cfg.algo.algo_params.policy_noise == pytest.approx(0.2)
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(1.0)
    assert cfg.reward.base_height_target == pytest.approx(0.3)


def test_offpolicy_td3_go1_joystick_flat_motrix_composes():
    cfg = _compose(
        "offpolicy",
        overrides=["algo=td3", "task=td3/go1_joystick_flat/motrix"],
    )

    assert cfg.training.task_name == "Go1JoystickFlat"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.algo == "td3"
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(1.0)


def test_offpolicy_g1_walk_flat_motrix_preserves_backend_specific_algo_value():
    mujoco_cfg = _compose("offpolicy", overrides=["algo=sac", "task=sac/g1_walk_flat/mujoco"])
    motrix_cfg = _compose("offpolicy", overrides=["algo=sac", "task=sac/g1_walk_flat/motrix"])

    assert mujoco_cfg.algo.use_symmetry is True
    assert motrix_cfg.algo.use_symmetry is False


def test_ppo_g1_backend_specific_hyperparams_remain_separate():
    mujoco_cfg = _compose("ppo", overrides=["task=g1_walk_flat/mujoco"])
    motrix_cfg = _compose("ppo", overrides=["task=g1_walk_flat/motrix"])

    assert mujoco_cfg.algo.max_iterations == 2200
    assert mujoco_cfg.algo.empirical_normalization is False
    assert mujoco_cfg.algo.obs_groups.actor == ["actor"]

    assert motrix_cfg.algo.max_iterations == 2200
    assert motrix_cfg.algo.empirical_normalization is True
    assert motrix_cfg.algo.obs_groups.actor == ["policy"]
    assert OmegaConf.select(motrix_cfg, "env.motrix_max_iterations") is None
    assert motrix_cfg.env.control_config.action_scale == pytest.approx(0.5)
    assert motrix_cfg.env.commands.vel_limit == [[0.4, 0.0, 0.0], [0.7, 0.0, 0.0]]
    assert motrix_cfg.env.gait_phase_init_mode == "offset_phase"
    assert motrix_cfg.reward.scales.tracking_lin_vel == pytest.approx(2.0)
    assert motrix_cfg.reward.scales.tracking_ang_vel == pytest.approx(0.25)
    assert motrix_cfg.reward.scales.forward_progress == pytest.approx(0.0)
    assert motrix_cfg.reward.scales.under_speed == pytest.approx(-0.2)
    assert motrix_cfg.reward.scales.penalty_feet_ori == pytest.approx(0.0)
    assert motrix_cfg.reward.scales.feet_phase == pytest.approx(1.2)
    assert motrix_cfg.reward.scales.feet_phase_contrast == pytest.approx(1.5)
    assert motrix_cfg.reward.scales.feet_phase_contact == pytest.approx(1.0)
    assert motrix_cfg.reward.scales.feet_double_stance == pytest.approx(-1.0)
    assert motrix_cfg.reward.scales.base_height == pytest.approx(-120.0)
    assert motrix_cfg.reward.scales.pose == pytest.approx(-0.05)
    assert motrix_cfg.reward.base_height_target == pytest.approx(0.765)
    assert motrix_cfg.reward.min_forward_speed_for_gait_reward == pytest.approx(0.05)
    assert motrix_cfg.reward.min_base_height == pytest.approx(0.5)
    assert motrix_cfg.reward.max_tilt_deg == pytest.approx(35.0)


@pytest.mark.parametrize(
    ("algo_dir", "overrides"),
    [
        ("ppo", ["task=g1_walk_flat/mujoco"]),
        ("ppo_him", ["task=go2_arm_manip_loco/mujoco"]),
        ("appo", ["task=g1_walk_flat/mujoco"]),
        ("offpolicy", ["algo=sac", "task=sac/g1_walk_flat/mujoco"]),
        ("offpolicy", ["algo=flashsac", "task=flashsac/g1_walk_flat/mujoco"]),
    ],
)
def test_post_step_forward_sensor_defaults_false_outside_sharpa_mujoco(
    algo_dir: str, overrides: list[str]
):
    cfg = _compose(algo_dir, overrides=overrides)

    assert cfg.env.post_step_forward_sensor is False


@pytest.mark.parametrize(
    ("algo_dir", "overrides"),
    [
        ("ppo", ["task=sharpa_inhand/mujoco"]),
        ("ppo", ["task=sharpa_inhand/mujoco_hora"]),
        ("ppo", ["task=sharpa_inhand_grasp/mujoco"]),
        ("appo", ["task=sharpa_inhand/mujoco"]),
        ("appo", ["task=sharpa_inhand/mujoco_hora"]),
        ("offpolicy", ["algo=sac", "task=sac/sharpa_inhand/mujoco_hora"]),
        ("hora_distill", ["task=sharpa_inhand/mujoco"]),
    ],
)
def test_post_step_forward_sensor_enabled_for_sharpa_mujoco(algo_dir: str, overrides: list[str]):
    cfg = _compose(algo_dir, overrides=overrides)

    assert cfg.env.post_step_forward_sensor is True


def test_mujoco_post_step_forward_sensor_can_be_overridden():
    override_cfg = _compose(
        "appo",
        overrides=["task=sharpa_inhand/mujoco_hora", "env.post_step_forward_sensor=false"],
    )

    assert override_cfg.env.post_step_forward_sensor is False


def test_appo_adaptive_lr_factors_are_overridden_only_by_dex_hand_owners():
    g1_cfg = _compose("appo", overrides=["task=g1_walk_flat/mujoco"])
    allegro_cfg = _compose("appo", overrides=["task=allegro_inhand/mujoco"])
    allegro_motrix_cfg = _compose("appo", overrides=["task=allegro_inhand/motrix"])
    sharpa_cfg = _compose("appo", overrides=["task=sharpa_inhand/mujoco"])
    sharpa_hora_cfg = _compose("appo", overrides=["task=sharpa_inhand/mujoco_hora"])

    assert g1_cfg.algo.algorithm.adaptive_kl_factor == pytest.approx(1.2)
    assert g1_cfg.algo.algorithm.adaptive_lr_factor == pytest.approx(1.1)
    assert allegro_cfg.algo.algorithm.adaptive_kl_factor == pytest.approx(2.0)
    assert allegro_cfg.algo.algorithm.adaptive_lr_factor == pytest.approx(1.5)
    assert allegro_motrix_cfg.algo.algorithm.adaptive_kl_factor == pytest.approx(2.0)
    assert allegro_motrix_cfg.algo.algorithm.adaptive_lr_factor == pytest.approx(1.5)
    assert sharpa_cfg.algo.algorithm.adaptive_kl_factor == pytest.approx(1.2)
    assert sharpa_cfg.algo.algorithm.adaptive_lr_factor == pytest.approx(1.1)
    assert sharpa_hora_cfg.algo.algorithm.adaptive_kl_factor == pytest.approx(1.2)
    assert sharpa_hora_cfg.algo.algorithm.adaptive_lr_factor == pytest.approx(1.1)


def test_ppo_go1_motrix_preserves_reward_and_algo_values():
    cfg = _compose("ppo", overrides=["task=go1_joystick_flat/motrix"])

    assert cfg.algo.max_iterations == 151
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.policy.init_noise_std == pytest.approx(0.5)
    assert cfg.algo.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(1.0)
    assert cfg.env.commands.vel_limit == [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]


def test_ppo_go2_motrix_preserves_backend_env_overrides():
    cfg = _compose("ppo", overrides=["task=go2_joystick_flat/motrix"])

    assert cfg.algo.num_envs == 1024
    assert cfg.algo.empirical_normalization is True
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False


def test_ppo_go2w_mujoco_uses_motor_owner_dr_path():
    cfg = _compose("ppo", overrides=["task=go2w_joystick_flat/mujoco"])

    assert cfg.training.task_name == "Go2WJoystickFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.env.commands.vel_limit == [[0.0, 0.0, -1.0], [1.0, 0.0, 1.0]]
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False
    assert cfg.env.control_config.action_scale == pytest.approx(0.5)
    assert cfg.env.control_config.Kp == pytest.approx(50.0)
    assert cfg.env.control_config.Kd == pytest.approx(1.5)
    assert cfg.env.control_config.wheel_action_scale == pytest.approx(10.0)
    assert cfg.env.control_config.wheel_Kd == pytest.approx(0.5)
    assert cfg.reward.scales.tracking_ang_vel == pytest.approx(0.75)
    assert cfg.reward.scales.orientation == pytest.approx(-2.0)
    assert cfg.reward.scales.upward == pytest.approx(1.0)
    assert cfg.reward.base_height_target == pytest.approx(0.4)
    assert cfg.reward.scales.torques < 0.0


def test_ppo_go2w_motrix_uses_motor_owner_dr_path():
    cfg = _compose("ppo", overrides=["task=go2w_joystick_flat/motrix"])

    assert cfg.training.task_name == "Go2WJoystickFlat"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.env.render_offset_mode == "zero"
    assert cfg.env.commands.vel_limit == [[0.0, 0.0, -1.0], [1.0, 0.0, 1.0]]
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False
    assert cfg.env.control_config.action_scale == pytest.approx(0.5)
    assert cfg.env.control_config.Kp == pytest.approx(50.0)
    assert cfg.env.control_config.Kd == pytest.approx(1.5)
    assert cfg.env.control_config.wheel_action_scale == pytest.approx(10.0)
    assert cfg.env.control_config.wheel_Kd == pytest.approx(0.5)
    assert cfg.reward.scales.tracking_ang_vel == pytest.approx(0.75)
    assert cfg.reward.scales.orientation == pytest.approx(-2.0)
    assert cfg.reward.scales.upward == pytest.approx(1.0)
    assert cfg.reward.scales.torques < 0.0


def test_ppo_go2w_motrix_uses_motor_owner_scene_path():
    cfg = _compose("ppo", overrides=["task=go2w_joystick_flat/motrix"])

    assert cfg.training.task_name == "Go2WJoystickFlat"
    assert cfg.training.sim_backend == "motrix"
    assert "model_file" not in cfg.env
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False
    assert cfg.env.control_config.wheel_action_scale == pytest.approx(10.0)
    assert cfg.reward.scales.torques < 0.0


def test_ppo_go2w_rough_mujoco_uses_terrain_generator():
    cfg = _compose("ppo", overrides=["task=go2w_joystick_rough/mujoco"])

    assert cfg.training.task_name == "Go2WJoystickRough"
    assert cfg.training.sim_backend == "mujoco"
    assert str(cfg.env.scene.model_file).endswith("src/unilab/assets/robots/go2w/go2w_mujoco.xml")
    assert cfg.env.scene.terrain.hfield_name == "terrain_hfield"
    assert cfg.env.scene.terrain.geom_name == "floor"
    assert cfg.env.terrain_scan.hfield_name == "terrain_hfield"
    assert cfg.env.terrain_scan.geom_name == "floor"
    assert cfg.env.commands.resampling_time == pytest.approx(10.0)
    assert cfg.env.commands.heading_command is True
    assert cfg.env.commands.vel_limit == [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
    assert cfg.env.commands.heading_range == pytest.approx([-3.141592653589793, 3.141592653589793])
    assert cfg.env.control_config.clip_actions == pytest.approx(100.0)
    assert cfg.env.control_config.action_scale == pytest.approx(0.25)
    assert cfg.env.control_config.hip_action_scale == pytest.approx(0.125)
    assert cfg.env.control_config.wheel_action_scale == pytest.approx(5.0)
    assert cfg.env.domain_rand.randomize_kp is True
    assert cfg.env.domain_rand.randomize_kd is True
    assert cfg.env.domain_rand.kp_multiplier_range == [0.5, 1.0]
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(3.0)
    assert cfg.reward.scales.hip_pos == pytest.approx(-2.0)
    assert cfg.reward.scales.joint_mirror == pytest.approx(-0.05)
    assert cfg.reward.only_positive_rewards is False
    assert cfg.algo.max_iterations == 1200


def test_ppo_go2w_rough_motrix_uses_yaw_reset_and_strong_control():
    cfg = _compose("ppo", overrides=["task=go2w_joystick_rough/motrix"])

    assert cfg.training.task_name == "Go2WJoystickRough"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.env.commands.vel_limit == [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
    assert cfg.env.commands.heading_range == pytest.approx([-3.141592653589793, 3.141592653589793])
    assert cfg.env.control_config.action_scale == pytest.approx(0.25)
    assert cfg.env.control_config.hip_action_scale == pytest.approx(0.125)
    assert cfg.env.control_config.wheel_action_scale == pytest.approx(5.0)
    assert cfg.env.domain_rand.randomize_kp is True
    assert cfg.env.domain_rand.randomize_kd is True
    assert cfg.reward.scales.orientation == pytest.approx(-2.0)
    assert cfg.reward.scales.hip_pos == pytest.approx(-0.5)
    assert cfg.reward.scales.upward == pytest.approx(1.0)
    assert cfg.algo.max_iterations == 1200


def test_offpolicy_g1_walk_flat_motrix_preserves_backend_env_overrides():
    cfg = _compose("offpolicy", overrides=["algo=sac", "task=sac/g1_walk_flat/motrix"])

    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 5000
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False


def test_offpolicy_flashsac_go2_joystick_mujoco_enables_full_dr_stack():
    mujoco_cfg = _compose(
        "offpolicy",
        overrides=["algo=flashsac", "task=flashsac/go2_joystick_flat/mujoco"],
    )

    assert mujoco_cfg.training.task_name == "Go2JoystickFlat"
    assert mujoco_cfg.training.sim_backend == "mujoco"

    assert mujoco_cfg.env.domain_rand.randomize_kp is True
    assert mujoco_cfg.env.domain_rand.randomize_kd is True
    assert mujoco_cfg.env.domain_rand.randomize_base_mass is True
    assert mujoco_cfg.env.domain_rand.random_com is True
    assert mujoco_cfg.env.domain_rand.randomize_gravity is True
    assert mujoco_cfg.env.domain_rand.push_robots is True
    assert mujoco_cfg.env.noise_config.level == pytest.approx(1.0)


def test_cli_override_beats_task_defaults():
    cfg = _compose(
        "ppo",
        overrides=["task=g1_walk_flat/motrix", "algo.max_iterations=1"],
    )

    assert cfg.algo.max_iterations == 1
    assert cfg.algo.empirical_normalization is True
