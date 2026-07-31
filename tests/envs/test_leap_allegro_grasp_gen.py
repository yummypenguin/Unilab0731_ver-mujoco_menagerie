from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.envs.manipulation.allegro_inhand.grasp_gen import (
    AllegroRotationGrasp,
)
from unilab.envs.manipulation.allegro_inhand.rotation import (
    AllegroRotationDomainRandomizationProvider,
    AllegroRotationPPO,
)
from unilab.envs.manipulation.leap_inhand.ball_grasp_allegro import (
    LeapAllegroGraspResetProvider,
    LeapInhandBallGraspAllegroCfg,
    LeapInhandBallGraspAllegroEnv,
    quantized_grasp_key,
)
from unilab.envs.manipulation.leap_inhand.ball_grasp_gen import (
    LeapInhandBallGraspEnv,
)

ROOT = Path(__file__).parent.parent.parent
CONF_DIR = ROOT / "conf" / "ppo"
SCENE = ROOT / "src" / "unilab" / "assets" / "robots" / "leap_hand" / "scene_ball.xml"
STRICT_CACHE = "robots/leap_hand/caches/ball_grasp_official_50k.npy"
NEW_CACHE = "robots/leap_hand/caches/ball_grasp_allegro_dedup_50k.npy"
SEED = np.asarray(
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


def _compose_cfg():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(
            "config",
            overrides=["task=leap_inhand_ball_grasp_allegro/mujoco"],
        )


def _provider_env(*, noise: float = 0.25):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            grasp_seed_qpos=SEED.tolist(),
            domain_rand=SimpleNamespace(joint_noise=noise),
        ),
        _NUM_HAND_DOF=16,
        _ctrl_lower=np.full(16, -10.0),
        _ctrl_upper=np.full(16, 10.0),
        nv=22,
    )


def _dedup_env(*, enabled: bool = True):
    env = object.__new__(LeapInhandBallGraspAllegroEnv)
    env._cfg = SimpleNamespace(
        grasp_dedup_enabled=enabled,
        grasp_dedup_joint_resolution=0.01,
        grasp_dedup_ball_position_resolution=0.001,
    )
    env._saved_grasp_keys = set()
    env._saved_grasping_states = []
    env._dedup_candidates = 0
    env._dedup_rejected = 0
    env._dedup_accepted = 0
    env._state = None
    return env


def _row() -> np.ndarray:
    row = np.zeros(23, dtype=np.float32)
    row[19] = 1.0
    return row


def _collector_env(tmp_path: Path, rows: np.ndarray):
    env = _dedup_env()
    env._cfg.grasp_quality_check = False
    env._cfg.grasp_collection_target = 50_000
    env._cfg.grasp_cache_path = str(tmp_path / "unused.npy")
    env._cfg.grasp_auto_save = False
    env._state = NpEnvState(
        obs={},
        reward=np.zeros(len(rows)),
        terminated=np.zeros(len(rows), dtype=bool),
        truncated=np.ones(len(rows), dtype=bool),
        info={
            "curr_dof_pos": rows[:, :16].copy(),
            "curr_ball_pos": rows[:, 16:19].copy(),
            "curr_ball_quat": rows[:, 19:23].copy(),
            "log": {},
        },
    )
    env._save_grasp_cache = lambda *args, **kwargs: None
    env._stop_collection = lambda: None
    env.get_hand_dof_pos = lambda: rows[:, :16]
    env.get_ball_pos = lambda: rows[:, 16:19]
    env.get_ball_quat = lambda: rows[:, 19:23]
    return env


def _load_inspector_module():
    path = ROOT / "scripts" / "inspect_leap_allegro_grasp_cache.py"
    spec = importlib.util.spec_from_file_location("inspect_leap_allegro_cache", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_registration_and_hydra_contract() -> None:
    registered = registry.list_registered_envs()
    assert "LeapInhandBallGraspAllegro" in registered
    assert set(registered["LeapInhandBallGraspAllegro"]["available_backends"]) == {
        "mujoco",
        "motrix",
    }

    cfg = _compose_cfg()
    seed = np.asarray(cfg.env.grasp_seed_qpos, dtype=np.float64)
    assert cfg.training.task_name == "LeapInhandBallGraspAllegro"
    assert seed.shape == (23,)
    assert np.isfinite(seed).all()
    np.testing.assert_array_equal(seed, SEED)
    assert np.linalg.norm(seed[19:23]) == pytest.approx(1.0)
    assert cfg.env.grasp_max_fingertip_distance == pytest.approx(0.1061)
    assert cfg.reward.reset_z_threshold == pytest.approx(0.6576576150346267)
    assert cfg.env.max_episode_seconds == pytest.approx(3.0)
    assert cfg.env.grasp_min_contacts == 2
    assert cfg.env.domain_rand.joint_noise == pytest.approx(0.25)
    assert cfg.env.grasp_cache_path == NEW_CACHE
    assert cfg.env.grasp_cache_path != STRICT_CACHE


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"grasp_seed_qpos": [0.0] * 22}, "shape"),
        ({"grasp_seed_qpos": [0.0] * 23}, "non-zero"),
        ({"grasp_max_fingertip_distance": 0.0}, "positive"),
        ({"grasp_collection_target": 0}, "positive"),
        ({"grasp_min_contacts": 5}, "within"),
        ({"grasp_dedup_joint_resolution": 0.0}, "positive"),
        ({"grasp_dedup_ball_position_resolution": 0.0}, "positive"),
    ],
)
def test_config_validation_rejects_invalid_values(overrides, match) -> None:
    cfg = LeapInhandBallGraspAllegroCfg(grasp_seed_qpos=SEED.tolist())
    for name, value in overrides.items():
        setattr(cfg, name, value)
    with pytest.raises(ValueError, match=match):
        cfg.validate()


def test_proposal_sampling_shapes_ranges_and_fixed_ball() -> None:
    provider = LeapAllegroGraspResetProvider()
    env = _provider_env()
    np.random.seed(123)

    hand_qpos, ball_pos, ball_quat, qvel = provider._sample_reset_state(env, 128)

    assert hand_qpos.shape == (128, 16)
    assert ball_pos.shape == (128, 3)
    assert ball_quat.shape == (128, 4)
    assert qvel.shape == (128, 22)
    offsets = hand_qpos - SEED[None, :16]
    assert np.all(offsets >= -0.25)
    assert np.all(offsets <= 0.25)
    np.testing.assert_array_equal(ball_pos, np.broadcast_to(SEED[16:19], (128, 3)))
    np.testing.assert_array_equal(ball_quat, np.broadcast_to(SEED[19:23], (128, 4)))
    assert not np.allclose(ball_quat, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(qvel, 0.0)


def test_proposal_sampling_clips_joint_limits(monkeypatch) -> None:
    provider = LeapAllegroGraspResetProvider()
    env = _provider_env()
    env._ctrl_lower = SEED[:16] - 0.1
    env._ctrl_upper = SEED[:16] + 0.1
    monkeypatch.setattr(
        np.random,
        "uniform",
        lambda *args, **kwargs: np.full(kwargs["size"], 0.25),
    )

    hand_qpos, _, _, _ = provider._sample_reset_state(env, 3)

    np.testing.assert_allclose(
        hand_qpos,
        np.broadcast_to(env._ctrl_upper, hand_qpos.shape),
    )


def test_inherited_info_updates_set_prev_ctrl_from_sampled_qpos() -> None:
    provider = LeapAllegroGraspResetProvider()
    hand = np.broadcast_to(SEED[:16], (2, 16)).copy()
    hand[1, 0] += 0.1
    ball_pos = np.broadcast_to(SEED[16:19], (2, 3)).copy()
    ball_quat = np.broadcast_to(SEED[19:23], (2, 4)).copy()
    env = SimpleNamespace(
        _dof_mid=np.zeros(16),
        _dof_range=np.ones(16),
        _NUM_LAG_STEPS=3,
        _NUM_OBS_PER_STEP=35,
        _num_action=16,
    )

    updates = provider._build_info_updates(env, hand, ball_pos, ball_quat)

    np.testing.assert_allclose(updates["prev_ctrl"], hand, rtol=1e-6)
    np.testing.assert_allclose(updates["init_pose"], hand, rtol=1e-6)
    np.testing.assert_allclose(updates["prev_dof_pos"], hand, rtol=1e-6)
    np.testing.assert_allclose(updates["prev_ball_pos"], ball_pos, rtol=1e-6)
    np.testing.assert_allclose(updates["prev_ball_quat"], ball_quat, rtol=1e-6)
    np.testing.assert_array_equal(updates["current_actions"], 0.0)
    np.testing.assert_array_equal(updates["last_actions"], 0.0)


def test_external_actions_are_ignored_and_sampled_target_is_held() -> None:
    env = object.__new__(LeapInhandBallGraspAllegroEnv)
    env._num_envs = 2
    env._num_action = 16
    env._np_dtype = np.float32
    env.default_angles = np.zeros(16, dtype=np.float32)
    env._ctrl_lower = np.full(16, -10.0)
    env._ctrl_upper = np.full(16, 10.0)
    env._cfg = SimpleNamespace(control_config=SimpleNamespace(action_scale=1.0 / 24.0))
    sampled = np.stack([SEED[:16], SEED[:16] + 0.01]).astype(np.float32)
    state = NpEnvState(
        obs={},
        reward=np.zeros(2),
        terminated=np.zeros(2, dtype=bool),
        truncated=np.zeros(2, dtype=bool),
        info={"prev_ctrl": sampled.copy()},
    )

    ctrl = env.apply_action(np.ones((2, 16), dtype=np.float32), state)

    np.testing.assert_array_equal(ctrl, sampled)
    np.testing.assert_array_equal(state.info["current_actions"], 0.0)


def test_three_conditions_use_strict_boundaries() -> None:
    env = object.__new__(LeapInhandBallGraspAllegroEnv)
    env._cfg = SimpleNamespace(
        grasp_max_fingertip_distance=0.1061,
        grasp_min_contacts=2,
    )
    env._reward_cfg = SimpleNamespace(reset_z_threshold=0.6576576150346267)
    ball = np.asarray([[0.0, 0.0, 0.6576576150346267]])
    tips = np.asarray([[[0.1061, 0.0, ball[0, 2]]] * 4])
    env.get_ball_pos = lambda: ball
    env.get_fingertip_pos = lambda: tips
    env._contact_count = lambda: np.asarray([1], dtype=np.int32)

    cond1, cond2, cond3 = env._compute_grasp_conditions()

    assert len((cond1, cond2, cond3)) == 3
    assert not cond1[0]
    assert not cond2[0]
    assert not cond3[0]

    tips[:, :, 0] = 0.10609
    ball[:, 2] += 1e-9
    env._contact_count = lambda: np.asarray([2], dtype=np.int32)
    cond1, cond2, cond3 = env._compute_grasp_conditions()
    assert cond1[0] and cond2[0] and cond3[0]


def test_contact_count_allows_index_middle_without_thumb_and_ignores_palm() -> None:
    env = object.__new__(LeapInhandBallGraspAllegroEnv)
    sensor_values = {
        "leap_index_contact": np.asarray([[1.0]]),
        "leap_middle_contact": np.asarray([[1.0]]),
        "leap_ring_contact": np.asarray([[0.0]]),
        "leap_thumb_contact": np.asarray([[0.0]]),
        "leap_palm_contact": np.asarray([[1.0]]),
    }
    env.get_sensor_data = lambda name: sensor_values[name]

    count = env._contact_count()

    np.testing.assert_array_equal(count, [2])
    assert "leap_palm_contact" not in env._CONTACT_SENSORS


def test_first_update_has_no_warmup_and_terminates(monkeypatch) -> None:
    env = object.__new__(LeapInhandBallGraspAllegroEnv)
    env._num_envs = 1
    env._np_dtype = np.float32
    env._enable_reward_log = False
    env._cfg = SimpleNamespace(grasp_quality_check=True)
    env._compute_grasp_conditions = lambda: (
        np.asarray([False]),
        np.asarray([True]),
        np.asarray([True]),
    )
    state = NpEnvState(
        obs={},
        reward=np.ones(1),
        terminated=np.zeros(1, dtype=bool),
        truncated=np.zeros(1, dtype=bool),
        info={"steps": np.zeros(1, dtype=np.uint32)},
    )
    monkeypatch.setattr(AllegroRotationPPO, "update_state", lambda self, value: value)

    result = AllegroRotationGrasp.update_state(env, state)

    assert result.terminated[0]
    np.testing.assert_array_equal(result.reward, 0.0)


def test_timeout_success_collector_uses_final_settled_rows(tmp_path) -> None:
    rows = np.stack([_row(), _row(), _row()])
    rows[0, 0] = 0.123
    rows[1, 0] = 0.456
    rows[2, 0] = 0.789
    env = _collector_env(tmp_path, rows)
    env.state.terminated[1] = True
    env.state.truncated[2] = False

    env._collect_successful_grasps(np.asarray([0, 1, 2], dtype=np.int32))

    assert env._total_saved_grasps() == 1
    assert env._saved_grasping_states[0].dtype == np.float32
    assert env._saved_grasping_states[0].shape == (1, 23)
    assert env._saved_grasping_states[0][0, 0] == pytest.approx(0.123)


def test_dedup_rejects_same_batch_cross_batch_and_quaternion_only_rows() -> None:
    env = _dedup_env()
    env._state = NpEnvState(
        obs={},
        reward=np.zeros(1),
        terminated=np.zeros(1, dtype=bool),
        truncated=np.zeros(1, dtype=bool),
        info={"log": {}},
    )
    first = _row()
    quaternion_only = first.copy()
    quaternion_only[19:23] = [0.0, 1.0, 0.0, 0.0]

    kept = env._filter_grasp_rows(np.stack([first, first, quaternion_only]))
    later = env._filter_grasp_rows(first[None, :])

    assert kept.shape == (1, 23)
    assert later.shape == (0, 23)
    assert env._dedup_candidates == 4
    assert env._dedup_accepted == 1
    assert env._dedup_rejected == 3
    assert len(env._saved_grasp_keys) == 1
    assert env.state.info["log"]["grasp/dedup_rejection_rate"] == pytest.approx(0.75)


def test_dedup_quantization_cells_and_quaternion_exclusion() -> None:
    base = _row()
    same_joint_cell = base.copy()
    same_joint_cell[0] = 0.004
    other_joint_cell = base.copy()
    other_joint_cell[0] = 0.006
    same_ball_cell = base.copy()
    same_ball_cell[16] = 0.0004
    other_ball_cell = base.copy()
    other_ball_cell[16] = 0.0006
    other_quaternion = base.copy()
    other_quaternion[19:23] = [0.0, 0.0, 1.0, 0.0]

    def key(row):
        return quantized_grasp_key(
            row,
            joint_resolution=0.01,
            ball_position_resolution=0.001,
        )

    assert key(base) == key(same_joint_cell)
    assert key(base) != key(other_joint_cell)
    assert key(base) == key(same_ball_cell)
    assert key(base) != key(other_ball_cell)
    assert key(base) == key(other_quaternion)
    assert len(key(base)) == 19


def test_duplicate_does_not_increase_saved_count() -> None:
    env = _dedup_env()
    row = _row()[None, :]
    first = env._filter_grasp_rows(row)
    env._saved_grasping_states.append(first)
    duplicate = env._filter_grasp_rows(row)
    if duplicate.shape[0]:
        env._saved_grasping_states.append(duplicate)

    assert env._total_saved_grasps() == 1


def test_dedup_disabled_keeps_every_row() -> None:
    env = _dedup_env(enabled=False)
    rows = np.stack([_row(), _row()])

    kept = env._filter_grasp_rows(rows)

    np.testing.assert_array_equal(kept, rows)
    assert env._dedup_candidates == 2
    assert env._dedup_accepted == 2
    assert env._dedup_rejected == 0


def test_allegro_default_filter_is_identity_and_collector_keeps_duplicates(tmp_path) -> None:
    rows = np.stack([_row(), _row()])
    env = object.__new__(AllegroRotationGrasp)
    assert env._filter_grasp_rows(rows) is rows
    env._cfg = SimpleNamespace(
        grasp_quality_check=False,
        grasp_collection_target=50_000,
        grasp_cache_path=str(tmp_path / "unused.npy"),
        grasp_auto_save=False,
    )
    env._saved_grasping_states = []
    env._state = NpEnvState(
        obs={},
        reward=np.zeros(2),
        terminated=np.zeros(2, dtype=bool),
        truncated=np.ones(2, dtype=bool),
        info={
            "curr_dof_pos": rows[:, :16],
            "curr_ball_pos": rows[:, 16:19],
            "curr_ball_quat": rows[:, 19:23],
            "log": {},
        },
    )
    env._save_grasp_cache = lambda *args, **kwargs: None
    env._stop_collection = lambda: None
    env.get_hand_dof_pos = lambda: rows[:, :16]
    env.get_ball_pos = lambda: rows[:, 16:19]
    env.get_ball_quat = lambda: rows[:, 19:23]

    env._collect_successful_grasps(np.asarray([0, 1], dtype=np.int32))

    assert env._total_saved_grasps() == 2


def test_new_environment_does_not_define_strict_leap_paths() -> None:
    forbidden_attributes = {
        "grasp_require_thumb_contact",
        "grasp_warmup_seconds",
        "grasp_max_fingertip_surface_gap",
        "grasp_max_ball_drift",
        "grasp_max_ball_linear_speed",
        "grasp_max_ball_angular_speed",
        "grasp_max_joint_speed",
        "grasp_max_abs_work",
        "grasp_max_self_penetration",
        "grasp_max_object_penetration",
        "grasp_frontier_fraction",
    }
    cfg_fields = LeapInhandBallGraspAllegroCfg.__dataclass_fields__
    assert forbidden_attributes.isdisjoint(cfg_fields)
    assert "_strict_quality_mask" not in LeapInhandBallGraspAllegroEnv.__dict__
    assert "_fingertip_surface_quality" not in LeapInhandBallGraspAllegroEnv.__dict__
    assert "_penetration_quality" not in LeapInhandBallGraspAllegroEnv.__dict__
    assert "replay_validate_grasp_cache_rows" not in LeapInhandBallGraspAllegroEnv.__dict__
    assert "update_state" not in LeapInhandBallGraspAllegroEnv.__dict__
    assert not issubclass(LeapInhandBallGraspAllegroEnv, LeapInhandBallGraspEnv)
    source = inspect.getsource(LeapAllegroGraspResetProvider._sample_reset_state)
    assert "frontier" not in source
    assert "ball_position_noise" not in source
    environment_source = inspect.getsource(LeapInhandBallGraspAllegroEnv)
    for forbidden_path in (
        "_backend.set_state",
        "_fingertip_surface_quality",
        "_penetration_quality",
        "replay_validate_grasp_cache_rows",
        "_strict_quality_mask",
        "grasp_require_thumb_contact",
    ):
        assert forbidden_path not in environment_source


def test_direct_save_uses_float32_unique_rows_and_temporary_path(tmp_path) -> None:
    env = _dedup_env()
    rows = np.stack([_row(), _row(), _row()])
    rows[2, 0] = 0.02
    unique = env._filter_grasp_rows(rows)
    env._saved_grasping_states.append(unique)
    output = tmp_path / "cache.npy"
    env._cfg.grasp_cache_path = str(output)
    env._cfg.grasp_collection_target = 2
    env._grasp_cache_saved = False

    AllegroRotationGrasp._save_grasp_cache(env, force=True)

    saved = np.load(output)
    assert saved.shape == (2, 23)
    assert saved.dtype == np.float32
    assert output.name != Path(STRICT_CACHE).name
    assert (
        len(
            {
                quantized_grasp_key(
                    row,
                    joint_resolution=0.01,
                    ball_position_resolution=0.001,
                )
                for row in saved
            }
        )
        == 2
    )


def test_cache_inspector_reports_quantized_and_quaternion_only_duplicates(tmp_path) -> None:
    inspector = _load_inspector_module()
    rows = np.stack([_row(), _row(), _row()])
    rows[1, 19:23] = [0.0, 1.0, 0.0, 0.0]
    rows[2, 0] = 0.02
    path = tmp_path / "inspect.npy"
    np.save(path, rows)

    report = inspector.inspect_cache(
        path,
        expected_rows=3,
        joint_resolution=0.01,
        ball_position_resolution=0.001,
    )

    assert report["file_exists"]
    assert report["shape"] == [3, 23]
    assert report["dtype"] == "float32"
    assert report["dtype_valid"]
    assert report["finite"]
    assert report["exact_duplicate_rows"] == 0
    assert report["quantized_unique_key_count"] == 2
    assert report["quantized_duplicate_count"] == 1
    assert report["quaternion_only_duplicate_group_count"] == 1
    assert report["expected_row_count_pass"]


def test_leap_ball_collision_radius_is_33_5_mm() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "leap_object_col")
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
    assert model.geom_size[geom_id, 0] == pytest.approx(0.0335)


def test_reset_provider_inherits_allegro_info_initialization() -> None:
    assert issubclass(
        LeapAllegroGraspResetProvider,
        AllegroRotationDomainRandomizationProvider,
    )
    assert "_build_info_updates" not in LeapAllegroGraspResetProvider.__dict__
