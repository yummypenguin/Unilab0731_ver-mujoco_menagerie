"""LEAP Hand asset, registry, and runtime contract tests."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.envs.manipulation.leap_inhand.allegro_faithful_rotation import (
    LeapInhandBallAllegroRotationCfg,
)
from unilab.envs.manipulation.leap_inhand.ball_grasp_gen import (
    LeapInhandBallGraspCfg,
    build_canonical_grasp_proposals,
    build_joint_coordinate_probes,
    deduplicate_grasp_cache_rows,
    fingertip_surface_gap_quality_mask,
    grasp_cache_row_key,
    normalize_grasp_cache_rows,
    penetration_quality_mask,
    resolve_grasp_proposal_center,
    sample_bounded_joint_offsets,
    save_grasp_cache_atomic,
    select_frontier_rows,
)
from unilab.envs.manipulation.leap_inhand.ball_rotation import LeapInhandBallRotationCfg
from unilab.envs.manipulation.leap_inhand.base import UNILAB_SIM_JOINT_ORDER
from unilab.envs.manipulation.leap_inhand.cache_rotation import (
    LeapInhandBallCacheRotationCfg,
    LeapInhandBallCacheRotationEnv,
)
from unilab.envs.manipulation.leap_inhand.direct_rotation import (
    DirectRotationRewardConfig,
    compute_direct_rotation_reward,
)
from unilab.envs.manipulation.leap_inhand.finger_gaiting_rotation import (
    FingerGaitingConfig,
    LeapInhandBallFingerGaitingRotationCfg,
    LeapInhandBallFingerGaitingRotationEnv,
    advance_finger_gaiting,
    normalize_finger_gaiting_observation,
)
from unilab.envs.manipulation.leap_inhand.rotation_v2 import compute_rotation_terms
from unilab.envs.manipulation.leap_inhand.sustained_cache_rotation import (
    LEAP_TIP_COLLISION_LOCAL_POS,
    STATE_A_SUPPORT_QPOS,
    SUPPORT_JOINT_INDICES,
    AllegroStyleRotationRewardConfig,
    LeapInhandBallSustainedCacheRotationCfg,
    LeapInhandBallSustainedCacheRotationEnv,
    apply_positive_spin_finger_participation,
    compute_allegro_style_obj_linvel_reward,
    compute_allegro_style_rotate_reward,
    compute_index_ring_opposition_quality,
    compute_position_error_reward,
    compute_support_pose_distance,
    compute_tip_collision_reference_positions,
)
from unilab.envs.manipulation.leap_inhand.sustained_rotation import (
    LeapInhandBallSustainedRotationCfg,
    LeapInhandBallSustainedRotationEnv,
    SustainedRotationCurriculumConfig,
    SustainedRotationRewardConfig,
    compute_anchor_proximity,
    compute_rotation_duration_valid,
    compute_spin_continuity_penalty,
    compute_stage_task_reward,
    compute_sustained_spin_terms,
)
from unilab.envs.manipulation.leap_inhand.toss import (
    TossRewardConfig,
    assisted_rebound_candidate,
    curriculum_level,
)

LEAP_ASSET_DIR = Path(ASSETS_ROOT_PATH) / "robots" / "leap_hand"

LEAP_COLLISION_VISUAL_GEOM_PAIRS = (
    ("palm_lower_collision", "palm_lower_visual"),
    ("index_mcp_col", "mcp_joint_visual"),
    ("index_pip_col", "pip_visual"),
    ("index_dip_col", "dip_visual"),
    ("index_tip_col", "fingertip_visual"),
    ("middle_mcp_col", "mcp_joint_2_visual"),
    ("middle_pip_col", "pip_2_visual"),
    ("middle_dip_col", "dip_2_visual"),
    ("middle_tip_col", "fingertip_2_visual"),
    ("ring_mcp_col", "mcp_joint_3_visual"),
    ("ring_pip_col", "pip_3_visual"),
    ("ring_dip_col", "dip_3_visual"),
    ("ring_tip_col", "fingertip_3_visual"),
    ("thumb_base_col", "pip_4_visual"),
    ("thumb_pip_col", "thumb_pip_visual"),
    ("thumb_dip_col", "thumb_dip_visual"),
    ("thumb_tip_col", "thumb_fingertip_visual"),
)

LEAP_ADJACENT_BODY_EXCLUDES = {
    frozenset(pair)
    for pair in (
        ("palm_lower", "mcp_joint"),
        ("mcp_joint", "pip"),
        ("pip", "dip"),
        ("dip", "fingertip"),
        ("palm_lower", "mcp_joint_2"),
        ("mcp_joint_2", "pip_2"),
        ("pip_2", "dip_2"),
        ("dip_2", "fingertip_2"),
        ("palm_lower", "mcp_joint_3"),
        ("mcp_joint_3", "pip_3"),
        ("pip_3", "dip_3"),
        ("dip_3", "fingertip_3"),
        ("palm_lower", "pip_4"),
        ("pip_4", "thumb_pip"),
        ("thumb_pip", "thumb_dip"),
        ("thumb_dip", "thumb_fingertip"),
    )
}


@pytest.mark.parametrize("scene_name", ["scene.xml", "scene_ball.xml", "scene_toss.xml"])
def test_leap_scene_compiles_with_aligned_joint_and_actuator_order(scene_name: str) -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(LEAP_ASSET_DIR / scene_name))

    assert model.nq == 23
    assert model.nv == 22
    assert model.nu == 16
    assert model.nkey == 1
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(16)]
    actuator_joint_ids = model.actuator_trnid[:16, 0]
    assert joint_names == [str(index) for index in UNILAB_SIM_JOINT_ORDER]
    np.testing.assert_array_equal(actuator_joint_ids, np.arange(16))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base") == -1

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    for collision_name, visual_name in LEAP_COLLISION_VISUAL_GEOM_PAIRS:
        collision_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, collision_name)
        visual_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, visual_name)
        assert model.geom_type[collision_id] == mujoco.mjtGeom.mjGEOM_MESH
        assert model.geom_dataid[collision_id] == model.geom_dataid[visual_id]
        assert model.geom_contype[collision_id] == 1
        assert model.geom_conaffinity[collision_id] == 1
        np.testing.assert_allclose(data.geom_xpos[collision_id], data.geom_xpos[visual_id])
        np.testing.assert_allclose(data.geom_xmat[collision_id], data.geom_xmat[visual_id])

    np.testing.assert_array_equal(model.actuator_forcelimited, np.ones(16, dtype=np.uint8))
    np.testing.assert_allclose(model.actuator_forcerange, np.tile([-0.5, 0.5], (16, 1)))

    mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()

    if scene_name == "scene_toss.xml":
        assert model.npair == 3
        sensor_names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)
        }
        assert {
            "leap_index_contact",
            "leap_middle_contact",
            "leap_ring_contact",
            "leap_thumb_contact",
        } <= sensor_names


def test_leap_objects_are_owned_by_leap_asset_directory() -> None:
    cube = (LEAP_ASSET_DIR / "cube.xml").read_text(encoding="utf-8")
    ball = (LEAP_ASSET_DIR / "ball.xml").read_text(encoding="utf-8")
    ball_scene = (LEAP_ASSET_DIR / "scene_ball.xml").read_text(encoding="utf-8")

    assert 'name="leap_object"' in cube
    assert 'size="0.0375 0.0375 0.0375"' in cube
    assert 'name="leap_object"' in ball
    assert 'size="0.04"' in ball
    assert 'diaginertia="0.0001 0.0001 0.0001"' in ball
    assert '<include file="ball.xml"/>' in ball_scene
    assert "allegro" not in cube.lower() + ball.lower() + ball_scene.lower()
    assert "sharpa" not in cube.lower() + ball.lower()


def test_leap_ball_rotation_marker_is_visual_only() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(LEAP_ASSET_DIR / "scene_ball.xml"))
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leap_object")
    collision_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "leap_object_col")
    marker_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "leap_ball_rotation_marker",
    )

    assert model.geom_type[collision_id] == mujoco.mjtGeom.mjGEOM_SPHERE
    assert model.geom_type[marker_id] == mujoco.mjtGeom.mjGEOM_CAPSULE
    assert model.geom_bodyid[marker_id] == object_body_id
    assert model.geom_contype[marker_id] == 0
    assert model.geom_conaffinity[marker_id] == 0
    assert model.body_mass[object_body_id] == pytest.approx(0.05)
    np.testing.assert_allclose(model.body_inertia[object_body_id], [0.0001, 0.0001, 0.0001])


def test_leap_ball_scene_defines_palm_contact_sensor() -> None:
    root = ET.parse(LEAP_ASSET_DIR / "scene_ball.xml").getroot()
    sensor = root.find("./sensor/contact[@name='leap_palm_contact']")

    assert sensor is not None
    assert sensor.attrib["geom1"] == "palm_lower_collision"
    assert sensor.attrib["geom2"] == "leap_object_col"
    assert sensor.attrib["data"] == "found"


def test_leap_contact_excludes_only_directly_connected_links() -> None:
    root = ET.parse(LEAP_ASSET_DIR / "leap_hand.xml").getroot()
    actual = {
        frozenset((exclude.attrib["body1"], exclude.attrib["body2"]))
        for exclude in root.findall("./contact/exclude")
    }

    assert actual == LEAP_ADJACENT_BODY_EXCLUDES
    assert frozenset(("palm_lower", "thumb_fingertip")) not in actual


def test_leap_cube_grasp_cache_matches_mujoco_layout() -> None:
    cache = np.load(LEAP_ASSET_DIR / "caches" / "cube_grasp_s10_1k.npy")

    assert cache.shape == (1024, 23)
    assert cache.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(cache[:, 19:23], axis=1), 1.0, atol=1e-5)
    assert np.mean(np.abs(cache[:, 19])) > 0.9  # MuJoCo identity component is w-first.


def test_leap_ball_grasp_cache_is_independent_and_matches_layout() -> None:
    cache_path = LEAP_ASSET_DIR / "caches" / "ball_grasp_s10_5k.npy"
    cache = np.load(cache_path)

    assert cache.shape == (5000, 23)
    assert cache.dtype == np.float32
    assert np.isfinite(cache).all()
    assert len(np.unique(cache, axis=0)) == 5000
    np.testing.assert_allclose(np.linalg.norm(cache[:, 19:23], axis=1), 1.0, atol=1e-6)
    assert LeapInhandBallRotationCfg().grasp_cache_path.endswith(
        "leap_hand/caches/ball_grasp_s10_5k.npy"
    )
    assert LeapInhandBallRotationCfg().reset_source == "cache"

    for backend in ("mujoco", "motrix"):
        owner = (Path("conf") / "ppo" / "task" / "leap_inhand_ball" / f"{backend}.yaml").read_text(
            encoding="utf-8"
        )
        cache_lines = [line.strip() for line in owner.splitlines() if "grasp_cache_path:" in line]
        assert cache_lines == ["grasp_cache_path: robots/leap_hand/caches/ball_grasp_s10_5k.npy"]
        assert all("cube" not in line for line in cache_lines)
        assert "reset_source: home" in owner


def test_leap_ball_home_matches_validated_candidate() -> None:
    mujoco = pytest.importorskip("mujoco")
    candidate = json.loads(
        (LEAP_ASSET_DIR / "canonical_poses" / "ball_candidate_01.json").read_text(encoding="utf-8")
    )
    model = mujoco.MjModel.from_xml_path(str(LEAP_ASSET_DIR / "scene_ball.xml"))
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    np.testing.assert_allclose(model.key_qpos[key_id], candidate["qpos"], atol=1e-7)
    np.testing.assert_allclose(model.key_ctrl[key_id], candidate["ctrl"], atol=1e-7)


def test_leap_registry_supports_mujoco_and_motrix() -> None:
    ensure_registries()
    registered = registry.list_registered_envs()

    assert registered["LeapInhandRotation"]["available_backends"] == ["mujoco", "motrix"]
    assert registered["LeapInhandBallRotation"]["available_backends"] == ["mujoco", "motrix"]
    assert registered["LeapInhandBallAllegroRotation"]["available_backends"] == [
        "mujoco",
        "motrix",
    ]
    assert registered["LeapInhandBallRotationV2"]["available_backends"] == [
        "mujoco",
        "motrix",
    ]
    assert registered["LeapInhandBallSustainedRotation"]["available_backends"] == [
        "mujoco",
        "motrix",
    ]
    assert registered["LeapInhandBallSustainedCacheRotation"]["available_backends"] == [
        "mujoco",
        "motrix",
    ]
    assert registered["LeapInhandBallFingerGaitingRotation"]["available_backends"] == [
        "mujoco",
        "motrix",
    ]
    assert registered["LeapInhandToss"]["available_backends"] == ["mujoco", "motrix"]
    assert registered["LeapInhandBallGrasp"]["available_backends"] == ["mujoco", "motrix"]


def test_leap_ball_grasp_rows_preserve_layout_and_normalize_quaternion() -> None:
    rows = np.zeros((2, 23), dtype=np.float64)
    rows[:, 19] = 2.0

    normalized = normalize_grasp_cache_rows(rows)

    assert normalized.shape == (2, 23)
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(normalized[:, 19:23], axis=1), 1.0)


def test_leap_ball_grasp_penetration_filter_applies_both_limits() -> None:
    valid = penetration_quality_mask(
        np.asarray([0.0, 0.001, 0.0011, np.nan]),
        np.asarray([0.001, 0.0011, 0.0, 0.0]),
        max_self_depth=0.001,
        max_object_depth=0.001,
    )

    np.testing.assert_array_equal(valid, [True, False, False, False])

    with pytest.raises(ValueError, match="same shape"):
        penetration_quality_mask(
            np.zeros(2),
            np.zeros(3),
            max_self_depth=0.001,
            max_object_depth=0.001,
        )


def test_leap_ball_grasp_surface_gap_filter_uses_signed_geom_distances() -> None:
    valid = fingertip_surface_gap_quality_mask(
        np.asarray(
            [
                [-0.001, 0.0, 0.005, 0.010],
                [0.0, 0.0, 0.0, 0.0101],
                [0.0, 0.0, np.nan, 0.0],
            ]
        ),
        max_gap=0.010,
    )

    np.testing.assert_array_equal(valid, [True, False, False])
    with pytest.raises(ValueError, match=r"shape \(\?, 4\)"):
        fingertip_surface_gap_quality_mask(np.zeros((2, 3)), max_gap=0.010)
    with pytest.raises(ValueError, match="non-negative and finite"):
        fingertip_surface_gap_quality_mask(np.zeros((2, 4)), max_gap=-0.001)


def test_leap_ball_grasp_proposals_use_canonical_state_without_cache() -> None:
    canonical = np.arange(23, dtype=np.float64) / 10.0
    joint_offsets = np.zeros((2, 16), dtype=np.float64)
    joint_offsets[1] = 0.25
    ball_offsets = np.asarray([[0.0, 0.0, 0.0], [0.01, -0.02, 0.03]])
    lower = np.full(16, -1.0)
    upper = np.full(16, 1.5)

    hand_qpos, ball_pos, ball_quat = build_canonical_grasp_proposals(
        canonical,
        joint_offsets,
        ball_offsets,
        joint_lower=lower,
        joint_upper=upper,
    )

    np.testing.assert_allclose(hand_qpos[0], np.clip(canonical[:16], lower, upper))
    np.testing.assert_allclose(hand_qpos[1], np.clip(canonical[:16] + 0.25, lower, upper))
    np.testing.assert_allclose(ball_pos, canonical[None, 16:19] + ball_offsets)
    np.testing.assert_allclose(ball_quat, np.broadcast_to(canonical[19:23], (2, 4)))


def test_leap_ball_grasp_uses_independent_configured_seed() -> None:
    canonical = np.arange(23, dtype=np.float64)
    seed = np.arange(23, dtype=np.float64) / 100.0
    seed[19:23] = [2.0, 0.0, 0.0, 0.0]

    center = resolve_grasp_proposal_center(canonical, seed.tolist())

    np.testing.assert_allclose(center[:19], seed[:19])
    np.testing.assert_allclose(center[19:23], [1.0, 0.0, 0.0, 0.0])
    cfg = LeapInhandBallGraspCfg()
    assert cfg.grasp_cache_path.endswith("ball_grasp_official_50k.npy")
    assert cfg.grasp_collection_target == 50_000
    assert cfg.grasp_joint_noise == pytest.approx(0.10)
    assert cfg.grasp_max_fingertip_surface_gap == pytest.approx(0.00995)


def test_leap_ball_grasp_cache_deduplicates_training_reset_equivalents() -> None:
    rows = np.zeros((3, 23), dtype=np.float64)
    rows[:, 19] = 1.0
    rows[1, 0] = 0.0004
    rows[1, 16] = 0.0002
    rows[2, 0] = 0.002

    unique, indices = deduplicate_grasp_cache_rows(rows)

    np.testing.assert_array_equal(indices, [0, 2])
    np.testing.assert_allclose(unique, rows[[0, 2]])
    assert grasp_cache_row_key(rows[0]) == grasp_cache_row_key(rows[1])
    assert grasp_cache_row_key(rows[0]) != grasp_cache_row_key(rows[2])


def test_leap_ball_generator_discards_duplicate_accepted_rows() -> None:
    from unilab.envs.manipulation.leap_inhand.ball_grasp_gen import (
        LeapInhandBallGraspEnv,
    )

    env = object.__new__(LeapInhandBallGraspEnv)
    env._state = SimpleNamespace(
        truncated=np.asarray([True, True]),
        terminated=np.asarray([False, False]),
        info={"log": {}},
    )
    env._saved_grasping_states = []
    env._saved_grasp_keys = set()
    env.nv = 22
    env._cfg = SimpleNamespace(
        grasp_min_contacts=2,
        grasp_require_thumb_contact=True,
    )
    env._backend = SimpleNamespace(
        nv=22,
        model=SimpleNamespace(nv=22),
        set_state=lambda env_ids, qpos, qvel: None,
    )
    hand_qpos = np.zeros((2, 16), dtype=np.float64)
    ball_pos = np.zeros((2, 3), dtype=np.float64)
    ball_quat = np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2)
    env._check_grasp_quality = lambda env_ids: np.ones(len(env_ids), dtype=bool)
    env.get_hand_dof_pos = lambda: hand_qpos
    env.get_ball_pos = lambda: ball_pos
    env.get_ball_quat = lambda: ball_quat
    env._contact_flags = lambda: np.ones((2, 4), dtype=bool)
    env._fingertip_surface_quality = lambda env_ids: (
        np.ones(len(env_ids), dtype=bool),
        np.zeros((len(env_ids), 4), dtype=np.float64),
    )
    env._penetration_quality = lambda env_ids: (
        np.ones(len(env_ids), dtype=bool),
        np.zeros(len(env_ids), dtype=np.float64),
        np.zeros(len(env_ids), dtype=np.float64),
        (),
    )
    env._update_grasp_frontier = lambda *args: None
    env._save_grasp_cache = lambda: None
    env._stop_collection = lambda: None

    env._collect_successful_grasps(np.asarray([0, 1], dtype=np.int32))

    assert len(env._saved_grasping_states) == 1
    assert env._saved_grasping_states[0].shape == (1, 23)
    assert len(env._saved_grasp_keys) == 1


def test_leap_ball_grasp_cache_atomic_save_refuses_implicit_overwrite(tmp_path: Path) -> None:
    rows = np.zeros((2, 23), dtype=np.float64)
    rows[:, 19] = 2.0
    output = tmp_path / "cache.npy"

    saved = save_grasp_cache_atomic(output, rows)

    assert saved == output
    loaded = np.load(output)
    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded[:, 19:23], [[1.0, 0.0, 0.0, 0.0]] * 2)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_grasp_cache_atomic(output, rows)


def test_leap_ball_grasp_samples_asymmetric_joint_offsets() -> None:
    lower = np.zeros(16, dtype=np.float64)
    upper = np.zeros(16, dtype=np.float64)
    lower[[0, 7, 8, 13]] = [-0.005, -0.002, -0.005, -0.005]
    upper[[0, 7, 8, 13]] = [0.005, 0.005, 0.005, 0.005]

    offsets = sample_bounded_joint_offsets(256, lower, upper)

    assert offsets.shape == (256, 16)
    assert np.all(offsets >= lower[None, :])
    assert np.all(offsets <= upper[None, :])
    np.testing.assert_array_equal(offsets[:, [1, 6, 9, 15]], 0.0)


def test_leap_ball_grasp_rejects_incomplete_joint_offset_bounds() -> None:
    cfg = LeapInhandBallGraspCfg(grasp_joint_offset_lower=[0.0] * 16)

    with pytest.raises(ValueError, match="must both be set or empty"):
        cfg.validate()

    with pytest.raises(ValueError, match="fingertip_surface_gap"):
        LeapInhandBallGraspCfg(grasp_max_fingertip_surface_gap=-0.001).validate()


def test_leap_ball_grasp_frontier_keeps_best_unique_rows() -> None:
    rows = np.zeros((4, 23), dtype=np.float64)
    rows[:, 19] = 1.0
    rows[1, 0] = 0.1
    rows[2, 0] = 0.2
    rows[3] = rows[1]

    selected, scores = select_frontier_rows(
        rows,
        np.asarray([4.0, 2.0, 1.0, 3.0]),
        capacity=2,
    )

    np.testing.assert_allclose(selected[:, 0], [0.2, 0.1])
    np.testing.assert_allclose(scores, [1.0, 2.0])


def test_leap_ball_grasp_joint_probes_apply_signed_clipped_deltas() -> None:
    hand_qpos = np.zeros(16, dtype=np.float64)
    hand_qpos[9] = 0.95
    lower = np.full(16, -1.0)
    upper = np.full(16, 1.0)

    probes, indices, deltas = build_joint_coordinate_probes(
        hand_qpos,
        np.asarray([9, 10]),
        np.asarray([0.1]),
        joint_lower=lower,
        joint_upper=upper,
    )

    assert probes.shape == (5, 16)
    np.testing.assert_array_equal(indices, [-1, 9, 9, 10, 10])
    np.testing.assert_allclose(deltas, [0.0, -0.1, 0.05, -0.1, 0.1])
    np.testing.assert_allclose(probes[0], hand_qpos)
    np.testing.assert_allclose(probes[1, 9], 0.85)
    np.testing.assert_allclose(probes[2, 9], 1.0)
    np.testing.assert_allclose(probes[3, 10], -0.1)
    np.testing.assert_allclose(probes[4, 10], 0.1)


def test_leap_ball_grasp_settling_diagnostic_uses_quality_contract() -> None:
    pytest.importorskip("mujoco")
    try:
        from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env is unavailable")

    from unilab.envs.manipulation.allegro_inhand.rotation import RewardConfigPPO
    from unilab.envs.manipulation.leap_inhand.ball_grasp_gen import (
        LeapInhandBallGraspCfg,
        LeapInhandBallGraspEnv,
    )

    cfg = LeapInhandBallGraspCfg(
        grasp_auto_save=False,
        grasp_collection_target=0,
        grasp_max_self_penetration=0.001,
        grasp_max_object_penetration=0.001,
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
            reset_z_threshold=0.4,
        ),
    )
    env = LeapInhandBallGraspEnv(cfg, num_envs=1, backend_type="mujoco")
    try:
        qpos = env._backend.get_keyframe_qpos("home")
        report = env.diagnose_grasp_state(qpos, settle_seconds=env.cfg.ctrl_dt)
        replay = env.replay_validate_grasp_cache_rows(
            qpos[None, :],
            settle_seconds=env.cfg.ctrl_dt,
        )
        probe_reports = env.diagnose_joint_coordinate_probes(
            qpos,
            joint_names=["7"],
            delta_magnitudes=np.asarray([0.005]),
            settle_seconds=env.cfg.ctrl_dt,
        )
    finally:
        env.close()

    assert report["settle_steps"] == 1
    assert set(report["conditions"]) == {
        "finite",
        "joint_limits",
        "height",
        "drift",
        "fingertips",
        "surface_gap",
        "contacts",
        "thumb",
        "ball_linvel",
        "ball_angvel",
        "joint_speed",
        "work",
    }
    assert set(report["contacts"]) == {
        "leap_index_contact",
        "leap_middle_contact",
        "leap_ring_contact",
        "leap_thumb_contact",
    }
    assert set(report["initial_contacts"]) == set(report["contacts"])
    assert set(report["tip_surface_distance"]) == {
        "index_tip_col",
        "middle_tip_col",
        "ring_tip_col",
        "thumb_tip_col",
    }
    for distances in report["tip_surface_distance"].values():
        assert np.isfinite(distances["initial"])
        assert np.isfinite(distances["settled"])
    assert set(report["penetration"]) >= {"valid", "self_depth", "object_depth"}
    assert isinstance(report["quality_valid"], bool)
    assert replay.settle_steps == 1
    assert replay.accepted.shape == (1,)
    assert replay.terminated_during_settle.shape == (1,)
    assert replay.contacts.shape == (1, 4)
    assert set(replay.conditions) == set(report["conditions"])
    assert set(replay.measurements) == set(report["measurements"])

    assert len(probe_reports) == 3
    assert [item["probe"]["joint"] for item in probe_reports] == ["baseline", "7", "7"]
    np.testing.assert_allclose(
        [item["probe"]["delta"] for item in probe_reports],
        [0.0, -0.005, 0.005],
    )


@pytest.mark.parametrize("shape", [(23,), (2, 22), (2, 24)])
def test_leap_ball_grasp_rows_reject_invalid_layout(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shape"):
        normalize_grasp_cache_rows(np.zeros(shape, dtype=np.float32))


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_leap_ball_rotation_home_reset_and_step(backend: str) -> None:
    if backend == "mujoco":
        pytest.importorskip("mujoco")
        try:
            from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
        except Exception:
            pytest.skip("mujoco.batch_env is unavailable")
    else:
        pytest.importorskip("motrixsim", reason="motrixsim not installed")

    ensure_registries()
    env = registry.make(
        "LeapInhandBallRotation",
        sim_backend=backend,
        num_envs=2,
        env_cfg_override={
            "sim_dt": 0.01 if backend == "motrix" else 1.0 / 120.0,
            "reset_source": "home",
            "reward_config": {
                "scales": {
                    "rotate": 1.25,
                    "obj_linvel": -0.3,
                    "pose_diff": -0.3,
                    "torque": -0.1,
                    "work": -2.0,
                    "drop": 0.0,
                },
                "angvel_clip_min": -0.5,
                "angvel_clip_max": 0.5,
                "reset_z_threshold": 0.4,
            },
        },
    )
    try:
        obs, info = env.reset(np.arange(2, dtype=np.int32))
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        assert obs["obs"].shape == (2, 105)
        assert np.isfinite(obs["obs"]).all()
        expected = np.asarray(
            json.loads(
                (LEAP_ASSET_DIR / "canonical_poses" / "ball_candidate_01.json").read_text(
                    encoding="utf-8"
                )
            )["qpos"],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            info["init_pose"], np.broadcast_to(expected[None, :16], (2, 16)), atol=1e-5
        )
        np.testing.assert_allclose(
            info["prev_ball_pos"], np.broadcast_to(expected[None, 16:19], (2, 3)), atol=1e-5
        )
        np.testing.assert_allclose(
            info["prev_ball_quat"], np.broadcast_to(expected[None, 19:23], (2, 4)), atol=1e-5
        )

        next_state = env.step(np.zeros((2, 16), dtype=np.float32))
        assert next_state.obs["obs"].shape == (2, 105)
        assert np.isfinite(next_state.obs["obs"]).all()
        assert np.isfinite(next_state.reward).all()
    finally:
        env.close()


def test_leap_ball_rotation_cache_reset_remains_available() -> None:
    pytest.importorskip("mujoco")
    try:
        from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env is unavailable")

    ensure_registries()
    env = registry.make(
        "LeapInhandBallRotation",
        sim_backend="mujoco",
        num_envs=2,
        env_cfg_override={
            "reset_source": "cache",
            "reward_config": {
                "scales": {
                    "rotate": 1.25,
                    "obj_linvel": -0.3,
                    "pose_diff": -0.3,
                    "torque": -0.1,
                    "work": -2.0,
                    "drop": 0.0,
                },
                "angvel_clip_min": -0.5,
                "angvel_clip_max": 0.5,
                "reset_z_threshold": 0.4,
            },
        },
    )
    try:
        _, info = env.reset(np.arange(2, dtype=np.int32))
        cache = np.load(LEAP_ASSET_DIR / "caches" / "ball_grasp_s10_5k.npy")
        reset_rows = np.concatenate(
            [info["init_pose"], info["prev_ball_pos"], info["prev_ball_quat"]], axis=1
        )
        for row in reset_rows:
            assert np.any(np.all(np.isclose(cache, row[None, :], atol=1e-6), axis=1))
    finally:
        env.close()


def test_leap_ball_rotation_rejects_unknown_reset_source() -> None:
    with pytest.raises(ValueError, match="reset_source"):
        LeapInhandBallRotationCfg(reset_source="unknown").validate()


def test_leap_ball_cache_rotation_has_independent_cache_and_drop_termination() -> None:
    cfg = LeapInhandBallCacheRotationCfg()
    assert cfg.reset_source == "cache"
    assert cfg.grasp_cache_path.endswith("ball_grasp_official_50k.npy")
    assert cfg.termination_drop_distance == pytest.approx(0.007)

    env = object.__new__(LeapInhandBallCacheRotationEnv)
    env._num_envs = 3
    env._cfg = SimpleNamespace(termination_drop_distance=0.007)
    env._termination_initial_ball_pos = np.asarray(
        [[0.0, 0.0, 0.670], [0.0, 0.0, 0.670], [0.0, 0.0, 0.680]],
        dtype=np.float64,
    )

    ball_pos = np.asarray(
        [[0.0, 0.0, 0.6631], [0.0, 0.0, 0.6630], [0.0, 0.0, 0.6720]],
        dtype=np.float64,
    )
    terminated = env._compute_terminated(ball_pos)

    np.testing.assert_array_equal(terminated, [False, True, True])


def test_allegro_faithful_config_uses_dense_seven_mm_drop_cost() -> None:
    cfg = LeapInhandBallAllegroRotationCfg()
    cfg.validate()
    assert cfg.reset_source == "home"
    assert cfg.sim_dt == pytest.approx(0.005)
    assert not hasattr(cfg, "pose_error_scales")

    owner = (Path("conf") / "ppo" / "task" / "leap_inhand_ball_allegro" / "mujoco.yaml").read_text(
        encoding="utf-8"
    )
    assert "drop: -1.0" in owner
    assert "reset_z_threshold: 0.66797318983078" in owner
    assert "sim_dt: 0.005" in owner


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_allegro_faithful_leap_ball_reset_and_step(backend: str) -> None:
    if backend == "mujoco":
        pytest.importorskip("mujoco")
        try:
            from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
        except Exception:
            pytest.skip("mujoco.batch_env is unavailable")
    else:
        pytest.importorskip("motrixsim", reason="motrixsim not installed")

    ensure_registries()
    env = registry.make(
        "LeapInhandBallAllegroRotation",
        sim_backend=backend,
        num_envs=2,
        env_cfg_override={
            "sim_dt": 0.01 if backend == "motrix" else 0.005,
            "reset_source": "home",
            "reward_config": {
                "scales": {
                    "rotate": 1.25,
                    "obj_linvel": -0.3,
                    "pose_diff": -0.3,
                    "torque": -0.1,
                    "work": -2.0,
                    "drop": -1.0,
                },
                "angvel_clip_min": -0.5,
                "angvel_clip_max": 0.5,
                "reset_z_threshold": 0.66797318983078,
            },
        },
    )
    try:
        obs, info = env.reset(np.arange(2, dtype=np.int32))
        assert obs["obs"].shape == (2, 105)
        np.testing.assert_allclose(
            info["init_pose"],
            np.broadcast_to(env.default_angles[None, :], (2, 16)),
            atol=1e-5,
        )
        palm_contact = np.asarray(env.get_sensor_data("leap_palm_contact")).reshape(2, -1)
        assert not np.any(palm_contact > 0.5)

        zeros_dof = np.zeros((2, 16), dtype=np.float32)
        zeros_vec = np.zeros((2, 3), dtype=np.float32)
        drop = env._reward_drop(
            {},
            zeros_dof,
            zeros_dof,
            np.asarray([[0.0, 0.0, 0.668], [0.0, 0.0, 0.667]], dtype=np.float32),
            zeros_vec,
            zeros_vec,
            zeros_dof,
            np.zeros(2, dtype=bool),
        )
        np.testing.assert_array_equal(drop, [0.0, 1.0])

        next_state = env.step(np.zeros((2, 16), dtype=np.float32))
        assert np.isfinite(next_state.reward).all()
        assert "diagnostic/ball_z_mean" in next_state.info["log"]
        assert "diagnostic/drop_rate" in next_state.info["log"]
        assert "diagnostic/palm_contact_rate" in next_state.info["log"]
        assert "diagnostic/raw_pose_l2_rms" in next_state.info["log"]
        assert "diagnostic/allegro_equivalent_pose_rms" not in next_state.info["log"]
        assert "diagnostic/torque_saturation_fraction" in next_state.info["log"]
    finally:
        env.close()


def test_leap_ball_rotation_v2_terms_require_directed_spin() -> None:
    angvel = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, -0.5],
            [0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    axis = np.broadcast_to(np.asarray([0.0, 0.0, 1.0], dtype=np.float32), angvel.shape)
    target = np.full(4, 0.5, dtype=np.float32)

    axis_speed, orthogonal_speed, progress, quality = compute_rotation_terms(
        angvel, axis, target, 0.5
    )

    np.testing.assert_allclose(axis_speed, [0.0, 0.5, -0.5, 0.0])
    np.testing.assert_allclose(orthogonal_speed, [0.0, 0.0, 0.0, 0.5])
    np.testing.assert_allclose(progress, [0.0, 1.0, -1.0, 0.0])
    np.testing.assert_allclose(quality, [0.0, 1.0, 0.0, 0.0])


def test_leap_sustained_rotation_keeps_off_axis_progress_observable() -> None:
    angvel = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, -0.5],
            [0.5, 0.0, 0.5],
        ],
        dtype=np.float32,
    )
    axis = np.broadcast_to(np.asarray([0.0, 0.0, 1.0], dtype=np.float32), angvel.shape)
    target = np.asarray([0.0, 0.5, 0.5, 0.5], dtype=np.float32)
    tolerance = np.full(4, 0.1, dtype=np.float32)

    axis_speed, orthogonal_speed, progress, visible_progress, purity = compute_sustained_spin_terms(
        angvel, axis, target, tolerance
    )

    np.testing.assert_allclose(axis_speed, [0.0, 0.5, -0.5, 0.5])
    np.testing.assert_allclose(orthogonal_speed, [0.0, 0.0, 0.0, 0.5])
    np.testing.assert_allclose(progress, [0.0, 1.0, -1.0, 1.0])
    np.testing.assert_allclose(visible_progress[:3], [0.0, 1.0, -1.0])
    assert visible_progress[3] == pytest.approx(0.5, abs=1e-7)
    assert purity[3] < 1e-8


def test_leap_sustained_rotation_does_not_reward_stationary_rotation_stage() -> None:
    task_reward, hold_reward, spin_reward, rotation_stability = compute_stage_task_reward(
        hold_stage=np.asarray([True, True, False, False, False]),
        stage_valid=np.asarray([True, False, True, True, True]),
        axis_progress=np.asarray([0.0, 0.0, 0.0, 1.0, -1.0]),
        visible_progress=np.asarray([0.0, 0.0, 0.0, 0.5, -1.0]),
        retention=np.ones(5),
        fingertip_support=np.ones(5),
        speed_tracking_quality=np.ones(5),
        stage_duration_progress=np.ones(5),
        positive_spin_retention_floor=np.full(5, 0.5),
        spin_progress_scale=1.5,
        retention_scale=0.5,
        fingertip_support_scale=0.25,
    )

    np.testing.assert_allclose(hold_reward, [0.75, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(spin_reward, [0.0, 0.0, 0.0, 0.75, -1.5])
    np.testing.assert_allclose(rotation_stability, [0.0, 0.0, 0.0, 0.75, 0.0])
    np.testing.assert_allclose(task_reward, [0.75, 0.0, 0.0, 1.5, -1.5])


def test_leap_sustained_rotation_softly_conditions_positive_spin_on_retention() -> None:
    task_reward, hold_reward, spin_reward, rotation_stability = compute_stage_task_reward(
        hold_stage=np.zeros(3, dtype=bool),
        stage_valid=np.asarray([True, False, False]),
        axis_progress=np.asarray([1.0, 1.0, -1.0]),
        visible_progress=np.asarray([0.5, 0.5, -0.5]),
        retention=np.asarray([1.0, 0.25, 0.25]),
        fingertip_support=np.ones(3),
        speed_tracking_quality=np.ones(3),
        stage_duration_progress=np.ones(3),
        positive_spin_retention_floor=np.full(3, 0.5),
        spin_progress_scale=2.0,
        retention_scale=0.0,
        fingertip_support_scale=0.0,
    )

    np.testing.assert_allclose(hold_reward, 0.0)
    np.testing.assert_allclose(spin_reward, [1.0, 0.625, -1.0])
    np.testing.assert_allclose(rotation_stability, 0.0)
    np.testing.assert_allclose(task_reward, spin_reward)


def test_leap_sustained_rotation_direct_spin_ignores_legacy_conditioning() -> None:
    task_reward, hold_reward, spin_reward, rotation_stability = compute_stage_task_reward(
        hold_stage=np.zeros(4, dtype=bool),
        stage_valid=np.zeros(4, dtype=bool),
        axis_progress=np.asarray([1.0, 0.25, -0.25, 0.5]),
        visible_progress=np.asarray([0.5, 0.125, -0.125, 0.5]),
        retention=np.asarray([1.0, 0.0, 0.0, 0.25]),
        fingertip_support=np.asarray([1.0, 0.0, 0.0, 0.5]),
        speed_tracking_quality=np.asarray([1.0, 0.0, 0.0, 0.5]),
        stage_duration_progress=np.asarray([1.0, 0.0, 0.0, 0.5]),
        positive_spin_retention_floor=np.asarray([0.5, 0.0, 0.0, 0.25]),
        spin_progress_scale=2.0,
        retention_scale=0.5,
        fingertip_support_scale=0.25,
        direct_spin_reward=True,
    )

    np.testing.assert_allclose(hold_reward, 0.0)
    np.testing.assert_allclose(spin_reward, [1.0, 0.0625, -0.0625, 0.5])
    np.testing.assert_allclose(rotation_stability, 0.0)
    np.testing.assert_allclose(task_reward, spin_reward)


def test_leap_sustained_rotation_can_tighten_late_stage_retention() -> None:
    task_reward, _hold_reward, spin_reward, _rotation_stability = compute_stage_task_reward(
        hold_stage=np.zeros(2, dtype=bool),
        stage_valid=np.zeros(2, dtype=bool),
        axis_progress=np.zeros(2),
        visible_progress=np.full(2, 0.5),
        retention=np.zeros(2),
        fingertip_support=np.zeros(2),
        speed_tracking_quality=np.ones(2),
        stage_duration_progress=np.ones(2),
        positive_spin_retention_floor=np.asarray([0.5, 0.25]),
        spin_progress_scale=2.0,
        retention_scale=0.0,
        fingertip_support_scale=0.0,
    )

    np.testing.assert_allclose(spin_reward, [0.5, 0.25])
    np.testing.assert_allclose(task_reward, spin_reward)


def test_leap_sustained_rotation_tracks_ema_target_speed() -> None:
    from unilab.envs.manipulation.leap_inhand.sustained_rotation import (
        compute_speed_tracking_quality,
    )

    quality = compute_speed_tracking_quality(
        axis_speed_ema=np.asarray([0.0, 0.10, 0.08, 0.05]),
        target_speed=np.asarray([0.0, 0.10, 0.10, 0.10]),
        tolerance_ratio=0.25,
    )

    np.testing.assert_allclose(
        quality,
        [0.0, 1.0, np.exp(-0.64), np.exp(-4.0)],
        atol=1e-7,
    )


def test_leap_sustained_rotation_penalizes_ema_speed_shortfall() -> None:
    penalty = compute_spin_continuity_penalty(
        axis_speed_ema=np.asarray([-0.10, 0.0, 0.15, 0.30, 0.60, 0.0]),
        target_speed=np.asarray([0.30, 0.30, 0.30, 0.30, 0.30, 0.0]),
        penalty_scale=0.05,
    )

    np.testing.assert_allclose(penalty, [-0.05, -0.05, -0.025, 0.0, 0.0, 0.0])


def test_leap_sustained_rotation_anchor_proximity_spans_grey_zone() -> None:
    quality = compute_anchor_proximity(
        position_error=np.asarray([0.0, 0.015, 0.0225, 0.030, 0.040]),
        gate_position_radius=0.015,
        failure_position_radius=0.030,
    )

    np.testing.assert_allclose(quality, [1.0, 1.0, 0.5, 0.0, 0.0])


def test_leap_sustained_rotation_rewards_target_speed_duration() -> None:
    task_reward, hold_reward, spin_reward, rotation_stability = compute_stage_task_reward(
        hold_stage=np.zeros(4, dtype=bool),
        stage_valid=np.asarray([False, True, True, False]),
        axis_progress=np.zeros(4),
        visible_progress=np.asarray([0.5, 0.5, 0.5, -0.5]),
        retention=np.ones(4),
        fingertip_support=np.zeros(4),
        speed_tracking_quality=np.asarray([0.0, 1.0, 1.0, 0.0]),
        stage_duration_progress=np.asarray([0.0, 0.0, 1.0, 0.0]),
        positive_spin_retention_floor=np.full(4, 0.5),
        spin_progress_scale=2.0,
        retention_scale=0.0,
        fingertip_support_scale=0.0,
    )

    np.testing.assert_allclose(hold_reward, 0.0)
    np.testing.assert_allclose(spin_reward, [0.25, 0.5, 1.0, -1.0])
    np.testing.assert_allclose(rotation_stability, 0.0)
    np.testing.assert_allclose(task_reward, spin_reward)


def test_leap_sustained_rotation_failure_counter_is_debounced() -> None:
    from unilab.envs.manipulation.leap_inhand.sustained_rotation import (
        LeapInhandBallSustainedRotationEnv,
    )

    counter = np.zeros(3, dtype=np.uint8)
    condition = np.asarray([True, False, True])
    counter = LeapInhandBallSustainedRotationEnv._advance_failure_counter(counter, condition)
    np.testing.assert_array_equal(counter, [1, 0, 1])
    counter = LeapInhandBallSustainedRotationEnv._advance_failure_counter(
        counter, np.asarray([True, True, False])
    )
    np.testing.assert_array_equal(counter, [2, 1, 0])


def test_leap_sustained_cache_rotation_uses_wide_workspace_termination() -> None:
    cfg = LeapInhandBallSustainedCacheRotationCfg(
        reward_config=AllegroStyleRotationRewardConfig()
    )
    assert cfg.reset_source == "cache"
    assert cfg.grasp_cache_path.endswith("ball_grasp_official_50k.npy")
    assert not hasattr(cfg, "termination_drop_distance")
    assert cfg.termination_workspace_radius == pytest.approx(0.05)

    env = object.__new__(LeapInhandBallSustainedCacheRotationEnv)
    env._cfg = SimpleNamespace(termination_workspace_radius=0.05)
    position_error = np.asarray([0.049, 0.050, 0.051])
    np.testing.assert_array_equal(
        env._compute_terminated(position_error),
        [False, False, True],
    )


def test_leap_sustained_cache_rotation_matches_allegro_rotate_logic() -> None:
    ball_angvel = np.asarray(
        [
            [0.0, 0.0, -0.75],
            [0.0, 0.0, -0.25],
            [0.0, 0.0, 0.25],
            [0.0, 0.0, 0.75],
        ]
    )
    clipped, reward = compute_allegro_style_rotate_reward(
        ball_angvel,
        np.asarray([0.0, 0.0, 1.0]),
        scale=1.25,
        clip_min=-0.5,
        clip_max=0.5,
    )

    np.testing.assert_allclose(clipped, [-0.5, -0.25, 0.25, 0.5])
    np.testing.assert_allclose(reward, [-0.625, -0.3125, 0.3125, 0.625])


def test_leap_sustained_cache_rotation_rewards_index_middle_participation() -> None:
    base_reward = np.asarray([0.625, 0.625, 0.625, 0.625, -0.625])
    contacts = np.asarray(
        [
            [False, False, True, True],
            [True, False, True, True],
            [False, True, True, True],
            [True, True, False, True],
            [False, False, True, True],
        ]
    )

    participation_scale, reward = apply_positive_spin_finger_participation(
        base_reward,
        contacts,
        base_contact_scale=0.25,
        index_contact_scale=0.375,
        middle_contact_scale=0.375,
    )

    np.testing.assert_allclose(participation_scale, [0.25, 0.625, 0.625, 1.0, 0.25])
    np.testing.assert_allclose(
        reward,
        [0.15625, 0.390625, 0.390625, 0.625, -0.625],
    )


def test_leap_sustained_cache_rotation_matches_allegro_obj_linvel_logic() -> None:
    ball_linvel = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, -0.2, 0.3],
            [-0.2, 0.0, -0.1],
        ]
    )
    linear_speed_l1, reward = compute_allegro_style_obj_linvel_reward(
        ball_linvel,
        scale=-0.3,
    )

    np.testing.assert_allclose(linear_speed_l1, [0.0, 0.6, 0.3])
    np.testing.assert_allclose(reward, [0.0, -0.18, -0.09])


def test_leap_sustained_cache_rotation_penalizes_position_error() -> None:
    anchor_pos = np.zeros((3, 3))
    ball_pos = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.006, 0.008, 0.0],
            [0.0, 0.0, 0.020],
        ]
    )
    position_error, reward = compute_position_error_reward(
        ball_pos,
        anchor_pos,
        scale=-5.0,
    )

    np.testing.assert_allclose(position_error, [0.0, 0.010, 0.020])
    np.testing.assert_allclose(reward, [0.0, -0.05, -0.10])


def test_leap_finger_gaiting_is_decoupled_from_sustained_cache_task() -> None:
    assert issubclass(LeapInhandBallFingerGaitingRotationCfg, LeapInhandBallSustainedRotationCfg)
    assert not issubclass(
        LeapInhandBallFingerGaitingRotationCfg,
        LeapInhandBallSustainedCacheRotationCfg,
    )
    assert issubclass(LeapInhandBallFingerGaitingRotationEnv, LeapInhandBallSustainedRotationEnv)
    assert not issubclass(
        LeapInhandBallFingerGaitingRotationEnv,
        LeapInhandBallSustainedCacheRotationEnv,
    )

    cfg = LeapInhandBallFingerGaitingRotationCfg()
    assert cfg.reset_source == "cache"
    assert cfg.grasp_cache_path.endswith("ball_grasp_official_50k.npy")
    assert cfg.termination_drop_distance == pytest.approx(0.007)

    env = object.__new__(LeapInhandBallFingerGaitingRotationEnv)
    env._cfg = cfg
    anchor = np.asarray([[0.0, 0.0, 0.650]], dtype=np.float64)
    ball_pos = np.asarray([[0.0, 0.0, 0.643]], dtype=np.float64)
    np.testing.assert_array_equal(env._compute_raw_drop(ball_pos, anchor), [True])


def test_leap_finger_gaiting_requires_debounced_safe_recontact() -> None:
    cfg = FingerGaitingConfig(
        minimum_release_steps=2,
        maximum_release_steps=5,
        handoff_cooldown_steps=3,
    )
    contacts = np.asarray([[False, True, True, True]])
    previous = np.ones((1, 4), dtype=bool)
    active = np.zeros((1, 4), dtype=bool)
    steps = np.zeros((1, 4), dtype=np.uint8)
    start_speed = np.zeros((1, 4), dtype=np.float32)
    cooldown = np.zeros(1, dtype=np.uint8)
    common = {
        "eligible": np.asarray([True]),
        "axis_speed_ema": np.asarray([0.1], dtype=np.float32),
        "target_speed": np.asarray([0.1], dtype=np.float32),
        "cfg": cfg,
    }

    transition = advance_finger_gaiting(
        contacts=contacts,
        previous_contacts=previous,
        active=active,
        release_steps=steps,
        release_start_speed=start_speed,
        cooldown_steps=cooldown,
        **common,
    )
    assert transition.active[0, 0]
    assert transition.release_steps[0, 0] == 1
    assert not np.any(transition.qualified_handoff)

    transition = advance_finger_gaiting(
        contacts=contacts,
        previous_contacts=contacts,
        active=transition.active,
        release_steps=transition.release_steps,
        release_start_speed=transition.release_start_speed,
        cooldown_steps=transition.cooldown_steps,
        **common,
    )
    assert transition.release_steps[0, 0] == 2

    recontact = np.ones((1, 4), dtype=bool)
    transition = advance_finger_gaiting(
        contacts=recontact,
        previous_contacts=contacts,
        active=transition.active,
        release_steps=transition.release_steps,
        release_start_speed=transition.release_start_speed,
        cooldown_steps=transition.cooldown_steps,
        **common,
    )
    np.testing.assert_array_equal(transition.qualified_handoff, [[True, False, False, False]])
    assert transition.cooldown_steps[0] == 3


def test_leap_finger_gaiting_rejects_contact_chatter() -> None:
    cfg = FingerGaitingConfig(minimum_release_steps=2)
    transition = advance_finger_gaiting(
        contacts=np.asarray([[True, True, True, True]]),
        previous_contacts=np.asarray([[False, True, True, True]]),
        active=np.asarray([[True, False, False, False]]),
        release_steps=np.asarray([[1, 0, 0, 0]], dtype=np.uint8),
        release_start_speed=np.asarray([[0.1, 0.0, 0.0, 0.0]], dtype=np.float32),
        cooldown_steps=np.zeros(1, dtype=np.uint8),
        eligible=np.asarray([True]),
        axis_speed_ema=np.asarray([0.1], dtype=np.float32),
        target_speed=np.asarray([0.1], dtype=np.float32),
        cfg=cfg,
    )

    assert not np.any(transition.qualified_handoff)
    assert not transition.active[0, 0]


def test_leap_finger_gaiting_allows_stationary_handoff_stage() -> None:
    cfg = FingerGaitingConfig(minimum_release_steps=2)
    transition = advance_finger_gaiting(
        contacts=np.asarray([[False, True, True, True]]),
        previous_contacts=np.asarray([[True, True, True, True]]),
        active=np.zeros((1, 4), dtype=bool),
        release_steps=np.zeros((1, 4), dtype=np.uint8),
        release_start_speed=np.zeros((1, 4), dtype=np.float32),
        cooldown_steps=np.zeros(1, dtype=np.uint8),
        eligible=np.asarray([True]),
        axis_speed_ema=np.asarray([-0.01], dtype=np.float32),
        target_speed=np.asarray([0.0], dtype=np.float32),
        cfg=cfg,
        stationary_handoff_allowed=np.asarray([True]),
    )
    assert transition.active[0, 0]

    transition = advance_finger_gaiting(
        contacts=np.asarray([[False, True, True, True]]),
        previous_contacts=np.asarray([[False, True, True, True]]),
        active=transition.active,
        release_steps=transition.release_steps,
        release_start_speed=transition.release_start_speed,
        cooldown_steps=transition.cooldown_steps,
        eligible=np.asarray([True]),
        axis_speed_ema=np.asarray([-0.02], dtype=np.float32),
        target_speed=np.asarray([0.0], dtype=np.float32),
        cfg=cfg,
        stationary_handoff_allowed=np.asarray([True]),
    )
    transition = advance_finger_gaiting(
        contacts=np.asarray([[True, True, True, True]]),
        previous_contacts=np.asarray([[False, True, True, True]]),
        active=transition.active,
        release_steps=transition.release_steps,
        release_start_speed=transition.release_start_speed,
        cooldown_steps=transition.cooldown_steps,
        eligible=np.asarray([True]),
        axis_speed_ema=np.asarray([-0.03], dtype=np.float32),
        target_speed=np.asarray([0.0], dtype=np.float32),
        cfg=cfg,
        stationary_handoff_allowed=np.asarray([True]),
    )

    assert transition.qualified_handoff[0, 0]


def test_leap_finger_gaiting_curriculum_requires_handoffs() -> None:
    cfg = LeapInhandBallFingerGaitingRotationCfg()

    assert cfg.reset_source == "cache"
    assert cfg.max_episode_seconds == pytest.approx(35.0)
    assert cfg.finger_gaiting.required_handoffs_by_stage == [0, 1, 1, 2, 2, 3, 4, 6]
    assert cfg.finger_gaiting.minimum_contacts_by_stage[0] == 3


def test_leap_finger_gaiting_observation_exposes_normalized_gate_state() -> None:
    observation = normalize_finger_gaiting_observation(
        release_active=np.asarray([[False, True, False, True], [True, False, True, False]]),
        release_steps=np.asarray([[0, 5, 10, 20], [2, 4, 6, 8]]),
        release_start_speed=np.asarray([[0.0, 0.25, 0.5, 0.75], [0.1, 0.2, 0.3, 0.4]]),
        cooldown_steps=np.asarray([0, 2]),
        stage_handoffs=np.asarray([0, 1]),
        required_handoffs=np.asarray([0, 2]),
        maximum_release_steps=10,
        maximum_target_speed=0.5,
        maximum_cooldown_steps=4,
    )

    assert observation.shape == (2, 14)
    np.testing.assert_allclose(observation[0, :4], [0.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(observation[0, 4:8], [0.0, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(observation[0, 8:12], [0.0, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(observation[:, 12], [0.0, 0.5])
    np.testing.assert_allclose(observation[:, 13], [0.0, 0.5])


def test_leap_rotation_duration_excludes_stationary_curriculum_stage() -> None:
    valid = compute_rotation_duration_valid(
        stage_valid=np.asarray([True, True, False]),
        target_speed=np.asarray([0.0, 0.04, 0.04], dtype=np.float32),
    )

    np.testing.assert_array_equal(valid, [False, True, False])


def test_leap_sustained_rotation_curriculum_bootstraps_low_speed_spin() -> None:
    curriculum = SustainedRotationCurriculumConfig()
    reward = SustainedRotationRewardConfig()

    assert curriculum.target_speeds == [
        0.0,
        0.04,
        0.07,
        0.085,
        0.10,
        0.16,
        0.25,
        0.50,
    ]
    assert curriculum.stage_durations_seconds == [
        1.0,
        1.0,
        1.5,
        1.0,
        2.0,
        2.0,
        4.0,
        10.0,
    ]
    assert curriculum.orthogonal_speed_tolerances == [
        0.10,
        0.08,
        0.065,
        0.058,
        0.05,
        0.06,
        0.075,
        0.10,
    ]
    assert curriculum.energy_level_scales == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.25,
        1.0,
    ]
    assert reward.stage_bonuses == [0.10, 0.175, 0.25, 0.30, 0.35, 0.50, 1.0]
    assert len(reward.stage_bonuses) == len(curriculum.target_speeds) - 1
    assert sum(curriculum.stage_durations_seconds) == 22.5


def test_leap_direct_rotation_reward_rejects_stall_and_reverse_motion() -> None:
    cfg = DirectRotationRewardConfig()
    common = {
        "target_speed": np.full(3, 0.30, dtype=np.float32),
        "orthogonal_speed_ema": np.zeros(3, dtype=np.float32),
        "orthogonal_tolerance": np.full(3, 0.30, dtype=np.float32),
        "position_error": np.zeros(3, dtype=np.float32),
        "previous_position_error": np.zeros(3, dtype=np.float32),
        "failure_position_radius": 0.03,
        "fingertip_contacts": np.ones((3, 4), dtype=bool),
        "palm_contact": np.zeros(3, dtype=bool),
        "no_failure_signal": np.ones(3, dtype=bool),
        "stage_valid": np.asarray([True, False, False]),
        "stage_duration_progress": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "object_linear_velocity": np.zeros((3, 3), dtype=np.float32),
        "elapsed_seconds": np.full(3, 2.0, dtype=np.float32),
        "cfg": cfg,
    }
    reward, terms = compute_direct_rotation_reward(
        axis_speed_ema=np.asarray([0.30, 0.0, -0.10], dtype=np.float32),
        **common,
    )

    assert reward[0] > reward[1] > reward[2]
    assert terms["stable_rotation"][0] == pytest.approx(6.0)
    assert terms["stall"][0] == pytest.approx(0.0)
    assert terms["stall"][1] == pytest.approx(-1.0)
    assert terms["reverse"][2] == pytest.approx(-4.0)


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_leap_sustained_rotation_reset_and_step(backend: str) -> None:
    if backend == "mujoco":
        pytest.importorskip("mujoco")
        try:
            from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
        except Exception:
            pytest.skip("mujoco.batch_env is unavailable")
    else:
        pytest.importorskip("motrixsim", reason="motrixsim not installed")

    ensure_registries()
    env = registry.make(
        "LeapInhandBallSustainedRotation",
        sim_backend=backend,
        num_envs=2,
        env_cfg_override={
            "sim_dt": 0.01 if backend == "motrix" else 0.005,
            "ctrl_dt": 0.05,
            "reset_source": "home",
            "reward_config": asdict(SustainedRotationRewardConfig()),
        },
    )
    try:
        obs, info = env.reset(np.arange(2, dtype=np.int32))
        assert obs["obs"].shape == (2, 100)
        assert np.isfinite(obs["obs"]).all()
        assert np.all(info["rotation_level"] == 0)
        assert np.all(info["rotation_stage_steps"] == 0)
        np.testing.assert_allclose(
            info["init_pose"],
            np.broadcast_to(env.default_angles[None, :], (2, 16)),
            atol=1e-5,
        )

        next_state = env.step(np.zeros((2, 16), dtype=np.float32))
        assert next_state.obs["obs"].shape == (2, 100)
        assert np.isfinite(next_state.obs["obs"]).all()
        assert np.isfinite(next_state.reward).all()
        assert next_state.terminated.dtype == bool
        assert next_state.truncated.dtype == bool
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_leap_sustained_cache_rotation_reset_and_step(backend: str) -> None:
    if backend == "mujoco":
        pytest.importorskip("mujoco")
        try:
            from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
        except Exception:
            pytest.skip("mujoco.batch_env is unavailable")
    else:
        pytest.importorskip("motrixsim", reason="motrixsim not installed")

    ensure_registries()
    env = registry.make(
        "LeapInhandBallSustainedCacheRotation",
        sim_backend=backend,
        num_envs=2,
        env_cfg_override={
            "sim_dt": 0.01 if backend == "motrix" else 0.005,
            "ctrl_dt": 0.05,
            "reset_source": "cache",
            "grasp_cache_path": "robots/leap_hand/caches/ball_grasp_official_50k.npy",
            "termination_workspace_radius": 0.05,
            "reward_config": asdict(AllegroStyleRotationRewardConfig()),
        },
    )
    try:
        obs, info = env.reset(np.arange(2, dtype=np.int32))
        assert obs["obs"].shape == (2, 100)
        assert np.isfinite(obs["obs"]).all()
        cache = np.load(LEAP_ASSET_DIR / "caches" / "ball_grasp_official_50k.npy")
        reset_rows = np.concatenate(
            [info["init_pose"], info["prev_ball_pos"], info["prev_ball_quat"]],
            axis=1,
        )
        for row in reset_rows:
            assert np.any(np.all(np.isclose(cache, row[None, :], atol=1e-6), axis=1))
        np.testing.assert_allclose(info["rotation_anchor_pos"], info["prev_ball_pos"], atol=1e-6)

        next_state = env.step(np.zeros((2, 16), dtype=np.float32))
        assert next_state.obs["obs"].shape == (2, 100)
        assert np.isfinite(next_state.obs["obs"]).all()
        assert np.isfinite(next_state.reward).all()
        assert not np.any(next_state.terminated)
        assert "reward/rotate" in next_state.info["log"]
        assert "reward/obj_linvel" in next_state.info["log"]
        assert "reward/position_error" in next_state.info["log"]
        assert "object/linvel_l1_m_s" in next_state.info["log"]
        assert "object/position_error_m" in next_state.info["log"]
        assert "contact/fingertip_count" in next_state.info["log"]
        assert "contact/index_rate" in next_state.info["log"]
        assert "contact/middle_rate" in next_state.info["log"]
        assert "contact/ring_rate" in next_state.info["log"]
        assert "contact/thumb_rate" in next_state.info["log"]
        assert "rotation/positive_spin_participation_scale" in next_state.info["log"]
        assert "reward/spin_progress" not in next_state.info["log"]
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_leap_direct_rotation_reset_and_step(backend: str) -> None:
    if backend == "mujoco":
        pytest.importorskip("mujoco")
        try:
            from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
        except Exception:
            pytest.skip("mujoco.batch_env is unavailable")
    else:
        pytest.importorskip("motrixsim", reason="motrixsim not installed")

    ensure_registries()
    env = registry.make(
        "LeapInhandBallDirectRotation",
        sim_backend=backend,
        num_envs=2,
        env_cfg_override={
            "sim_dt": 0.01 if backend == "motrix" else 0.005,
            "ctrl_dt": 0.05,
            "reset_source": "cache",
            "grasp_cache_path": "robots/leap_hand/caches/ball_grasp_official_50k.npy",
            "termination_workspace_radius": 0.05,
            "reward_config": asdict(DirectRotationRewardConfig()),
        },
    )
    try:
        obs, info = env.reset(np.arange(2, dtype=np.int32))
        assert obs["obs"].shape == (2, 100)
        assert np.isfinite(obs["obs"]).all()
        assert np.all(info["rotation_level"] == 0)
        assert np.all(info["direct_natural_handoffs"] == 0)

        next_state = env.step(np.zeros((2, 16), dtype=np.float32))
        assert next_state.obs["obs"].shape == (2, 100)
        assert np.isfinite(next_state.obs["obs"]).all()
        assert np.isfinite(next_state.reward).all()
        assert "reward/direct_stable_rotation" in next_state.info["log"]
        assert "reward/direct_stall" in next_state.info["log"]
        assert "gaiting/natural_handoff_rate" in next_state.info["log"]
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_leap_finger_gaiting_rotation_reset_and_step(backend: str) -> None:
    if backend == "mujoco":
        pytest.importorskip("mujoco")
        try:
            from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
        except Exception:
            pytest.skip("mujoco.batch_env is unavailable")
    else:
        pytest.importorskip("motrixsim", reason="motrixsim not installed")

    ensure_registries()
    env = registry.make(
        "LeapInhandBallFingerGaitingRotation",
        sim_backend=backend,
        num_envs=2,
        env_cfg_override={
            "sim_dt": 0.01 if backend == "motrix" else 0.005,
            "ctrl_dt": 0.05,
            "reset_source": "cache",
            "grasp_cache_path": "robots/leap_hand/caches/ball_grasp_official_50k.npy",
            "termination_drop_distance": 0.007,
            "reward_config": asdict(SustainedRotationRewardConfig()),
        },
    )
    try:
        obs, info = env.reset(np.arange(2, dtype=np.int32))
        assert obs["obs"].shape == (2, 114)
        assert np.isfinite(obs["obs"]).all()
        np.testing.assert_allclose(obs["obs"][:, -14:], 0.0)
        assert np.all(info["gaiting_stage_handoffs"] == 0)

        next_state = env.step(np.zeros((2, 16), dtype=np.float32))
        assert next_state.obs["obs"].shape == (2, 114)
        assert np.isfinite(next_state.obs["obs"]).all()
        assert np.isfinite(next_state.reward).all()
        assert "diagnostic/level_0_axis_speed_ema" in next_state.info["log"]
        assert "diagnostic/level_2_speed_ok_fraction" in next_state.info["log"]
        assert "diagnostic/level_2_completion_ready_fraction" in next_state.info["log"]
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_leap_ball_rotation_v2_reset_and_step(backend: str) -> None:
    if backend == "mujoco":
        pytest.importorskip("mujoco")
        try:
            from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
        except Exception:
            pytest.skip("mujoco.batch_env is unavailable")
    else:
        pytest.importorskip("motrixsim", reason="motrixsim not installed")

    ensure_registries()
    env = registry.make(
        "LeapInhandBallRotationV2",
        sim_backend=backend,
        num_envs=2,
        env_cfg_override={
            "sim_dt": 0.01 if backend == "motrix" else 1.0 / 120.0,
            "reward_config": {
                "axis_progress_scale": 2.0,
                "directional_quality_scale": 0.5,
                "retention_scale": 0.25,
                "action_rate_scale": 0.001,
                "torque_scale": 0.01,
                "work_scale": 0.1,
                "sustained_rotation_bonus": 0.2,
                "quarter_turn_bonus": 0.25,
                "drop_penalty": 5.0,
                "retention_sigma": 0.03,
                "orthogonal_speed_tolerance": 0.5,
                "reset_z_threshold": 0.4,
                "workspace_radius": 0.12,
            },
        },
    )
    try:
        obs, info = env.reset(np.arange(2, dtype=np.int32))
        assert obs["obs"].shape == (2, 99)
        assert np.isfinite(obs["obs"]).all()
        assert np.all(info["rotation_level"] == 0)

        next_state = env.step(np.zeros((2, 16), dtype=np.float32))
        assert next_state.obs["obs"].shape == (2, 99)
        assert np.isfinite(next_state.obs["obs"]).all()
        assert np.isfinite(next_state.reward).all()
        assert next_state.terminated.dtype == bool
    finally:
        env.close()


def test_leap_toss_curriculum_levels_are_monotonic() -> None:
    level_steps = [0, 1000, 2500, 4000]

    assert curriculum_level(0, level_steps) == 0
    assert curriculum_level(999, level_steps) == 0
    assert curriculum_level(1000, level_steps) == 1
    assert curriculum_level(3999, level_steps) == 2
    assert curriculum_level(4000, level_steps) == 3
    assert curriculum_level(999999, level_steps) == 3


def test_assisted_rebound_candidate_allows_active_contact_with_incoming_speed() -> None:
    contacts = np.asarray(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [False, False, False],
        ]
    )
    outward_speed = np.asarray([0.10, 0.10, 0.01, 0.10])

    candidate = assisted_rebound_candidate(
        contacts,
        outward_speed,
        min_impact_speed=0.05,
    )

    np.testing.assert_array_equal(candidate, [True, True, False, False])


def test_leap_reset_and_step() -> None:
    pytest.importorskip("mujoco")
    try:
        from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env is unavailable")

    ensure_registries()
    env = registry.make(
        "LeapInhandRotation",
        sim_backend="mujoco",
        num_envs=2,
        env_cfg_override={
            "reward_config": {
                "scales": {
                    "rotate": 1.25,
                    "obj_linvel": -0.3,
                    "pose_diff": -0.1,
                    "torque": -0.1,
                    "work": -1.0,
                    "drop": -10.0,
                },
                "angvel_clip_min": -0.25,
                "angvel_clip_max": 0.25,
                "reset_z_threshold": 0.4,
            }
        },
    )
    try:
        obs, info = env.reset(np.arange(2, dtype=np.int32))
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        assert obs["obs"].shape == (2, 105)
        assert np.isfinite(obs["obs"]).all()

        next_state = env.step(np.zeros((2, 16), dtype=np.float32))
        assert next_state.obs["obs"].shape == (2, 105)
        assert np.isfinite(next_state.obs["obs"]).all()
        assert np.isfinite(next_state.reward).all()
        assert next_state.terminated.dtype == bool
        assert next_state.truncated.dtype == bool
    finally:
        env.close()


def test_leap_toss_reset_and_step() -> None:
    pytest.importorskip("mujoco")
    try:
        from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env is unavailable")

    ensure_registries()
    env = registry.make(
        "LeapInhandToss",
        sim_backend="mujoco",
        num_envs=2,
        env_cfg_override={
            "reward_config": asdict(TossRewardConfig()),
            "curriculum": {"enabled": False},
        },
    )
    try:
        obs, info = env.reset(np.arange(2, dtype=np.int32))
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        assert obs["obs"].shape == (2, 75)
        assert np.isfinite(obs["obs"]).all()
        np.testing.assert_array_equal(env._phase_timeout_steps(), [100, 24, 10, 50, 60])

        next_state = env.step(np.zeros((2, 16), dtype=np.float32))
        assert next_state.obs["obs"].shape == (2, 75)
        assert np.isfinite(next_state.obs["obs"]).all()
        assert np.isfinite(next_state.reward).all()
        assert next_state.terminated.dtype == bool
        assert next_state.truncated.dtype == bool
    finally:
        env.close()


def test_leap_motrix_reset_and_step() -> None:
    pytest.importorskip("motrixsim", reason="motrixsim not installed")

    ensure_registries()
    env = registry.make(
        "LeapInhandRotation",
        sim_backend="motrix",
        num_envs=2,
        env_cfg_override={
            "sim_dt": 0.01,
            "reward_config": {
                "scales": {
                    "rotate": 1.25,
                    "obj_linvel": -0.3,
                    "pose_diff": -0.1,
                    "torque": -0.1,
                    "work": -1.0,
                    "drop": -10.0,
                },
                "angvel_clip_min": -0.25,
                "angvel_clip_max": 0.25,
                "reset_z_threshold": 0.4,
            },
        },
    )
    try:
        obs, info = env.reset(np.arange(2, dtype=np.int32))
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        assert obs["obs"].shape == (2, 105)
        assert np.isfinite(obs["obs"]).all()

        next_state = env.step(np.zeros((2, 16), dtype=np.float32))
        assert next_state.obs["obs"].shape == (2, 105)
        assert np.isfinite(next_state.obs["obs"]).all()
        assert np.isfinite(next_state.reward).all()
        assert next_state.terminated.dtype == bool
        assert next_state.truncated.dtype == bool
    finally:
        env.close()


def test_leap_toss_motrix_reset_and_step() -> None:
    pytest.importorskip("motrixsim", reason="motrixsim not installed")

    ensure_registries()
    env = registry.make(
        "LeapInhandToss",
        sim_backend="motrix",
        num_envs=2,
        env_cfg_override={
            "sim_dt": 0.01,
            "reward_config": asdict(TossRewardConfig()),
            "curriculum": {"enabled": False},
        },
    )
    try:
        obs, info = env.reset(np.arange(2, dtype=np.int32))
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        assert obs["obs"].shape == (2, 75)
        assert np.isfinite(obs["obs"]).all()

        next_state = env.step(np.zeros((2, 16), dtype=np.float32))
        assert next_state.obs["obs"].shape == (2, 75)
        assert np.isfinite(next_state.obs["obs"]).all()
        assert np.isfinite(next_state.reward).all()
    finally:
        env.close()


def test_support_pose_distance_state_a_exact_and_scaled_delta() -> None:
    ctrl_lower = np.full(16, -1.0, dtype=np.float64)
    ctrl_upper = np.full(16, 1.0, dtype=np.float64)
    dof_pos = np.zeros((1, 16), dtype=np.float64)
    dof_pos[0, SUPPORT_JOINT_INDICES] = STATE_A_SUPPORT_QPOS

    dist_exact = compute_support_pose_distance(dof_pos, ctrl_lower, ctrl_upper)
    np.testing.assert_allclose(dist_exact, [0.0], atol=1e-7)

    # Shift each selected joint by 10% of its joint range (ctrl_upper - ctrl_lower = 2.0 -> 0.2)
    dof_pos_shifted = dof_pos.copy()
    dof_pos_shifted[0, SUPPORT_JOINT_INDICES] += 0.20
    dist_shifted = compute_support_pose_distance(dof_pos_shifted, ctrl_lower, ctrl_upper)
    np.testing.assert_allclose(dist_shifted, [0.10], atol=1e-7)


def test_support_pose_progress_sign_and_clipping() -> None:
    scale = 0.25
    clip = 0.04
    prev_dist = np.asarray([0.20, 0.10], dtype=np.float64)
    curr_dist = np.asarray([0.15, 0.14], dtype=np.float64)
    raw_progress = prev_dist - curr_dist
    clipped_progress = np.clip(raw_progress, -clip, clip)
    reward = scale * clipped_progress

    np.testing.assert_allclose(raw_progress, [0.05, -0.04])
    np.testing.assert_allclose(clipped_progress, [0.04, -0.04])
    np.testing.assert_allclose(reward, [0.01, -0.01])


def test_tip_collision_reference_transform_identity_quat() -> None:
    fingertip_body_pos = np.asarray(
        [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]],
        dtype=np.float64,
    )
    fingertip_body_quat = np.asarray(
        [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
        dtype=np.float64,
    )

    tip_pos = compute_tip_collision_reference_positions(fingertip_body_pos, fingertip_body_quat)
    expected = fingertip_body_pos + LEAP_TIP_COLLISION_LOCAL_POS

    np.testing.assert_allclose(tip_pos, expected)


def test_opposition_quality_geometry() -> None:
    ball_pos = np.zeros((1, 3), dtype=np.float64)

    # 180 degrees (opposite) -> quality = 1.0
    index_180 = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
    ring_180 = np.asarray([[-1.0, 0.0, 0.0]], dtype=np.float64)
    q_180 = compute_index_ring_opposition_quality(index_180, ring_180, ball_pos)
    np.testing.assert_allclose(q_180, [1.0], atol=1e-7)

    # 90 degrees -> quality = 0.5
    index_90 = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
    ring_90 = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64)
    q_90 = compute_index_ring_opposition_quality(index_90, ring_90, ball_pos)
    np.testing.assert_allclose(q_90, [0.5], atol=1e-7)

    # 0 degrees (same side) -> quality = 0.0
    index_0 = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
    ring_0 = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
    q_0 = compute_index_ring_opposition_quality(index_0, ring_0, ball_pos)
    np.testing.assert_allclose(q_0, [0.0], atol=1e-7)


def test_contact_weighted_opposition_potential() -> None:
    quality = 0.8
    contacts = np.asarray(
        [
            [True, False, True, False],
            [True, False, False, False],
            [False, False, True, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )
    index_contact = contacts[:, 0]
    ring_contact = contacts[:, 2]
    potential = (index_contact & ring_contact).astype(np.float64) * quality

    np.testing.assert_allclose(potential, [0.8, 0.0, 0.0, 0.0])


def test_opposition_progress_sign_and_clipping() -> None:
    scale = 0.20
    clip = 0.05
    prev_potential = np.asarray([0.0, 0.8], dtype=np.float64)
    curr_potential = np.asarray([0.8, 0.0], dtype=np.float64)
    raw_progress = curr_potential - prev_potential
    clipped_progress = np.clip(raw_progress, -clip, clip)
    reward = scale * clipped_progress

    np.testing.assert_allclose(raw_progress, [0.8, -0.8])
    np.testing.assert_allclose(clipped_progress, [0.05, -0.05])
    np.testing.assert_allclose(reward, [0.01, -0.01])


def test_sustained_cache_rotation_reward_config_defaults() -> None:
    cfg = AllegroStyleRotationRewardConfig()

    assert cfg.positive_spin_base_contact_scale == 1.0
    assert cfg.positive_spin_index_contact_scale == 0.0
    assert cfg.positive_spin_middle_contact_scale == 0.0
    assert cfg.support_pose_progress_scale == 0.25
    assert cfg.support_pose_progress_clip == 0.04
    assert cfg.opposition_progress_scale == 0.20
    assert cfg.opposition_progress_clip == 0.05

