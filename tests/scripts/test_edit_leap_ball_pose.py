from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "edit_leap_ball_pose.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("edit_leap_ball_pose", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_normalize_pose_qpos_normalizes_only_ball_quaternion():
    mod = _load_script()
    qpos = np.arange(23, dtype=np.float64) / 10.0
    qpos[19:23] = [2.0, 0.0, 0.0, 0.0]

    normalized = mod.normalize_pose_qpos(qpos)

    np.testing.assert_allclose(normalized[:19], qpos[:19])
    np.testing.assert_allclose(normalized[19:23], [1.0, 0.0, 0.0, 0.0])
    assert normalized is not qpos


def test_apply_joint_delta_clips_and_synchronizes_ctrl():
    mod = _load_script()
    qpos = np.zeros(23, dtype=np.float64)
    ctrl = np.zeros(16, dtype=np.float64)
    lower = np.full(16, -0.5)
    upper = np.full(16, 0.5)

    value = mod.apply_joint_delta(qpos, ctrl, 13, 0.75, lower, upper)

    assert value == pytest.approx(0.5)
    assert qpos[13] == pytest.approx(0.5)
    assert ctrl[13] == pytest.approx(0.5)
    np.testing.assert_allclose(qpos[:13], 0.0)
    np.testing.assert_allclose(qpos[14:], 0.0)


def test_apply_ball_delta_changes_only_selected_world_axis():
    mod = _load_script()
    qpos = np.zeros(23, dtype=np.float64)
    qpos[19] = 1.0

    value = mod.apply_ball_delta(qpos, axis=2, delta=0.001)

    assert value == pytest.approx(0.001)
    np.testing.assert_allclose(qpos[16:19], [0.0, 0.0, 0.001])
    np.testing.assert_allclose(qpos[19:23], [1.0, 0.0, 0.0, 0.0])


def test_pose_helpers_reject_invalid_indices_and_zero_quaternion():
    mod = _load_script()
    qpos = np.zeros(23, dtype=np.float64)
    ctrl = np.zeros(16, dtype=np.float64)
    lower = np.full(16, -1.0)
    upper = np.full(16, 1.0)

    with pytest.raises(ValueError, match="non-zero length"):
        mod.normalize_pose_qpos(qpos)
    with pytest.raises(IndexError, match="Hand qpos index"):
        mod.apply_joint_delta(qpos, ctrl, 16, 0.1, lower, upper)
    with pytest.raises(IndexError, match="Ball position axis"):
        mod.apply_ball_delta(qpos, axis=3, delta=0.001)


def test_joint_slider_degrees_clamp_and_sync_resolved_actuator():
    mod = _load_script()
    qpos = np.zeros(12, dtype=np.float64)
    ctrl = np.zeros(10, dtype=np.float64)
    joint = mod.HandJointMetadata(
        name="test_joint",
        joint_id=2,
        qpos_address=4,
        actuator_id=7,
        lower=-0.5,
        upper=0.5,
    )

    applied = mod.set_hand_joint_degrees(qpos, ctrl, joint, 20.0)

    assert applied == pytest.approx(np.deg2rad(20.0))
    assert qpos[4] == pytest.approx(np.deg2rad(20.0))
    assert ctrl[7] == pytest.approx(np.deg2rad(20.0))
    np.testing.assert_allclose(np.delete(qpos, 4), 0.0)
    np.testing.assert_allclose(np.delete(ctrl, 7), 0.0)

    clamped = mod.set_hand_joint_degrees(qpos, ctrl, joint, 90.0)

    assert clamped == pytest.approx(0.5)
    assert qpos[4] == pytest.approx(0.5)
    assert ctrl[7] == pytest.approx(0.5)


def test_ball_position_uses_resolved_address_and_preserves_quaternion():
    mod = _load_script()
    qpos = np.zeros(15, dtype=np.float64)
    ball_qpos_address = 5
    qpos[ball_qpos_address + 3 : ball_qpos_address + 7] = [0.5, 0.5, 0.5, 0.5]
    before_quat = qpos[ball_qpos_address + 3 : ball_qpos_address + 7].copy()

    mod.set_ball_position(qpos, ball_qpos_address, [-0.03, 0.04, 0.57])

    np.testing.assert_allclose(qpos[ball_qpos_address : ball_qpos_address + 3], [-0.03, 0.04, 0.57])
    np.testing.assert_array_equal(qpos[ball_qpos_address + 3 : ball_qpos_address + 7], before_quat)


def test_identity_quaternion_is_zero_zyx_euler():
    mod = _load_script()

    euler = mod.quat_wxyz_to_euler_zyx([1.0, 0.0, 0.0, 0.0])
    quat = mod.euler_zyx_to_quat_wxyz(0.0, 0.0, 0.0)

    np.testing.assert_allclose(euler, [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(quat, [1.0, 0.0, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize(
    "euler",
    [
        (0.2, -0.3, 0.4),
        (-1.1, 0.7, -2.0),
        (np.deg2rad(90.0), np.deg2rad(30.0), np.deg2rad(-45.0)),
    ],
)
def test_euler_quaternion_round_trip_and_normalization(euler):
    mod = _load_script()

    quat = mod.euler_zyx_to_quat_wxyz(*euler)
    round_trip = mod.quat_wxyz_to_euler_zyx(quat)

    assert np.linalg.norm(quat) == pytest.approx(1.0)
    np.testing.assert_allclose(round_trip, euler, atol=1e-12)


def test_quaternion_uses_mujoco_wxyz_order():
    mod = _load_script()

    quat = mod.euler_zyx_to_quat_wxyz(np.pi / 2.0, 0.0, 0.0)

    np.testing.assert_allclose(
        quat,
        [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0],
        atol=1e-12,
    )


def test_set_ball_euler_writes_normalized_quaternion_only():
    mod = _load_script()
    qpos = np.arange(15, dtype=np.float64)
    ball_qpos_address = 5
    position_before = qpos[ball_qpos_address : ball_qpos_address + 3].copy()

    quat = mod.set_ball_euler_zyx(qpos, ball_qpos_address, [0.4, -0.2, 1.3])

    np.testing.assert_array_equal(qpos[ball_qpos_address : ball_qpos_address + 3], position_before)
    np.testing.assert_allclose(qpos[ball_qpos_address + 3 : ball_qpos_address + 7], quat)
    assert np.linalg.norm(quat) == pytest.approx(1.0)


def test_reset_pose_arrays_restores_qpos_ctrl_and_zeroes_qvel():
    mod = _load_script()
    initial_qpos = np.linspace(-1.0, 1.0, 23)
    initial_ctrl = np.linspace(-0.5, 0.5, 16)
    qpos = np.full(23, 9.0)
    ctrl = np.full(16, 8.0)
    qvel = np.full(22, 7.0)

    mod.reset_pose_arrays(qpos, ctrl, qvel, initial_qpos, initial_ctrl)

    np.testing.assert_array_equal(qpos, initial_qpos)
    np.testing.assert_array_equal(ctrl, initial_ctrl)
    np.testing.assert_array_equal(qvel, 0.0)


def test_leap_scene_metadata_forward_and_settle_smoke():
    mujoco = pytest.importorskip("mujoco")
    mod = _load_script()
    model = mujoco.MjModel.from_xml_path(str(mod.DEFAULT_SCENE.resolve()))
    data = mujoco.MjData(model)

    ball_qpos_address = mod.find_ball_freejoint_qpos_address(model)
    joints = mod.find_hand_joint_metadata(model)

    assert ball_qpos_address == int(
        model.jnt_qposadr[
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                "leap_object_joint",
            )
        ]
    )
    assert len(joints) == 16
    assert len({joint.qpos_address for joint in joints}) == 16
    assert len({joint.actuator_id for joint in joints}) == 16
    for index, joint in enumerate(joints):
        target_degrees = np.rad2deg(joint.lower + 0.25 * (joint.upper - joint.lower))
        expected = mod.set_hand_joint_degrees(
            data.qpos,
            data.ctrl,
            joint,
            target_degrees,
        )
        assert data.qpos[joint.qpos_address] == pytest.approx(expected)
        assert data.ctrl[joint.actuator_id] == pytest.approx(expected)
        assert index < mod.HAND_DOF
    mujoco.mj_forward(model, data)
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert data.time == pytest.approx(10 * model.opt.timestep)
