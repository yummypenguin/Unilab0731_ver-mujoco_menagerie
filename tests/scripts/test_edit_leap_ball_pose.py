from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_PATH = (
    Path(__file__).parent.parent.parent / "scripts" / "edit_leap_ball_pose.py"
)


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
