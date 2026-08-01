from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from unilab.envs.manipulation.leap_inhand.base import MENAGERIE_SIM_JOINT_NAMES

_SCRIPT_PATH = (
    Path(__file__).parent.parent.parent
    / "scripts"
    / "diagnose_leap_ball_rotation_authority.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "diagnose_leap_ball_rotation_authority", _SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_load_candidate_validates_and_normalizes_quaternion(tmp_path: Path):
    mod = _load_script()
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(
            {
                "coordinate_contract": {
                    "qpos_joint_names": list(MENAGERIE_SIM_JOINT_NAMES)
                },
                "qpos": [*np.arange(19, dtype=float), 2.0, 0.0, 0.0, 0.0],
                "ctrl": np.arange(16, dtype=float).tolist(),
            }
        ),
        encoding="utf-8",
    )

    qpos, ctrl, joint_names = mod.load_candidate(path)

    np.testing.assert_allclose(qpos[19:23], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(ctrl, np.arange(16, dtype=float))
    assert joint_names == MENAGERIE_SIM_JOINT_NAMES


def test_load_candidate_rejects_duplicate_joint_names(tmp_path: Path):
    mod = _load_script()
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(
            {
                "coordinate_contract": {"qpos_joint_names": ["if_mcp"] * 16},
                "qpos": [*np.zeros(19), 1.0, 0.0, 0.0, 0.0],
                "ctrl": np.zeros(16).tolist(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="16 unique names"):
        mod.load_candidate(path)


def test_build_probe_targets_creates_signed_single_joint_batch():
    mod = _load_script()
    ctrl = np.zeros(16)
    limits = np.tile([-0.1, 0.1], (16, 1))

    targets, probes = mod.build_probe_targets(
        ctrl,
        MENAGERIE_SIM_JOINT_NAMES,
        limits,
        0.04,
    )

    assert targets.shape == (33, 16)
    np.testing.assert_allclose(targets[0], 0.0)
    assert probes[1]["joint_name"] == "if_mcp"
    assert probes[1]["applied_delta_rad"] == pytest.approx(-0.04)
    assert probes[2]["applied_delta_rad"] == pytest.approx(0.04)
    assert np.count_nonzero(targets[1]) == 1
    assert np.count_nonzero(targets[2]) == 1


def test_build_probe_targets_reports_limit_clipping():
    mod = _load_script()
    ctrl = np.zeros(16)
    ctrl[0] = 0.09
    limits = np.tile([-0.1, 0.1], (16, 1))

    targets, probes = mod.build_probe_targets(
        ctrl,
        MENAGERIE_SIM_JOINT_NAMES,
        limits,
        0.04,
    )

    assert targets[2, 0] == pytest.approx(0.1)
    assert probes[2]["requested_delta_rad"] == pytest.approx(0.04)
    assert probes[2]["applied_delta_rad"] == pytest.approx(0.01)


def test_classify_probe_requires_safety_and_directional_motion():
    mod = _load_script()
    common = {
        "finite": True,
        "minimum_ball_height": 0.67,
        "maximum_ball_displacement": 0.002,
        "minimum_contact_count": 2,
        "thumb_contact_retained": True,
        "maximum_self_penetration": 0.0005,
        "maximum_object_penetration": 0.0002,
        "minimum_height": 0.4,
        "maximum_displacement": 0.005,
        "maximum_penetration": 0.001,
        "minimum_axis_rotation": 0.0025,
        "minimum_axis_speed": 0.05,
    }

    positive = mod.classify_probe(
        axis_rotation=0.004,
        signed_peak_axis_speed=0.08,
        **common,
    )
    negative = mod.classify_probe(
        axis_rotation=-0.004,
        signed_peak_axis_speed=-0.08,
        **common,
    )
    unsafe = mod.classify_probe(
        axis_rotation=0.004,
        signed_peak_axis_speed=0.08,
        **{**common, "thumb_contact_retained": False},
    )

    assert positive == {
        "safe": True,
        "positive_authority": True,
        "negative_authority": False,
    }
    assert negative == {
        "safe": True,
        "positive_authority": False,
        "negative_authority": True,
    }
    assert unsafe == {
        "safe": False,
        "positive_authority": False,
        "negative_authority": False,
    }


def test_select_safe_extreme_excludes_stronger_unsafe_probe():
    mod = _load_script()
    reports = [
        {"safe": False, "axis_rotation_rad": -0.02, "joint_name": "unsafe"},
        {"safe": True, "axis_rotation_rad": -0.01, "joint_name": "safe_negative"},
        {"safe": True, "axis_rotation_rad": 0.015, "joint_name": "safe_positive"},
    ]

    assert mod.select_safe_extreme(reports, direction="negative")["joint_name"] == (
        "safe_negative"
    )
    assert mod.select_safe_extreme(reports, direction="positive")["joint_name"] == (
        "safe_positive"
    )
    with pytest.raises(ValueError, match="direction"):
        mod.select_safe_extreme(reports, direction="sideways")
