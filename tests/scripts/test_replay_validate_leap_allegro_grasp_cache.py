from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.parent
SCRIPT_PATH = ROOT / "scripts" / "replay_validate_leap_allegro_grasp_cache.py"


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("replay_validate_leap_cache", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture
def validator():
    return _load_script()


def _rows(count: int) -> np.ndarray:
    rows = np.zeros((count, 23), dtype=np.float32)
    rows[:, 19] = 1.0
    return rows


@pytest.mark.parametrize("shape", [(23,), (2, 22), (2, 24), (0, 23)])
def test_invalid_shape_or_empty_cache_is_rejected(validator, shape) -> None:
    with pytest.raises(validator.ValidationInputError):
        validator.validate_cache_array(
            np.zeros(shape, dtype=np.float32),
            batch_size=2,
            settle_seconds=3.0,
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nan_and_inf_are_rejected(validator, value: float) -> None:
    rows = _rows(2)
    rows[1, 3] = value
    with pytest.raises(validator.ValidationInputError, match="NaN or Inf"):
        validator.validate_cache_array(rows, batch_size=2, settle_seconds=3.0)


def test_zero_quaternion_is_rejected(validator) -> None:
    rows = _rows(2)
    rows[1, 19:23] = 0.0
    with pytest.raises(validator.ValidationInputError, match="quaternion"):
        validator.validate_cache_array(rows, batch_size=2, settle_seconds=3.0)


@pytest.mark.parametrize(
    ("batch_size", "settle_seconds"),
    [(0, 3.0), (-1, 3.0), (2, 0.0), (2, -1.0), (2, np.inf)],
)
def test_invalid_batch_size_and_settle_seconds_are_rejected(
    validator, batch_size: int, settle_seconds: float
) -> None:
    with pytest.raises(validator.ValidationInputError):
        validator.validate_cache_array(
            _rows(2),
            batch_size=batch_size,
            settle_seconds=settle_seconds,
        )


def test_loading_is_read_only_and_creates_no_npy(
    validator, tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "input.npy"
    expected = _rows(2)
    np.save(path, expected)
    before = {item.name for item in tmp_path.glob("*.npy")}

    def forbidden_save(*args, **kwargs):
        del args, kwargs
        raise AssertionError("validator attempted to save an npy")

    monkeypatch.setattr(validator.np, "save", forbidden_save)
    loaded = validator.load_cache(path, batch_size=2, settle_seconds=3.0)

    assert not loaded.flags.writeable
    np.testing.assert_array_equal(loaded, expected)
    assert {item.name for item in tmp_path.glob("*.npy")} == before


def test_initialize_uses_row_targets_and_zero_qvel(validator) -> None:
    rows = _rows(2).astype(np.float64)
    rows[0, :16] = np.arange(16)
    rows[1, :16] = np.arange(16) + 20
    captured: dict[str, np.ndarray] = {}

    class Backend:
        def set_state(self, env_ids, qpos, qvel):
            captured["env_ids"] = env_ids.copy()
            captured["qpos"] = qpos.copy()
            captured["qvel"] = qvel.copy()

    class Provider:
        def _build_info_updates(self, env, hand, ball_pos, ball_quat):
            del env, ball_pos, ball_quat
            return {"prev_ctrl": np.full_like(hand, -999.0), "provider_used": True}

    env = SimpleNamespace(
        nv=22,
        _backend=Backend(),
        state=SimpleNamespace(
            info={},
            terminated=np.ones(2, dtype=bool),
            truncated=np.ones(2, dtype=bool),
        ),
    )
    validator.initialize_replay_batch(env, Provider(), rows)

    np.testing.assert_array_equal(captured["qpos"], rows)
    np.testing.assert_array_equal(captured["qvel"], np.zeros((2, 22)))
    np.testing.assert_array_equal(env.state.info["prev_ctrl"], rows[:, :16])
    np.testing.assert_array_equal(env.state.info["init_pose"], rows[:, :16])
    np.testing.assert_array_equal(env.state.info["prev_dof_pos"], rows[:, :16])
    assert env.state.info["provider_used"] is True
    assert not env.state.terminated.any()
    assert not env.state.truncated.any()


def _measurement_env(validator, conditions) -> Any:
    calls = {"production": 0}

    def production():
        calls["production"] += 1
        return tuple(np.asarray(value, dtype=bool) for value in conditions)

    env = SimpleNamespace(
        _compute_grasp_conditions=production,
        get_ball_pos=lambda: np.asarray([[0.0, 0.0, 0.7], [0.0, 0.0, 0.6]]),
        get_fingertip_pos=lambda: np.zeros((2, 4, 3)),
        _sensor_scalar=lambda value: value,
        get_sensor_data=lambda name: {
            "a": np.asarray([1.0, 0.0]),
            "b": np.asarray([1.0, 1.0]),
            "c": np.asarray([0.0, 0.0]),
            "d": np.asarray([0.0, 0.0]),
        }[name],
        _CONTACT_SENSORS=("a", "b", "c", "d"),
        cfg=SimpleNamespace(grasp_max_fingertip_distance=1.0),
        _reward_cfg=SimpleNamespace(reset_z_threshold=0.65),
        calls=calls,
    )
    return env


def test_measurement_uses_production_conditions_and_records_initial_invalid(validator) -> None:
    env = _measurement_env(
        validator,
        ([True, False], [True, True], [True, False]),
    )
    snapshot = validator.measure_conditions(env, 2)

    assert env.calls["production"] == 1
    np.testing.assert_array_equal(snapshot.valid, [True, False])
    np.testing.assert_array_equal(snapshot.contact_count, [2, 1])
    assert snapshot.height_margin[0] == pytest.approx(0.05)
    assert snapshot.height_margin[1] == pytest.approx(-0.05)
    assert validator._global_failed_indices(~snapshot.valid, 100) == [101]


def test_ever_terminated_final_valid_is_not_timeout_success(validator) -> None:
    success = validator.compute_timeout_success(
        final_truncated=np.asarray([True]),
        ever_terminated=np.asarray([True]),
        final_cond1=np.asarray([True]),
        final_cond2=np.asarray([True]),
        final_cond3=np.asarray([True]),
    )
    assert not success[0]


def test_final_not_truncated_is_not_timeout_success(validator) -> None:
    success = validator.compute_timeout_success(
        final_truncated=np.asarray([False]),
        ever_terminated=np.asarray([False]),
        final_cond1=np.asarray([True]),
        final_cond2=np.asarray([True]),
        final_cond3=np.asarray([True]),
    )
    assert not success[0]


def test_cache_save_guard_raises_and_records_call(validator) -> None:
    guard = validator.CacheSaveGuard()
    with pytest.raises(RuntimeError, match="forbids"):
        guard(force=True)
    assert guard.called


def test_validator_source_has_no_cache_or_strict_paths(validator) -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "np.save" not in source
    assert "replay_validate_grasp_cache_rows" not in source
    assert "surface" not in source
    assert "penetration" not in source
    assert "serialization" not in source
    assert "_filter_grasp_rows" not in source
    assert inspect.getsource(validator.measure_conditions).count(
        "_compute_grasp_conditions"
    ) == 1


def test_padding_last_batch_does_not_change_active_rows(validator) -> None:
    rows = _rows(2)
    rows[0, 0] = 1.0
    rows[1, 0] = 2.0
    padded = validator._pad_final_batch(rows, 4)

    assert padded.shape == (4, 23)
    np.testing.assert_array_equal(padded[:2], rows)
    np.testing.assert_array_equal(padded[2], rows[1])
    np.testing.assert_array_equal(padded[3], rows[1])
    padded[0, 0] = 99.0
    assert rows[0, 0] == 1.0


def test_failed_indices_keep_original_global_index_for_partial_batch(validator) -> None:
    failed = np.asarray([False, True])
    assert validator._global_failed_indices(failed, 256) == [257]
