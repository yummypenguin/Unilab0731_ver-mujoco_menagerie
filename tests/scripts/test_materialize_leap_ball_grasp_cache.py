from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_PATH = (
    Path(__file__).parent.parent.parent
    / "scripts"
    / "materialize_leap_ball_grasp_cache.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "materialize_leap_ball_grasp_cache", _SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _raw_rows() -> np.ndarray:
    rows = np.zeros((4, 23), dtype=np.float32)
    rows[:, 19] = 1.0
    rows[1, 0] = 0.0004  # Reset-equivalent duplicate of row zero.
    rows[2, 0] = 0.01
    rows[3, 0] = 0.02
    return rows


def _pass_report(count: int) -> dict[str, int]:
    return {"input_rows": count, "accepted_rows": count, "rejected_rows": 0}


def test_materializer_runs_two_replays_before_atomic_publication(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_script()
    source = tmp_path / "raw.npy"
    output = tmp_path / "formal.npy"
    np.save(source, _raw_rows())
    calls: list[int] = []

    def accept_all(rows, args):
        del args
        calls.append(len(rows))
        return rows, np.ones((len(rows), 4), dtype=bool), _pass_report(len(rows))

    monkeypatch.setattr(mod, "_validate_pass", accept_all)
    args = mod._parse_args(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--target",
            "2",
            "--replay-passes",
            "2",
        ]
    )

    report, complete = mod.materialize(args)

    assert complete is True
    assert calls == [3, 3]
    assert report["raw_rows"] == 4
    assert report["unique_rows"] == 3
    assert report["duplicate_rows"] == 1
    assert len(report["passes"]) == 2
    assert np.load(output).shape == (2, 23)
    assert output.with_suffix(".json").exists()


def test_materializer_does_not_publish_when_stable_rows_are_short(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_script()
    source = tmp_path / "raw.npy"
    output = tmp_path / "formal.npy"
    np.save(source, _raw_rows())

    def keep_one(rows, args):
        del args
        kept = rows[:1]
        return kept, np.ones((1, 4), dtype=bool), _pass_report(len(rows))

    monkeypatch.setattr(mod, "_validate_pass", keep_one)
    args = mod._parse_args(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--target",
            "2",
            "--replay-passes",
            "2",
        ]
    )

    report, complete = mod.materialize(args)

    assert complete is False
    assert report["stable_rows_before_selection"] == 1
    assert report["shortfall_rows"] == 1
    assert not output.exists()
    assert output.with_suffix(".json").exists()


def test_materializer_rejects_output_that_overwrites_raw_input(tmp_path: Path) -> None:
    mod = _load_script()
    source = tmp_path / "raw.npy"
    np.save(source, _raw_rows())
    args = mod._parse_args(
        [
            "--input",
            str(source),
            "--output",
            str(source),
            "--target",
            "2",
        ]
    )

    with pytest.raises(ValueError, match="different from every raw input"):
        mod.materialize(args)
