from __future__ import annotations

import numpy as np
import pytest

from unilab.visualization.reward_telemetry import (
    RewardTelemetry,
    parse_reward_value_keys,
)


def test_parse_reward_value_keys_normalizes_and_deduplicates() -> None:
    assert parse_reward_value_keys("rotate, reward/retention,rotate") == (
        "reward/rotate",
        "reward/retention",
    )


def test_reward_telemetry_tracks_actual_return_and_logged_term_deltas() -> None:
    telemetry = RewardTelemetry(max_terms=4)
    telemetry.update(
        np.array([0.3]),
        {
            "steps": np.array([1]),
            "log": {
                "reward/rotate": 0.2,
                "reward/retention": 0.1,
                "diagnostic/ignored": 99.0,
                "reward/total": 0.3,
            },
        },
        advanced=True,
    )
    telemetry.update(
        np.array([0.4]),
        {
            "steps": np.array([2]),
            "log": {
                "reward/rotate": 0.35,
                "reward/retention": 0.05,
                "reward/total": 0.4,
            },
        },
        advanced=True,
    )

    assert telemetry.episode_return == pytest.approx(0.7)
    assert telemetry.visible_terms() == [
        ("rotate", pytest.approx(0.35), pytest.approx(0.15)),
        ("retention", pytest.approx(0.05), pytest.approx(-0.05)),
        ("total", pytest.approx(0.4), pytest.approx(0.1)),
    ]


def test_reward_telemetry_does_not_count_paused_frames_and_resets_episode_return() -> None:
    telemetry = RewardTelemetry(selected_keys=("total",))
    first_info = {"steps": np.array([8]), "log": {"reward/total": 0.25}}
    telemetry.update(np.array([0.25]), first_info, advanced=True)
    telemetry.update(np.array([0.25]), first_info, advanced=False)
    telemetry.update(
        np.array([-0.5]),
        {"steps": np.array([0]), "log": {"reward/total": -0.5}},
        advanced=True,
    )

    assert telemetry.episode_return == pytest.approx(-0.5)
    assert telemetry.visible_terms() == [("total", -0.5, -0.75)]


def test_reward_telemetry_rejects_non_finite_log_values() -> None:
    telemetry = RewardTelemetry()
    telemetry.update(
        np.array([0.0]),
        {
            "steps": np.array([1]),
            "log": {
                "reward/nan": np.nan,
                "reward/finite": 0.5,
            },
        },
        advanced=True,
    )

    assert telemetry.visible_terms() == [("finite", 0.5, 0.0)]


def test_reward_telemetry_refreshes_equal_new_samples_and_keeps_total_visible() -> None:
    telemetry = RewardTelemetry(max_terms=2)
    telemetry.update(
        np.array([0.0]),
        {
            "steps": np.array([1]),
            "log": {
                "reward/first": 1.0,
                "reward/second": 2.0,
                "reward/total": 3.0,
            },
        },
        advanced=True,
    )
    telemetry.update(
        np.array([0.0]),
        {
            "steps": np.array([2]),
            "log": {
                "reward/first": 1.0,
                "reward/second": 2.0,
                "reward/total": 3.0,
            },
        },
        advanced=True,
    )

    assert telemetry.visible_terms() == [
        ("first", 1.0, 0.0),
        ("total", 3.0, 0.0),
    ]
