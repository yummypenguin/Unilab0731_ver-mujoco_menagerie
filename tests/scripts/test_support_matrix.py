from __future__ import annotations

from pathlib import Path

from unilab.utils.support_matrix import EvidenceLevel, build_support_rows


def _row(entrypoint_label: str, task_slug: str):
    root = Path(__file__).resolve().parents[2]
    for row in build_support_rows(root):
        if row.entrypoint_label == entrypoint_label and row.task_slug == task_slug:
            return row
    raise AssertionError(f"Missing support row: {entrypoint_label} / {task_slug}")


def test_support_matrix_marks_go2_ppo_backends_as_tested():
    row = _row("PPO (torch)", "go2_joystick_flat")

    assert row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert row.cells["motrix"].level == EvidenceLevel.TESTED


def test_support_matrix_marks_appo_go1_backends_as_tested():
    row = _row("APPO (torch)", "go1_joystick_flat")

    assert row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert row.cells["motrix"].level == EvidenceLevel.TESTED


def test_support_matrix_keeps_uncovered_mlx_tasks_at_configured():
    row = _row("PPO (mlx)", "g1_motion_tracking")

    assert row.cells["mujoco"].level == EvidenceLevel.CONFIGURED
    assert row.cells["motrix"].level == EvidenceLevel.CONFIGURED


def test_support_matrix_marks_sharpa_motrix_phase1_support():
    row = _row("PPO (torch)", "sharpa_inhand")

    assert row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert row.cells["motrix"].level == EvidenceLevel.TESTED

    appo_row = _row("APPO (torch)", "sharpa_inhand")

    assert appo_row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert appo_row.cells["motrix"].level == EvidenceLevel.TESTED
    allegro_appo_row = _row("APPO (torch)", "allegro_inhand")

    assert allegro_appo_row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert allegro_appo_row.cells["motrix"].level == EvidenceLevel.TESTED


def test_support_matrix_marks_leap_ppo_backends_as_tested():
    row = _row("PPO (torch)", "leap_inhand")

    assert row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert row.cells["motrix"].level == EvidenceLevel.TESTED

    ball_row = _row("PPO (torch)", "leap_inhand_ball")

    assert ball_row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert ball_row.cells["motrix"].level == EvidenceLevel.TESTED

    ball_v2_row = _row("PPO (torch)", "leap_inhand_ball_rotation_v2")

    assert ball_v2_row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert ball_v2_row.cells["motrix"].level == EvidenceLevel.TESTED

    sustained_row = _row("PPO (torch)", "leap_inhand_ball_sustained")

    assert sustained_row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert sustained_row.cells["motrix"].level == EvidenceLevel.TESTED

    sustained_cache_row = _row(
        "PPO (torch)", "leap_inhand_ball_sustained_cache"
    )

    assert sustained_cache_row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert sustained_cache_row.cells["motrix"].level == EvidenceLevel.TESTED

    allegro_faithful_row = _row("PPO (torch)", "leap_inhand_ball_allegro")

    assert allegro_faithful_row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert allegro_faithful_row.cells["motrix"].level == EvidenceLevel.TESTED

    toss_row = _row("PPO (torch)", "leap_inhand_toss")

    assert toss_row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert toss_row.cells["motrix"].level == EvidenceLevel.TESTED
