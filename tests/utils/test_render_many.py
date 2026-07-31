"""Tests for MuJoCo GL backend resolution in unilab.visualization.render_many."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="GitHub Actions runners do not provide stable EGL/GLFW rendering backends.",
)


def _reload_render_many(monkeypatch):
    monkeypatch.setitem(sys.modules, "mujoco", types.SimpleNamespace())
    sys.modules.pop("unilab.visualization.render_many", None)
    return importlib.import_module("unilab.visualization.render_many")


def test_resolve_gl_backend_uses_egl_when_probe_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)

    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "_egl_runtime_usable", lambda: True)

    assert render_many._resolve_gl_backend() == "egl"


def test_resolve_gl_backend_uses_osmesa_when_headless_and_egl_unavailable(monkeypatch) -> None:
    # Headless host (no DISPLAY): glfw cannot work, so software rendering wins.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)

    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "_egl_runtime_usable", lambda: False)

    assert render_many._resolve_gl_backend() == "osmesa"


def test_resolve_gl_backend_uses_glfw_when_display_present_and_egl_unavailable(monkeypatch) -> None:
    # A display is available: glfw can create an off-screen context.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")

    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "_egl_runtime_usable", lambda: False)

    assert render_many._resolve_gl_backend() == "glfw"


def test_resolve_gl_backend_preserves_explicit_safe_value(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("MUJOCO_GL", "osmesa")

    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "_egl_runtime_usable", lambda: False)

    assert render_many._resolve_gl_backend() == "osmesa"


def test_resolve_gl_backend_uses_glfw_on_windows_without_display(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)

    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "_egl_runtime_usable", lambda: False)

    assert render_many._resolve_gl_backend() == "glfw"


def test_resolve_gl_backend_rejects_linux_only_backend_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("MUJOCO_GL", "osmesa")

    render_many = _reload_render_many(monkeypatch)

    assert render_many._resolve_gl_backend() == "glfw"


def test_egl_runtime_usable_sets_default_device_id(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)

    def _fake_run(cmd, env, check, stdout, stderr, timeout):
        assert cmd[0] == sys.executable
        assert env["MUJOCO_GL"] == "egl"
        assert env["MUJOCO_EGL_DEVICE_ID"] == "0"
        assert check is True
        assert stdout is subprocess.DEVNULL
        assert stderr is subprocess.DEVNULL
        assert timeout == 10
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(render_many.subprocess, "run", _fake_run)

    assert render_many._egl_runtime_usable() is True
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "0"


def test_egl_runtime_usable_returns_false_on_probe_failure(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)

    def _fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(render_many.subprocess, "run", _fake_run)

    assert render_many._egl_runtime_usable() is False


def _reload_render_many_with_geom_enums(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        types.SimpleNamespace(
            mjtGeom=types.SimpleNamespace(mjGEOM_PLANE=0, mjGEOM_HFIELD=1, mjGEOM_BOX=6),
        ),
    )
    sys.modules.pop("unilab.visualization.render_many", None)
    return importlib.import_module("unilab.visualization.render_many")


def test_replicable_terrain_geom_indices_selects_worldbody_box(monkeypatch) -> None:
    # The x2 wall-flip render twin declares the wall as a group-0 worldbody box
    # geom precisely so this selector picks it up and the grid renderer
    # replicates one wall per env cell. Lock that contract in.
    render_many = _reload_render_many_with_geom_enums(monkeypatch)

    model = types.SimpleNamespace(
        ngeom=4,
        # 0: floor plane (worldbody)  1: robot geom (body 5)
        # 2: wall box (worldbody)     3: group-2 worldbody box (non-default group)
        geom_group=np.array([0, 0, 0, 2], dtype=np.int32),
        geom_bodyid=np.array([0, 5, 0, 0], dtype=np.int32),
        geom_type=np.array([0, 6, 6, 6], dtype=np.int32),
    )

    indices = render_many._replicable_terrain_geom_indices(model)

    # Only the worldbody box wall (geom 2) is replicable: the plane is skipped,
    # the body-attached robot geom is skipped, and the non-group-0 geom is skipped.
    assert indices.tolist() == [2]


def test_offset_freejoint_object_qpos_handles_arbitrary_object_body(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)

    model = types.SimpleNamespace(
        nbody=4,
        body_jntadr=np.array([-1, 0, 1, -1], dtype=np.int32),
        body_jntnum=np.array([0, 1, 1, 0], dtype=np.int32),
        jnt_type=np.array([0, 0], dtype=np.int32),
        jnt_qposadr=np.array([0, 7], dtype=np.int32),
    )
    data = types.SimpleNamespace(qpos=np.zeros((14,), dtype=np.float32))

    shifted = render_many._offset_freejoint_object_qpos(
        model, data, np.array([1.5, -2.0], dtype=np.float32)
    )

    assert shifted == {2}
    assert data.qpos[0] == pytest.approx(0.0)
    assert data.qpos[1] == pytest.approx(0.0)
    assert data.qpos[7] == pytest.approx(1.5)
    assert data.qpos[8] == pytest.approx(-2.0)


def test_render_backend_usable_reflects_resolved_backend(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)

    seen: dict[str, str] = {}

    def _fake_probe(backend: str) -> bool:
        seen["backend"] = backend
        return backend == "egl"

    monkeypatch.setattr(render_many, "_gl_backend_runtime_usable", _fake_probe)

    monkeypatch.setenv("MUJOCO_GL", "egl")
    assert render_many.render_backend_usable() is True
    assert seen["backend"] == "egl"

    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    assert render_many.render_backend_usable() is False


def test_render_states_get_frames_skips_when_backend_unusable(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "render_backend_usable", lambda: False)

    frames = render_many.render_states_get_frames(
        [np.zeros((1, 8), dtype=np.float32)],
        "/no/such/model.xml",
        num_processes=4,
    )

    assert frames == []


def test_render_states_get_frames_tracking_skips_when_backend_unusable(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "render_backend_usable", lambda: False)

    frames = render_many.render_states_get_frames_tracking(
        [np.zeros((1, 8), dtype=np.float32)],
        "/no/such/model.xml",
    )

    assert frames == []


def test_render_process_count_defaults_to_one_on_windows(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many.sys, "platform", "win32")
    monkeypatch.delenv("UNILAB_RENDER_PROCESSES", raising=False)

    assert render_many._resolve_render_process_count(None, frame_count=200) == 1
    assert render_many._resolve_render_process_count(8, frame_count=200) == 1


def test_render_process_count_defaults_to_eight_on_linux(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many.sys, "platform", "linux")
    monkeypatch.delenv("UNILAB_RENDER_PROCESSES", raising=False)

    assert render_many._resolve_render_process_count(None, frame_count=200) == 8


def test_render_process_override_takes_precedence(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many.sys, "platform", "win32")
    monkeypatch.setenv("UNILAB_RENDER_PROCESSES", "2")

    assert render_many._resolve_render_process_count(8, frame_count=200) == 2


@pytest.mark.parametrize("value", ["0", "-1", "abc", "1.5"])
def test_invalid_render_process_override_uses_safe_default(
    monkeypatch, capsys, value: str
) -> None:
    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many.sys, "platform", "win32")
    monkeypatch.setenv("UNILAB_RENDER_PROCESSES", value)

    assert render_many._resolve_render_process_count(8, frame_count=200) == 1
    assert "Ignoring invalid UNILAB_RENDER_PROCESSES" in capsys.readouterr().err


def test_invalid_render_process_override_ignores_requested_count_on_linux(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many.sys, "platform", "linux")
    monkeypatch.setenv("UNILAB_RENDER_PROCESSES", "invalid")

    assert render_many._resolve_render_process_count(2, frame_count=200) == 8


def test_render_process_count_is_bounded_by_frame_count(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many.sys, "platform", "linux")
    monkeypatch.setenv("UNILAB_RENDER_PROCESSES", "8")

    assert render_many._resolve_render_process_count(None, frame_count=3) == 3
    assert render_many._resolve_render_process_count(None, frame_count=0) == 1


def test_init_worker_uses_requested_framebuffer_size(monkeypatch) -> None:
    render_many = _reload_render_many_with_geom_enums(monkeypatch)
    model = types.SimpleNamespace(
        vis=types.SimpleNamespace(global_=types.SimpleNamespace(offwidth=0, offheight=0)),
        ngeom=0,
    )
    renderer_calls: list[tuple[int, int]] = []

    monkeypatch.setattr(
        render_many.mujoco,
        "MjModel",
        types.SimpleNamespace(from_xml_path=lambda path: model),
        raising=False,
    )
    monkeypatch.setattr(render_many.mujoco, "MjData", lambda loaded: object(), raising=False)

    class FakeRenderer:
        def __init__(self, loaded, *, height, width):
            renderer_calls.append((width, height))

        def close(self):
            pass

    monkeypatch.setattr(render_many.mujoco, "Renderer", FakeRenderer, raising=False)

    render_many.init_worker("model.xml", (1280, 720))
    try:
        assert model.vis.global_.offwidth == 1280
        assert model.vis.global_.offheight == 720
        assert renderer_calls == [(1280, 720)]
    finally:
        render_many._close_worker()


def test_close_worker_is_idempotent(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    close_calls: list[bool] = []
    render_many._worker_ctx.update(
        {
            "renderer": types.SimpleNamespace(close=lambda: close_calls.append(True)),
            "models": [object()],
            "data_list": [object()],
            "terrain_geom_indices": [object()],
        }
    )

    render_many._close_worker()
    render_many._close_worker()

    assert close_calls == [True]
    assert render_many._worker_ctx == {}


def test_partial_worker_init_failure_clears_context(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    models = [
        types.SimpleNamespace(vis=types.SimpleNamespace(global_=types.SimpleNamespace())),
        types.SimpleNamespace(vis=types.SimpleNamespace(global_=types.SimpleNamespace())),
    ]
    loaded = iter(models)
    data_calls = 0

    monkeypatch.setattr(
        render_many.mujoco,
        "MjModel",
        types.SimpleNamespace(from_xml_path=lambda path: next(loaded)),
        raising=False,
    )

    def fake_data(model):
        nonlocal data_calls
        data_calls += 1
        if data_calls == 2:
            raise MemoryError("simulated allocation failure")
        return object()

    monkeypatch.setattr(render_many.mujoco, "MjData", fake_data, raising=False)

    with pytest.raises(MemoryError, match="simulated allocation failure"):
        render_many.init_worker(["first.xml", "second.xml"], (1280, 720))

    assert render_many._worker_ctx == {}


def test_num_processes_one_does_not_create_process_pool(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "render_backend_usable", lambda: True)
    monkeypatch.setattr(
        render_many,
        "_try_render_tasks_serial",
        lambda model_path, shape, tasks, render_job: ["serial"],
    )

    class ForbiddenPool:
        def __init__(self, *args, **kwargs):
            raise AssertionError("ProcessPoolExecutor must not be created")

    monkeypatch.setattr(render_many, "ProcessPoolExecutor", ForbiddenPool)

    frames = render_many.render_states_get_frames(
        [np.zeros((1, 8), dtype=np.float32)],
        "model.xml",
        num_processes=1,
    )

    assert frames == ["serial"]


def test_windows_default_uses_serial_path(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many.sys, "platform", "win32")
    monkeypatch.delenv("UNILAB_RENDER_PROCESSES", raising=False)
    monkeypatch.setattr(render_many, "render_backend_usable", lambda: True)
    serial_calls: list[int] = []
    monkeypatch.setattr(
        render_many,
        "_try_render_tasks_serial",
        lambda model_path, shape, tasks, render_job: serial_calls.append(len(tasks))
        or ["serial"],
    )
    monkeypatch.setattr(
        render_many,
        "_render_tasks_parallel",
        lambda *args, **kwargs: pytest.fail("parallel path must not run on Windows by default"),
    )

    frames = render_many.render_states_get_frames(
        [np.zeros((1, 8), dtype=np.float32)],
        "model.xml",
        num_processes=8,
    )

    assert frames == ["serial"]
    assert serial_calls == [1]


def test_parallel_failure_falls_back_to_serial_once(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many.sys, "platform", "linux")
    monkeypatch.delenv("UNILAB_RENDER_PROCESSES", raising=False)
    monkeypatch.setattr(render_many, "render_backend_usable", lambda: True)
    monkeypatch.setattr(
        render_many,
        "_render_tasks_parallel",
        lambda *args, **kwargs: (_ for _ in ()).throw(render_many.BrokenExecutor("boom")),
    )
    serial_calls: list[int] = []
    monkeypatch.setattr(
        render_many,
        "_try_render_tasks_serial",
        lambda model_path, shape, tasks, render_job: serial_calls.append(len(tasks))
        or ["recovered"],
    )

    states = [np.zeros((1, 8), dtype=np.float32) for _ in range(2)]
    frames = render_many.render_states_get_frames(states, "model.xml", num_processes=2)

    assert frames == ["recovered"]
    assert serial_calls == [2]


def test_parallel_and_serial_failure_returns_empty_list(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many.sys, "platform", "linux")
    monkeypatch.delenv("UNILAB_RENDER_PROCESSES", raising=False)
    monkeypatch.setattr(render_many, "render_backend_usable", lambda: True)
    monkeypatch.setattr(
        render_many,
        "_render_tasks_parallel",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("parallel failed")),
    )
    serial_calls: list[bool] = []
    monkeypatch.setattr(
        render_many,
        "_try_render_tasks_serial",
        lambda *args, **kwargs: serial_calls.append(True) or [],
    )

    states = [np.zeros((1, 8), dtype=np.float32) for _ in range(2)]
    frames = render_many.render_states_get_frames(states, "model.xml", num_processes=2)

    assert frames == []
    assert serial_calls == [True]


def test_render_states_get_frames_fails_fast_on_worker_init_error(monkeypatch) -> None:
    """A failing pool initializer must NOT respawn workers forever (issue #605).

    ProcessPoolExecutor raises BrokenProcessPool quickly instead of hanging, and
    render_states_get_frames degrades to an empty result + warning.
    """
    # Skip the EGL probe in spawned workers (they inherit MUJOCO_GL via os.environ).
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    # Windows defaults to serial; explicitly opt in so this legacy regression
    # test still exercises ProcessPoolExecutor initializer failure.
    monkeypatch.setenv("UNILAB_RENDER_PROCESSES", "2")
    render_many = _reload_render_many(monkeypatch)
    # Bypass the parent pre-flight so we exercise the pool's fail-fast path.
    monkeypatch.setattr(render_many, "render_backend_usable", lambda: True)

    frames = render_many.render_states_get_frames(
        [np.zeros((1, 8), dtype=np.float32)],
        "/nonexistent/model/path.xml",  # init_worker raises while loading this
        num_processes=2,
    )

    assert frames == []
