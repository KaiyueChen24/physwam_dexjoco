"""GDB must debug the saved evaluation child without changing original results."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/debug_fastwam_egl.py"
if not SCRIPT.exists():
    SCRIPT = Path(__file__).with_name("debug_fastwam_egl.py")
spec = importlib.util.spec_from_file_location("debug_fastwam_egl", SCRIPT)
debug = importlib.util.module_from_spec(spec)
spec.loader.exec_module(debug)


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(debug, "ROOT", tmp_path)
    (tmp_path / "environment.yaml").write_text("env_name: water_plant\n")
    (tmp_path / "server.json").write_text("{}")
    path = tmp_path / "result.json"
    command = [sys.executable, "-m", "dexjoco_fastwam_client.evaluate",
               "--config", "environment.yaml", "--metadata", "server.json",
               "--output", "original_videos", "--episodes", "2", "--seed", "0",
               "--host", "127.0.0.1", "--port", "8002", "--no-video"]
    path.write_text(json.dumps({"status": "failed", "command": command}))
    return path, command


def test_debugger_targets_child_and_preserves_policy_settings_and_results(recorded, monkeypatch):
    path, original_command = recorded
    original = path.read_bytes()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    destination = path.parent / "new_debug"
    command, env = debug.build_command(path, destination, 3, "/usr/bin/gdb")
    child = command[command.index("--args") + 1:]
    assert child[:3] == original_command[:3]
    assert child[child.index("--port") + 1] == "8002"
    assert child[child.index("--episodes") + 1] == "2"
    assert "--no-video" in child
    assert child[child.index("--output") + 1] == str(destination / "metrics")
    assert env["MUJOCO_GL"] == env["PYOPENGL_PLATFORM"] == "egl"
    assert env["MUJOCO_EGL_DEVICE_ID"] == env["CUDA_VISIBLE_DEVICES"] == "3"
    assert "thread apply all bt 30" in command and "info sharedlibrary" in command
    assert not destination.exists() and path.read_bytes() == original


def test_dry_run_needs_no_gdb_and_creates_no_files(recorded, monkeypatch, capsys):
    path, _ = recorded
    monkeypatch.setattr(debug.shutil, "which", lambda _: None)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--result", str(path), "--egl-device-id", "3", "--dry-run"])
    debug.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["egl_device_id"] == 3
    assert not Path(plan["output"]).exists()


def test_gdb_log_and_plan_are_kept_separate_from_official_results(recorded, monkeypatch):
    path, _ = recorded
    original = path.read_bytes()
    monkeypatch.setattr(debug.shutil, "which", lambda _: "/usr/bin/gdb")
    def run(command, **kwargs):
        assert "--args" in command
        kwargs["stdout"].write("Thread 1 received signal SIGABRT\n#0 abort\n#1 mjr_readPixels\n")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(debug.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--result", str(path), "--egl-device-id", "3"])
    debug.main()
    plans = list(path.parent.glob("egl_debug/*/debug_plan.json"))
    assert len(plans) == 1
    folder = plans[0].parent
    assert "SIGABRT" in (folder / "gdb.log").read_text()
    assert json.loads((folder / "debugger_exit.json").read_text()) == {"returncode": 0}
    assert path.read_bytes() == original
    assert not (folder / "result.json").exists()


def test_rejects_non_evaluator_commands(recorded):
    path, _ = recorded
    path.write_text(json.dumps({"command": [sys.executable, "-c", "arbitrary_code"]}))
    with pytest.raises(ValueError, match="evaluation child"):
        debug.build_command(path, path.parent / "debug", 0, "gdb")


def test_missing_saved_assets_fail_before_starting_debugger(recorded):
    path, _ = recorded
    (path.parent / "server.json").unlink()
    with pytest.raises(FileNotFoundError, match="saved evaluation asset"):
        debug.build_command(path, path.parent / "debug", 0, "gdb")
