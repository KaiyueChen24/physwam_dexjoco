"""Rendering selection reaches the subprocess and remains part of result provenance."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_group as evaluate
from summarize_results import summarize


@pytest.fixture
def evaluation(monkeypatch, tmp_path):
    import websockets.sync.client
    from openpi_client import msgpack_numpy
    config = tmp_path / "environment.yaml"
    config.write_text("env_name: water_plant\nrobot_type: single_arm\n")
    task = {"name": "water_plant", "eval_config": str(config),
            "cameras": {"base": "front", "wrist": "wrist"}, "prompt": "water the plant"}
    run = {"run_dir": str(tmp_path), "run_name": "test",
           "group": {"name": "single_arm", "robot_type": "single_arm", "tasks": [task]}}
    metadata = {"group": "single_arm", "run_name": "test", "action_horizon": 30,
                "checkpoint": "checkpoint", "tasks": ["water_plant"]}
    monkeypatch.setattr(evaluate, "read_run", lambda _: run)
    monkeypatch.setattr(evaluate.shutil, "which", lambda _: "/fixture/dexjoco-openpi-eval")
    socket = SimpleNamespace(recv=lambda **kw: msgpack_numpy.packb(metadata))
    monkeypatch.setattr(websockets.sync.client, "connect", lambda *a, **kw: nullcontext(socket))
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            calls.append(kwargs["env"])
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            (output / "success_rate_1_2.txt").touch()
        def wait(self, **kwargs):
            return 0
    monkeypatch.setattr(evaluate.subprocess, "Popen", Process)
    base_args = ["evaluate_group.py", "--run-dir", str(tmp_path), "--seeds", "0", "--episodes", "2",
                 "--evaluation-name", "backend_test"]
    def execute(backend):
        monkeypatch.setattr(sys, "argv", [*base_args, "--gl-backend", backend])
        evaluate.main()
    return execute, tmp_path / "evaluation/backend_test", calls


@pytest.mark.parametrize("backend", ["egl", "osmesa"])
def test_backend_sets_both_child_environment_variables_and_is_saved(evaluation, backend):
    execute, directory, calls = evaluation
    execute(backend)
    assert calls[0]["MUJOCO_GL"] == backend
    assert calls[0]["PYOPENGL_PLATFORM"] == backend
    assert calls[0]["CUDA_VISIBLE_DEVICES"] == "3"  # Simulator choice must not alter inference GPU selection.
    plan = json.loads((directory / "plan.json").read_text())
    result = json.loads((directory / "water_plant_seed0/result.json").read_text())
    assert plan["gl_backend"] == result["gl_backend"] == backend
    summary = summarize(directory)
    assert summary["gl_backend"] == backend
    assert summary["complete"]
    assert summary["task_mean_success_rate"] == 0.5


def test_backend_change_cannot_reuse_an_evaluation_name(evaluation):
    execute, directory, calls = evaluation
    execute("egl")
    with pytest.raises(ValueError, match="rendering backend changed"):
        execute("osmesa")
    assert len(calls) == 1


def test_summary_rejects_mixed_backend_results(evaluation):
    execute, directory, _ = evaluation
    execute("osmesa")
    path = directory / "water_plant_seed0/result.json"
    result = json.loads(path.read_text())
    result["gl_backend"] = "egl"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="rendering backend does not match"):
        summarize(directory)


def test_legacy_egl_plans_still_resume(evaluation):
    execute, directory, calls = evaluation
    execute("egl")
    for path in [directory / "plan.json", directory / "water_plant_seed0/result.json"]:
        data = json.loads(path.read_text())
        del data["gl_backend"]
        path.write_text(json.dumps(data))
    execute("egl")
    assert len(calls) == 1  # Legacy completed jobs are skipped, not rerun.
    assert summarize(directory)["gl_backend"] == "egl"
