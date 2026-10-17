"""Rendering selection reaches the subprocess and remains part of result provenance."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT.parent / "fastwam"),
                str(ROOT.parent / "fastwam/src")]
import evaluate_group as evaluate
from summarize_results import summarize


@pytest.fixture(params=["pi05", "fastwam"])
def evaluation(monkeypatch, tmp_path, request):
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
    evaluator = evaluate
    if request.param == "fastwam":
        import evaluate_fastwam_group as evaluator
        from experiments.dexjoco import config as fastwam_config
        task["environment"] = {"env_name": "water_plant", "robot_type": "single_arm"}
        config = {"framework": {"group": run["group"], "run_name": "test",
                                "experiment": {"execute_steps": 8, "action_horizon": 32}}}
        stats = tmp_path / "dataset_stats.json"
        stats.write_text("{}")
        metadata.update(model="fastwam", action_horizon=32,
                        config_signature=fastwam_config.signature(config),
                        stats_sha256=hashlib.sha256(stats.read_bytes()).hexdigest())
        monkeypatch.setattr(fastwam_config, "read_run", lambda _: (tmp_path, config))
    else:
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
            execute.commands.append(command)
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            if execute.exit_code == 0:
                (output / "success_rate_1_2.txt").touch()
            else:
                (output / "episodes.jsonl").write_text(
                    json.dumps({"episode_id": 0, "steps": 301, "success": True}) + "\n")
        def wait(self, **kwargs):
            return execute.exit_code
    monkeypatch.setattr(evaluator.subprocess, "Popen", Process)
    base_args = [evaluator.__file__, "--run-dir", str(tmp_path), "--seeds", "0", "--episodes", "2",
                 "--evaluation-name", "backend_test"]
    def execute(backend, *, no_video=False):
        argv = [*base_args, "--gl-backend", backend]
        if no_video:
            argv.append("--no-video")
        monkeypatch.setattr(sys, "argv", argv)
        evaluator.main()
    execute.exit_code = 0
    execute.commands = []
    execute.kind = request.param
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
        data.pop("record_video", None)
        path.write_text(json.dumps(data))
    execute("egl")
    assert len(calls) == 1  # Legacy completed jobs are skipped, not rerun.
    assert summarize(directory)["gl_backend"] == "egl"


def test_native_abort_after_one_success_is_incomplete_not_a_success_rate(evaluation):
    execute, directory, _ = evaluation
    execute.exit_code = -6
    with pytest.raises(SystemExit, match="failed"):
        execute("egl")
    result = json.loads((directory / "water_plant_seed0/result.json").read_text())
    assert result["status"] == "failed"
    assert "SIGABRT" in result["error"]
    partial = list((directory / "water_plant_seed0").glob("*/videos/episodes.jsonl"))
    assert len(partial) == 1 and json.loads(partial[0].read_text())["success"]
    summary = summarize(directory)
    assert not summary["complete"]
    assert summary["task_mean_success_rate"] is None


def test_fastwam_can_disable_video_without_changing_rendering_or_counts(evaluation):
    execute, directory, calls = evaluation
    if execute.kind != "fastwam":
        pytest.skip("FastWAM recording option")
    execute("egl", no_video=True)
    assert calls[0]["MUJOCO_GL"] == calls[0]["PYOPENGL_PLATFORM"] == "egl"
    assert calls[0]["CUDA_VISIBLE_DEVICES"] == "3"
    assert "--no-video" in execute.commands[0]
    plan = json.loads((directory / "plan.json").read_text())
    result = json.loads((directory / "water_plant_seed0/result.json").read_text())
    assert plan["record_video"] is result["record_video"] is False
    summary = summarize(directory)
    assert summary["record_video"] is False
    assert summary["complete"] and summary["task_mean_success_rate"] == 0.5
    assert result["episodes"] == 2


def test_fastwam_video_setting_cannot_change_under_same_evaluation_name(evaluation):
    execute, _, calls = evaluation
    if execute.kind != "fastwam":
        pytest.skip("FastWAM recording option")
    execute("egl")
    assert "--no-video" not in execute.commands[0]
    with pytest.raises(ValueError, match="changed"):
        execute("egl", no_video=True)
    assert len(calls) == 1


def test_fastwam_summary_rejects_mixed_recording_settings(evaluation):
    execute, directory, _ = evaluation
    if execute.kind != "fastwam":
        pytest.skip("FastWAM recording option")
    execute("egl", no_video=True)
    path = directory / "water_plant_seed0/result.json"
    data = json.loads(path.read_text())
    data["record_video"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="video recording setting does not match"):
        summarize(directory)
