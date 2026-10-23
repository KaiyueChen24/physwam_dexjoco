"""Protect the data, camera and evaluation boundary between rand_obj and rand_full."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from group_common import build_train_config, load_group, resolve, training_parser
from prepare_group_data import build_plan
import evaluate_group
from summarize_results import summarize


@pytest.mark.parametrize("group,count,action_dim", [("single_arm", 6, 32), ("dual_arm", 5, 44)])
def test_full_training_profile_uses_randomized_data_and_separate_run(group, count, action_dim, monkeypatch):
    arm = "single" if group == "single_arm" else "dual"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.chdir(ROOT)
    r = resolve(training_parser("test", f"pi05_{arm}_rand_full").parse_args([]))
    assert r["group"]["name"] == group and len(r["group"]["tasks"]) == count
    assert r["group"]["data_regime"] == "rand_full"
    assert Path(r["group"]["dataset_root"]).name == "dexjoco_lerobot_datasets_rand_full"
    assert "rand_full" in r["run_name"]
    assert r["experiment"]["finetune_mode"] == "full"
    assert r["experiment"]["validation_ratio"] == 0.1
    for task in r["group"]["tasks"]:
        assert task["cameras"]["base"] == "random_camera"
        assert Path(task["eval_config"]).parent.name == "rand_full"
    c = build_train_config(r)
    assert c.model.action_dim == action_dim
    assert c.model.paligemma_variant == "gemma_2b"
    assert c.policy_metadata["data_regime"] == "rand_full"
    assert c.data.create(c.assets_dirs, c.model).multitask["data_regime"] == "rand_full"


def test_randomized_group_does_not_mutate_existing_obj_configuration():
    original = load_group("single_arm")
    load_group("single_arm", "rand_full")
    assert load_group("single_arm") == original
    assert original["tasks"][2]["cameras"]["base"] == "ego_right"
    assert "data_regime" not in original  # Existing resolved snapshots still resume unchanged.
    with pytest.raises(ValueError, match="data_regime"):
        load_group("single_arm", "rand_ful")


@pytest.mark.parametrize("regime,prefix", [("rand_obj", "dexjoco_lerobot_datasets"), ("rand_full", "dexjoco_lerobot_datasets_rand_full")])
def test_download_selects_all_tasks_from_only_the_requested_regime(regime, prefix):
    groups, plan = build_plan("all", regime)
    assert len(groups) == 2 and len(plan["data_patterns"]) == 11
    assert all(path.startswith(prefix + "/") for path in plan["data_patterns"])
    assert set(plan["model_patterns"]) == {"pi05_base/*", "pi05_base_action_dim_44/*"}


@pytest.mark.parametrize("trained,tested", [("rand_obj", "rand_full"), ("rand_full", "rand_full"), ("rand_full", "rand_obj")])
def test_evaluation_switches_real_environment_flag_and_camera(trained, tested, tmp_path, monkeypatch):
    from openpi_client import msgpack_numpy
    import websockets.sync.client
    configs = tmp_path / "configs"
    for regime in ("rand_obj", "rand_full"):
        p = configs / regime / "water_plant.yaml"
        p.parent.mkdir(parents=True)
        p.write_text(yaml.safe_dump({"env_name": "water_plant", "robot_type": "single_arm",
                                    "camera_mapping": {"base": "random_camera" if regime == "rand_full" else "front", "wrist": "wrist"}}))
    task = {"name": "water_plant", "prompt": "water the plant", "eval_config": f"configs/{trained}/water_plant.yaml",
            "cameras": {"base": "random_camera" if trained == "rand_full" else "front", "wrist": "wrist"}}
    saved = {"run_name": "test", "run_dir": str(tmp_path / "run"), "experiment": {"data_regime": trained},
             "group": {"name": "single_arm", "robot_type": "single_arm", "data_regime": trained, "tasks": [task]}}
    meta = {"run_name": "test", "group": "single_arm", "checkpoint": "checkpoint", "action_horizon": 30, "tasks": ["water_plant"]}
    monkeypatch.setattr(evaluate_group, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_group, "repo_path", lambda p: tmp_path / p)
    monkeypatch.setattr(evaluate_group, "read_run", lambda _: saved)
    monkeypatch.setattr(evaluate_group.shutil, "which", lambda _: "/fixture/dexjoco-openpi-eval")
    socket = type("Socket", (), {"recv": lambda self, **kw: msgpack_numpy.packb(meta)})()
    monkeypatch.setattr(websockets.sync.client, "connect", lambda *a, **kw: nullcontext(socket))
    calls = []
    class Process:
        def __init__(self, command, **kw):
            calls.append(command)
            output = Path(command[command.index("--output") + 1]);output.mkdir()
            (output / "success_rate_0_2.txt").touch()
        def wait(self, **kw):
            return 0
    monkeypatch.setattr(evaluate_group.subprocess, "Popen", Process)
    args = ["evaluate_group.py", "--run-dir", "unused", "--seeds", "0", "--episodes", "2", "--evaluation-name", "check"]
    if tested != trained:
        args += ["--data-regime", tested]
    monkeypatch.setattr(sys, "argv", args)
    evaluate_group.main()
    assert ("--rand-full" in calls[0]) == (tested == "rand_full")
    config = yaml.safe_load(Path(calls[0][calls[0].index("--config") + 1]).read_text())
    assert config["camera_mapping"]["base"] == ("random_camera" if tested == "rand_full" else "front")
    result = summarize(tmp_path / "run/evaluation/check")
    assert result["complete"] and result["task_mean_success_rate"] == 0.0
    assert result["data_regime"] == tested
    p = tmp_path / "run/evaluation/check/water_plant_seed0/result.json"
    bad = json.loads(p.read_text());bad["data_regime"] = "rand_obj" if tested == "rand_full" else "rand_full";p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="randomization regime"):
        summarize(p.parent.parent)


@pytest.mark.parametrize("group,action,state", [("single_arm", 22, 23), ("dual_arm", 44, 46)])
def test_random_camera_frames_are_mapped_to_the_model_input_without_swapping_wrists(group, action, state, tmp_path, monkeypatch):
    import importlib.util
    import lerobot.datasets.lerobot_dataset as lerobot
    spec = load_group(group, "rand_full")
    spec["tasks"] = spec["tasks"][:1]
    spec["dataset_root"] = str(tmp_path)
    task = spec["tasks"][0]
    meta = tmp_path / task["dataset"] / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}")
    camera_values = {"random_camera": 9, "wrist": 2, "wrist_left": 3, "wrist_right": 7}
    class Metadata:
        def __init__(self, *a, **kw):
            self.features = {"action": {"shape": [action]}, "observation.state": {"shape": [state]},
                             **{f"observation.images.{c}": {} for c in task["cameras"].values()}}
            self.fps = 30
            self.total_episodes = 100
            self.info = {"codebase_version": "v3.0"}
    class Dataset:
        def __init__(self, *a, **kw):
            self.hf_dataset = [{"action": np.zeros(action), "observation.state": np.zeros(state)}]
        def __len__(self):
            return 1
        def __getitem__(self, _):
            return {"action": np.zeros((30, action)), "observation.state": np.zeros(state),
                    **{f"observation.images.{c}": np.full((3, 224, 224), camera_values[c], dtype=np.uint8)
                       for c in task["cameras"].values()}}
    monkeypatch.setattr(lerobot, "LeRobotDatasetMetadata", Metadata)
    monkeypatch.setattr(lerobot, "LeRobotDataset", Dataset)
    module_spec = importlib.util.spec_from_file_location("random_camera_dataset", ROOT / "openpi/src/openpi/training/dexjoco_multitask.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    sample = module.MultiTaskDataset(spec, 30)[0]
    for key, destination in module.canonical_cameras(spec).items():
        assert sample[f"observation.images.{destination}"].mean() == camera_values[task["cameras"][key]]
