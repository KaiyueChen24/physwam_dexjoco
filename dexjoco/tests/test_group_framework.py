"""Behavioral checks for mixing, data contracts, and incomplete evaluation handling."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
# Load the dataset module without importing upstream config.py or creating a GPU context.
module_spec = importlib.util.spec_from_file_location("group_dataset", ROOT / "openpi/src/openpi/training/dexjoco_multitask.py")
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
from group_common import load_group
from evaluate_group import parse_success
from summarize_results import summarize


@pytest.mark.parametrize("group,count,state,action", [("single_arm", 6, 23, 22), ("dual_arm", 5, 46, 44)])
def test_group_shapes_and_task_prompts(group, count, state, action, tmp_path, monkeypatch):
    import lerobot.datasets.lerobot_dataset as lerobot
    spec = load_group(group)
    spec["dataset_root"] = str(tmp_path)
    cameras = {camera for task in spec["tasks"] for camera in task["cameras"].values()}
    for task in spec["tasks"]:
        meta = tmp_path / task["dataset"] / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text("{}")

    class Metadata:
        def __init__(self, *args, **kwargs):
            self.features = {"action": {"shape": [action]}, "observation.state": {"shape": [state]},
                             **{f"observation.images.{camera}": {} for camera in cameras}}
            self.fps = 30
            self.total_episodes = 100
            self.info = {"codebase_version": "v3.0"}

    class Dataset:
        def __init__(self, *args, **kwargs):
            self.hf_dataset = [{"action": np.zeros(action), "observation.state": np.zeros(state)}] * 7

        def __len__(self):
            return 7

        def __getitem__(self, index):
            return {"action": np.zeros((30, action)), "observation.state": np.zeros(state),
                    **{f"observation.images.{camera}": np.zeros((3, 224, 224)) for camera in cameras}}

    monkeypatch.setattr(lerobot, "LeRobotDatasetMetadata", Metadata)
    monkeypatch.setattr(lerobot, "LeRobotDataset", Dataset)
    dataset = module.MultiTaskDataset(spec, 30)
    assert len(dataset) == count * 7
    for task_index, task in enumerate(spec["tasks"]):
        sample = dataset[task_index * 7]
        assert sample["prompt"] == task["prompt"]
        assert sample["observation.state"].shape == (state,)
        assert sample["action"].shape == (30, action)
        for camera in module.canonical_cameras(spec).values():
            assert sample[f"observation.images.{camera}"].shape == (3, 224, 224)
        assert dataset.low_dim(task_index * 7)["actions"].shape == (action,)
    assert dataset.locate(7) == (1, 0)
    assert dataset.locate(-1) == (count - 1, 6)
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_missing_real_dataset_fails_instead_of_generating_frames(tmp_path):
    spec = {**load_group("single_arm"), "dataset_root": str(tmp_path)}
    with pytest.raises(FileNotFoundError, match="download this task first"):
        module.MultiTaskDataset(spec, 30)


def test_balanced_sampling_does_not_favor_long_tasks():
    lengths = [10, 1000, 30]
    weights = module.frame_weights(lengths)
    offsets = np.cumsum([0] + lengths)
    masses = [weights[a:b].sum() for a, b in zip(offsets[:-1], offsets[1:])]
    np.testing.assert_allclose(masses, [1 / 3] * 3)
    samples = np.random.default_rng(42).choice(sum(lengths), 30000, p=weights)
    frequencies = np.bincount(np.searchsorted(offsets[1:], samples, side="right"), minlength=3) / len(samples)
    np.testing.assert_allclose(frequencies, [1 / 3] * 3, atol=0.015)


def test_real_adapter_maps_mouse_camera_and_preserves_episode_selection(tmp_path, monkeypatch):
    import lerobot.datasets.lerobot_dataset as lerobot
    spec = load_group("single_arm")
    spec["tasks"] = [next(t for t in spec["tasks"] if t["name"] == "click_mouse")]
    spec["tasks"][0]["episodes"] = [3]
    spec["dataset_root"] = str(tmp_path)
    root = tmp_path / "click_mouse/meta"
    root.mkdir(parents=True)
    (root / "info.json").write_text("{}")
    calls = []

    class Metadata:
        def __init__(self, *args, **kwargs):
            self.features = {"action": {"shape": [22]}, "observation.state": {"shape": [23]},
                             "observation.images.ego_right": {}, "observation.images.wrist": {}}
            self.fps = 30
            self.total_episodes = 100
            self.info = {"codebase_version": "v3.0"}

    class Dataset:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)
            self.hf_dataset = [{"action": np.zeros(22), "observation.state": np.zeros(23)}] * 2

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {"action": np.zeros((30, 22)), "observation.state": np.zeros(23),
                    "task": "instruction from dataset", "observation.images.ego_right": np.ones((3, 224, 224)),
                    "observation.images.wrist": np.zeros((3, 224, 224))}

    monkeypatch.setattr(lerobot, "LeRobotDatasetMetadata", Metadata)
    monkeypatch.setattr(lerobot, "LeRobotDataset", Dataset)
    dataset = module.MultiTaskDataset(spec, 30)
    sample = dataset[0]
    assert sample["observation.images.front"].mean() == 1
    assert sample["prompt"] == "instruction from dataset"
    assert calls[0]["episodes"] == [3]
    assert calls[0]["delta_timestamps"]["action"][-1] == 29 / 30
    assert dataset.low_dim(0)["actions"].shape == (22,)


def test_success_marker_rejects_partial_episodes(tmp_path):
    (tmp_path / "success_rate_2_5.txt").touch()
    assert parse_success(tmp_path, 5) == (2, 5)
    with pytest.raises(ValueError):
        parse_success(tmp_path, 50)


def test_summary_keeps_missing_tasks_visible(tmp_path):
    plan = {"group": "single_arm", "run_name": "test", "server": {"checkpoint": "checkpoint",
            "tasks": ["a", "b"], "data_source": "lerobot"},
            "jobs": [{"task": t, "seed": 0, "episodes": 5} for t in ("a", "b")]}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    job = tmp_path / "a_seed0"
    job.mkdir()
    (job / "result.json").write_text(json.dumps({"status": "complete", "checkpoint": "checkpoint",
                                                 "successes": 4, "episodes": 5}))
    result = summarize(tmp_path)
    assert not result["complete"]
    assert result["task_mean_success_rate"] is None
    assert result["tasks"][1]["missing_or_failed_seeds"] == [0]
    job = tmp_path / "b_seed0"
    job.mkdir()
    (job / "result.json").write_text(json.dumps({"status": "complete", "checkpoint": "checkpoint",
                                                 "successes": 2, "episodes": 5}))
    result = summarize(tmp_path)
    assert result["complete"]
    assert result["task_mean_success_rate"] == pytest.approx(0.6)


def test_saved_run_paths_relocate_without_changing_external_data(tmp_path, monkeypatch):
    import yaml
    import group_common
    source = Path("/old/checkout")
    destination = tmp_path / "checkout"
    run = destination / "runs/test"
    run.mkdir(parents=True)
    saved = {"schema_version": 1, "run_dir": str(source / "runs/test"),
             "group": {"dataset_root": str(source / "datasets"), "synthetic": False},
             "experiment": {"initialization_params": str(source / "checkpoints/base/params"),
                            "synthetic": False, "tiny": False}}
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(saved))
    monkeypatch.setattr(group_common, "ROOT", destination)
    resolved = group_common.read_run(run)
    assert "synthetic" not in resolved["group"]
    assert "tiny" not in resolved["experiment"]
    assert resolved["group"]["dataset_root"] == str(destination / "datasets")
    assert resolved["experiment"]["initialization_params"] == str(destination / "checkpoints/base/params")
    saved["group"]["dataset_root"] = "/scratch/training_data"
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(saved))
    assert group_common.read_run(run)["group"]["dataset_root"] == "/scratch/training_data"


@pytest.mark.parametrize("legacy_mode", ["synthetic", "tiny"])
def test_legacy_framework_runs_cannot_be_loaded_as_real_policies(tmp_path, legacy_mode):
    import yaml
    from group_common import read_run
    saved = {"schema_version": 1, "run_dir": str(tmp_path), "group": {},
             "experiment": {legacy_mode: True}}
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump(saved))
    with pytest.raises(ValueError, match="no longer supported"):
        read_run(tmp_path)
