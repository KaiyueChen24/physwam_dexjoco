"""Real FastWAM CLI must select randomized demonstrations and matching cameras."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "fastwam"))
from experiments.dexjoco.config import build_config, select_group


def dry_run(arm, *options):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/train_fastwam_group.py"),
         "--experiment", f"configs/experiments/fastwam_{arm}_rand_full.yaml",
         "--run-name", "rand_full_contract", "--dry-run", *options],
        capture_output=True, text=True, check=True,
    )
    return yaml.safe_load(result.stdout)


@pytest.mark.parametrize("arm,count,state_dim,action_dim", [
    ("single", 6, 23, 22), ("dual", 5, 46, 44),
])
def test_training_cli_selects_full_data_cameras_and_saved_environment(arm, count, state_dim, action_dim):
    config = dry_run(arm)
    group = config["framework"]["group"]
    assert group["data_regime"] == "rand_full"
    expected_root = ROOT / "datasets/dexjoco_lerobot_datasets_rand_full"
    assert Path(group["dataset_root"]) == expected_root
    assert len(group["tasks"]) == count
    for task in group["tasks"]:
        assert Path(task["dataset_path"]) == expected_root / task["dataset"]
        assert task["cameras"]["base"] == "random_camera"
        assert task["environment"]["camera_mapping"] == task["cameras"]
        assert Path(task["eval_config"]).parent.name == "rand_full"
        assert all(value == key for key, value in task["cameras"].items() if key != "base")
    assert config["data"]["train"]["tasks"] == group["tasks"]
    assert config["model"]["proprio_dim"] == state_dim
    assert config["model"]["action_dit_config"]["action_dim"] == action_dim
    assert config["framework"]["launch"]["num_processes"] == 8
    assert config["framework"]["launch"]["effective_global_batch_size"] == config["batch_size"] * 8
    assert config["wandb"]["name"] == f"fastwam_{arm}_rand_full"


@pytest.mark.parametrize("mode", ["joint", "per_task"])
def test_cli_dataset_override_keeps_full_randomization_and_task_selection(tmp_path, mode):
    base = tmp_path / "randomized_data"
    config = dry_run("single", "--dataset-root", str(base), "--tasks", "click_mouse",
                     "--gpus", "2,3", "--mode", mode)
    group = config["framework"]["group"]
    assert Path(group["dataset_root"]) == base
    assert len(group["tasks"]) == 1
    assert group["tasks"][0]["name"] == "click_mouse"
    assert group["tasks"][0]["cameras"]["base"] == "random_camera"
    assert config["framework"]["launch"]["num_processes"] == 2


def test_yaml_dataset_root_and_cli_precedence(tmp_path):
    profile = yaml.safe_load((ROOT / "configs/experiments/fastwam_single_rand_full.yaml").read_text())
    profile["dataset_root"] = str(tmp_path / "yaml_data")
    config = build_config(ROOT, "single_arm", profile, "yaml_full")
    assert Path(config["framework"]["group"]["dataset_root"]) == tmp_path / "yaml_data"
    config = build_config(ROOT, "single_arm", profile, "cli_full", dataset_root=str(tmp_path / "cli_data"))
    assert Path(config["framework"]["group"]["dataset_root"]) == tmp_path / "cli_data"


def test_rand_obj_defaults_are_unchanged_and_not_mutated():
    original = select_group(ROOT, "single_arm")
    select_group(ROOT, "single_arm", data_regime="rand_full")
    assert select_group(ROOT, "single_arm") == original
    assert "data_regime" not in original  # Preserve old run configuration signatures.
    assert Path(original["dataset_root"]).name == "dexjoco_lerobot_datasets"
    assert original["tasks"][2]["cameras"]["base"] == "ego_right"
    assert all(Path(t["eval_config"]).parent.name == "rand_obj" for t in original["tasks"])
    with pytest.raises(ValueError, match="data_regime"):
        select_group(ROOT, "single_arm", data_regime="rand_ful")
