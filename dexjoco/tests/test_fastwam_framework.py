"""CPU contract tests; no model weights, simulator, downloads or GPU required."""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(os.environ.get("DEXJOCO_TEST_ROOT", Path(__file__).resolve().parents[1]))
FASTWAM_ROOT = Path(os.environ.get("FASTWAM_TEST_ROOT", ROOT.parent / "fastwam"))
sys.path[:0] = [str(FASTWAM_ROOT), str(FASTWAM_ROOT / "src"),
                str(Path(__file__).resolve().parents[1] / "dexjoco"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from experiments.dexjoco.config import build_config, select_group
from experiments.dexjoco.data import compute_stats, inspect_task
from experiments.dexjoco.transforms import MinMaxNormalizer, pack_images
from dexjoco_fastwam_client.evaluate import execute_episode, validate_metadata


@pytest.fixture
def experiment():
    path = Path(__file__).resolve().parents[1] / "configs/experiments/fastwam_single.yaml"
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("profile,steps,gpus,tasks,global_batch", [
    ("fastwam_single", 30000, "0,1,2,3,4,5,6,7",
     ["water_plant", "hammer_nail", "click_mouse", "pick_bucket", "pinch_tongs", "fold_glasses"], 128),
    ("fastwam_smoke", 20, "0,1,2,3", ["water_plant"], 4),
])
def test_cli_reads_profile_without_explicit_launch_flags(profile, steps, gpus, tasks, global_batch):
    script = Path(__file__).resolve().parents[1] / "scripts/train_fastwam_group.py"
    result = subprocess.run([sys.executable, str(script), "--experiment",
                             f"configs/experiments/{profile}.yaml", "--run-name", "profile_contract", "--dry-run"],
                            capture_output=True, text=True, check=True)
    config = yaml.safe_load(result.stdout)
    framework = config["framework"]
    assert framework["launch"] == {"gpus": gpus, "num_processes": len(gpus.split(",")),
                                   "effective_global_batch_size": global_batch}
    assert framework["group"]["name"] == "single_arm"
    assert [t["name"] for t in framework["group"]["tasks"]] == tasks
    assert config["max_steps"] == steps
    assert config["model"]["action_dit_config"]["action_dim"] == 22
    if profile == "fastwam_single":
        profile_config = yaml.safe_load((ROOT / "configs/experiments/fastwam_single.yaml").read_text())
        expected_name = profile_config["wandb_name"] if profile_config["wandb_name"] is not None else "profile_contract"
        assert config["wandb"]["name"] == expected_name
    if profile == "fastwam_smoke":
        assert config["save_every"] == 10
        assert framework["experiment"]["validation_max_batches"] == 2



@pytest.mark.parametrize("selection", ["explicit", "run_name", "legacy"])
def test_experiment_wandb_name_override_and_legacy_compatibility(experiment, selection):
    run_name = "wandb_name_contract"
    if selection == "explicit":
        experiment["wandb_name"] = "single_arm_joint_custom"
        expected = "single_arm_joint_custom"
    elif selection == "run_name":
        experiment["wandb_name"] = None
        expected = run_name
    else:
        experiment.pop("wandb_name", None)
        expected = yaml.safe_load((FASTWAM_ROOT / "configs/train.yaml").read_text())["wandb"]["name"]
    config = build_config(ROOT, "single_arm", experiment, run_name)
    assert config["wandb"]["name"] == expected
    assert config["framework"]["run_name"] == run_name
    assert Path(config["output_dir"]).name == run_name


@pytest.mark.parametrize("value", ["", "   ", 123, False])
def test_invalid_experiment_wandb_name_fails_before_training(experiment, value):
    experiment["wandb_name"] = value
    with pytest.raises(ValueError, match="wandb_name"):
        build_config(ROOT, "single_arm", experiment, "invalid_wandb_name")


def test_cli_flags_override_profile_gpu_task_group_and_budget():
    script = Path(__file__).resolve().parents[1] / "scripts/train_fastwam_group.py"
    result = subprocess.run([sys.executable, str(script), "--run-name", "profile_override", "--dry-run",
                             "--group", "dual_arm", "--tasks", "bimanual_assembly",
                             "--gpus", "4,5", "--batch-size", "2", "--steps", "30"],
                            capture_output=True, text=True, check=True)
    config = yaml.safe_load(result.stdout)
    assert config["framework"]["launch"] == {"gpus": "4,5", "num_processes": 2, "effective_global_batch_size": 4}
    assert config["model"]["action_dit_config"]["action_dim"] == 44
    assert config["max_steps"] == 30
    assert [t["name"] for t in config["framework"]["group"]["tasks"]] == ["bimanual_assembly"]


@pytest.mark.parametrize("gpus", ["0,0", "0,,2", "not-a-gpu"])
def test_invalid_launch_gpu_selection_fails_during_dry_run(gpus):
    script = Path(__file__).resolve().parents[1] / "scripts/train_fastwam_group.py"
    result = subprocess.run([sys.executable, str(script), "--run-name", "invalid_launch", "--dry-run", "--gpus", gpus],
                            capture_output=True, text=True)
    assert result.returncode == 2


@pytest.mark.parametrize("group,dims,count", [("single_arm", (23, 22), 6), ("dual_arm", (46, 44), 5)])
def test_group_dimensions_do_not_inherit_pi05_padding(group, dims, count, experiment):
    cfg = build_config(ROOT, group, experiment, "fastwam_test")
    assert len(cfg["framework"]["group"]["tasks"]) == count
    assert (cfg["model"]["proprio_dim"], cfg["model"]["action_dit_config"]["action_dim"]) == dims
    assert cfg["model"]["load_text_encoder"] is True
    assert cfg["eval_every"] == 0


def test_horizon_and_image_constraints(experiment):
    for patch in ({"action_horizon": 30}, {"execute_steps": 40}, {"image_size": [225, 224]}):
        with pytest.raises(ValueError):
            build_config(ROOT, "single_arm", {**experiment, **patch}, "test")
    with pytest.raises(ValueError):
        build_config(ROOT, "single_arm", experiment, "../bad")
    with pytest.raises(ValueError):
        select_group(ROOT, "single_arm", ["bimanual_assembly"])


def test_config_resolves_through_hydra_without_instantiating_model(experiment):
    from omegaconf import OmegaConf
    cfg = build_config(ROOT, "dual_arm", experiment, "test")
    resolved = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
    assert resolved["model"]["video_dit_config"]["action_dim"] == 44
    assert resolved["data"]["train"]["_recursive_"] is False


def test_minmax_and_constant_dimensions_match_upstream():
    import torch
    from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
    stats = {field: {"default": {"global_min": [2., -1.], "global_max": [2., 3.]}}
             for field in ("state", "action")}
    normalizer = MinMaxNormalizer(stats)
    official = SingleFieldLinearNormalizer({"min": torch.tensor([2., -1.]), "max": torch.tensor([2., 3.])})
    raw = np.array([[2., -1.], [2., 1.], [2., 3.]], np.float32)
    normalized = normalizer.forward("action", raw)
    np.testing.assert_allclose(normalized, official.forward(torch.from_numpy(raw)).numpy())
    np.testing.assert_allclose(normalizer.backward("action", normalized), raw)
    with pytest.raises(ValueError):
        normalizer.forward("state", [np.nan, 0.])


def make_dataset(root):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import av
    cameras = ["front", "wrist"]
    for folder in ("meta/episodes/chunk-000", "data/chunk-000"):
        (root / folder).mkdir(parents=True)
    fields = {"action": {"dtype": "float32", "shape": [22]},
              "observation.state": {"dtype": "float32", "shape": [23]}}
    for key, dtype in (("episode_index", "int64"), ("task_index", "int64"), ("timestamp", "float32")):
        fields[key] = {"dtype": dtype, "shape": [1]}
    fields.update({f"observation.images.{c}": {"dtype": "video", "shape": [16, 16, 3],
                                               "names": ["height", "width", "channels"]} for c in cameras})
    info = {"codebase_version": "v3.0", "fps": 30, "total_episodes": 2, "total_frames": 12,
            "features": fields, "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    (root / "meta/info.json").write_text(json.dumps(info))
    pd.DataFrame({"task_index": [0]}, index=["dataset instruction"]).to_parquet(root / "meta/tasks.parquet")
    episodes = []
    for ep in range(2):
        row = {"episode_index": ep, "length": 6, "data/chunk_index": 0, "data/file_index": 0}
        for camera in cameras:
            prefix = f"videos/observation.images.{camera}"
            row.update({prefix + "/chunk_index": 0, prefix + "/file_index": ep,
                        prefix + "/from_timestamp": 0.0})
            video = root / f"videos/observation.images.{camera}/chunk-000/file-{ep:03d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(video), "w") as container:
                stream = container.add_stream("libx264", rate=30)
                stream.width = stream.height = 16
                stream.pix_fmt = "yuv420p"
                for i in range(6):
                    frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), 20 + ep * 150 + i, np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        episodes.append(row)
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet")
    rows = [{"episode_index": ep, "task_index": 0, "timestamp": i / 30,
             "observation.state": [float(ep * 100 + i)] * 23,
             "action": [float(ep * 100 + i)] * 22} for ep in range(2) for i in range(6)]
    pq.write_table(pa.Table.from_pylist(rows), root / "data/chunk-000/file-000.parquet")
    return {"name": "water_plant", "dataset_path": str(root), "episodes": [0],
            "prompt": "Water the plant.", "cameras": {"base": "front", "wrist": "wrist"}}


def test_selected_episode_stats_and_real_video_sequence(tmp_path):
    from experiments.dexjoco.dataset import DexJoCoVideoDataset
    task = make_dataset(tmp_path / "dataset")
    stats = compute_stats([task], 23, 22)
    assert stats["action"]["default"]["global_max"] == [5.] * 22  # held-out episode is ~100
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(stats))
    dataset = DexJoCoVideoDataset(tasks=[task], state_dim=23, action_dim=22, action_horizon=4,
                                 video_stride=1, camera_order=["base", "wrist"], image_size=[16, 16],
                                 sampling="task_balanced", stats_path=stats_path, control_hz=30)
    end = dataset[5]  # Every future query must clamp to the final frame of episode 0.
    assert end["video"].shape == (3, 5, 16, 32)
    assert end["action"].shape == (4, 22)
    assert end["proprio"].shape == (4, 23)
    assert end["action_is_pad"].tolist() == [False, True, True, True]
    np.testing.assert_allclose(end["action"].numpy(), 1.)
    assert end["video"].max() < 0  # episode 1's bright frames were not decoded
    assert end["prompt"].endswith(task["prompt"])
    task["episodes"] = [5]
    with pytest.raises(ValueError):
        compute_stats([task], 23, 22)


def test_balanced_virtual_epoch_covers_tasks_equally():
    from experiments.dexjoco.dataset import DexJoCoVideoDataset
    ds = DexJoCoVideoDataset.__new__(DexJoCoVideoDataset)
    ds.sampling, ds.tasks, ds.lengths, ds.ends = "task_balanced", [1, 2], [2, 5], [2, 7]
    pairs = [ds.locate(i) for i in range(len(ds))]
    assert [sum(t == i for t, _ in pairs) for i in range(2)] == [5, 5]
    assert {f for t, f in pairs if t == 1} == set(range(5))


class FakeEnv:
    def reset(self):
        self.steps, self.is_done, self.is_success = 0, False, False
    def get_obs(self):
        return {"prompt": "test"}
    def step(self, action):
        self.steps += 1
        self.is_done = self.steps >= 5
        self.is_success = self.is_done


def test_closed_loop_replans_and_rejects_stale_chunks():
    requests = []
    def infer(obs):
        requests.append(obs.copy())
        return {**obs, "actions": np.zeros((4, 22))}
    env = FakeEnv()
    result = execute_episode(env, infer, task="water", episode_id=7, horizon=4, action_dim=22,
                             execute_steps=2, max_steps=20)
    assert result["success"] and result["steps"] == 5
    assert [r["step_id"] for r in requests] == [0, 2, 4]
    with pytest.raises(ValueError, match="Stale"):
        execute_episode(env, lambda obs: {**infer(obs), "episode_id": 6}, task="water", episode_id=7,
                        horizon=4, action_dim=22, execute_steps=2, max_steps=20)


def test_policy_uses_shared_images_and_returns_physical_actions(tmp_path, experiment):
    import torch
    from experiments.dexjoco.policy import DexJoCoFastWAMPolicy
    config = build_config(ROOT, "single_arm", experiment, "test", task_names=["water_plant"])
    (tmp_path / "fastwam_config.yaml").write_text(yaml.safe_dump(config))
    stats = {"state": {"default": {"global_min": [0.] * 23, "global_max": [2.] * 23}},
             "action": {"default": {"global_min": [2.] * 22, "global_max": [4.] * 22}}}
    (tmp_path / "dataset_stats.json").write_text(json.dumps(stats))
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"test checkpoint identity")
    calls = []
    class Model:
        def encode_prompt(self, prompt):
            return torch.zeros(1), torch.ones(1)
        def infer_action(self, **kwargs):
            calls.append(kwargs)
            return {"action": torch.zeros(32, 22)}
    policy = DexJoCoFastWAMPolicy(tmp_path, checkpoint, model=Model())
    images = {k: np.full((20, 30, 3), 80, np.uint8) for k in ("base", "wrist")}
    obs = {"images": images, "state": np.ones(23), "task": "water_plant",
           "prompt": config["framework"]["group"]["tasks"][0]["prompt"], "episode_id": 0, "step_id": 0}
    result = policy.infer(obs)
    np.testing.assert_allclose(result["actions"], 3.)
    np.testing.assert_allclose(calls[0]["input_image"].numpy(), pack_images(images, ["base", "wrist"], [224, 224]))
    np.testing.assert_allclose(calls[0]["proprio"].numpy(), 0.)
    assert result["episode_id"] == result["step_id"] == 0
    bad = copy.deepcopy(policy.metadata)
    bad["action_horizon"] = 30
    with pytest.raises(ValueError, match="action_horizon"):
        validate_metadata(bad, policy.metadata)
