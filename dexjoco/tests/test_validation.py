"""留出集成员、训练统计隔离、只前向 loss 与尾批次完整性测试。"""
import contextlib
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FAST = ROOT.parent / "fastwam"
sys.path[:0] = [str(ROOT / "scripts"), str(FAST), str(FAST / "src")]
from dataset_split import available_episodes, save_split, split_group, validation_settings


def make_group(tmp_path, counts=(10, 7)):
    tasks = []
    for i, count in enumerate(counts):
        root = tmp_path / f"task{i}" / "meta"
        root.mkdir(parents=True, exist_ok=True)
        (root / "info.json").write_text(json.dumps({"total_episodes": count}))
        (root / "episodes.jsonl").write_text("\n".join(json.dumps({"episode_index": j}) for j in range(count)))
        tasks.append({"name": f"task{i}", "dataset": f"task{i}", "episodes": None})
    return {"name": "single_arm", "dataset_root": str(tmp_path), "tasks": tasks, "sampling": "task_balanced"}


def test_episode_split_is_reproducible_disjoint_and_shared_across_task_subsets(tmp_path):
    group = make_group(tmp_path)
    original = copy.deepcopy(group)
    options = validation_settings({"validation_ratio": 0.2})
    train, val, manifest = split_group(group, options)
    assert group == original
    assert manifest == split_group(group, options)[2]
    assert [len(t["episodes"]) for t in val["tasks"]] == [2, 2]
    for source, a, b in zip(manifest["tasks"], train["tasks"], val["tasks"]):
        assert not set(a["episodes"]) & set(b["episodes"])
        assert set(a["episodes"]) | set(b["episodes"]) == set(source["source_episodes"])
    subset = {**group, "tasks": list(reversed(group["tasks"]))[:1]}
    assert split_group(subset, options)[2]["tasks"][0] == manifest["tasks"][1]
    save_split(tmp_path, manifest)
    save_split(tmp_path, manifest)
    with pytest.raises(ValueError, match="different"):
        save_split(tmp_path, {**manifest, "seed": 100})


def test_split_honors_explicit_candidate_episodes_and_rejects_small_or_invalid_sources(tmp_path):
    group = make_group(tmp_path, (10,))
    group["tasks"][0]["episodes"] = [1, 3, 7]
    options = validation_settings({"validation_ratio": 0.1})
    train, val, manifest = split_group(group, options)
    assert set(train["tasks"][0]["episodes"] + val["tasks"][0]["episodes"]) == {1, 3, 7}
    assert manifest["tasks"][0]["validation_count"] == 1
    group["tasks"][0]["episodes"] = [1]
    with pytest.raises(ValueError, match="at least two"):
        split_group(group, options)
    group["tasks"][0]["episodes"] = [100, 101]
    with pytest.raises(ValueError, match="not present"):
        split_group(group, options)
    assert split_group(group, validation_settings({}))[1:] == (None, None)


@pytest.mark.parametrize("change", [{"validation_ratio": -0.1}, {"validation_ratio": 1.0},
    {"validation_ratio": float('nan')}, {"validation_interval": 0}, {"validation_batch_size": 0},
    {"validation_max_batches": 0}, {"validation_seed": -1}])
def test_bad_validation_settings_fail_before_training(change):
    with pytest.raises(ValueError):
        validation_settings(change)


def test_real_v3_split_stats_and_video_reader_exclude_validation_episodes(tmp_path):
    from test_fastwam_framework import make_dataset
    from experiments.dexjoco.data import compute_stats
    from experiments.dexjoco.dataset import DexJoCoVideoDataset
    task = make_dataset(tmp_path / "data")
    task["episodes"] = None
    assert available_episodes(task["dataset_path"]) == [0, 1]
    group = {"name": "single_arm", "tasks": [task], "sampling": "task_balanced"}
    train, val, _ = split_group(group, validation_settings({"validation_ratio": 0.1}))
    assert train["tasks"][0]["episodes"] != val["tasks"][0]["episodes"]
    stats = compute_stats(train["tasks"], 23, 22)
    train_id = train["tasks"][0]["episodes"][0]
    assert stats["action"]["default"]["global_min"] == [train_id * 100.] * 22
    assert stats["action"]["default"]["global_max"] == [train_id * 100. + 5] * 22
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats))
    common = dict(state_dim=23, action_dim=22, action_horizon=4, video_stride=1,
                  camera_order=["base", "wrist"], image_size=[16, 16], sampling="frame_uniform",
                  stats_path=path, control_hz=30)
    train_ds = DexJoCoVideoDataset(tasks=train["tasks"], **common)
    val_ds = DexJoCoVideoDataset(tasks=val["tasks"], **common)
    assert set(train_ds.readers[0].episodes).isdisjoint(val_ds.readers[0].episodes)
    assert len(train_ds) == len(val_ds) == 6
    # 最后一帧的动作块仍由 reader 限制在各自 episode 内，不会跨到另一划分。
    for dataset, selected in [(train_ds, train_id), (val_ds, 1-train_id)]:
        raw = dataset.readers[0][5]
        np.testing.assert_array_equal(raw["action"].numpy(), np.full((4, 22), selected * 100. + 5))
        dataset[5]  # 同时覆盖真实视频的尾帧解码与 padding。


def test_pi05_loss_validation_counts_tail_samples_and_never_mutates_params(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT / "openpi")
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.training import dexjoco_validation as mod

    class Model(nnx.Module):
        def __init__(self):
            self.weight = nnx.Param(jnp.asarray(0.0))
        def compute_loss(self, rng, obs, actions, train=False):
            assert train is False
            return jnp.square(self.weight.value - actions).mean(axis=-1)

    class Data:
        lengths = [3, 5]
        ends = [3, 8]
        def __getitem__(self, index):
            return {"state": np.zeros(1, np.float32), "actions": np.full((2, 1), 1 if index < 3 else 3, np.float32),
                    "image": {"base_0_rgb": np.zeros((2, 2, 3), np.float32)},
                    "image_mask": {"base_0_rgb": np.asarray(True)}}
    monkeypatch.setattr(mod, "MultiTaskDataset", lambda *a: Data())
    monkeypatch.setattr(mod, "make_data_factory", lambda *a: SimpleNamespace(create=lambda *a: None))
    monkeypatch.setattr(mod.data_loader, "transform_dataset", lambda raw, cfg: raw)
    graph, params = nnx.split(Model())
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("batch",))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
    state = SimpleNamespace(model_def=graph)
    state_sharding = SimpleNamespace(params=jax.tree.map(lambda _: replicated, params))
    options = {"group": {"tasks": [{"name": "a"}, {"name": "b"}]}, "batch_size": 4,
               "seed": 42, "sampling": "task_balanced", "max_batches": None,
               "output_path": str(tmp_path / "validation_loss.jsonl")}
    config = SimpleNamespace(validation=options, model=SimpleNamespace(action_horizon=2), assets_dirs=tmp_path)
    before = [np.asarray(x).copy() for x in jax.tree.leaves(params)]
    runner = mod.ValidationRunner(config, state, state_sharding, shard)
    result = runner.evaluate(params, 2)
    assert result["task_samples"] == {"a": 3, "b": 5}
    assert result["task_losses"] == {"a": 1., "b": 9.}
    assert result["val_loss"] == pytest.approx(5.)
    assert result["full_validation"]
    assert runner.evaluate(params, 3)["val_loss"] == result["val_loss"]
    for a, b in zip(before, jax.tree.leaves(params)):
        np.testing.assert_array_equal(a, np.asarray(b))


def test_fastwam_loss_validation_preserves_params_gradients_rng_and_mixed_modes(tmp_path):
    import torch
    from experiments.dexjoco.validation import evaluate_loss

    class Options(dict):
        __getattr__ = dict.__getitem__
    class Data:
        sampling = "frame_uniform"
        lengths = [3, 5]
        ends = [3, 8]
        tasks = [{"name": "a"}, {"name": "b"}]
        def __getitem__(self, index):
            return {"action": torch.full((2, 1), 1. if index < 3 else 3.), "prompt": "Task"}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.))
            self.child = torch.nn.Dropout()
        def training_loss(self, sample):
            assert not torch.is_grad_enabled()
            assert not self.training and not self.child.training
            torch.rand(1)  # 验证产生的随机数不能影响后续训练。
            np.random.rand()
            random.random()
            loss = (self.weight - sample["action"]).square().mean()
            return loss, {"loss_action": float(loss)}
    model = Model()
    model.train()
    model.child.eval()  # 与冻结的编码器/可训练分支混合模式一致。
    model.weight.grad = torch.tensor(7.)
    accelerator = SimpleNamespace(device=torch.device("cpu"), process_index=0, num_processes=1,
        is_main_process=True, unwrap_model=lambda m: m, autocast=contextlib.nullcontext,
        reduce=lambda value, **kw: value)
    options = Options(seed=42, batch_size=4, max_batches=None, sampling="task_balanced")
    trainer = SimpleNamespace(model=model, accelerator=accelerator, cfg=SimpleNamespace(validation_loss=options),
                              train_dataset=object(), val_dataset=Data(), global_step=10, output_dir=tmp_path)
    torch_state, numpy_state, python_state = torch.random.get_rng_state(), np.random.get_state(), random.getstate()
    result = evaluate_loss(trainer)
    assert result["task_samples"] == {"a": 3, "b": 5}
    assert result["val_loss"] == pytest.approx(5.)
    assert result["full_validation"]
    assert model.weight.item() == 0 and model.weight.grad.item() == 7
    assert model.training and not model.child.training
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert all(np.array_equal(a, b) for a, b in zip(numpy_state, np.random.get_state()))
    assert python_state == random.getstate()
    options["sampling"] = "frame_uniform"
    assert evaluate_loss(trainer)["val_loss"] == pytest.approx(6.)


def test_fastwam_prepare_wires_heldout_data_and_train_only_stats(tmp_path):
    import yaml
    from test_fastwam_framework import make_dataset
    from experiments.dexjoco.config import build_config
    from train_fastwam_group import prepare
    from group_common import load_group
    task = make_dataset(tmp_path / "datasets/water_plant")
    exp = yaml.safe_load((ROOT / "configs/experiments/fastwam_single.yaml").read_text())
    def configuration():
        cfg = build_config(ROOT, "single_arm", exp, "validation_test", task_names=["water_plant"],
                           dataset_root=tmp_path / "datasets")
        cfg["output_dir"] = str(tmp_path / "run")
        return cfg
    config = configuration()
    prepare(config, ROOT, FAST)
    training = config["data"]["train"]["tasks"][0]["episodes"]
    validation = config["data"]["val"]["tasks"][0]["episodes"]
    assert set(training).isdisjoint(validation)
    assert config["validation_loss"]["enabled"]
    assert config["eval_every"] == 0  # 只计算 loss，不开启视频生成评测。
    assert config["data"]["train"]["stats_path"] == config["data"]["val"]["stats_path"]
    stats = json.loads((tmp_path / "run/dataset_stats.json").read_text())
    assert stats["provenance"]["tasks"]["water_plant"]["episodes"] == training
    assert stats["action"]["default"]["global_max"] == [training[0] * 100. + 5] * 22
    pi_group = load_group("single_arm")
    pi_group["tasks"] = pi_group["tasks"][:1]
    pi_group["dataset_root"] = str(tmp_path / "datasets")
    assert split_group(pi_group, validation_settings(exp))[2] == config["framework"]["data_split"]
    prepare(configuration(), ROOT, FAST)  # 重复 prepare 不改变划分或统计。
    changed = configuration()
    changed["framework"]["experiment"]["validation_seed"] += 1
    with pytest.raises(ValueError, match="changed"):
        prepare(changed, ROOT, FAST)


def _distributed_loss_worker(rank, folder):
    import datetime
    import torch
    from experiments.dexjoco.validation import evaluate_loss
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method="file://" + str(Path(folder) / "rendezvous"),
        rank=rank, world_size=2, timeout=datetime.timedelta(seconds=30))
    try:
        class Options(dict):
            __getattr__ = dict.__getitem__
        class Data:
            sampling = "frame_uniform"
            lengths = [3, 5]
            ends = [3, 8]
            tasks = [{"name": "a"}, {"name": "b"}]
            def __getitem__(self, i):
                return {"action": torch.tensor([1. if i < 3 else 3.])}
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.))
            def training_loss(self, sample):
                assert not torch.is_grad_enabled()
                loss = (self.weight - sample["action"]).square().mean()
                return loss, {"loss_action": loss.item()}
        def reduce(value, **kwargs):
            torch.distributed.all_reduce(value)
            return value
        accelerator = SimpleNamespace(device=torch.device("cpu"), process_index=rank, num_processes=2,
            is_main_process=rank == 0, unwrap_model=lambda m: m, autocast=contextlib.nullcontext, reduce=reduce)
        trainer = SimpleNamespace(model=Model(), accelerator=accelerator, train_dataset=object(), val_dataset=Data(),
            cfg=SimpleNamespace(validation_loss=Options(seed=42, batch_size=2, max_batches=None, sampling="task_balanced")),
            global_step=2, output_dir=folder)
        result = evaluate_loss(trainer)
        assert result["task_samples"] == {"a": 3, "b": 5}  # 尾批次不足卡数时也不重复计数。
        assert result["val_loss"] == pytest.approx(5.)
        assert trainer.model.weight.grad is None
    finally:
        torch.distributed.destroy_process_group()


def test_fastwam_loss_validation_two_cpu_ranks_count_each_frame_once(tmp_path):
    import torch
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    torch.multiprocessing.spawn(_distributed_loss_worker, args=(str(tmp_path),), nprocs=2, join=True)
    lines = (tmp_path / "validation_loss.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["full_validation"]
