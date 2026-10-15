"""π0.5 留出集 loss：只前向计算、逐任务统计，不更新参数、优化器或训练 RNG。"""
from __future__ import annotations

import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.training import data_loader
from openpi.training.dexjoco_multitask import MultiTaskDataset, make_data_factory


class ValidationRunner:
    def __init__(self, config, state, state_sharding, batch_sharding):
        self.options = config.validation
        self.group = self.options["group"]
        self.batch_size = self.options["batch_size"]
        if self.batch_size % jax.device_count():
            raise ValueError("validation_batch_size must be divisible by visible GPU count")
        self.raw = MultiTaskDataset(self.group, config.model.action_horizon)
        # 与训练共用资产目录和归一化参数，仅替换 episode 成员。
        data_config = make_data_factory(self.group).create(config.assets_dirs, config.model)
        self.dataset = data_loader.transform_dataset(self.raw, data_config)
        self.sharding = batch_sharding
        model_def = state.model_def

        def loss(params, rng, batch):
            model = nnx.merge(model_def, params)
            model.eval()
            observation, actions = batch
            values = model.compute_loss(rng, observation, actions, train=False)
            return jnp.mean(values, axis=tuple(range(1, values.ndim)))

        self.loss = jax.jit(loss, in_shardings=(state_sharding.params,
                           jax.sharding.NamedSharding(batch_sharding.mesh, jax.sharding.PartitionSpec()),
                           batch_sharding), out_shardings=batch_sharding)

    def evaluate(self, params, step):
        task_losses, samples, available = {}, {}, {}
        for task_index, task in enumerate(self.group["tasks"]):
            start = self.raw.ends[task_index - 1] if task_index else 0
            length = self.raw.lengths[task_index]
            indices = np.arange(start, start + length)
            available[task["name"]] = length
            maximum = self.options.get("max_batches")
            if maximum is not None and length > maximum * self.batch_size:
                # 固定随机子集，避免仅验证每条轨迹开头；全量时每帧只验证一次。
                indices = np.random.default_rng(self.options["seed"] + task_index).choice(
                    indices, maximum * self.batch_size, replace=False)
            total, count = 0.0, 0
            for batch_index, offset in enumerate(range(0, len(indices), self.batch_size)):
                chosen = indices[offset:offset + self.batch_size]
                valid = len(chosen)
                items = [self.dataset[int(i)] for i in chosen]
                items.extend([items[-1]] * (self.batch_size - valid))
                batch = data_loader._collate_fn(items)
                batch = (model_lib.Observation.from_dict(batch), batch["actions"])
                batch = jax.tree.map(lambda x: jax.make_array_from_process_local_data(self.sharding, x), batch)
                rng = jax.random.fold_in(jax.random.key(self.options["seed"]), task_index)
                values = np.asarray(self.loss(params, jax.random.fold_in(rng, batch_index), batch))[:valid]
                if not np.isfinite(values).all():
                    raise FloatingPointError(f"Nonfinite validation loss for {task['name']}")
                total += float(values.sum(dtype=np.float64))
                count += valid
            task_losses[task["name"]] = total / count
            samples[task["name"]] = count
        if self.options["sampling"] == "task_balanced":
            value = float(np.mean(list(task_losses.values())))
        else:
            value = sum(task_losses[t] * samples[t] for t in samples) / sum(samples.values())
        result = {"step": int(step), "val_loss": value, "task_losses": task_losses,
                  "task_samples": samples, "available_samples": available,
                  "full_validation": samples == available, "aggregation": self.options["sampling"]}
        with Path(self.options["output_path"]).open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        return result
