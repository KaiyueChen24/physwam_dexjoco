"""FastWAM 留出集只计算原始 training_loss，不反向、不生成视频、不更新训练 RNG。"""
from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import default_collate


def evaluate_loss(trainer):
    options = trainer.cfg.validation_loss
    dataset = trainer.val_dataset
    if dataset is None or dataset is trainer.train_dataset:
        raise ValueError("Loss validation requires a separate held-out dataset")
    if dataset.sampling != "frame_uniform":
        raise ValueError("Validation dataset must enumerate original held-out frames")
    accelerator = trainer.accelerator
    model = accelerator.unwrap_model(trainer.model)
    loss_model = trainer.model if hasattr(trainer.model, "training_loss") else model
    modes = [(module, module.training) for module in model.modules()]
    device = accelerator.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    numpy_state, python_state = np.random.get_state(), random.getstate()
    task_losses, task_samples, available, components = {}, {}, {}, {}
    try:
        model.eval()
        # fork_rng 会在退出时恢复状态，验证不会消耗训练噪声采样流。
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(options.seed) + accelerator.process_index)
            np.random.seed(int(options.seed))
            random.seed(int(options.seed))
            for task_index, task in enumerate(dataset.tasks):
                start = dataset.ends[task_index - 1] if task_index else 0
                length = dataset.lengths[task_index]
                available[task["name"]] = length
                indices = np.arange(start, start + length)
                global_batch = int(options.batch_size) * accelerator.num_processes
                maximum = options.get("max_batches")
                if maximum is not None and length > int(maximum) * global_batch:
                    indices = np.random.default_rng(int(options.seed) + task_index).choice(
                        indices, int(maximum) * global_batch, replace=False)
                totals = {"loss": 0.0}
                count = 0
                for offset in range(0, len(indices), global_batch):
                    batch_ids = indices[offset:offset + global_batch]
                    local_start = accelerator.process_index * int(options.batch_size)
                    chosen = batch_ids[local_start:local_start + int(options.batch_size)]
                    valid = len(chosen)
                    # 尾批次缺样本的 rank 仍前向一次保持分布式调用对齐，但权重为 0。
                    batch = default_collate([dataset[int(i)] for i in (chosen if valid else batch_ids[:1])])
                    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                    with accelerator.autocast():
                        loss, values = loss_model.training_loss(batch)
                    values = {"loss": float(loss.detach().float().item()), **{k: float(v) for k, v in values.items()}}
                    for key, value in values.items():
                        totals[key] = totals.get(key, 0.0) + value * valid
                    count += valid
                keys = sorted(totals)
                packed = torch.tensor([totals[k] for k in keys] + [count], dtype=torch.float64, device=device)
                packed = accelerator.reduce(packed, reduction="sum").cpu().numpy()
                if not np.isfinite(packed).all() or packed[-1] <= 0:
                    raise FloatingPointError(f"Invalid validation metrics for {task['name']}")
                means = {key: float(packed[i] / packed[-1]) for i, key in enumerate(keys)}
                task_losses[task["name"]] = means.pop("loss")
                components[task["name"]] = means
                task_samples[task["name"]] = int(packed[-1])
    finally:
        for module, training in modes:
            module.training = training
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    if options.sampling == "task_balanced":
        value = float(np.mean(list(task_losses.values())))
    else:
        value = sum(task_losses[t] * task_samples[t] for t in task_samples) / sum(task_samples.values())
    result = {"step": int(trainer.global_step), "val_loss": value, "task_losses": task_losses,
              "task_samples": task_samples, "available_samples": available, "task_components": components,
              "full_validation": task_samples == available, "aggregation": options.sampling}
    if accelerator.is_main_process:
        with (Path(trainer.output_dir) / "validation_loss.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
    return result
