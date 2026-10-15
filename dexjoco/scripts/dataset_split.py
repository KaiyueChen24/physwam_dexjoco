"""按任务、完整 episode 划分训练/验证；π0.5 与 FastWAM 共用同一算法。"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random


def validation_settings(experiment):
    """[验证/loss] ratio=0 关闭；不启用模拟器或生成式视频评测。"""
    settings = {
        "ratio": experiment.get("validation_ratio", 0.0),
        "seed": experiment.get("validation_seed", 42),
        "interval": experiment.get("validation_interval", 1000),
        "batch_size": experiment.get("validation_batch_size", 1),
        "max_batches": experiment.get("validation_max_batches"),
    }
    ratio = settings["ratio"]
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 <= ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    for key in ("seed", "interval", "batch_size"):
        if type(settings[key]) is not int or settings[key] < (0 if key == "seed" else 1):
            raise ValueError(f"validation {key} must be a valid integer")
    if settings["max_batches"] is not None and (type(settings["max_batches"]) is not int or settings["max_batches"] < 1):
        raise ValueError("validation_max_batches must be null or a positive integer")
    return settings


def available_episodes(root):
    """只读取 LeRobot 元数据，支持 v2 JSONL 与 v3 Parquet，不解码视频。"""
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    episode_file = root / "meta/episodes.jsonl"
    parquet = sorted((root / "meta/episodes").rglob("*.parquet"))
    if episode_file.is_file():
        ids = [int(json.loads(line)["episode_index"]) for line in episode_file.read_text().splitlines() if line.strip()]
    elif parquet:
        import pyarrow.parquet as pq
        ids = [int(i) for path in parquet for i in pq.read_table(path, columns=["episode_index"])["episode_index"].to_pylist()]
    else:
        # 标准 LeRobot episode ID 为连续整数；旧数据只有 info.json 时使用此约定。
        ids = list(range(int(info["total_episodes"])))
    if not ids or len(set(ids)) != len(ids) or min(ids) < 0:
        raise ValueError(f"Invalid episode metadata: {root}")
    return sorted(ids)


def split_group(group, settings):
    """每任务留出 ceil(N*ratio) 条 episode，至少各留一条；划分不依赖任务顺序。"""
    if not settings["ratio"]:
        return copy.deepcopy(group), None, None
    train, validation = copy.deepcopy(group), copy.deepcopy(group)
    validation["sampling"] = "frame_uniform"  # 验证逐帧遍历，不进行任务重采样。
    records = []
    for source, train_task, val_task in zip(group["tasks"], train["tasks"], validation["tasks"], strict=True):
        root = source.get("dataset_path") or Path(group["dataset_root"]) / source["dataset"]
        all_ids = available_episodes(root)
        requested = source.get("episodes")
        ids = all_ids if requested is None else sorted(requested)
        if (not ids or len(ids) != len(set(ids)) or any(type(i) is not int for i in ids)
                or not set(ids) <= set(all_ids)):
            raise ValueError(f"{source['name']}: selected episodes not present in metadata")
        if len(ids) < 2:
            raise ValueError(f"{source['name']}: at least two episodes are required for validation; use validation_ratio=0 to disable")
        # 使用稳定 hash，使两个模型、任务子集、不同 Python 进程得到相同划分。
        task_seed = int.from_bytes(hashlib.sha256(f"{settings['seed']}:{source['name']}".encode()).digest()[:8], "big")
        shuffled = list(ids)
        random.Random(task_seed).shuffle(shuffled)
        n_val = min(len(ids) - 1, max(1, math.ceil(len(ids) * settings["ratio"])))
        val_ids = sorted(shuffled[:n_val])
        train_ids = sorted(shuffled[n_val:])
        train_task["episodes"], val_task["episodes"] = train_ids, val_ids
        records.append({"task": source["name"], "source_episodes": ids,
                        "train_episodes": train_ids, "validation_episodes": val_ids,
                        "source_count": len(ids), "train_count": len(train_ids), "validation_count": len(val_ids)})
    manifest = {"schema_version": 1, "unit": "episode", "validation_ratio": settings["ratio"],
                "seed": settings["seed"], "rounding": "ceil, clamped to [1, N-1]", "tasks": records}
    return train, validation, manifest


def save_split(run_dir, manifest):
    """保存并核对划分；恢复和重复 prepare 不得静默变更验证成员。"""
    if manifest is None:
        return
    path = Path(run_dir) / "data_split.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing run has a different train/validation split; use a new run-name")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
