"""DexJoCo task groups with canonical camera fields and balanced frame sampling."""
from __future__ import annotations

import bisect
import dataclasses
from pathlib import Path

import numpy as np


def frame_weights(lengths: list[int], balanced: bool = True) -> np.ndarray:
    """Each task gets equal probability regardless of its trajectory lengths."""
    if not lengths or any(n <= 0 for n in lengths):
        raise ValueError("Every task must contain at least one frame")
    return np.concatenate([
        np.full(n, 1.0 / (len(lengths) * n) if balanced else 1.0 / sum(lengths))
        for n in lengths
    ])


class MultiTaskDataset:
    def __init__(self, spec: dict, action_horizon: int):
        self.spec = spec
        self.horizon = action_horizon
        self.datasets = []
        self.manifest = []
        lengths = []
        for task in spec["tasks"]:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

            root = Path(spec["dataset_root"]) / task["dataset"]
            # Never silently download data for a misspelled local path.
            if not (root / "meta/info.json").is_file():
                raise FileNotFoundError(f"Missing {root / 'meta/info.json'}; download this task first")
            meta = LeRobotDatasetMetadata("local_repo", root=root)
            expected = {"action": spec["action_dim"], "observation.state": spec["state_dim"]}
            for key, dim in expected.items():
                if list(meta.features[key]["shape"]) != [dim]:
                    raise ValueError(f"{task['name']}: {key} must have shape [{dim}]")
            for camera in task["cameras"].values():
                if f"observation.images.{camera}" not in meta.features:
                    raise ValueError(f"{task['name']}: missing camera {camera}")
            dataset = LeRobotDataset(
                "local_repo", root=root, episodes=task.get("episodes"),
                delta_timestamps={"action": [t / meta.fps for t in range(action_horizon)]},
                video_backend=spec.get("video_backend", "pyav"),
            )
            self.datasets.append(dataset)
            length = len(dataset)
            record = {"name": task["name"], "root": str(root), "frames": length,
                      "episodes": task.get("episodes"), "available_episodes": meta.total_episodes,
                      "fps": meta.fps, "codebase_version": meta.info["codebase_version"],
                      "cameras": task["cameras"], "prompt_fallback": task["prompt"]}
            lengths.append(length)
            self.manifest.append(record)
        self.lengths = lengths
        self.ends = np.cumsum(lengths).tolist()
        self.sampling_weights = frame_weights(lengths, spec.get("sampling") == "task_balanced")

    def __len__(self):
        return self.ends[-1]

    def locate(self, index: int):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        task_index = bisect.bisect_right(self.ends, index)
        local_index = index - (self.ends[task_index - 1] if task_index else 0)
        return task_index, local_index

    def __getitem__(self, index):
        task_index, local_index = self.locate(index)
        task = self.spec["tasks"][task_index]
        raw = self.datasets[task_index][local_index]
        sample = {"observation.state": np.asarray(raw["observation.state"], dtype=np.float32),
                  "action": np.asarray(raw["action"], dtype=np.float32),
                  "prompt": raw.get("task", task["prompt"])}
        if not isinstance(sample["prompt"], str) or not sample["prompt"].strip():
            raise ValueError(f"{task['name']}: empty language instruction")
        for policy_key, destination in canonical_cameras(self.spec).items():
            sample[f"observation.images.{destination}"] = raw[f"observation.images.{task['cameras'][policy_key]}"]
        if sample["action"].shape != (self.horizon, self.spec["action_dim"]):
            raise ValueError(f"{task['name']}: invalid action chunk shape {sample['action'].shape}")
        return sample

    def low_dim(self, index):
        """Read one state/action frame without decoding video or constructing a chunk."""
        task_index, local_index = self.locate(index)
        raw = self.datasets[task_index].hf_dataset[local_index]
        return {"state": np.asarray(raw["observation.state"], dtype=np.float32),
                "actions": np.asarray(raw["action"], dtype=np.float32)}


def canonical_cameras(spec):
    if spec["robot_type"] == "single_arm":
        return {"base": "front", "wrist": "wrist"}
    return {"base": "ego", "wrist_left": "wrist_left", "wrist_right": "wrist_right"}


def make_data_factory(spec: dict):
    # Lazy import avoids a config.py <-> data-loader circular dependency.
    from openpi.training import config as cfg

    @dataclasses.dataclass(frozen=True)
    class GroupDataConfig(cfg.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            cls = cfg.SingleArmDataConfig if spec["robot_type"] == "single_arm" else cfg.DualArmDataConfig
            factory = cls(root=Path(spec["dataset_root"]), repo_id=spec["name"],
                          assets=cfg.AssetsConfig(asset_id=spec["name"]))
            data = factory.create(assets_dirs, model_config)
            return dataclasses.replace(data, multitask=spec, prompt_from_task=False)

    return GroupDataConfig(repo_id=spec["name"])
