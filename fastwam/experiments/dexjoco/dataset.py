"""Task-aware sequence adapter around FastWAM's vendored LeRobot v2/v3 readers."""
from __future__ import annotations

import bisect
import json

import numpy as np
import torch

from .data import inspect_task
from .transforms import MinMaxNormalizer, PROMPT_TEMPLATE, pack_images


class DexJoCoVideoDataset(torch.utils.data.Dataset):
    def __init__(self, tasks, state_dim, action_dim, action_horizon, video_stride,
                 camera_order, image_size, sampling, stats_path, control_hz,
                 processor=None, **_):
        self.tasks = tasks
        self.state_dim, self.action_dim = int(state_dim), int(action_dim)
        self.horizon, self.video_stride = int(action_horizon), int(video_stride)
        self.camera_order, self.image_size = list(camera_order), list(image_size)
        self.sampling = sampling
        self.normalizer = MinMaxNormalizer(json.loads(open(stats_path).read()))
        self.readers, self.lengths = [], []
        for task in tasks:
            meta = inspect_task(task, self.state_dim, self.action_dim)
            if abs(meta["fps"] - float(control_hz)) > 1e-6:
                raise ValueError(f"{task['name']}: dataset fps differs from configured control_hz")
            if meta["version"].startswith("v3."):
                from fastwam.datasets.lerobot3.lerobot_dataset import LeRobotDataset
            else:
                from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset
            deltas = {key: [i / meta["fps"] for i in range(self.horizon)]
                      for key in ("action", "observation.state")}
            for camera in task["cameras"].values():
                deltas[f"observation.images.{camera}"] = [
                    i / meta["fps"] for i in range(0, self.horizon + 1, self.video_stride)]
            reader = LeRobotDataset(repo_id=task["name"], root=task["dataset_path"],
                                    episodes=task.get("episodes"), delta_timestamps=deltas, video_backend="pyav")
            self.readers.append(reader)
            self.lengths.append(len(reader))
        if not self.lengths or min(self.lengths) <= 0:
            raise ValueError("All selected tasks must contain frames")
        self.ends = np.cumsum(self.lengths).tolist()

    def __len__(self):
        if self.sampling == "task_balanced":
            return max(self.lengths) * len(self.tasks)
        return self.ends[-1]

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.sampling == "task_balanced":
            task = index % len(self.tasks)
            return task, (index // len(self.tasks)) % self.lengths[task]
        task = bisect.bisect_right(self.ends, index)
        return task, index - (self.ends[task - 1] if task else 0)

    def __getitem__(self, index):
        task_index, frame = self.locate(index)
        task = self.tasks[task_index]
        raw = self.readers[task_index][frame]  # Reader clamps queries within the selected episode.
        image_count = self.horizon // self.video_stride + 1
        images = {}
        for camera in self.camera_order:
            value = raw[f"observation.images.{task['cameras'][camera]}"]
            if value.ndim != 4 or value.shape[0] != image_count or value.shape[1] != 3:
                raise ValueError(f"{task['name']}: invalid camera sequence {tuple(value.shape)}")
            pixels = value.detach().cpu().numpy()
            if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
                raise ValueError("LeRobot video reader must return RGB in [0,1]")
            images[camera] = np.rint(pixels.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        video = np.stack([pack_images({k: v[i] for k, v in images.items()}, self.camera_order, self.image_size)
                          for i in range(image_count)], axis=1)
        action = self.normalizer.forward("action", raw["action"].numpy())
        state = self.normalizer.forward("state", raw["observation.state"].numpy())
        if action.shape != (self.horizon, self.action_dim) or state.shape != (self.horizon, self.state_dim):
            raise ValueError("Unexpected sequence length or robot dimensions")
        first_camera = f"observation.images.{task['cameras'][self.camera_order[0]]}"
        return {"video": torch.from_numpy(video), "action": torch.from_numpy(action),
                "proprio": torch.from_numpy(state), "prompt": PROMPT_TEMPLATE.format(task=task["prompt"]),
                "image_is_pad": raw[first_camera + "_is_pad"],
                "action_is_pad": raw["action_is_pad"], "proprio_is_pad": raw["observation.state_is_pad"]}
