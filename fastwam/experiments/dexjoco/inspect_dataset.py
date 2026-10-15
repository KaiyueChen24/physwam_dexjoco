"""Decode one sequence per task without loading Wan or downloading model weights."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import read_run
from .dataset import DexJoCoVideoDataset


def inspect(run_dir):
    run_dir, config = read_run(run_dir)
    kwargs = {k: v for k, v in config["data"]["train"].items() if not k.startswith("_")}
    dataset = DexJoCoVideoDataset(**kwargs)
    target = run_dir / "inspection"
    target.mkdir(exist_ok=True)
    summaries = []
    for index, task in enumerate(dataset.tasks):
        offset = index if dataset.sampling == "task_balanced" else (dataset.ends[index - 1] if index else 0)
        sample = dataset[offset]
        row = {"task": task["name"], "prompt": sample["prompt"],
               "video": list(sample["video"].shape), "action": list(sample["action"].shape),
               "proprio": list(sample["proprio"].shape), "selected_frames": dataset.lengths[index]}
        if not all(np.isfinite(sample[k].numpy()).all() for k in ("video", "action", "proprio")):
            raise ValueError(f"Nonfinite sequence: {task['name']}")
        video = sample["video"].numpy()
        for t in (0, video.shape[1] - 1):
            pixels = np.rint((video[:, t].transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(target / f"{task['name']}_frame{t}.png")
        summaries.append(row)
        print(json.dumps(row), flush=True)
    (target / "samples.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help='[配置与实验] 已有实验目录，从中读取配置快照和资产。')
    inspect(parser.parse_args().run_dir)
