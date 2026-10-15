"""Validate local LeRobot metadata and compute statistics on selected episodes only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def inspect_task(task, state_dim, action_dim):
    root = Path(task["dataset_path"])
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"{task['name']}: missing {info_path}; download/extract the LeRobot dataset first")
    info = json.loads(info_path.read_text())
    version = str(info.get("codebase_version", ""))
    if not version.startswith(("v2.", "v3.")):
        raise ValueError(f"{root}: unsupported LeRobot version {version!r}")
    features = info["features"]
    for key, dim in (("action", action_dim), ("observation.state", state_dim)):
        if features.get(key, {}).get("shape") != [dim]:
            raise ValueError(f"{root}: {key} must be [{dim}], got {features.get(key)}; remove privileged state/padding in the converter")
    for camera in task["cameras"].values():
        key = f"observation.images.{camera}"
        if key not in features or features[key].get("dtype") != "video":
            raise ValueError(f"{root}: expected video feature {key}")
    fps = float(info["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{root}: invalid fps")
    files = sorted((root / "data").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {root / 'data'}")
    return {"name": task["name"], "path": str(root), "version": version, "fps": fps,
            "episodes": task.get("episodes"), "cameras": task["cameras"],
            "info_sha256": hashlib.sha256(info_path.read_bytes()).hexdigest(),
            "files": [{"path": str(p.relative_to(root)), "size": p.stat().st_size,
                       "mtime_ns": p.stat().st_mtime_ns} for p in files]}


def low_dim_batches(task, state_dim, action_dim):
    import pyarrow.dataset as pads

    inspect_task(task, state_dim, action_dim)
    files = sorted(str(p) for p in Path(task["dataset_path"]).glob("data/*/*.parquet"))
    selection = task.get("episodes")
    predicate = pads.field("episode_index").isin(selection) if selection is not None else None
    scanner = pads.dataset(files, format="parquet").scanner(
        columns=["observation.state", "action", "episode_index"], filter=predicate, batch_size=4096)
    for batch in scanner.to_batches():
        values = batch.to_pydict()
        arrays = {field: np.asarray(values[key], dtype=np.float32)
                  for field, key in (("state", "observation.state"), ("action", "action"))}
        if not len(arrays["action"]):
            continue
        for field, dim in (("state", state_dim), ("action", action_dim)):
            if arrays[field].shape != (len(arrays["action"]), dim) or not np.isfinite(arrays[field]).all():
                raise ValueError(f"{task['name']}: invalid {field} values")
        yield arrays, np.asarray(values["episode_index"], dtype=np.int64)


def compute_stats(tasks, state_dim, action_dim):
    extrema = {field: {"global_min": np.full(dim, np.inf), "global_max": np.full(dim, -np.inf)}
               for field, dim in (("state", state_dim), ("action", action_dim))}
    counts = {}
    for task in tasks:
        count, episodes = 0, set()
        for arrays, ids in low_dim_batches(task, state_dim, action_dim):
            count += len(ids)
            episodes.update(map(int, ids))
            for field in extrema:
                extrema[field]["global_min"] = np.minimum(extrema[field]["global_min"], arrays[field].min(0))
                extrema[field]["global_max"] = np.maximum(extrema[field]["global_max"], arrays[field].max(0))
        if not count or (task.get("episodes") is not None and episodes != set(task["episodes"])):
            raise ValueError(f"{task['name']}: empty data or requested episodes not present")
        counts[task["name"]] = {"frames": count, "episodes": sorted(episodes)}
    stats = {field: {"default": {key: value.astype(np.float32).tolist() for key, value in values.items()}}
             for field, values in extrema.items()}
    stats["provenance"] = {"method": "exact min/max of selected training frames; no video decoding", "tasks": counts}
    return stats
