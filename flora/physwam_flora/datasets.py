"""Read DexJoCo metadata without importing simulation, LeRobot, or a policy."""
from __future__ import annotations

import json
from pathlib import Path

from .common import WORKSPACE, read_jsonl, resolve_path, validate_manifest, write_json, write_jsonl
from .media import extract_episode


def task_group(config):
    import yaml

    group = yaml.safe_load(resolve_path(config["task_group"]).read_text())
    selected = config.get("tasks")
    names = {t["name"] for t in group["tasks"]}
    if selected is not None and (not selected or set(selected) - names):
        raise ValueError(f"Unknown or empty task selection: {selected}; available: {sorted(names)}")
    tasks = [t for t in group["tasks"] if selected is None or t["name"] in selected]
    return group, tasks


def episode_selection(records, source):
    wanted = source.get("episodes")
    records = sorted(records, key=lambda r: int(r["episode_index"]))
    if len({int(r["episode_index"]) for r in records}) != len(records):
        raise ValueError("Duplicate episode_index in metadata")
    if wanted is not None:
        missing = set(wanted) - {int(r["episode_index"]) for r in records}
        if missing:
            raise ValueError(f"Requested episodes do not exist: {sorted(missing)}")
        records = [r for r in records if int(r["episode_index"]) in wanted]
    limit = source.get("limit_per_task")
    if limit is not None:
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit_per_task must be a positive integer or null")
        records = records[:limit]
    return records


def lerobot_rows(config, media_root):
    group, tasks = task_group(config)
    source = config["source"]
    root = resolve_path(source["root"]) if source.get("root") else WORKSPACE / "dexjoco" / group["dataset_root"]
    output = []
    for task in tasks:
        dataset = root / task["dataset"]
        info_path = dataset / "meta/info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Download/extract the dataset first: {info_path}")
        info = json.loads(info_path.read_text())
        version = info.get("codebase_version", "")
        if version.startswith("v2."):
            records = read_jsonl(dataset / "meta/episodes.jsonl")
        elif version.startswith("v3."):
            import pyarrow.parquet as pq

            files = sorted(dataset.glob("meta/episodes/**/*.parquet"))
            if not files:
                raise FileNotFoundError(f"No v3 episode metadata under {dataset}")
            records = [row for file in files for row in pq.read_table(file).to_pylist()]
        else:
            raise ValueError(f"Unsupported LeRobot version: {version!r}")
        for record in episode_selection(records, source):
            index = int(record["episode_index"])
            name = f"{task['name']}_ep{index:06d}"
            cameras, provenance = {}, {}
            for role, camera in task["cameras"].items():
                key = f"observation.images.{camera}"
                if info["features"].get(key, {}).get("dtype") != "video":
                    raise ValueError(f"{dataset}: missing video feature {key}")
                if version.startswith("v2."):
                    values = dict(episode_chunk=index // int(info["chunks_size"]),
                                  episode_index=index, video_key=key)
                    path = dataset / info["video_path"].format(**values)
                    provenance[role] = {"path": str(path.resolve())}
                else:
                    prefix = f"videos/{key}"
                    values = dict(chunk_index=int(record[prefix + "/chunk_index"]),
                                  file_index=int(record[prefix + "/file_index"]), video_key=key)
                    source_path = dataset / info["video_path"].format(**values)
                    start, end = float(record[prefix + "/from_timestamp"]), float(record[prefix + "/to_timestamp"])
                    path = media_root / name / f"{role}.mp4"
                    extract_episode(source_path, path, start, end, float(info["fps"]), int(record["length"]))
                    provenance[role] = {"path": str(source_path.resolve()), "start_s": start, "end_s": end,
                                        "export": "native_fps_h264_crf0_episode_relative"}
                cameras[role] = str(path.resolve())
            # Configured DexJoCo task prompt is canonical. Do not fabricate success
            # labels merely because a dataset is called an expert dataset.
            output.append({"id": name, "task_name": task["name"], "instruction": task["prompt"],
                           "robot": group["robot_type"], "episode_index": index,
                           "fps": float(info["fps"]), "frames": int(record["length"]),
                           "cameras": cameras, "success": None,
                           "source": {"kind": "lerobot", "version": version,
                                      "dataset": str(dataset.resolve()), "cameras": provenance}})
    return output


def rollout_rows(config):
    group, tasks = task_group(config)
    if len(tasks) != 1:
        raise ValueError("rollout source requires exactly one selected task")
    task, source = tasks[0], config["source"]
    if not source.get("root"):
        raise ValueError("Set source.root to an evaluation attempt or its videos directory")
    root = resolve_path(source["root"])
    root = root / "videos" if (root / "videos").is_dir() else root
    records = read_jsonl(root / "episodes.jsonl")
    records = [{**r, "episode_index": r["episode_id"]} for r in records]
    env_path = root.parent / "environment.yaml"
    camera_mapping, instruction = task["cameras"], task["prompt"]
    if env_path.is_file():
        import yaml

        environment = yaml.safe_load(env_path.read_text())
        if environment.get("env_name") != task["name"]:
            raise ValueError("Selected task differs from the evaluation environment")
        camera_mapping = environment["camera_mapping"]
        instruction = environment["prompt"]
    output = []
    for record in episode_selection(records, source):
        if record.get("task", task["name"]) != task["name"]:
            raise ValueError("Rollout task does not match selected task")
        if not isinstance(record.get("success"), bool):
            raise ValueError("Rollout record must contain a boolean success label")
        index = int(record["episode_index"])
        cameras = {role: str(root / f"episode_{index:03d}_{camera}.mp4")
                   for role, camera in camera_mapping.items()}
        if not all(Path(path).is_file() for path in cameras.values()):
            raise FileNotFoundError(f"{root}: episode {index} has no complete camera videos; --no-video runs cannot be scored")
        output.append({"id": f"{task['name']}_ep{index:06d}", "task_name": task["name"],
                       "instruction": instruction, "robot": group["robot_type"],
                       "episode_index": index, "cameras": cameras, "success": record["success"],
                       "source": {"kind": "fastwam_rollout", "root": str(root), "seed": record.get("seed")}})
    return output


def prepare(config, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new preparation directory: {output}")
    kind = config["source"]["kind"]
    if kind not in {"lerobot", "rollout", "manifest", "egodex"}:
        raise ValueError(f"Unknown source kind: {kind}")
    output.mkdir(parents=True)
    if kind == "lerobot":
        rows = lerobot_rows(config, output / "media")
    elif kind == "rollout":
        rows = rollout_rows(config)
    elif kind == "egodex":
        from .egodex import egodex_rows

        rows = egodex_rows(config)
    else:
        from .common import read_manifest

        rows = read_manifest(resolve_path(config["source"]["path"]))
    validate_manifest(rows)
    write_jsonl(output / "episodes.jsonl", rows)
    write_json(output / "prepare.json", {"schema_version": 1, "config": config, "episodes": len(rows)})
    return {"manifest": str(output / "episodes.jsonl"), "episodes": len(rows)}
