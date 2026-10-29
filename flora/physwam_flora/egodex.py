"""Reference EgoDex files selected by download_egodex.py, without pose dependencies."""
from __future__ import annotations

import json
from pathlib import PurePosixPath
import re
import zlib

from .common import resolve_path, sha256
from .datasets import episode_selection
from .media import inspect_episode


def verify_file(root, spec):
    relative = PurePosixPath(spec["path"])
    path = root.joinpath(*relative.parts).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"EgoDex member escapes dataset directory: {relative}")
    if not path.is_file() or path.stat().st_size != spec["bytes"]:
        raise ValueError(f"EgoDex download missing or incomplete: {path}")
    checksum = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            checksum = zlib.crc32(chunk, checksum)
    if f"{checksum:08x}" != spec["crc32"]:
        raise ValueError(f"EgoDex download CRC mismatch: {path}")
    return path


def egodex_rows(config):
    caption = config.get("caption", {})
    if caption.get("profile") != "human" or caption.get("views") != "main" or caption.get("yam32", False):
        raise ValueError("EgoDex requires caption.profile=human, views=main, yam32=false")
    source = config["source"]
    root = resolve_path(source["root"])
    manifest = root / "download_manifest.json"
    download = json.loads(manifest.read_text())
    if download.get("schema_version") != 1 or not isinstance(download.get("files"), list):
        raise ValueError("Expected download_egodex.py's completed download_manifest.json")
    files = {}
    records = {}
    for spec in download["files"]:
        name = spec["path"]
        match = re.fullmatch(r"test/([a-z0-9_]+)/(\d+)\.(mp4|hdf5)", name)
        if not match or name in files:
            raise ValueError(f"Invalid/duplicate EgoDex download member: {name}")
        files[name] = spec
        task, index, extension = match.groups()
        if extension == "mp4":
            records.setdefault(task, []).append({"episode_index": int(index), "video": name})
    tasks = config.get("tasks")
    tasks = list(records) if tasks is None else tasks
    if not tasks or len(tasks) != len(set(tasks)) or set(tasks) - records.keys():
        raise ValueError(f"Unknown/empty/duplicate EgoDex tasks: {tasks}; available: {sorted(records)}")
    digest, rows = sha256(manifest), []
    for task in tasks:
        selected = episode_selection(records[task], source)
        limit = source.get("limit_per_task")
        if limit is not None and len(selected) != limit:
            raise ValueError(f"{task}: requested {limit} episodes, available selection has {len(selected)}")
        for record in selected:
            index, video_name = record["episode_index"], record["video"]
            hdf5_name = str(PurePosixPath(video_name).with_suffix(".hdf5"))
            if hdf5_name not in files:
                raise ValueError(f"Missing paired HDF5 entry: {hdf5_name}")
            print(f"Checking EgoDex: {task}/{index} (CRC and full video decode)", flush=True)
            video = verify_file(root, files[video_name])
            hdf5 = verify_file(root, files[hdf5_name])
            row = {"id": f"egodex_{task}_ep{index:06d}", "task_name": task,
                   "robot": "human", "episode_index": index, "instruction": "",
                   "instruction_mode": "none", "cameras": {"base": str(video)}, "success": None,
                   "source": {"kind": "egodex", "split": "test", "video": str(video),
                              "hdf5": str(hdf5), "download_manifest": str(manifest),
                              "download_manifest_sha256": digest,
                              "reference_text_supplied_to_model": False}}
            # No re-encoding, cropping or temporal resampling. Reject unsupported media.
            info = inspect_episode(row, stage="caption", caption_views="main")["cameras"]["base"]
            row.update(fps=info["fps"], frames=info["frames"])
            rows.append(row)
    return rows
