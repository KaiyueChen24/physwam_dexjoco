from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

FLORA_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = FLORA_ROOT.parent
VENDOR = FLORA_ROOT / "vendor"


def read_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        rows.append(row)
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    with Path(path).open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_path(value, base=WORKSPACE):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(base) / path).resolve()


def load_config(path):
    import yaml

    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Config must be an object with schema_version: 1")
    # All configuration paths are workspace-relative, independent of cwd.
    return config


def validate_manifest(rows):
    if not rows:
        raise ValueError("No episodes selected")
    ids = set()
    for row in rows:
        name = row.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", name):
            raise ValueError(f"Invalid episode id: {name!r}")
        if name in ids:
            raise ValueError(f"Duplicate episode id: {name}")
        ids.add(name)
        if not isinstance(row.get("instruction"), str) or not row["instruction"].strip():
            raise ValueError(f"{name}: an explicit task instruction is required")
        cameras = row.get("cameras")
        if not isinstance(cameras, dict) or not cameras or "base" not in cameras:
            raise ValueError(f"{name}: cameras must include base")
        if set(cameras) - {"base", "wrist", "wrist_left", "wrist_right"}:
            raise ValueError(f"{name}: unsupported camera roles: {list(cameras)}")
        for value in cameras.values():
            if not isinstance(value, str) or not Path(value).is_file():
                raise FileNotFoundError(f"{name}: video missing: {value}")
        success = row.get("success")
        if success is not None and not isinstance(success, bool):
            raise ValueError(f"{name}: success must be bool or null; never infer it from a filename")
        fps = row.get("fps")
        if fps is not None and (not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0):
            raise ValueError(f"{name}: invalid fps")
    return rows


def read_manifest(path):
    path = Path(path).resolve()
    rows = read_jsonl(path)
    for row in rows:
        row["cameras"] = {role: str(resolve_path(value, path.parent))
                          for role, value in row.get("cameras", {}).items()}
    return validate_manifest(rows)


def verify_vendor():
    lock = json.loads((VENDOR / "sources.lock.json").read_text())
    count = 0
    for folder, spec in lock["sources"].items():
        for relative, expected in spec["files"].items():
            path = VENDOR / folder / relative
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Official source integrity check failed: {path}")
            count += 1
    return {"verified_files": count, "sources": {
        folder: {key: value for key, value in spec.items() if key != "files"}
        for folder, spec in lock["sources"].items()}}
