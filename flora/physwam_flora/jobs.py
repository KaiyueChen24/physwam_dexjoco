"""Translate a local episode manifest to the exact upstream JSONL contracts."""
from __future__ import annotations

import json
import math

from .common import resolve_path


def reward_roles(cameras):
    if {"wrist_left", "wrist_right"} <= cameras.keys():
        return ["wrist_left", "base", "wrist_right"]
    if "wrist_left" in cameras or "wrist_right" in cameras:
        raise ValueError("A bimanual episode needs both wrist cameras")
    return ["base", "wrist"] if "wrist" in cameras else ["base"]


def validate_rubric(steps):
    if not isinstance(steps, list) or not steps:
        raise ValueError("Rubric must be a nonempty list of steps")
    maxima = []
    for step in steps:
        if not isinstance(step.get("action"), str) or not step["action"].strip():
            raise ValueError("Every rubric step requires an action")
        grading = step.get("grading")
        if not isinstance(grading, list) or not 2 <= len(grading) <= 5:
            raise ValueError("Rubric steps need 2-5 explicitly labelled grades")
        for grade in grading:
            if not isinstance(grade.get("label"), str) or not grade["label"].strip():
                raise ValueError("Rubric grades require labels")
            value = grade.get("points")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Rubric points must be finite and nonnegative")
        maxima.append(max(g["points"] for g in grading))
    if not math.isclose(sum(maxima), 1.0, abs_tol=1e-6):
        raise ValueError("Rubric step maxima must sum to 1")


def build_jobs(rows, config, stage):
    jobs, labels = [], []
    captions = config.get("caption", {})
    views = captions.get("views", "auto")
    profile = captions.get("profile", "dex")
    if views not in {"auto", "main", "three"} or profile not in {"human", "dex", "yam", "tianji"}:
        raise ValueError("Invalid caption views/profile")
    rubric_map = config.get("rubric", {}).get("files", {})
    for row in rows:
        cameras = row["cameras"]
        roles = reward_roles(cameras)
        if stage == "caption":
            three = views == "three" or (views == "auto" and "wrist_left" in cameras)
            if three and not {"wrist_left", "wrist_right"} <= cameras.keys():
                raise ValueError("Caption three-view mode requires base and both wrist cameras")
            if three and profile == "human":
                raise ValueError("Human caption profile only accepts one camera")
            if captions.get("yam32", False) and (profile != "yam" or not three):
                raise ValueError("yam32 requires YAM and three views")
            job = {"id": row["id"], "video": cameras["base"], "profile": profile,
                   "instruction": row["instruction"], "yam32": bool(captions.get("yam32", False))}
            if three:
                job.update(left_wrist=cameras["wrist_left"], right_wrist=cameras["wrist_right"])
        else:
            if not row["instruction"].strip():
                raise ValueError(f"{row['id']}: {stage} requires a task instruction; blank instructions are caption-only")
            job = {"id": row["id"], "task": row["instruction"], "videos": [cameras[r] for r in roles]}
            if stage == "rubric":
                if row.get("task_name") not in rubric_map:
                    raise ValueError(f"Provide rubric.files.{row.get('task_name')} with task-specific grading criteria")
                steps = json.loads(resolve_path(rubric_map[row["task_name"]]).read_text())
                validate_rubric(steps)
                descriptions = {"base": "a main scene camera", "wrist": "a wrist camera",
                                "wrist_left": "a left-wrist camera", "wrist_right": "a right-wrist camera"}
                job.update(robot=row.get("robot", "robot"), cameras=[descriptions[r] for r in roles], rubric=steps)
        jobs.append(job)
        # No ground-truth label enters any upstream model job.
        labels.append({"id": row["id"], "task_name": row.get("task_name"), "success": row.get("success")})
    return jobs, labels
