"""Ground-truth labels are joined only after upstream inference completes."""
from __future__ import annotations

import math
from statistics import mean
from statistics import median
import json
from pathlib import Path

from .common import read_jsonl


def unique_rows(path):
    rows = read_jsonl(path)
    result = {r["id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate ids in {path}")
    return result


def terminal_metrics(predictions, labels, threshold=5.0):
    if not math.isfinite(threshold) or not 1 <= threshold <= 5:
        raise ValueError("Success threshold must be within [1,5]")
    pred, gold = unique_rows(predictions), unique_rows(labels)
    extra = set(pred) - set(gold)
    if extra:
        raise ValueError(f"Predictions contain unknown ids: {sorted(extra)}")
    evaluated, invalid = [], []
    for name, p in pred.items():
        value = p.get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 1 <= value <= 5:
            invalid.append(name)
            continue
        label = gold[name].get("success")
        if label is not None:
            if not isinstance(label, bool):
                raise ValueError("Ground truth success must be bool or null")
            evaluated.append((name, label, float(value)))
    tp = sum(y and s >= threshold for _, y, s in evaluated)
    tn = sum(not y and s < threshold for _, y, s in evaluated)
    fp = sum(not y and s >= threshold for _, y, s in evaluated)
    fn = sum(y and s < threshold for _, y, s in evaluated)
    positives = [s for _, y, s in evaluated if y]
    negatives = [s for _, y, s in evaluated if not y]
    # Rank statistic with average ranks for ties, O(n log n).
    auc = None
    if positives and negatives:
        ordered = sorted((s, y) for _, y, s in evaluated)
        rank_sum, i = 0.0, 0
        while i < len(ordered):
            j = i + 1
            while j < len(ordered) and ordered[j][0] == ordered[i][0]:
                j += 1
            rank_sum += ((i + 1 + j) / 2) * sum(y for _, y in ordered[i:j])
            i = j
        auc = (rank_sum - len(positives) * (len(positives) + 1) / 2) / (len(positives) * len(negatives))
    return {"threshold": threshold, "score_field": "score", "expected_episodes": len(gold),
            "predicted_episodes": len(pred), "evaluated_episodes": len(evaluated),
            "missing_predictions": sorted(set(gold) - set(pred)), "invalid_predictions": invalid,
            "unlabelled_episodes": sum(r.get("success") is None for r in gold.values()),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "accuracy": (tp + tn) / len(evaluated) if evaluated else None,
            "balanced_accuracy": (tp / len(positives) + tn / len(negatives)) / 2 if positives and negatives else None,
            "failure_recall": tn / len(negatives) if negatives else None,
            "auroc": auc, "mean_success_score": mean(positives) if positives else None,
            "mean_failure_score": mean(negatives) if negatives else None,
            "note": "Threshold is a local evaluation choice; scores are not calibrated success probabilities."}


def dense_quality(path):
    rows = list(unique_rows(path).values())
    invalid = sum(r["invalid_clips"] for r in rows)
    return {"episodes": len(rows), "clips": sum(len(r["clips"]) for r in rows),
            "invalid_clips": invalid, "episodes_with_invalid_clips": [r["id"] for r in rows if r["invalid_clips"]],
            "usable_without_parse_fallback": invalid == 0,
            "note": "Upstream replaces unparseable dense windows with normal labels. Inspect invalid_clips before interpreting scores."}


def caption_timing(run_dir, group="all"):
    """Summarize native elapsed_s, without inferring timings from file mtimes."""
    root = Path(run_dir).resolve()
    metadata = json.loads((root / "run.json").read_text())
    status = json.loads((root / "status.json").read_text())
    if metadata["plan"]["stage"] != "caption" or metadata["plan"].get("mode") != "inference" or status.get("status") != "complete":
        raise ValueError("Timing requires a completed caption inference run")
    groups = {"single": "single_arm", "dual": "dual_arm", "human": "human"}
    if group not in {"all", *groups}:
        raise ValueError(f"Unknown timing group: {group}")
    rows = [row for row in metadata["episodes"] if group == "all" or row.get("robot") == groups[group]]
    if not rows:
        raise ValueError(f"No episodes for timing group: {group}")
    ids, timings, by_task = set(), [], {}
    for row in rows:
        name = row["id"]
        path = (root / "annotations" / f"{name}.json").resolve()
        if name in ids or not path.is_relative_to(root / "annotations"):
            raise ValueError(f"Invalid/duplicate caption episode: {name}")
        ids.add(name)
        annotation = json.loads(path.read_text())
        elapsed, duration = annotation.get("elapsed_s"), annotation.get("duration_s")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
               for value in (elapsed, duration)):
            raise ValueError(f"{name}: missing/invalid elapsed_s or duration_s; refusing a partial average")
        item = {"id": name, "elapsed_s": elapsed, "video_s": duration,
                "segments": len(annotation["segments"])}
        timings.append(item)
        by_task.setdefault(row["task_name"], []).append(item)

    def summarize(items):
        times = [item["elapsed_s"] for item in items]
        seconds = sum(times)
        video_seconds = sum(item["video_s"] for item in items)
        return {"episodes": len(items), "mean_s": round(mean(times), 3),
                "median_s": round(median(times), 3), "min_s": min(times), "max_s": max(times),
                "total_s": round(seconds, 3), "mean_video_s": round(video_seconds / len(items), 3),
                "processing_s_per_video_s": round(seconds / video_seconds, 3),
                "segments": sum(item["segments"] for item in items)}

    return {"run": str(root), "group": group, **summarize(timings),
            "by_task": {task: summarize(items) for task, items in by_task.items()},
            "scope": "Official annotation elapsed_s: model loading, temporal segmentation and caption generation; "
                     "excludes entry media validation, final session cleanup and review-video export."}
