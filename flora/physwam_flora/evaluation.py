"""Ground-truth labels are joined only after upstream inference completes."""
from __future__ import annotations

import math
from statistics import mean

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
