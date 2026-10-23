"""Check evaluation uncertainty across seeds, including incomplete results."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_results import summarize


def write_plan(directory, counts, *, missing=None):
    plan = {"group": "single_arm", "run_name": "test", "server": {
        "checkpoint": "checkpoint", "tasks": list(counts), "data_source": "lerobot"},
        "jobs": [{"task": task, "seed": seed, "episodes": 50}
                 for task, values in counts.items() for seed in range(len(values))]}
    (directory / "plan.json").write_text(json.dumps(plan))
    for task, values in counts.items():
        for seed, successes in enumerate(values):
            if (task, seed) == missing:
                continue
            job = directory / f"{task}_seed{seed}"
            job.mkdir()
            (job / "result.json").write_text(json.dumps({
                "status": "complete", "checkpoint": "checkpoint",
                "successes": successes, "episodes": 50}))


def test_overall_std_is_across_seed_task_means_not_average_task_std(tmp_path):
    write_plan(tmp_path, {"a": [10, 20, 30], "b": [40, 30, 20]})
    result = summarize(tmp_path)
    a, b = result["tasks"]
    assert a["success_rate_mean"] == pytest.approx(0.4)
    assert b["success_rate_mean"] == pytest.approx(0.6)
    assert a["success_rate_std"] == pytest.approx(0.2)
    assert b["success_rate_std"] == pytest.approx(0.2)
    assert result["seed_mean_success_rate"] == pytest.approx(0.5)
    assert result["seed_mean_success_rate_std"] == pytest.approx(0.0)
    assert result["success_rate_std_ddof"] == 1
    assert result["task_mean_success_rate"] == pytest.approx(0.5)


def test_missing_seed_does_not_produce_final_statistics(tmp_path):
    write_plan(tmp_path, {"a": [10, 20, 30], "b": [40, 30, 20]}, missing=("b", 2))
    result = summarize(tmp_path)
    assert not result["complete"]
    assert result["tasks"][1]["success_rate_mean"] is None
    assert result["tasks"][1]["success_rate_std"] is None
    assert result["tasks"][1]["missing_or_failed_seeds"] == [2]
    assert result["seed_mean_success_rate"] is None
    assert result["seed_mean_success_rate_std"] is None


def test_one_seed_does_not_claim_zero_uncertainty(tmp_path):
    write_plan(tmp_path, {"a": [10], "b": [40]})
    result = summarize(tmp_path)
    assert result["complete"]
    assert result["tasks"][0]["success_rate_mean"] == pytest.approx(0.2)
    assert result["tasks"][0]["success_rate_std"] is None
    assert result["seed_mean_success_rate_std"] is None
