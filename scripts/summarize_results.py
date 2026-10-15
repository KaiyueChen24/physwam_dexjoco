"""Summarize one evaluation plan without silently averaging missing tasks."""
import argparse
import json
from pathlib import Path

from group_common import repo_path


def summarize(directory):
    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text())
    task_names = list(dict.fromkeys(job["task"] for job in plan["jobs"]))
    tasks = []
    for task in task_names:
        successes = episodes = 0
        missing = []
        latency = []
        for job in [j for j in plan["jobs"] if j["task"] == task]:
            path = directory / f"{task}_seed{job['seed']}" / "result.json"
            result = json.loads(path.read_text()) if path.exists() else {}
            if result.get("status") != "complete":
                missing.append(job["seed"])
                continue
            if result.get("checkpoint") != plan["server"]["checkpoint"] or result.get("episodes") != job["episodes"]:
                raise ValueError("Result provenance/count does not match evaluation plan")
            if not 0 <= result["successes"] <= result["episodes"]:
                raise ValueError("Invalid success count")
            successes += result["successes"]
            episodes += result["episodes"]
            if result.get("latency"):
                latency.append({"seed": job["seed"], **result["latency"]})
        tasks.append({"task": task, "successes": successes, "episodes": episodes,
                      "success_rate": successes / episodes if episodes else None,
                      "complete": not missing, "missing_or_failed_seeds": missing, "latency": latency})
    complete = all(t["complete"] for t in tasks)
    return {"group": plan["group"], "run_name": plan["run_name"], "checkpoint": plan["server"]["checkpoint"],
            "complete": complete, "benchmark_eligible": not (plan["server"].get("synthetic") or plan["server"].get("tiny")),
            "evaluated_tasks": task_names,
            "all_group_tasks_evaluated": set(task_names) == set(plan["server"]["tasks"]),
            "task_mean_success_rate": sum(t["success_rate"] for t in tasks) / len(tasks) if complete else None,
            "tasks": tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True)
    args = parser.parse_args()
    directory = repo_path(args.evaluation_dir)
    result = summarize(directory)
    (directory / "summary.json").write_text(json.dumps(result, indent=2))
    lines = ["task,successes,episodes,success_rate,complete"]
    for task in result["tasks"]:
        rate = "" if task["success_rate"] is None else str(task["success_rate"])
        lines.append(f"{task['task']},{task['successes']},{task['episodes']},{rate},{task['complete']}")
    (directory / "summary.csv").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))
    if not result["complete"]:
        raise SystemExit("Evaluation is incomplete; task mean has not been reported")


if __name__ == "__main__":
    main()
