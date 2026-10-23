"""Summarize one evaluation plan without silently averaging missing tasks."""
import argparse
import json
import statistics
from pathlib import Path

from group_common import repo_path


def summarize(directory):
    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text())
    task_names = list(dict.fromkeys(job["task"] for job in plan["jobs"]))
    gl_backend = plan.get("gl_backend", "egl")
    record_video = plan.get("record_video", True)
    regime = plan.get("data_regime", "rand_full" if plan.get("rand_full") else "rand_obj")
    tasks = []
    for task in task_names:
        successes = episodes = 0
        missing = []
        latency = []
        seed_results = []
        for job in [j for j in plan["jobs"] if j["task"] == task]:
            path = directory / f"{task}_seed{job['seed']}" / "result.json"
            result = json.loads(path.read_text()) if path.exists() else {}
            if result.get("status") != "complete":
                missing.append(job["seed"])
                continue
            if result.get("checkpoint") != plan["server"]["checkpoint"] or result.get("episodes") != job["episodes"]:
                raise ValueError("Result provenance/count does not match evaluation plan")
            if result.get("gl_backend", "egl") != gl_backend:
                raise ValueError("Result rendering backend does not match evaluation plan")
            if result.get("record_video", True) != record_video:
                raise ValueError("Result video recording setting does not match evaluation plan")
            if not 0 <= result["successes"] <= result["episodes"]:
                raise ValueError("Invalid success count")
            result_regime = result.get("data_regime", "rand_full" if result.get("rand_full", "--rand-full" in result.get("command", [])) else "rand_obj")
            if result_regime != regime:
                raise ValueError("Result randomization regime does not match evaluation plan")
            successes += result["successes"]
            episodes += result["episodes"]
            seed_results.append({"seed": job["seed"], "successes": result["successes"],
                                 "episodes": result["episodes"],
                                 "success_rate": result["successes"] / result["episodes"]})
            if result.get("latency"):
                latency.append({"seed": job["seed"], **result["latency"]})
        tasks.append({"task": task, "successes": successes, "episodes": episodes,
                      "success_rate": successes / episodes if episodes else None,
                      "complete": not missing, "missing_or_failed_seeds": missing, "latency": latency,
                      "seed_results": seed_results,
                      "success_rate_mean": statistics.mean(r["success_rate"] for r in seed_results)
                          if not missing and seed_results else None,
                      "success_rate_std": statistics.stdev(r["success_rate"] for r in seed_results)
                          if not missing and len(seed_results) > 1 else None})
    complete = all(t["complete"] for t in tasks)
    seeds = list(dict.fromkeys(job["seed"] for job in plan["jobs"]))
    seed_means = []
    for seed in seeds:
        rates = [next((r["success_rate"] for r in task["seed_results"] if r["seed"] == seed), None)
                 for task in tasks]
        if rates and all(rate is not None for rate in rates):
            seed_means.append({"seed": seed, "success_rate": statistics.mean(rates)})
    seed_stats_complete = complete and len(seed_means) == len(seeds) and bool(seed_means)
    return {"group": plan["group"], "run_name": plan["run_name"], "checkpoint": plan["server"]["checkpoint"],
            "gl_backend": gl_backend, "record_video": record_video, "data_regime": regime, "regime": regime, "rand_full": regime == "rand_full", "complete": complete, "benchmark_eligible": not (plan["server"].get("synthetic") or plan["server"].get("tiny")),
            "evaluated_tasks": task_names,
            "all_group_tasks_evaluated": set(task_names) == set(plan["server"]["tasks"]),
            "task_mean_success_rate": sum(t["success_rate"] for t in tasks) / len(tasks) if complete else None,
            "seed_mean_success_rates": seed_means,
            "seed_mean_success_rate": statistics.mean(r["success_rate"] for r in seed_means)
                if seed_stats_complete else None,
            "seed_mean_success_rate_std": statistics.stdev(r["success_rate"] for r in seed_means)
                if seed_stats_complete and len(seed_means) > 1 else None,
            "success_rate_std_ddof": 1,
            "tasks": tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True, help='[闭环评测] 包含 plan.json 和逐任务结果的评测目录，汇总写回该目录。')
    args = parser.parse_args()
    directory = repo_path(args.evaluation_dir)
    result = summarize(directory)
    (directory / "summary.json").write_text(json.dumps(result, indent=2))
    lines = ["task,successes,episodes,success_rate,complete,success_rate_mean,success_rate_std"]
    for task in result["tasks"]:
        rate = "" if task["success_rate"] is None else str(task["success_rate"])
        mean = "" if task["success_rate_mean"] is None else str(task["success_rate_mean"])
        std = "" if task["success_rate_std"] is None else str(task["success_rate_std"])
        lines.append(f"{task['task']},{task['successes']},{task['episodes']},{rate},{task['complete']},{mean},{std}")
    (directory / "summary.csv").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))
    if not result["complete"]:
        raise SystemExit("Evaluation is incomplete; task mean has not been reported")


if __name__ == "__main__":
    main()
