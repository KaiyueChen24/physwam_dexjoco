"""Evaluate every task with one shared policy and persist resumable per-job results."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import uuid

import numpy as np
import yaml

from group_common import ROOT, read_run, repo_path


def parse_success(directory, expected_episodes):
    markers = list(Path(directory).glob("success_rate_*.txt"))
    if len(markers) != 1:
        raise ValueError("Expected exactly one completed success-rate marker")
    match = re.fullmatch(r"success_rate_(\d+)_(\d+)\.txt", markers[0].name)
    if not match:
        raise ValueError("Malformed success-rate marker")
    successes, episodes = map(int, match.groups())
    if episodes != expected_episodes or not 0 <= successes <= episodes:
        raise ValueError("Incomplete evaluation or invalid success count")
    return successes, episodes


def latency_summary(path):
    if not path.is_file():
        return None
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # One initial chunk is excluded from steady-state latency; its timing remains in JSONL.
    steady = records[1:]
    result = {"chunks_total": len(records), "warmup_chunks_excluded": 1 if records else 0}
    for key in ("roundtrip_ms", "server_infer_ms"):
        values = [r[key] for r in steady if r.get(key) is not None]
        if values:
            result[key] = {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
                           "p95": float(np.percentile(values, 95))}
    return result


def rendering_environment(backend):
    if backend not in {"egl", "osmesa"}:
        raise ValueError(f"Unsupported rendering backend: {backend}")
    # Set before the child imports MuJoCo/PyOpenGL; both variables must agree.
    return {**os.environ, "MUJOCO_GL": backend, "PYOPENGL_PLATFORM": backend}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help='[配置与实验] 已有实验目录，从中读取配置快照和资产。')
    parser.add_argument("--group", choices=("single_arm", "dual_arm"), help='[任务选择] 选择单臂或双臂任务组。')
    parser.add_argument("--host", default="127.0.0.1", help='[部署与连接] 策略服务地址。')
    parser.add_argument("--port", type=int, default=8000, help='[部署与连接] 策略服务端口。')
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help='[闭环评测] 评测环境随机种子列表；与训练 seed 分开设置。')
    parser.add_argument("--episodes", type=int, default=50, help='[闭环评测] 每个任务、每个 seed 的评测回合数。')
    parser.add_argument("--tasks", nargs="+", help='[任务选择] 仅选择列出的任务；默认使用所选任务组或快照中的全部任务。')
    parser.add_argument("--evaluation-name", default="rand_obj", help='[闭环评测] 本次评测结果子目录名；仅命名，不会自动切换随机化。')
    parser.add_argument("--gl-backend", choices=("egl", "osmesa"), default="egl",
                        help="[评测/渲染] egl 使用 GPU；osmesa 使用 CPU 软件渲染，供 EGL 故障诊断。")
    parser.add_argument("--timeout", type=float, default=7200, help='[闭环评测] 每个任务与 seed 组合的进程超时，单位秒。')
    parser.add_argument("--retry-failed", action="store_true", help='[闭环评测] 重跑已记录失败的评测任务；已完成任务继续跳过。')
    parser.add_argument("--dry-run", action="store_true", help='[配置与实验] 仅打印配置或计划，不启动训练、推理或评测。')
    randomization = parser.add_mutually_exclusive_group()
    randomization.add_argument("--data-regime", choices=("rand_obj", "rand_full"),
                               help="[评测/随机化] 默认沿用训练快照；可显式覆盖以做跨条件测试。")
    randomization.add_argument("--rand-full", dest="data_regime", action="store_const", const="rand_full",
                               help="[评测/随机化] 显式启用 rand_full，等同 --data-regime rand_full。")
    args = parser.parse_args()
    if args.episodes <= 0 or args.timeout <= 0 or len(args.seeds) != len(set(args.seeds)):
        parser.error("Episodes/timeout must be positive and seeds must be unique")
    if Path(args.evaluation_name).name != args.evaluation_name or args.evaluation_name in (".", ".."):
        parser.error("evaluation-name must be a simple directory name")
    resolved = read_run(args.run_dir)
    group = resolved["group"]
    trained_regime = group.get("data_regime", resolved.get("experiment", {}).get("data_regime", "rand_obj"))
    regime = args.data_regime or trained_regime
    if regime not in ("rand_obj", "rand_full"):
        raise ValueError("Unknown evaluation data regime")
    if args.group and args.group != group["name"]:
        parser.error("Requested group does not match training run")
    available = {t["name"] for t in group["tasks"]}
    if args.tasks and not set(args.tasks) <= available:
        parser.error("Requested task is outside this group")
    tasks = [t for t in group["tasks"] if not args.tasks or t["name"] in args.tasks]
    jobs = [{"task": t["name"], "seed": s, "episodes": args.episodes} for t in tasks for s in args.seeds]
    if args.dry_run:
        print(json.dumps({"group": group["name"], "gl_backend": args.gl_backend, "data_regime": regime, "rand_full": regime == "rand_full", "jobs": jobs}, indent=2))
        return
    executable = shutil.which("dexjoco-openpi-eval")
    if not executable:
        parser.error("Activate dexjoco environment; dexjoco-openpi-eval is missing")
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect

    with connect(f"ws://{args.host}:{args.port}", open_timeout=15, max_size=None) as socket:
        metadata = msgpack_numpy.unpackb(socket.recv(timeout=15))
    if metadata.get("group") != group["name"] or metadata.get("run_name") != resolved["run_name"]:
        raise ValueError("Server is serving a different group/run; start serve_group.py for this run")
    if metadata.get("action_horizon") != 30:
        raise ValueError("The current DexJoCo evaluator requires action_horizon=30")
    if metadata.get("synthetic") or metadata.get("tiny"):
        raise ValueError("Synthetic/tiny checkpoints cannot be used for benchmark reporting")
    evaluation_dir = Path(resolved["run_dir"]) / "evaluation" / args.evaluation_name
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    plan = {"group": group["name"], "run_name": resolved["run_name"], "server": metadata, "jobs": jobs, "gl_backend": args.gl_backend, "data_regime": regime, "rand_full": regime == "rand_full"}
    plan_path = evaluation_dir / "plan.json"
    if plan_path.exists():
        old_plan = json.loads(plan_path.read_text())
        old_plan.setdefault("gl_backend", "egl")  # Legacy plans always forced EGL.
        old_regime = old_plan.get("data_regime", "rand_full" if old_plan.get("rand_full") else "rand_obj")
        old_plan.setdefault("data_regime", old_regime)
        old_plan.setdefault("rand_full", old_regime == "rand_full")
        if old_plan != plan:
            raise ValueError("Evaluation plan/checkpoint/rendering backend changed; use a new evaluation-name")
    plan_path.write_text(json.dumps(plan, indent=2))
    failures = []
    for job in jobs:
        task = next(t for t in tasks if t["name"] == job["task"])
        job_dir = evaluation_dir / f"{job['task']}_seed{job['seed']}"
        job_dir.mkdir(exist_ok=True)
        result_path = job_dir / "result.json"
        if result_path.exists():
            old = json.loads(result_path.read_text())
            if old.get("status") == "complete":
                print("Skip completed:", job, flush=True)
                continue
            if not args.retry_failed:
                failures.append(job)
                print("Skip failed job (use --retry-failed):", job, flush=True)
                continue
        attempt = job_dir / f"attempt_{uuid.uuid4().hex[:8]}"
        attempt.mkdir()
        config_source = task["eval_config"] if regime == trained_regime else f"configs/{regime}/{task['name']}.yaml"
        evaluation_config = yaml.safe_load(repo_path(config_source).read_text())
        if evaluation_config["env_name"] != task["name"] or evaluation_config["robot_type"] != group["robot_type"]:
            raise ValueError("Evaluation task/robot type does not match group")
        if regime == trained_regime:
            evaluation_config["camera_mapping"] = task["cameras"]
        if set(evaluation_config["camera_mapping"]) != set(task["cameras"]):
            raise ValueError("Evaluation camera keys do not match the policy")
        if regime == "rand_full" and evaluation_config["camera_mapping"].get("base") != "random_camera":
            raise ValueError("rand_full requires random_camera for the main view")
        evaluation_config["prompt"] = task["prompt"]
        config_path = attempt / "environment.yaml"
        config_path.write_text(yaml.safe_dump(evaluation_config, sort_keys=False))
        output = attempt / "videos"
        latency_path = attempt / "latency.jsonl"
        command = [executable, "--config", str(config_path), "--host", args.host, "--port", str(args.port),
                   "--seed", str(job["seed"]), "--episodes", str(args.episodes),
                   "--output", str(output), "--metrics-output", str(latency_path),
                   "--render-mode", "rgb_array"]
        if regime == "rand_full":
            command.append("--rand-full")
        result = {**job, "group": group["name"], "checkpoint": metadata["checkpoint"],
                  "data_regime": regime, "rand_full": regime == "rand_full",
                  "data_source": "lerobot", "gl_backend": args.gl_backend,
                  "command": command, "output": str(output), "status": "failed"}
        print("Evaluate:", job, "log:", attempt / "eval.log", flush=True)
        try:
            with (attempt / "eval.log").open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=rendering_environment(args.gl_backend),
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    returncode = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)
            successes, episodes = parse_success(output, args.episodes)
            result.update(status="complete", successes=successes, episodes=episodes,
                          success_rate=successes / episodes, latency=latency_summary(latency_path))
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            result["error"] = str(exc)
            failures.append(job)
        result_path.write_text(json.dumps(result, indent=2))
        print(result["status"], job, flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed or remain failed; inspect per-job logs")


if __name__ == "__main__":
    main()
