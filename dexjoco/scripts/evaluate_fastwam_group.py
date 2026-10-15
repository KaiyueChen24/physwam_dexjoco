"""Evaluate the saved FastWAM task snapshot, keeping the pi05 result schema."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

import yaml

from fastwam_common import ROOT, add_fastwam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help='[配置与实验] 已有实验目录，从中读取配置快照和资产。')
    parser.add_argument("--fastwam-root", help='[配置与实验] FastWAM 源码目录；默认使用 DexJoCo 的同级 fastwam/。')
    parser.add_argument("--tasks", nargs="+", help='[任务选择] 仅选择列出的任务；默认使用所选任务组或快照中的全部任务。')
    parser.add_argument("--host", default="127.0.0.1", help='[部署与连接] 策略服务地址。')
    parser.add_argument("--port", type=int, default=8001, help='[部署与连接] 策略服务端口。')
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help='[闭环评测] 评测环境随机种子列表；与训练 seed 分开设置。')
    parser.add_argument("--episodes", type=int, default=50, help='[闭环评测] 每个任务、每个 seed 的评测回合数。')
    parser.add_argument("--max-steps", type=int, default=3000, help='[闭环评测] 每个回合最多执行的环境动作步数，不是训练更新步数。')
    parser.add_argument("--execute-steps", type=int, help='[闭环评测] 每次预测后执行多少步再观察；必须不大于 action_horizon。')
    parser.add_argument("--evaluation-name", default="rand_obj", help='[闭环评测] 本次评测结果子目录名；仅命名，不会自动切换随机化。')
    parser.add_argument("--rand-full", action="store_true", help='[闭环评测] 启用环境的完整视觉/场景随机化；不改变训练数据。')
    parser.add_argument("--timeout", type=float, default=86400, help='[闭环评测] 每个任务与 seed 组合的进程超时，单位秒。')
    parser.add_argument("--retry-failed", action="store_true", help='[闭环评测] 重跑已记录失败的评测任务；已完成任务继续跳过。')
    parser.add_argument("--dry-run", action="store_true", help='[配置与实验] 仅打印配置或计划，不启动训练、推理或评测。')
    args = parser.parse_args()
    add_fastwam(args.fastwam_root)
    from experiments.dexjoco.config import read_run, signature
    run_dir, config = read_run(args.run_dir)
    group = config["framework"]["group"]
    exp = config["framework"]["experiment"]
    if (min(args.episodes, args.max_steps, args.timeout) <= 0 or len(args.seeds) != len(set(args.seeds))
            or Path(args.evaluation_name).name != args.evaluation_name or args.evaluation_name in (".", "..")):
        parser.error("Invalid episode/step/timeout/seed/evaluation-name settings")
    if args.tasks and not set(args.tasks) <= {t["name"] for t in group["tasks"]}:
        parser.error("Requested task not in training snapshot")
    steps = args.execute_steps if args.execute_steps is not None else exp["execute_steps"]
    if not 0 < steps <= exp["action_horizon"]:
        parser.error("execute-steps must be within action horizon")
    tasks = [t for t in group["tasks"] if not args.tasks or t["name"] in args.tasks]
    jobs = [{"task": t["name"], "seed": seed, "episodes": args.episodes} for t in tasks for seed in args.seeds]
    if args.dry_run:
        print(json.dumps({"jobs": jobs, "execute_steps": steps, "action_horizon": exp["action_horizon"]}, indent=2))
        return
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect
    from evaluate_group import parse_success, latency_summary
    with connect(f"ws://{args.host}:{args.port}", open_timeout=15, max_size=None) as socket:
        metadata = msgpack_numpy.unpackb(socket.recv(timeout=15))
    if metadata.get("model") != "fastwam" or metadata.get("config_signature") != signature(config):
        raise ValueError("Server model/config does not match this FastWAM run")
    if metadata.get("stats_sha256") != hashlib.sha256((run_dir / "dataset_stats.json").read_bytes()).hexdigest():
        raise ValueError("Server normalization statistics differ from this run")
    destination = run_dir / "evaluation" / args.evaluation_name
    destination.mkdir(parents=True, exist_ok=True)
    plan = {"group": group["name"], "run_name": config["framework"]["run_name"], "server": metadata,
            "jobs": jobs, "execute_steps": steps, "max_steps": args.max_steps, "rand_full": args.rand_full}
    plan_path = destination / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Plan/checkpoint/settings changed; use a new evaluation-name")
    plan_path.write_text(json.dumps(plan, indent=2))
    failures = []
    for job in jobs:
        folder = destination / f"{job['task']}_seed{job['seed']}"
        folder.mkdir(exist_ok=True)
        result_path = folder / "result.json"
        if result_path.exists():
            old = json.loads(result_path.read_text())
            if old.get("status") == "complete":
                continue
            if not args.retry_failed:
                failures.append(job)
                continue
        attempt = folder / uuid.uuid4().hex[:10]
        attempt.mkdir()
        task = next(t for t in tasks if t["name"] == job["task"])
        (attempt / "environment.yaml").write_text(yaml.safe_dump(task["environment"], sort_keys=False))
        (attempt / "server.json").write_text(json.dumps(metadata))
        command = [sys.executable, "-m", "dexjoco_fastwam_client.evaluate", "--config", str(attempt / "environment.yaml"),
                   "--metadata", str(attempt / "server.json"), "--output", str(attempt / "videos"),
                   "--host", args.host, "--port", str(args.port), "--seed", str(job["seed"]),
                   "--episodes", str(args.episodes), "--max-steps", str(args.max_steps), "--execute-steps", str(steps)]
        if args.rand_full:
            command.append("--rand-full")
        env = dict(os.environ, MUJOCO_GL="egl")
        env["PYTHONPATH"] = str(ROOT / "dexjoco") + os.pathsep + env.get("PYTHONPATH", "")
        result = {**job, "checkpoint": metadata["checkpoint"], "status": "failed", "command": command}
        try:
            with (attempt / "eval.log").open("w") as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    returncode = process.wait(timeout=args.timeout)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)
            successes, count = parse_success(attempt / "videos", args.episodes)
            result.update(status="complete", successes=successes, episodes=count,
                          success_rate=successes / count, latency=latency_summary(attempt / "videos/latency.jsonl"))
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            result["error"] = str(exc)
            failures.append(job)
        result_path.write_text(json.dumps(result, indent=2))
        print(job, result["status"], flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} evaluation jobs failed; inspect eval.log")


if __name__ == "__main__":
    main()
