"""Serve a task-group checkpoint using the exact saved architecture and assets."""
import argparse
import json
import logging
import os

from group_common import build_train_config, latest_checkpoint, read_run, repo_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help='[配置与实验] 已有实验目录，从中读取配置快照和资产。')
    parser.add_argument("--checkpoint", help='[权重与结果保存] 完整 π0.5 checkpoint 目录；默认使用本次实验最新的完整保存点。')
    parser.add_argument("--gpus", default="0", help='[GPU 与并行] 可见 GPU 编号，逗号分隔，例如 0,1；不是 GPU 数量。')
    parser.add_argument("--host", default="127.0.0.1", help='[部署与连接] 服务监听地址；127.0.0.1 为本机，0.0.0.0 监听所有接口。')
    parser.add_argument("--port", type=int, default=8000, help='[部署与连接] 策略服务端口。')
    parser.add_argument("--allow-smoke-checkpoint", action="store_true", help='[闭环评测] 允许合成或 tiny 权重用于流程检查，不能用于报告正式成绩。')
    parser.add_argument("--dry-run", action="store_true", help='[配置与实验] 仅打印配置或计划，不启动训练、推理或评测。')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    resolved = read_run(args.run_dir)
    if (resolved["experiment"]["synthetic"] or resolved["experiment"].get("tiny")) and not args.allow_smoke_checkpoint:
        parser.error("Synthetic/tiny checkpoint is for pipeline testing; use --allow-smoke-checkpoint explicitly")
    config = build_train_config(resolved)
    checkpoint = repo_path(args.checkpoint) if args.checkpoint else latest_checkpoint(config)
    if not (checkpoint / "params").is_dir() or not (checkpoint / "assets").is_dir():
        raise FileNotFoundError(f"Checkpoint must contain params and assets: {checkpoint}")
    metadata = {**config.policy_metadata, "checkpoint": str(checkpoint), "run_name": resolved["run_name"]}
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    logging.basicConfig(level=logging.INFO, force=True)
    policy = create_trained_policy(config, checkpoint)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
