"""Serve a task-group checkpoint using the exact saved architecture and assets."""
import argparse
import json
import logging
import os

from group_common import build_train_config, latest_checkpoint, read_run, repo_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", help="Completed checkpoint directory; default latest in this run")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
