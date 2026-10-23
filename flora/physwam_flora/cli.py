from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .common import load_config, read_manifest, verify_vendor, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="DexJoCo offline evaluation with pinned official FLORA code")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify-sources", help="Verify the bundled upstream SHA-256 manifest")
    prepare = commands.add_parser("prepare", help="Build an episode manifest from LeRobot or recorded rollout data")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--output", required=True)
    inspect = commands.add_parser("inspect", help="CPU decode/synchronization checks, no network/model loading")
    inspect.add_argument("--manifest", required=True)
    inspect.add_argument("--stage", choices=["terminal", "dense", "rubric", "caption"], default="terminal")
    inspect.add_argument("--caption-views", choices=["auto", "main", "three"], default="auto")
    run = commands.add_parser("run", help="Invoke an official backend in an isolated subprocess")
    run.add_argument("--config", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--stage", choices=["terminal", "dense", "rubric", "caption"], required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--gpus")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print commands only; no decoding, network, CUDA or output writes")
    mode.add_argument("--native-check", action="store_true", help="Official CPU input/prompt check; Rewarder may download processor files")
    summary = commands.add_parser("summarize", help="Evaluate terminal labels or report dense parse failures")
    summary.add_argument("--predictions", required=True)
    summary.add_argument("--kind", choices=["terminal", "dense"], default="terminal")
    summary.add_argument("--labels")
    summary.add_argument("--threshold", type=float, default=5.0)
    summary.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-sources":
            result = verify_vendor()
        elif args.command == "prepare":
            from .datasets import prepare

            result = prepare(load_config(args.config), args.output)
        elif args.command == "inspect":
            from .media import inspect_episode

            result = [inspect_episode(row, args.stage, args.caption_views) for row in read_manifest(args.manifest)]
        elif args.command == "run":
            from .runner import run

            result = run(load_config(args.config), args.manifest, args.output, args.stage,
                         gpus=args.gpus, dry_run=args.dry_run, native_check=args.native_check)
        else:
            from .evaluation import dense_quality, terminal_metrics

            if args.kind == "terminal" and not args.labels:
                raise ValueError("terminal summary requires --labels")
            result = terminal_metrics(args.predictions, args.labels, args.threshold) if args.kind == "terminal" else dense_quality(args.predictions)
            if args.output:
                write_json(Path(args.output), result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f"FLORA: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
