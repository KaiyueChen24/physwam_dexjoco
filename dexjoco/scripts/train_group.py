"""Prepare, normalize and train a single-arm or dual-arm DexJoCo task group."""
import contextlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from group_common import (Tee, build_train_config, load_trainer, resolve, training_parser,
                          validate_gpu, write_run, resolve_training_split)


def prepare_data(config, resolved, check_images=False):
    from openpi.training.dexjoco_multitask import MultiTaskDataset

    dataset = MultiTaskDataset(resolved["group"], config.model.action_horizon)
    manifest = {"group": resolved["group"]["name"], "data_source": "lerobot",
                "sampling": resolved["experiment"]["sampling"], "tasks": dataset.manifest,
                "normalization": "seeded frame sampling using the training task mixture; training episodes only",
                "data_split": resolved.get("data_split")}
    if check_images:
        for index, task in enumerate(resolved["group"]["tasks"]):
            offset = dataset.ends[index - 1] if index else 0
            sample = dataset[offset]
            print(task["name"], "state", sample["observation.state"].shape,
                  "action", sample["action"].shape, "prompt", sample["prompt"], flush=True)
    download_manifest = Path(resolved["group"]["dataset_root"]).parent / "download_manifest.json"
    if download_manifest.exists():
        manifest["download_provenance"] = json.loads(download_manifest.read_text())
    (Path(resolved["run_dir"]) / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    return dataset


def compute_stats(config, resolved, dataset):
    from openpi.shared import normalize

    exp = resolved["experiment"]
    rng = np.random.default_rng(exp["seed"])
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for start in range(0, exp["stats_samples"], exp["stats_batch_size"]):
        count = min(exp["stats_batch_size"], exp["stats_samples"] - start)
        indices = rng.choice(len(dataset), size=count, p=dataset.sampling_weights)
        frames = [dataset.low_dim(int(i)) for i in indices]
        for key in stats:
            batch = np.stack([frame[key] for frame in frames])
            if not np.isfinite(batch).all():
                raise ValueError(f"Nonfinite {key} in normalization samples")
            stats[key].update(batch)
    path = config.assets_dirs / resolved["group"]["name"]
    normalize.save(path, {key: value.get_statistics() for key, value in stats.items()})
    (path / "provenance.yaml").write_text(yaml.safe_dump({
        "group": resolved["group"], "samples": exp["stats_samples"], "seed": exp["seed"],
        "sampling": exp["sampling"], "method": "training frame mixture, one action per frame"}, sort_keys=False))
    print("Normalization statistics:", path, flush=True)


def main():
    parser = training_parser(__doc__)
    parser.add_argument("--stage", choices=("prepare", "stats", "train"), default="train", help='[训练] prepare 检查数据；stats 计算归一化；train 执行训练（缺统计时自动计算）。')
    args = parser.parse_args()
    resolved = resolve(args)
    if args.dry_run:
        print(yaml.safe_dump(resolved, sort_keys=False))
        return
    resolve_training_split(resolved)
    config = build_train_config(resolved, args.resume)
    if args.stage == "train":
        validate_gpu(config)
        if config.checkpoint_dir.exists() and not args.resume:
            raise FileExistsError("Training run already has checkpoints; use --resume or another run-name")
    run_dir = write_run(resolved)
    with (run_dir / "train.log").open("a", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print(f"Stage: {args.stage}, group: {resolved['group']['name']}", flush=True)
            dataset = prepare_data(config, resolved, check_images=args.stage == "prepare")
            if args.stage == "prepare":
                return
            stats_path = config.assets_dirs / resolved["group"]["name"] / "norm_stats.json"
            if args.stage == "stats":
                if args.resume:
                    raise ValueError("Do not recompute normalization statistics when resuming")
                compute_stats(config, resolved, dataset)
                return
            if not stats_path.is_file():
                if args.resume:
                    raise FileNotFoundError("Resume requires the run's original normalization assets")
                compute_stats(config, resolved, dataset)
            del dataset
            trainer = load_trainer()
            import logging
            logging.basicConfig(level=logging.INFO, force=True)
            trainer.main(config)


if __name__ == "__main__":
    main()
