"""Prepare and train task groups with the official FastWAM trainer."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from dataset_split import save_split, split_group, validation_settings

from fastwam_common import ROOT, add_fastwam, resolve_launch_options


def prepare(config, dexjoco_root, fastwam_root):
    from experiments.dexjoco.config import git_version, signature
    from experiments.dexjoco.data import compute_stats, inspect_task

    # [验证/loss] 先固定 episode 划分，再检查数据和计算仅含训练帧的统计。
    group = config["framework"]["group"]
    options = validation_settings(config["framework"]["experiment"])
    train, validation, split = split_group(group, options)
    config["framework"]["group"] = group = train
    config["data"]["train"]["tasks"] = train["tasks"]
    if validation is not None:
        config["framework"]["validation_group"] = validation
        config["framework"]["data_split"] = split
        import copy
        config["data"]["val"] = copy.deepcopy(config["data"]["train"])
        config["data"]["val"].update(tasks=validation["tasks"], sampling="frame_uniform")
    config["validation_loss"] = {"enabled": validation is not None, "interval": options["interval"],
                                 "seed": options["seed"], "batch_size": options["batch_size"],
                                 "max_batches": options["max_batches"], "sampling": config["framework"]["experiment"]["sampling"]}
    manifests = [inspect_task(t, group["state_dim"], group["action_dim"]) for t in group["tasks"]]
    if any(abs(m["fps"] - config["framework"]["experiment"]["control_hz"]) > 1e-6 for m in manifests):
        raise ValueError("Dataset fps differs from control_hz; resample explicitly before training")
    run_dir = Path(config["output_dir"])
    if (run_dir / "fastwam_config.yaml").exists():
        old = yaml.safe_load((run_dir / "fastwam_config.yaml").read_text())
        if signature(old) != signature(config):
            raise ValueError("Run config changed; use a new run-name")
        manifest_path = run_dir / "dataset_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifests:
            raise ValueError("Dataset metadata or parquet files changed; use a new run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_split(run_dir, split)
    (run_dir / "fastwam_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config["framework"], sort_keys=False))
    (run_dir / "dataset_manifest.json").write_text(json.dumps(manifests, indent=2))
    versions = run_dir / "versions.json"
    if not versions.exists():
        versions.write_text(json.dumps({"fastwam": git_version(fastwam_root),
                                       "dexjoco": git_version(dexjoco_root), "python": sys.version}, indent=2))
    stats = run_dir / "dataset_stats.json"
    if not stats.exists():
        value = compute_stats(group["tasks"], group["state_dim"], group["action_dim"])
        temp = stats.with_suffix(".tmp")
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(stats)
    checksum = hashlib.sha256(stats.read_bytes()).hexdigest()
    checksum_path = run_dir / "dataset_stats.sha256"
    if checksum_path.exists() and checksum_path.read_text().strip() != checksum:
        raise ValueError("Normalization statistics changed inside an existing run")
    checksum_path.write_text(checksum + "\n")
    (run_dir / "model_init.yaml").write_text(yaml.safe_dump(config["model"], sort_keys=False))
    print(f"Prepared: {run_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("single_arm", "dual_arm"), help='[任务选择] 覆盖实验 YAML 的 group，选择单臂或双臂任务组。')
    parser.add_argument("--run-name", required=True, help='[配置与实验] 实验目录名，结果写入 runs/<名称>/。')
    parser.add_argument("--fastwam-root", help='[配置与实验] FastWAM 源码目录；默认使用 DexJoCo 的同级 fastwam/。')
    parser.add_argument("--dataset-root", help='[数据与 CPU] 覆盖任务数据根目录；相对路径以 DexJoCo 仓库根目录为准。')
    parser.add_argument("--experiment", default="configs/experiments/fastwam_single.yaml", help='[配置与实验] 实验 YAML 路径；默认八卡单臂六任务联合训练配方，相对路径以 DexJoCo 仓库根目录为准。')
    parser.add_argument("--tasks", nargs="+", help='[任务选择] 覆盖实验 YAML 的 tasks；YAML 中 tasks 为 null 时使用全部任务。')
    parser.add_argument("--mode", choices=("joint", "per_task"), default="joint", help='[训练] joint 为组内共享一个策略；per_task 为逐任务分别训练。')
    parser.add_argument("--stage", choices=("prepare", "inspect", "initialize", "train"), default="prepare", help='[训练] prepare 检查与统计；inspect 检查序列；initialize 构造动作骨干；train 启动官方训练器。')
    parser.add_argument("--gpus", help='[GPU 与并行] 覆盖实验 YAML 的 gpus，例如 4,5,6,7；默认从配方读取。')
    parser.add_argument("--batch-size", type=int, help='[训练] 覆盖 YAML 的每张 GPU batch；有效全局 batch 还需乘卡数和梯度累积。')
    parser.add_argument("--steps", type=int, help='[训练] 覆盖 YAML 的优化器更新总步数，不是 epoch。')
    parser.add_argument("--resume", help='[权重与结果保存] 显式指定本次实验 checkpoints/state/step_XXXXXX 完整状态目录；仅支持 joint 的 train 阶段。')
    parser.add_argument("--dry-run", action="store_true", help='[配置与实验] 仅打印配置或计划，不启动训练、推理或评测。')
    args = parser.parse_args()
    if args.resume and (args.stage != "train" or args.mode != "joint"):
        parser.error("--resume requires --stage train --mode joint")
    fastwam_root = add_fastwam(args.fastwam_root)
    from experiments.dexjoco.config import build_config, read_yaml, select_group
    experiment_path = Path(args.experiment).expanduser()
    experiment = read_yaml(experiment_path if experiment_path.is_absolute() else ROOT / experiment_path)
    if experiment.get("model") != "fastwam" or experiment.get("schema_version") != 1:
        raise ValueError("Expected a FastWAM experiment")
    try:
        args.group, args.tasks, args.gpus = resolve_launch_options(args.group, args.tasks, args.gpus, experiment)
    except ValueError as error:
        parser.error(str(error))
    # Record actual CLI/YAML selections in the run snapshot.
    experiment.update(group=args.group, tasks=args.tasks, gpus=args.gpus)
    validation_settings(experiment)
    for arg, key in ((args.batch_size, "batch_size"), (args.steps, "max_steps")):
        if arg is not None:
            experiment[key] = arg
    selections = [args.tasks]
    if args.mode == "per_task":
        if args.stage == "initialize":
            parser.error("Initialization is shared; use --mode joint")
        selections = [[t["name"]] for t in select_group(
            ROOT, args.group, args.tasks, args.dataset_root or experiment.get("dataset_root"),
            data_regime=experiment.get("data_regime", "rand_obj"))["tasks"]]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(fastwam_root / "checkpoints"))
    os.environ["PYTHONPATH"] = os.pathsep.join([str(fastwam_root), str(fastwam_root / "src"), os.environ.get("PYTHONPATH", "")])
    for selected in selections:
        name = args.run_name + ("_" + selected[0] if args.mode == "per_task" else "")
        config = build_config(ROOT, args.group, experiment, name, task_names=selected, dataset_root=args.dataset_root)
        config["framework"]["launch"] = {
            "gpus": args.gpus, "num_processes": len(args.gpus.split(",")),
            "effective_global_batch_size": config["batch_size"] * len(args.gpus.split(",")) * config["gradient_accumulation_steps"],
        }
        if args.dry_run:
            print(yaml.safe_dump(config, sort_keys=False))
            continue
        run_dir = Path(config["output_dir"])
        if args.resume:
            resume = Path(args.resume).expanduser().resolve()
            if not resume.is_dir() or not (resume / "trainer_state.json").is_file() or not resume.is_relative_to(run_dir / "checkpoints/state"):
                raise ValueError("Resume must name this run's complete trainer-state directory")
            if not (run_dir / "dataset_stats.json").is_file():
                raise FileNotFoundError("Resume requires the original normalization statistics")
        elif args.stage == "train" and (run_dir / "checkpoints").exists():
            raise FileExistsError("Checkpoints already exist; use --resume or another run-name")
        prepare(config, ROOT, fastwam_root)
        if args.stage == "prepare":
            continue
        if args.stage == "inspect":
            from experiments.dexjoco.inspect_dataset import inspect
            inspect(run_dir)
            continue
        if args.stage == "initialize":
            output = Path(config["model"]["action_dit_pretrained_path"])
            if output.exists():
                raise FileExistsError(f"Initialization already exists: {output}")
            command = [sys.executable, str(fastwam_root / "scripts/preprocess_action_dit_backbone.py"),
                       "--model-config", str(run_dir / "model_init.yaml"), "--output", str(output),
                       "--device", "cuda", "--dtype", "bfloat16"]
        else:
            if not Path(config["model"]["action_dit_pretrained_path"]).is_file():
                raise FileNotFoundError("Prepare the ActionDiT backbone with --stage initialize first")
            accelerate = shutil.which("accelerate")
            if not accelerate:
                raise RuntimeError("Activate the FastWAM environment; accelerate is missing")
            command = [accelerate, "launch", "--config_file", "scripts/accelerate_configs/accelerate_zero1_ds.yaml",
                       "--num_processes", str(len(args.gpus.split(","))), "scripts/train.py",
                       "--config-path", str(run_dir), "--config-name", "fastwam_config"]
            if args.resume:
                command.append("resume=" + str(resume))
        print("Launch:", command, flush=True)
        with (run_dir / (args.stage + ".log")).open("a") as log:
            subprocess.run(command, cwd=fastwam_root, check=True, stdout=log, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
