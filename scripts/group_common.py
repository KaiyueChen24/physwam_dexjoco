"""Shared configuration and provenance helpers. Safe to import in dexjoco env."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

import yaml

from dataset_split import save_split, split_group, validation_settings

ROOT = Path(__file__).resolve().parents[1]


def repo_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_yaml(path):
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"{path}: expected schema_version: 1 and a mapping")
    return data


def load_group(name):
    path = repo_path(name if str(name).endswith(".yaml") else f"configs/task_groups/{name}.yaml")
    group = read_yaml(path)
    dual = group["robot_type"] == "dual_arm"
    expected = (46, 44, 44) if dual else (23, 22, 32)
    if group["robot_type"] not in ("single_arm", "dual_arm"):
        raise ValueError("robot_type must be single_arm or dual_arm")
    actual = tuple(group[k] for k in ("state_dim", "action_dim", "model_action_dim"))
    if actual != expected:
        raise ValueError(f"Group dimensions {actual} do not match {expected}")
    names = [t["name"] for t in group["tasks"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Tasks must be nonempty and unique")
    cameras = {"base", "wrist_left", "wrist_right"} if dual else {"base", "wrist"}
    for task in group["tasks"]:
        if set(task["cameras"]) != cameras or not task["prompt"].strip():
            raise ValueError(f"{task['name']}: invalid cameras or prompt")
        if Path(task["dataset"]).is_absolute() or ".." in Path(task["dataset"]).parts:
            raise ValueError("Task dataset must be a relative child of dataset_root")
        episodes = task.get("episodes")
        if episodes is not None and (not episodes or len(set(episodes)) != len(episodes)
                                     or any(not isinstance(i, int) or i < 0 for i in episodes)):
            raise ValueError(f"{task['name']}: invalid episode selection")
    return group


def training_parser(description, default_experiment="h100_lora"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--group", choices=("single_arm", "dual_arm"), required=True, help='[任务选择] 选择单臂或双臂任务组。')
    parser.add_argument("--experiment", default=f"configs/experiments/{default_experiment}.yaml", help='[配置与实验] 实验 YAML 路径；相对路径以 DexJoCo 仓库根目录为准。')
    parser.add_argument("--run-name", help='[配置与实验] 实验目录名，结果写入 runs/<名称>/。')
    parser.add_argument("--gpus", default="0", help='[GPU 与并行] 可见 GPU 编号，逗号分隔，例如 0,1；不是 GPU 数量。')
    parser.add_argument("--dataset-root", help='[数据与 CPU] 覆盖任务数据根目录；相对路径以 DexJoCo 仓库根目录为准。')
    parser.add_argument("--checkpoint", help='[权重与结果保存] π0.5 初始化 params 目录，覆盖 YAML 中该任务组的路径。')
    parser.add_argument("--batch-size", type=int, help='[训练] 覆盖 YAML 的全局 batch（所有 GPU 合计）。')
    parser.add_argument("--steps", type=int, help='[训练] 覆盖 YAML 的优化器更新总步数，不是 epoch。')
    parser.add_argument("--num-workers", type=int, help='[数据与 CPU] 覆盖 CPU 数据加载进程数；0 表示主进程加载。')
    parser.add_argument("--fsdp-devices", type=int, help='[GPU 与并行] 每个参数分片组的 GPU 数；1 为不分片，须整除可见卡数。')
    parser.add_argument("--resume", action="store_true", help='[权重与结果保存] 从该实验最近的完整 checkpoint 恢复模型和优化器状态。')
    parser.add_argument("--dry-run", action="store_true", help='[配置与实验] 仅打印配置或计划，不启动训练、推理或评测。')
    parser.add_argument("--tiny", action="store_true", help='[训练] 仅对合成实验使用 dummy 小模型验证框架，不是完整 π0.5 训练。')
    return parser


def resolve(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    group = load_group(args.group)
    experiment = read_yaml(repo_path(args.experiment))
    if experiment.get("model") != "pi05":
        raise ValueError("Only pi05 is implemented by this training entry point")
    for flag, key in (("batch_size", "batch_size"), ("steps", "num_train_steps"),
                      ("num_workers", "num_workers"), ("fsdp_devices", "fsdp_devices")):
        value = getattr(args, flag, None)
        if value is not None:
            experiment[key] = value
    validation_settings(experiment)  # [验证/loss] 在访问数据或启动 GPU 前校验参数。
    experiment["tiny"] = args.tiny
    if args.tiny and not experiment["synthetic"]:
        raise ValueError("--tiny requires a synthetic experiment")
    for key in ("batch_size", "num_train_steps", "action_horizon", "fsdp_devices", "stats_samples",
                "stats_batch_size", "log_interval", "save_interval", "synthetic_frames_per_task", "synthetic_episode_count"):
        if key in experiment and experiment[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if experiment["num_workers"] < 0 or experiment["warmup_steps"] < 0:
        raise ValueError("num_workers and warmup_steps must be nonnegative")
    if experiment["sampling"] not in ("task_balanced", "frame_uniform"):
        raise ValueError("Unknown sampling mode")
    if experiment["stats_samples"] < len(group["tasks"]):
        raise ValueError("stats_samples must cover all tasks")
    group["dataset_root"] = str(repo_path(args.dataset_root or experiment.get("dataset_root")
                                         or group["dataset_root"]))
    for key in ("synthetic", "sampling", "seed", "video_backend", "synthetic_frames_per_task", "synthetic_episode_count"):
        if key in experiment:
            group[key] = experiment[key]
    checkpoint = args.checkpoint or experiment["checkpoint_paths"][group["name"]]
    experiment["initialization_params"] = str(repo_path(checkpoint))
    if args.tiny:
        experiment["initialization_params"] = None
    run_name = args.run_name or f"pi05_{args.group}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("run-name must be a simple directory name")
    return {"schema_version": 1, "group": group, "experiment": experiment, "run_name": run_name,
            "gpus": args.gpus, "run_dir": str(ROOT / "runs" / run_name)}


def resolve_training_split(resolved):
    # [数据/验证划分] 此操作需要本地元数据；dry-run 保持只展示计划，不访问数据。
    settings = validation_settings(resolved["experiment"])
    train, validation, manifest = split_group(resolved["group"], settings)
    resolved["group"] = train
    if validation is not None:
        resolved["validation_group"] = validation
        resolved["data_split"] = manifest
    return resolved


def activate_openpi():
    os.chdir(ROOT / "openpi")  # upstream config import reads config.yaml from cwd
    sys.path.insert(0, str(ROOT / "openpi/src"))


def build_train_config(resolved, resume=False):
    activate_openpi()
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config as cfg, optimizer, weight_loaders
    from openpi.training.dexjoco_multitask import make_data_factory
    from openpi.shared import nnx_utils

    group, exp = resolved["group"], resolved["experiment"]
    tiny = exp.get("tiny", False)
    validation = None
    if resolved.get("validation_group"):
        options = validation_settings(exp)
        validation = {**options, "group": resolved["validation_group"], "sampling": exp["sampling"],
                      "output_path": str(Path(resolved["run_dir"]) / "validation_loss.jsonl")}
    # [模型/LoRA] 语言侧 rank=16、alpha=16；动作侧 rank=32、alpha=32，定义在 gemma.py。
    model = Pi0Config(pi05=True, action_dim=group["model_action_dim"],
                      action_horizon=exp["action_horizon"], max_token_len=exp["max_token_len"],
                      paligemma_variant="dummy" if tiny else "gemma_2b_lora",
                      action_expert_variant="dummy" if tiny else "gemma_300m_lora")
    loader = (weight_loaders.NoOpWeightLoader() if tiny else
              weight_loaders.CheckpointWeightLoader(exp["initialization_params"]))
    run_dir = Path(resolved["run_dir"])
    return cfg.TrainConfig(
        name=f"pi05_{group['name']}_multitask", exp_name=resolved["run_name"], validation=validation,
        model=model, data=make_data_factory(group), weight_loader=loader,
        # [训练/冻结] 正式模式沿用模型过滤器，视觉编码器仍可训练；tiny 冻结整个 PaliGemma。
        # [训练/EMA] ema_decay=None 关闭参数指数移动平均。
        freeze_filter=nnx_utils.PathRegex("PaliGemma/.*") if tiny else model.get_freeze_filter(), ema_decay=None,
        # [训练/学习率] 预热后保持常数：peak_lr 与 decay_lr 使用同一个 YAML 值。
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=exp["warmup_steps"], peak_lr=exp["learning_rate"],
            decay_steps=max(exp["num_train_steps"], exp["warmup_steps"] + 1),
            decay_lr=exp["learning_rate"]),
        # [训练/优化器] 默认 β1=0.9、β2=0.95、eps=1e-8、weight_decay=1e-10；梯度裁剪为 1。
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        batch_size=exp["batch_size"], num_workers=exp["num_workers"],
        num_train_steps=exp["num_train_steps"], seed=exp["seed"], fsdp_devices=exp["fsdp_devices"],
        log_interval=exp["log_interval"], save_interval=exp["save_interval"],
        keep_period=exp.get("keep_period"), wandb_enabled=exp["wandb_enabled"], resume=resume,
        assets_base_dir=str(run_dir / "assets"), checkpoint_base_dir=str(run_dir / "checkpoints"),
        policy_metadata={"group": group["name"], "robot_type": group["robot_type"],
                         "tasks": [t["name"] for t in group["tasks"]],
                         "synthetic": bool(exp["synthetic"]), "tiny": tiny,
                         "action_horizon": exp["action_horizon"]},
    )


def load_trainer():
    path = ROOT / "openpi/scripts/train.py"
    spec = importlib.util.spec_from_file_location("dexjoco_openpi_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_gpu(config, tiny=False):
    import jax
    devices = jax.devices()
    print("Devices:", devices, flush=True)
    if not tiny and not all(d.platform == "gpu" for d in devices):
        raise RuntimeError("Full pi05 training requires visible GPU devices")
    if config.batch_size % len(devices) or len(devices) % config.fsdp_devices:
        raise ValueError("Global batch and fsdp_devices must divide the visible GPU layout")
    if not tiny and not config.resume and not Path(config.weight_loader.params_path).is_dir():
        raise FileNotFoundError(f"Initialization checkpoint missing: {config.weight_loader.params_path}")


def write_run(resolved):
    run_dir = Path(resolved["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "resolved_config.yaml"
    if snapshot.exists():
        old = read_run(run_dir)
        # A run's data, architecture, statistics and optimizer cannot silently change.
        old_exp, new_exp = dict(old["experiment"]), dict(resolved["experiment"])
        old_exp.pop("num_train_steps", None)
        new_exp.pop("num_train_steps", None)
        if (old["group"] != resolved["group"] or old_exp != new_exp
                or old.get("validation_group") != resolved.get("validation_group")
                or old.get("data_split") != resolved.get("data_split")):
            raise ValueError("Existing run has different data/model settings; use another run-name")
    save_split(run_dir, resolved.get("data_split"))
    snapshot.write_text(yaml.safe_dump(resolved, sort_keys=False))
    if not (run_dir / "environment.txt").exists():
        details = [f"Python: {sys.version}", f"Executable: {sys.executable}",
                   f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}"]
        for command in (["git", "rev-parse", "HEAD"], ["git", "diff", "--stat"],
                        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                        [sys.executable, "-m", "pip", "freeze"]):
            try:
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
                details.extend([" ".join(command), result.stdout, result.stderr])
            except (OSError, subprocess.TimeoutExpired) as exc:
                details.append(str(exc))
        (run_dir / "environment.txt").write_text("\n".join(details))
    return run_dir


class Tee:
    def __init__(self, terminal, file):
        self.terminal, self.file = terminal, file

    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def isatty(self):
        return False


def latest_checkpoint(config):
    candidates = [p for p in config.checkpoint_dir.glob("*")
                  if p.name.isdigit() and (p / "params").is_dir()
                  and (p / "_CHECKPOINT_METADATA").exists()]
    if not candidates:
        raise FileNotFoundError(f"No completed checkpoint under {config.checkpoint_dir}")
    return max(candidates, key=lambda p: int(p.name))


def read_run(run_dir):
    run_dir = repo_path(run_dir)
    resolved = read_yaml(run_dir / "resolved_config.yaml")
    origin_root = Path(resolved["run_dir"]).parent.parent
    # Relocate only paths originally inside the source checkout. External data paths stay explicit.
    containers = [(resolved["group"], "dataset_root"),
                  (resolved["experiment"], "initialization_params")]
    if resolved.get("validation_group"):
        containers.append((resolved["validation_group"], "dataset_root"))
    for container, key in containers:
        value = container.get(key)
        if value and Path(value).is_relative_to(origin_root):
            container[key] = str(ROOT / Path(value).relative_to(origin_root))
    resolved["run_dir"] = str(run_dir)
    return resolved
