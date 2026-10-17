"""Configuration/provenance helpers usable without Torch, JAX or MuJoCo."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

FASTWAM_ROOT = Path(__file__).resolve().parents[2]


def read_yaml(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def select_group(dexjoco_root, group_name, task_names=None, dataset_root=None):
    root = Path(dexjoco_root).resolve()
    group = read_yaml(root / "configs/task_groups" / f"{group_name}.yaml")
    expected = {"single_arm": (23, 22, ["base", "wrist"]),
                "dual_arm": (46, 44, ["base", "wrist_left", "wrist_right"])}
    state_dim, action_dim, camera_order = expected[group_name]
    if group.get("schema_version") != 1 or group.get("robot_type") != group_name or (group.get("state_dim"), group.get("action_dim")) != (state_dim, action_dim):
        raise ValueError("Task group robot type or dimensions do not match")
    group = copy.deepcopy(group)
    # model_action_dim in existing groups is specific to pi05, not a FastWAM setting.
    group["model_action_dim"] = action_dim
    base = Path(dataset_root or group["dataset_root"]).expanduser()
    base = base.resolve() if base.is_absolute() else (root / base).resolve()
    tasks = group["tasks"]
    names = [t["name"] for t in tasks]
    if not names or len(names) != len(set(names)):
        raise ValueError("Task names must be nonempty and unique")
    if task_names and not set(task_names) <= set(names):
        raise ValueError(f"Unknown task selection: {set(task_names) - set(names)}")
    tasks = [t for t in tasks if not task_names or t["name"] in task_names]
    for task in tasks:
        if set(task["cameras"]) != set(camera_order) or not task["prompt"].strip():
            raise ValueError(f"{task['name']}: invalid cameras/prompt")
        child = Path(task["dataset"])
        if child.is_absolute() or ".." in child.parts:
            raise ValueError("Dataset must be a relative child of dataset_root")
        task["dataset_path"] = str(base / child)
        evaluation = read_yaml(root / task["eval_config"])
        if evaluation["env_name"] != task["name"] or evaluation["robot_type"] != group_name:
            raise ValueError(f"{task['name']}: evaluation environment/type mismatch")
        evaluation.update(prompt=task["prompt"], camera_mapping=task["cameras"])
        task["environment"] = evaluation  # Snapshot: serving needs no original checkout/data.
        episodes = task.get("episodes")
        if episodes is not None and (not episodes or len(episodes) != len(set(episodes)) or
                                     any(type(i) is not int or i < 0 for i in episodes)):
            raise ValueError(f"{task['name']}: invalid episode IDs")
    group.update(tasks=tasks, dataset_root=str(base), camera_order=camera_order)
    return group


def build_config(dexjoco_root, group_name, experiment, run_name, *, task_names=None, dataset_root=None):
    if not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("run-name must be a simple directory name")
    group = select_group(dexjoco_root, group_name, task_names, dataset_root)
    exp = copy.deepcopy(experiment)
    for key in ("action_horizon", "video_stride", "control_hz", "batch_size", "max_steps", "learning_rate",
                "gradient_accumulation_steps", "num_inference_steps", "execute_steps", "save_every", "sigma_shift"):
        if not isinstance(exp[key], (int, float)) or exp[key] <= 0:
            raise ValueError(f"{key} must be positive")
    horizon, stride = exp["action_horizon"], exp["video_stride"]
    if type(horizon) is not int or type(stride) is not int or horizon % stride or (horizon // stride) % 4:
        raise ValueError("action_horizon/video_stride must be an integer multiple of 4 for Wan VAE")
    if exp["execute_steps"] > horizon or exp["num_workers"] < 0:
        raise ValueError("Invalid execute_steps or num_workers")
    if exp["sampling"] not in ("task_balanced", "frame_uniform"):
        raise ValueError("Unknown task sampling")
    if len(exp["image_size"]) != 2 or any(type(v) is not int or v <= 0 or v % 16 for v in exp["image_size"]):
        raise ValueError("Per-camera image dimensions must be positive multiples of 16")
    if exp.get("mixed_precision") not in ("bf16", "fp16", "no"):
        raise ValueError("Invalid mixed_precision")
    run_dir = Path(dexjoco_root).resolve() / "runs" / run_name
    config = read_yaml(FASTWAM_ROOT / "configs/train.yaml")
    model = read_yaml(FASTWAM_ROOT / "configs/model/fastwam.yaml")
    # [模型/状态] 状态与动作维度来自任务组，不继承 π0.5 的 padding。
    # [模型/文本] 在线加载 T5 编码器；[GPU/显存] 重算开关来自实验 YAML，关闭训练编译。
    # [初始化/权重] 使用官方预处理构造的 ActionDiT backbone，不是训练完成的策略。
    model.update(proprio_dim=group["state_dim"], load_text_encoder=True,
                 mot_checkpoint_mixed_attn=bool(exp["gradient_checkpointing"]),
                 compile_training_denoise=False,
                 action_dit_pretrained_path=str(FASTWAM_ROOT / "checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"))
    for branch in ("video_dit_config", "action_dit_config"):
        model[branch]["action_dim"] = group["action_dim"]
        model[branch]["use_gradient_checkpointing"] = bool(exp["gradient_checkpointing"])
    config.pop("defaults", None)
    for key in ("batch_size", "num_workers", "max_steps", "learning_rate", "save_every", "log_every",
                "gradient_accumulation_steps", "mixed_precision", "seed", "weight_decay"):
        config[key] = exp[key]
    # [保存/路径] 运行产物位于 DexJoCo/runs/<名称>/；恢复路径由训练命令显式设置。
    # [评测/eval] eval_every=0 仅关闭生成式视频评测；prepare 另行接入留出集 loss 验证，闭环评测使用独立入口。
    # [训练/预算] num_epochs 仅供 max_steps 缺失时估算总步数；本入口由实验 YAML 的 max_steps 决定结束。
    config.update(output_dir=str(run_dir), model=model, resume=None, eval_every=0, num_epochs=100000)
    config["wandb"]["enabled"] = bool(exp["wandb_enabled"])
    # Optional experiment override; absent fields preserve legacy run snapshots.
    if "wandb_name" in exp:
        wandb_name = exp["wandb_name"]
        if wandb_name is not None and (not isinstance(wandb_name, str) or not wandb_name.strip()):
            raise ValueError("wandb_name must be a nonempty string or null (use run-name)")
        config["wandb"]["name"] = run_name if wandb_name is None else wandb_name
    data = read_yaml(FASTWAM_ROOT / "configs/data" / f"dexjoco_{'single' if group_name == 'single_arm' else 'dual'}.yaml")
    data["train"].update(tasks=group["tasks"], action_horizon=horizon, video_stride=stride,
                         camera_order=group["camera_order"], image_size=exp["image_size"], sampling=exp["sampling"],
                         stats_path=str(run_dir / "dataset_stats.json"), control_hz=exp["control_hz"])
    config["data"] = data
    config["framework"] = {"schema_version": 1, "model": "fastwam", "run_name": run_name,
                           "group": group, "experiment": exp}
    return config


def git_version(root):
    def run(*args):
        value = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return value.stdout.strip() if value.returncode == 0 else None
    return {"commit": run("rev-parse", "HEAD"), "status": run("status", "--short")}


def read_run(run_dir):
    run_dir = Path(run_dir).expanduser().resolve()
    config = read_yaml(run_dir / "fastwam_config.yaml")
    if config.get("framework", {}).get("model") != "fastwam":
        raise ValueError("Expected a FastWAM run")
    return run_dir, config
