"""Convert the DexJoCo wire contract to official FastWAM action inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import read_run, signature
from .transforms import MinMaxNormalizer, PROMPT_TEMPLATE, pack_images


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DexJoCoFastWAMPolicy:
    def __init__(self, run_dir, checkpoint, device="cuda", model=None):
        import torch

        self.run_dir, self.config = read_run(run_dir)
        framework = self.config["framework"]
        self.group, self.experiment = framework["group"], framework["experiment"]
        self.normalizer = MinMaxNormalizer(json.loads((self.run_dir / "dataset_stats.json").read_text()))
        stats_digest = checkpoint_digest(self.run_dir / "dataset_stats.json")
        checksum_path = self.run_dir / "dataset_stats.sha256"
        if checksum_path.exists() and checksum_path.read_text().strip() != stats_digest:
            raise ValueError("Run normalization statistics checksum mismatch")
        self.tasks = {t["name"]: t for t in self.group["tasks"]}
        if model is None:
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[self.config["mixed_precision"]]
            model_config = dict(self.config["model"])
            model_config.update(skip_dit_load_from_pretrain=True, action_dit_pretrained_path=None)
            model = instantiate(OmegaConf.create(model_config), model_dtype=dtype, device=device)
            # Upstream's loader uses strict=False for legacy compatibility. Deployment requires a full policy.
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.mot.load_state_dict(payload["mot"], strict=True)
            model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            model.eval()
        self.model = model
        self.context_cache = {}
        self.metadata = {"schema_version": 1, "model": "fastwam", "run_name": framework["run_name"],
                         "group": self.group["name"], "tasks": list(self.tasks),
                         "config_signature": signature(self.config),
                         "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": checkpoint_digest(checkpoint),
                         "stats_sha256": stats_digest,
                         "action_dim": self.group["action_dim"], "state_dim": self.group["state_dim"],
                         "action_horizon": self.experiment["action_horizon"],
                         "camera_order": self.group["camera_order"], "image_size": self.experiment["image_size"],
                         "control_hz": self.experiment["control_hz"], "execution_mode": "synchronous"}

    def infer(self, observation):
        import torch

        task = self.tasks.get(observation.get("task"))
        if task is None:
            raise ValueError("Task is not present in this checkpoint's task group")
        if observation.get("prompt") != task["prompt"]:
            raise ValueError("Prompt differs from the training task snapshot")
        episode, step = observation.get("episode_id"), observation.get("step_id")
        if type(episode) is not int or type(step) is not int or episode < 0 or step < 0:
            raise ValueError("episode_id/step_id must be nonnegative integers")
        seed = observation.get("seed", 0)
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        random_key = f"{self.experiment['seed']}:{task['name']}:{seed}:{episode}:{step}".encode()
        action_seed = int.from_bytes(hashlib.sha256(random_key).digest()[:4], "little")
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (self.group["state_dim"],):
            raise ValueError("Incorrect proprioception shape; privileged state/padding is not accepted")
        image = pack_images(observation["images"], self.group["camera_order"], self.experiment["image_size"])
        state = self.normalizer.forward("state", state)
        with torch.no_grad():
            if task["name"] not in self.context_cache:
                self.context_cache[task["name"]] = self.model.encode_prompt(PROMPT_TEMPLATE.format(task=task["prompt"]))
            context, context_mask = self.context_cache[task["name"]]
            result = self.model.infer_action(
                prompt=None, input_image=torch.from_numpy(image.copy()),
                proprio=torch.from_numpy(state.copy()), context=context, context_mask=context_mask,
                action_horizon=self.experiment["action_horizon"],
                num_inference_steps=self.experiment["num_inference_steps"],
                sigma_shift=self.experiment["sigma_shift"], text_cfg_scale=1.0, compile_action_infer=False,
                seed=action_seed)
        action = result["action"].detach().float().cpu().numpy()
        if action.shape != (self.experiment["action_horizon"], self.group["action_dim"]):
            raise ValueError(f"Unexpected model output: {action.shape}")
        return {"actions": self.normalizer.backward("action", action), "episode_id": episode, "step_id": step}
