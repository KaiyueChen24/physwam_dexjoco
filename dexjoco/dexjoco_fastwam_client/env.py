"""Reuse DexJoCo action conversion, expose raw RGB for shared FastWAM transforms."""
import numpy as np

from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv


class DexJoCoFastWAMEnv(DexJoCoOpenPIEnv):
    def reset(self):
        super().reset()
        if self.env_name == "click_mouse":
            action = np.array([
                -4.4294e-01, 1.3729e-06, 1.5170e00,
                -3.14156462e00, -6.91584035e-05, -1.40317984e-03,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0.263, 0, 0, 0,
            ], dtype=np.float32)
            for _ in range(30):
                self.step(action)
                if self.is_done:
                    break

    def _process_obs(self, env_obs):
        dim = 46 if self.dual_arm else 23
        state = np.asarray(env_obs["state"][:dim], dtype=np.float32)
        images = {key: np.asarray(env_obs[raw_key]).copy() for key, raw_key in self.camera_mapping.items()}
        if state.shape != (dim,) or not np.isfinite(state).all():
            raise ValueError("Invalid robot proprioception")
        if any(im.ndim != 3 or im.shape[-1] != 3 or im.dtype != np.uint8 for im in images.values()):
            raise ValueError("Simulator must provide uint8 RGB HWC images")
        return {"images": images, "state": state, "prompt": self.prompt}

    def step(self, action):
        if self.env is None:
            raise RuntimeError("Environment has not started")
        obs, _, terminated, truncated, info = self.env.step(self._process_action(action))
        self._done = bool(terminated or truncated)
        self._success = bool(info.get("succeed", False))
        self._update_raw_obs(obs)
        self.obs = self._process_obs(obs)
        return info.get("pressed_digits")
