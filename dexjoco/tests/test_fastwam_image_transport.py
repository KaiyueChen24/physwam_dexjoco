"""Image transport must preserve policy pixels, raw recordings and wire metadata."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "dexjoco"), str(ROOT.parent / "fastwam")]

from dexjoco_fastwam_client import evaluate as client
from experiments.dexjoco.transforms import pack_images, resize_rgb, resize_rgb_uint8


def original_resize_rgb(image, size):
    """Independent reference for the pre-optimization server preprocessing."""
    from PIL import Image
    height, width = size
    ratio = min(height / image.shape[0], width / image.shape[1])
    resized = Image.fromarray(image).resize(
        (max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio))),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (width, height))
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return np.asarray(canvas).transpose(2, 0, 1).astype(np.float32) / 255.0


@pytest.mark.parametrize("shape,size", [
    ((640, 640, 3), (224, 224)),
    ((480, 640, 3), (224, 224)),
    ((641, 479, 3), (224, 224)),
    ((224, 224, 3), (224, 224)),
    ((17, 31, 3), (112, 224)),
    ((301, 3, 3), (224, 224)),
])
def test_client_resize_and_server_reprocessing_preserve_exact_policy_pixels(shape, size):
    from openpi_client import msgpack_numpy
    rng = np.random.default_rng(52)
    images = {key: rng.integers(0, 256, shape, dtype=np.uint8)
              for key in ("base", "wrist")}
    observation = {"images": images, "state": np.arange(23, dtype=np.float32),
                   "task": "water_plant", "prompt": "water the plant", "episode_id": 4,
                   "step_id": 56, "seed": 2}
    originals = {key: image.copy() for key, image in images.items()}
    resized = client.resize_observation_images(observation, size)
    received = msgpack_numpy.unpackb(msgpack_numpy.packb(resized))
    expected = np.concatenate([original_resize_rgb(images[key], size)
                               for key in ("base", "wrist")], axis=-1) * 2 - 1
    np.testing.assert_array_equal(pack_images(received["images"], ["base", "wrist"], size), expected)
    for key in images:
        np.testing.assert_array_equal(resize_rgb(images[key], size), original_resize_rgb(images[key], size))
        assert received["images"][key].shape == (*size, 3)
        assert received["images"][key].dtype == np.uint8
        np.testing.assert_array_equal(observation["images"][key], originals[key])
    assert resized is not observation and resized["images"] is not images
    np.testing.assert_array_equal(received["state"], observation["state"])
    for key in ("task", "prompt", "episode_id", "step_id", "seed"):
        assert received[key] == observation[key]
    if shape == (640, 640, 3):
        assert len(msgpack_numpy.packb(resized)) < len(msgpack_numpy.packb(observation)) / 8


@pytest.mark.parametrize("bad", [np.zeros((20, 30, 3), np.float32),
                                  np.zeros((20, 30), np.uint8),
                                  np.zeros((20, 30, 4), np.uint8)])
def test_wire_resize_keeps_rgb_uint8_validation(bad):
    with pytest.raises(ValueError, match="uint8 RGB HWC"):
        resize_rgb_uint8(bad, (224, 224))


@pytest.mark.parametrize("resizing", [False, True])
def test_real_client_message_path_records_small_payload_but_full_size_video(tmp_path, monkeypatch, resizing):
    import imageio.v2 as imageio
    import websockets.sync.client
    from openpi_client import msgpack_numpy

    metadata = {"model": "fastwam", "group": "single_arm", "run_name": "test",
                "config_signature": "config", "checkpoint_sha256": "checkpoint",
                "stats_sha256": "stats", "action_dim": 22, "state_dim": 23,
                "action_horizon": 32, "camera_order": ["base", "wrist"],
                "control_hz": 30, "image_size": [224, 224]}
    raw = {key: np.random.default_rng(index).integers(0, 256, (640, 640, 3), dtype=np.uint8)
           for index, key in enumerate(("base", "wrist"))}
    requests, frames, closed_writers = [], [], []

    class Env:
        def __init__(self, **kwargs):
            self.is_done = self.is_success = False
            self.env = SimpleNamespace(hz=30)
        def start(self): pass
        def reset(self): self.is_done = self.is_success = False
        def close(self): pass
        def get_obs(self):
            return {"images": raw, "state": np.zeros(23, np.float32), "prompt": "water the plant"}
        def get_raw_images(self): return {"front": raw["base"], "wrist": raw["wrist"]}
        def step(self, action): self.is_done = self.is_success = True

    monkeypatch.setitem(sys.modules, "dexjoco_fastwam_client.env", SimpleNamespace(DexJoCoFastWAMEnv=Env))
    class Socket:
        def send(self, payload): requests.append(msgpack_numpy.unpackb(payload))
        def recv(self, **kwargs):
            if not requests: return msgpack_numpy.packb(metadata)
            return msgpack_numpy.packb({"actions": np.zeros((32, 22), np.float32),
                                       "episode_id": 0, "step_id": 0,
                                       "server_timing": {"infer_ms": 0.1}})
    def connect(*args, **kwargs):
        assert kwargs["ping_timeout"] == 180
        return nullcontext(Socket())
    monkeypatch.setattr(websockets.sync.client, "connect", connect)
    class Writer:
        def append_data(self, image): frames.append(image.copy())
        def close(self): closed_writers.append(True)
    monkeypatch.setattr(imageio, "get_writer", lambda *a, **kw: Writer())
    config = tmp_path / "environment.yaml"
    config.write_text(yaml.safe_dump({"env_name": "water_plant", "robot_type": "single_arm",
                                     "camera_mapping": {"base": "front", "wrist": "wrist"},
                                     "prompt": "water the plant"}))
    metadata_path = tmp_path / "server.json"
    metadata_path.write_text(json.dumps(metadata))
    output = tmp_path / "videos"
    argv = ["evaluate", "--config", str(config), "--metadata", str(metadata_path),
            "--output", str(output), "--episodes", "1"]
    if resizing:
        argv.extend(["--resize-before-send", "--fastwam-root", str(ROOT.parent / "fastwam")])
    monkeypatch.setattr(sys, "argv", argv)
    client.main()
    assert len(requests) == 1
    expected_shape = (224, 224, 3) if resizing else (640, 640, 3)
    assert all(image.shape == expected_shape for image in requests[0]["images"].values())
    np.testing.assert_array_equal(pack_images(requests[0]["images"], ["base", "wrist"], [224, 224]),
                                  pack_images(raw, ["base", "wrist"], [224, 224]))
    assert len(frames) == len(closed_writers) == 2
    for frame, original in zip(frames, raw.values()):
        np.testing.assert_array_equal(frame, original)
    metrics = json.loads((output / "latency.jsonl").read_text())
    assert metrics["request_bytes"] == len(msgpack_numpy.packb(requests[0]))
    assert metrics["request_bytes"] < 310_000 if resizing else metrics["request_bytes"] > 2_450_000
    assert (output / "success_rate_1_1.txt").is_file()
