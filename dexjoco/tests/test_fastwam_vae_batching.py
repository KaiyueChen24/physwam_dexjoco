"""Keep the static VAE capture isolated from validation/tail input shapes."""
import contextlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(os.environ.get("FASTWAM_TEST_ROOT", Path(__file__).resolve().parents[2] / "fastwam"))
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.wan_video_vae import VideoVAE38_


def fake_static_capture(monkeypatch):
    calls = []
    compiled_calls = []
    def compile_function(fn, **kwargs):
        calls.append(kwargs)
        captured = None
        buffer = None
        def replay(video, scale):
            nonlocal captured, buffer
            spec = (tuple(video.shape), tuple(video.stride()), video.dtype, video.device)
            if captured is None:
                captured = spec
            if spec != captured:
                raise RuntimeError("Static CUDA graph cannot replay a different input shape")
            compiled_calls.append(tuple(video.shape))
            result = fn(video, scale)
            if buffer is None:
                buffer = result.clone()
            else:
                buffer.copy_(result)
            return buffer
        return replay
    monkeypatch.setattr(torch, "compile", compile_function)
    return calls, compiled_calls


@pytest.mark.parametrize("real_vae", [False, True])
def test_training_capture_survives_validation_tail_and_return_to_training(monkeypatch, real_vae):
    calls, compiled_calls = fake_static_capture(monkeypatch)
    if real_vae:
        encoder = VideoVAE38_(dim=2, z_dim=2, dec_dim=2, dim_mult=[1, 2, 2, 2],
                              num_res_blocks=1).eval().requires_grad_(False)
        scale = [torch.zeros(2), torch.ones(2)]
        spatial = 16
    else:
        class Encoder:
            def encode(self, video, scale):
                return video[:, :1] * scale[1] + scale[0]
        encoder, scale, spatial = Encoder(), [0., 2.], 4
    model = SimpleNamespace(device=torch.device("cpu"), vae=SimpleNamespace(model=encoder, scale=scale))
    previous = None
    for size in (16, 8, 6, 1, 8, 16):
        video = torch.randn(size, 3, 5, spatial, spatial)
        with torch.no_grad():
            expected = encoder.encode(video, scale).clone()
        actual = FastWAM._encode_video_latents(model, video)
        torch.testing.assert_close(actual, expected)
        assert actual.shape[0] == size
        if previous is not None:
            torch.testing.assert_close(previous[0], previous[1])  # Capture replay must not overwrite old outputs.
        previous = actual, actual.clone()
    assert len(calls) == 1
    assert calls[0] == {"backend": "cudagraphs", "fullgraph": True, "dynamic": False}
    assert [shape[0] for shape in compiled_calls] == [16, 16]


def test_different_temporal_extent_or_dtype_does_not_reuse_capture(monkeypatch):
    calls, compiled_calls = fake_static_capture(monkeypatch)
    encoder = SimpleNamespace(encode=lambda video, scale: video[:, :1] * 2)
    model = SimpleNamespace(device=torch.device("cpu"), vae=SimpleNamespace(model=encoder, scale=[0., 1.]))
    for video in (torch.randn(8, 3, 5, 4, 4), torch.randn(8, 3, 9, 4, 4),
                  torch.randn(8, 3, 5, 4, 4, dtype=torch.float64)):
        actual = FastWAM._encode_video_latents(model, video)
        torch.testing.assert_close(actual, encoder.encode(video, model.vae.scale))
    assert len(calls) == len(compiled_calls) == 1


def test_validation_keeps_every_tail_sample_and_restores_training_capture(monkeypatch, tmp_path):
    from experiments.dexjoco.validation import evaluate_loss
    _, compiled_calls = fake_static_capture(monkeypatch)
    class Options(dict):
        __getattr__ = dict.__getitem__
    class Data:
        sampling = "frame_uniform"
        lengths = [22, 1]
        ends = [22, 23]
        tasks = [{"name": "a"}, {"name": "b"}]
        def __getitem__(self, index):
            return {"video": torch.full((3, 5, 4, 4), float(index + 1))}
    class Model(torch.nn.Module):
        _encode_video_latents = FastWAM._encode_video_latents
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.))
            self.device = torch.device("cpu")
            self.vae = SimpleNamespace(model=SimpleNamespace(encode=lambda video, scale: video[:, :1]), scale=[0., 1.])
        def training_loss(self, sample):
            latents = self._encode_video_latents(sample["video"])
            loss = (self.weight - latents).square().mean()
            return loss, {"loss_video": loss.item()}
    model = Model()
    model.weight.grad = torch.tensor(7.)
    sample = {"video": torch.ones(16, 3, 5, 4, 4)}
    model.training_loss(sample)
    accelerator = SimpleNamespace(device=torch.device("cpu"), process_index=0, num_processes=1,
        is_main_process=True, unwrap_model=lambda m: m, autocast=contextlib.nullcontext,
        reduce=lambda value, **kwargs: value)
    trainer = SimpleNamespace(model=model, accelerator=accelerator, train_dataset=object(), val_dataset=Data(),
        cfg=SimpleNamespace(validation_loss=Options(seed=42, batch_size=8, max_batches=50, sampling="task_balanced")),
        global_step=1000, output_dir=tmp_path)
    rng = torch.random.get_rng_state().clone()
    result = evaluate_loss(trainer)
    assert result["task_samples"] == {"a": 22, "b": 1}
    assert result["full_validation"]
    assert result["task_losses"] == pytest.approx({"a": np.square(np.arange(1, 23)).mean(), "b": 529.})
    assert result["val_loss"] == pytest.approx((172.5 + 529.) / 2)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert model.training and model.weight.grad.item() == 7
    model.training_loss(sample)
    assert [shape[0] for shape in compiled_calls] == [16, 16]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for the actual CUDA Graph backend")
def test_real_cuda_graph_with_validation_shapes():
    encoder = VideoVAE38_(dim=2, z_dim=2, dec_dim=2, dim_mult=[1, 2, 2, 2],
                         num_res_blocks=1).cuda().eval().requires_grad_(False)
    model = SimpleNamespace(device=torch.device("cuda"),
        vae=SimpleNamespace(model=encoder, scale=[torch.zeros(2, device="cuda"), torch.ones(2, device="cuda")]))
    saved = []
    for size in (16, 8, 6, 1, 8, 16):
        video = torch.randn(size, 3, 5, 16, 16, device="cuda")
        with torch.no_grad():
            expected = encoder.encode(video, model.vae.scale).clone()
        actual = FastWAM._encode_video_latents(model, video)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected)
        saved.append((actual, actual.clone()))
    for actual, expected in saved:
        torch.testing.assert_close(actual, expected)
