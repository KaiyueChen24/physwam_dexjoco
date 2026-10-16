"""Exercise actual Wan/Action DiT blocks without weights, downloads or CUDA."""
import copy
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

ROOT = Path(os.environ.get("FASTWAM_TEST_ROOT", Path(__file__).resolve().parents[2] / "fastwam"))
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT


def make_model(mixed=False, video=False, action=False):
    torch.manual_seed(17)
    shared = dict(ffn_dim=48, text_dim=8, freq_dim=8, eps=1e-6,
                  num_heads=2, attn_head_dim=6, num_layers=2)
    return MoT({
        "video": WanVideoDiT(hidden_dim=24, in_dim=4, out_dim=4,
                             patch_size=(1, 1, 1), has_image_input=False,
                             use_gradient_checkpointing=video, **shared),
        "action": ActionDiT(hidden_dim=16, action_dim=3,
                            use_gradient_checkpointing=action, **shared),
    }, mot_checkpoint_mixed_attn=mixed)


def make_inputs(model, requires_grad=True, per_token=False):
    torch.manual_seed(29)
    def rand(*shape):
        return torch.randn(*shape, requires_grad=requires_grad)
    # Video tokens cannot see action tokens; both experts also have masked text.
    attention_mask = torch.ones(7, 7, dtype=torch.bool)
    attention_mask[:4, 4:] = False
    return dict(
        video_tokens=rand(2, 4, 24), action_tokens=rand(2, 3, 16),
        video_freqs=model.mixtures["video"].get_freqs(1, 2, 2),
        action_freqs=model.mixtures["action"].get_freqs(3),
        video_t_mod=rand(2, 4, 6, 24) if per_token else rand(2, 6, 24),
        action_t_mod=rand(2, 6, 16),
        video_context=rand(2, 5, 24), action_context=rand(2, 5, 16),
        video_context_mask=torch.tensor([[[True, True, True, False, False]]]).expand(2, 4, 5),
        action_context_mask=torch.tensor([[[True, True, True, False, False]]]).expand(2, 3, 5),
        attention_mask=attention_mask,
    )


def run_backward(model, inputs):
    output = model.forward_joint_core(**inputs)
    sum(x.square().mean() for x in output).backward()
    return output


@pytest.mark.parametrize("flags", [(True, True, True), (True, False, False),
                                  (False, True, False), (False, False, True)])
@pytest.mark.parametrize("requires_grad,per_token", [(True, False), (False, True)])
def test_checkpoint_preserves_outputs_and_gradients_and_recomputes_layers(flags, requires_grad, per_token):
    reference = make_model()
    model = make_model(*flags)
    reference_inputs = make_inputs(reference, requires_grad, per_token)
    inputs = copy.deepcopy(reference_inputs)
    expected = run_backward(reference, reference_inputs)
    with patch.object(model, "_forward_joint_layer", wraps=model._forward_joint_layer) as layer:
        actual = run_backward(model, inputs)
        # Two distinct layers each run once forward and once during backward.
        assert layer.call_count == 4
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right)
    reference_params = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        expected_grad = reference_params[name].grad
        if expected_grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected_grad)
    for key, value in inputs.items():
        if value.requires_grad:
            torch.testing.assert_close(value.grad, reference_inputs[key].grad)
    # Check that every block actually contributes to the gradients being compared.
    for expert in model.mixtures.values():
        for block in expert.blocks:
            assert block.self_attn.q.weight.grad.abs().sum() > 0


def test_checkpoint_reduces_saved_activation_storage():
    def saved_elements(model):
        sizes = []
        def pack(tensor):
            sizes.append(tensor.numel())
            return tensor
        inputs = make_inputs(model)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = model.forward_joint_core(**inputs)
        sum(x.square().mean() for x in output).backward()
        return sum(sizes)
    assert saved_elements(make_model(True, True, True)) < saved_elements(make_model())


@pytest.mark.parametrize("eval_mode", [False, True])
def test_no_checkpoint_recomputation_in_no_grad_or_eval(eval_mode):
    reference = make_model()
    model = make_model(True, True, True)
    if eval_mode:
        reference.eval()
        model.eval()
        expected = run_backward(reference, make_inputs(reference))
        with patch.object(model, "_forward_joint_layer", wraps=model._forward_joint_layer) as layer:
            actual = run_backward(model, make_inputs(model))
    else:
        with torch.no_grad():
            expected = reference.forward_joint_core(**make_inputs(reference))
            with patch.object(model, "_forward_joint_layer", wraps=model._forward_joint_layer) as layer:
                actual = model.forward_joint_core(**make_inputs(model))
    assert layer.call_count == 2
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right)


def test_compiled_training_rejects_checkpointing_with_actionable_message():
    model = make_model(True, True, True)
    model.compile_training_layers = True
    with pytest.raises(ValueError, match="compile_training_denoise=False"):
        model.forward_joint_core(**make_inputs(model))


def test_training_loss_and_optimizer_step_with_real_tiny_experts():
    from fastwam.models.wan22.fastwam import FastWAM

    def policy(checkpointing):
        mot = make_model(checkpointing, checkpointing, checkpointing)
        mot.mixtures["video"].seperated_timestep = True
        return FastWAM(video_expert=mot.mixtures["video"], action_expert=mot.mixtures["action"],
                       mot=mot, vae=torch.nn.Identity(), text_dim=8, compile_training_denoise=False)
    reference, model = policy(False), policy(True)
    torch.manual_seed(31)
    # Encoded VAE/T5 inputs only; denoising, schedules, masks and losses remain real.
    encoded = dict(input_latents=torch.randn(2, 4, 2, 2, 2),
                   first_frame_latents=torch.randn(2, 4, 1, 2, 2),
                   context=torch.randn(2, 5, 8), context_mask=torch.ones(2, 5, dtype=torch.bool),
                   action=torch.randn(2, 4, 3), action_is_pad=None, image_is_pad=None,
                   fuse_vae_embedding_in_latents=True)
    def loss_and_grad(policy):
        torch.manual_seed(43)
        with patch.object(policy, "build_inputs", return_value=copy.deepcopy(encoded)):
            loss, metrics = policy.training_loss({})
        assert torch.isfinite(loss)
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        loss.backward()
        return loss
    expected_loss = loss_and_grad(reference)
    actual_loss = loss_and_grad(model)
    torch.testing.assert_close(actual_loss, expected_loss)
    reference_params = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        expected_grad = reference_params[name].grad
        if expected_grad is not None:
            torch.testing.assert_close(parameter.grad, expected_grad)
    before = model.action_expert.head.weight.detach().clone()
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    assert not torch.equal(model.action_expert.head.weight, before)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_dictionary_path_mixed_attention_checkpoint_preserves_gradients():
    reference, model = make_model(), make_model(True, True, True)
    reference_inputs = make_inputs(reference)
    inputs = copy.deepcopy(reference_inputs)
    def forward(policy, data):
        return policy(
            embeds_all={"video": data["video_tokens"], "action": data["action_tokens"]},
            freqs_all={"video": data["video_freqs"], "action": data["action_freqs"]},
            t_mod_all={"video": data["video_t_mod"], "action": data["action_t_mod"]},
            context_all={name: {"context": data[f"{name}_context"], "mask": data[f"{name}_context_mask"]}
                         for name in ("video", "action")},
            attention_mask=data["attention_mask"],
        )
    expected, actual = forward(reference, reference_inputs), forward(model, inputs)
    sum(x.square().mean() for x in expected.values()).backward()
    sum(x.square().mean() for x in actual.values()).backward()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
    for key, value in inputs.items():
        if value.requires_grad:
            torch.testing.assert_close(value.grad, reference_inputs[key].grad)
    reference_params = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        if reference_params[name].grad is not None:
            torch.testing.assert_close(parameter.grad, reference_params[name].grad)
