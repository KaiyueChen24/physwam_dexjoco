"""Verify actual model variants, trainable modules and saved-run compatibility."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from group_common import (build_train_config, get_finetune_mode, load_group, read_run,
                          resolve, training_parser, write_run)


@pytest.mark.parametrize("experiment,batch,steps", [("pi05_full", 32, 60000), ("pi05_smoke", 4, 20)])
def test_profiles_select_full_single_arm_group_and_four_gpus(experiment, batch, steps, monkeypatch):
    # resolve changes CUDA visibility, so restore the environment after each test.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # Budget values are explicit overrides so users can tune YAML without breaking architecture checks.
    resolved = resolve(training_parser("test", experiment).parse_args([
        "--batch-size", str(batch), "--steps", str(steps)]))
    assert resolved["group"]["name"] == "single_arm"
    assert resolved["group"]["tasks"] == load_group("single_arm")["tasks"]
    assert len(resolved["group"]["tasks"]) == 6
    assert resolved["gpus"] == "0,1,2,3"
    assert resolved["experiment"]["finetune_mode"] == "full"
    assert resolved["experiment"]["fsdp_devices"] == 4
    assert resolved["experiment"]["batch_size"] == batch
    assert resolved["experiment"]["num_train_steps"] == steps


def test_cli_can_override_profile_group_and_gpu_layout(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    args = training_parser("test").parse_args([
        "--group", "dual_arm", "--gpus", "4,5", "--fsdp-devices", "2", "--batch-size", "8"])
    resolved = resolve(args)
    assert resolved["group"]["name"] == "dual_arm"
    assert len(resolved["group"]["tasks"]) == 5
    assert resolved["gpus"] == "4,5"
    assert resolved["experiment"]["fsdp_devices"] == 2
    assert resolved["experiment"]["batch_size"] == 8


@pytest.mark.parametrize("flags,error", [
    (["--gpus", "0,0,1,2"], "duplicate"),
    (["--gpus", "0,"], "nonnegative"),
    (["--gpus", "0,1"], "fsdp_devices"),
    (["--batch-size", "3"], "Global batch"),
    (["--gpus", "0,1,2,3,4,5,6,7", "--batch-size", "8"], "validation_batch_size"),
])
def test_invalid_layout_fails_before_accessing_data_or_gpu(flags, error):
    # Smoke validation batch=4, so 8 visible cards also requires an override in YAML.
    with pytest.raises(ValueError, match=error):
        resolve(training_parser("test", "pi05_smoke").parse_args(flags))


def test_invalid_finetuning_mode_is_rejected():
    with pytest.raises(ValueError, match="finetune_mode"):
        get_finetune_mode({"finetune_mode": "ful"})


@pytest.mark.parametrize("mode", ["full", "lora", None])
def test_model_and_optimizer_filter_match_mode_including_legacy_snapshots(mode, tmp_path, monkeypatch):
    # Only construct configuration objects and a tiny parameter tree; no model weights are allocated.
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    resolved = resolve(training_parser("test").parse_args([]))
    resolved["run_dir"] = str(tmp_path / "run")
    if mode is None:
        resolved["experiment"].pop("finetune_mode")
    else:
        resolved["experiment"]["finetune_mode"] = mode
    config = build_train_config(resolved)
    from flax import nnx
    expected_mode = mode or "lora"
    suffix = "" if expected_mode == "full" else "_lora"
    assert config.model.paligemma_variant == f"gemma_2b{suffix}"
    assert config.model.action_expert_variant == f"gemma_300m{suffix}"
    assert config.model.pi05 and config.model.action_dim == 32
    assert config.policy_metadata["finetune_mode"] == expected_mode
    assert config.policy_metadata["tasks"] == [t["name"] for t in load_group("single_arm")["tasks"]]
    paths = {
        "vision": ("PaliGemma", "img", "embedding", "kernel"),
        "language": ("PaliGemma", "llm", "layers", "attn", "q_einsum", "w"),
        "action_expert": ("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "w"),
        "action_projection": ("action_out_proj", "kernel"),
    }
    if expected_mode == "lora":
        paths["language_adapter"] = (*paths["language"][:-1], "lora_a")
        paths["action_adapter"] = (*paths["action_expert"][:-1], "lora_a")
    state = nnx.State.from_flat_path({path: nnx.Param(np.zeros(1, dtype=np.float32)) for path in paths.values()})
    selected = set(state.filter(config.trainable_filter).flat_state())
    expected = set(paths.values())
    if expected_mode == "full":
        assert not state.filter(config.freeze_filter).flat_state()
    else:
        expected -= {paths["language"], paths["action_expert"]}
    assert selected == expected


def test_old_lora_run_keeps_its_mode_and_cannot_resume_as_full(tmp_path):
    resolved = resolve(training_parser("test").parse_args(["--run-name", "compatibility"]))
    resolved["run_dir"] = str(tmp_path)
    resolved["experiment"].pop("finetune_mode")
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump(resolved))
    (tmp_path / "environment.txt").write_text("test environment")
    restored = read_run(tmp_path)
    assert restored["experiment"]["finetune_mode"] == "lora"
    write_run(restored)
    changed = copy.deepcopy(restored)
    changed["experiment"]["finetune_mode"] = "full"
    with pytest.raises(ValueError, match="different data/model settings"):
        write_run(changed)


def test_full_mode_round_trips_through_saved_run(tmp_path):
    resolved = resolve(training_parser("test").parse_args(["--run-name", "full_round_trip"]))
    resolved["run_dir"] = str(tmp_path)
    (tmp_path / "environment.txt").write_text("test environment")
    write_run(resolved)
    restored = read_run(tmp_path)
    assert restored["experiment"]["finetune_mode"] == "full"
    assert len(restored["group"]["tasks"]) == 6


@pytest.mark.parametrize("resume", [False, True])
def test_worker_count_change_requires_explicit_resume_and_is_recorded(tmp_path, resume):
    resolved = resolve(training_parser("test").parse_args(["--run-name", "workers"]))
    resolved["run_dir"] = str(tmp_path)
    resolved["experiment"]["num_workers"] = 4
    (tmp_path / "environment.txt").write_text("fixture")
    write_run(resolved)
    changed = copy.deepcopy(resolved)
    changed["experiment"]["num_workers"] = 8
    if not resume:
        with pytest.raises(ValueError, match="different data/model settings"):
            write_run(changed)
        assert not (tmp_path / "runtime_changes.jsonl").exists()
        assert read_run(tmp_path)["experiment"]["num_workers"] == 4
        return
    write_run(changed, resume=True)
    assert read_run(tmp_path)["experiment"]["num_workers"] == 8
    records = [json.loads(line) for line in (tmp_path / "runtime_changes.jsonl").read_text().splitlines()]
    assert records[0]["num_workers"] == {"old": 4, "new": 8}
    write_run(changed, resume=True)
    assert len((tmp_path / "runtime_changes.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("field,value", [("batch_size", 256), ("learning_rate", 1e-4),
                                       ("sampling", "frame_uniform"), ("finetune_mode", "lora")])
def test_worker_tuning_does_not_allow_other_resume_changes(tmp_path, field, value):
    resolved = resolve(training_parser("test").parse_args(["--run-name", "workers"]))
    resolved["run_dir"] = str(tmp_path)
    (tmp_path / "environment.txt").write_text("fixture")
    write_run(resolved)
    changed = copy.deepcopy(resolved)
    changed["experiment"]["num_workers"] = 16
    changed["experiment"][field] = value
    with pytest.raises(ValueError, match="different data/model settings"):
        write_run(changed, resume=True)
    assert not (tmp_path / "runtime_changes.jsonl").exists()
    assert read_run(tmp_path)["experiment"][field] == resolved["experiment"][field]
