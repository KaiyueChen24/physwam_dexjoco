"""Train, save, restore, then continue pi05 on synthetic task-group data."""
import contextlib
import json
import logging
from pathlib import Path
import sys

import numpy as np
import yaml

from group_common import (Tee, build_train_config, load_trainer, resolve, training_parser,
                          validate_gpu, write_run)
from train_group import compute_stats, prepare_data


def run_smoke(config, resolved):
    import jax
    from openpi.training import checkpoints, data_loader, sharding

    trainer = load_trainer()
    logging.basicConfig(level=logging.INFO, force=True)
    if not resolved["experiment"]["synthetic"]:
        raise ValueError("Smoke entry point requires synthetic data")
    if config.checkpoint_dir.exists():
        raise FileExistsError("Use a new run-name for smoke; existing checkpoints are preserved")
    validate_gpu(config, resolved["experiment"].get("tiny", False))
    dataset = prepare_data(config, resolved, check_images=True)
    compute_stats(config, resolved, dataset)
    del dataset
    mesh = sharding.make_mesh(config.fsdp_devices)
    batch_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    loader = data_loader.create_data_loader(config, sharding=batch_sharding, shuffle=True)
    iterator = iter(loader)
    rng, init_rng = jax.random.split(jax.random.key(config.seed))
    state, state_sharding = trainer.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    initial = [np.asarray(x).copy() for x in jax.tree.leaves(state.params.filter(config.trainable_filter))]
    train_step = jax.jit(
        lambda rng, state, batch: trainer.train_step(config, rng, state, batch),
        in_shardings=(jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
                     state_sharding, batch_sharding), out_shardings=(state_sharding, None),
        donate_argnums=(1,),
    )
    records = []
    for step in range(config.num_train_steps):
        with sharding.set_mesh(mesh):
            state, info = train_step(rng, state, next(iterator))
        metrics = {key: float(value) for key, value in jax.device_get(info).items()}
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"Nonfinite metrics at step {step}: {metrics}")
        records.append(metrics)
        print(f"Smoke step {step + 1}/{config.num_train_steps}: {metrics}", flush=True)
    final = [np.asarray(x) for x in jax.tree.leaves(state.params.filter(config.trainable_filter))]
    delta = max(float(np.max(np.abs(a - b))) for a, b in zip(initial, final, strict=True))
    if not np.isfinite(delta) or delta <= 0:
        raise AssertionError("No finite update to trainable parameters")
    manager, _ = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=None, overwrite=False, resume=False)
    try:
        checkpoints.save_state(manager, state, loader, int(state.step))
        manager.wait_until_finished()
        restored = checkpoints.restore_state(manager, state, loader)
        for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        print("Checkpoint restore matches saved training state", flush=True)
        with sharding.set_mesh(mesh):
            restored, info = train_step(rng, restored, next(iterator))
        continuation = {key: float(value) for key, value in jax.device_get(info).items()}
        if not all(np.isfinite(v) for v in continuation.values()):
            raise FloatingPointError("Nonfinite metrics after restore")
        if int(restored.step) != config.num_train_steps + 1:
            raise AssertionError("Step counter did not continue after restore")
        checkpoints.save_state(manager, restored, loader, int(restored.step))
        manager.wait_until_finished()
    finally:
        manager.close()
    result = {"passed": True, "group": resolved["group"]["name"],
              "model": "dummy framework test" if resolved["experiment"].get("tiny") else "full pi05 LoRA",
              "synthetic": True, "steps_before_restore": config.num_train_steps,
              "step_after_restore": int(restored.step), "max_parameter_update": delta,
              "metrics": records, "continuation_metrics": continuation,
              "checkpoint_dir": str(config.checkpoint_dir)}
    (Path(resolved["run_dir"]) / "smoke_result.json").write_text(json.dumps(result, indent=2))
    print("SMOKE PASSED:", json.dumps({k: v for k, v in result.items() if k != "metrics"}), flush=True)


def main():
    parser = training_parser(__doc__, default_experiment="local_smoke")
    args = parser.parse_args()
    resolved = resolve(args)
    if args.resume:
        parser.error("Smoke verifies restore itself; do not pass --resume")
    if args.dry_run:
        print(yaml.safe_dump(resolved, sort_keys=False))
        return
    config = build_train_config(resolved)
    # Fail before creating run artifacts if the full-model checkpoint/GPU is absent.
    validate_gpu(config, args.tiny)
    run_dir = write_run(resolved)
    with (run_dir / "train.log").open("a", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            run_smoke(config, resolved)


if __name__ == "__main__":
    main()
