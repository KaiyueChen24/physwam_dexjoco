"""正式训练的 W&B 步数、项目配置和 GPU 监控；CPU 小数组，无云端连接。"""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from group_common import build_train_config, load_trainer, resolve, training_parser


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.chdir(ROOT / 'openpi')
    return load_trainer()


def test_formal_profile_enables_wandb_and_passes_project_to_trainer(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    resolved = resolve(training_parser('test').parse_args(['--run-name', 'logging-test']))
    config = build_train_config(resolved)
    assert config.wandb_enabled
    assert config.project_name == 'physwam_dexjoco'
    assert config.validation is None  # split is deliberately not resolved without real data
    assert resolved['experiment']['validation_max_batches'] is None


@pytest.mark.parametrize('visible,expected', [('0,1,2,3', [0, 1, 2, 3]),
    ('4,5,6,7', [4, 5, 6, 7]), ('', []), (None, None)])
def test_wandb_new_run_and_resume_monitor_visible_cards(trainer, tmp_path, monkeypatch, visible, expected):
    if visible is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    calls = []
    fake = SimpleNamespace(Settings=lambda **kw: kw, init=lambda **kw: calls.append(kw),
                           run=SimpleNamespace(id='fixture-run'))
    monkeypatch.setattr(trainer, 'wandb', fake)
    config = SimpleNamespace(checkpoint_dir=tmp_path, exp_name='fixture', project_name='fixture-project')
    # Config must be a dataclass because the real initialization serializes it.
    monkeypatch.setattr(trainer.dataclasses, 'asdict', lambda _: {'fixture': True})
    trainer.init_wandb(config, resuming=False)
    assert calls[-1]['project'] == 'fixture-project'
    assert calls[-1]['settings']['x_stats_gpu_device_ids'] == expected
    assert (tmp_path / 'wandb_id.txt').read_text() == 'fixture-run'
    trainer.init_wandb(config, resuming=True)
    assert calls[-1]['id'] == 'fixture-run'
    assert calls[-1]['resume'] == 'must'
    assert calls[-1]['settings']['x_stats_gpu_device_ids'] == expected


@pytest.mark.parametrize('total,log_interval,save_interval,validation_interval,expected_logs,expected_saves', [
    (4, 2, 2, 2, [2, 4], [2, 4]),
    (5, 3, 3, 2, [2, 3, 4, 5], [3, 5]),
])
def test_training_validation_and_checkpoints_use_completed_updates(
        trainer, tmp_path, monkeypatch, total, log_interval, save_interval,
        validation_interval, expected_logs, expected_saves):
    import jax.numpy as jnp
    import openpi.training.dexjoco_validation as validation_module

    history, saves, evaluations = [], [], []
    cursor = [0]
    def log(values, step):
        # Match W&B's default commit behavior: a committed step cannot be logged twice.
        assert step >= cursor[0], (step, cursor[0])
        cursor[0] = step + 1
        history.append((step, dict(values)))
    monkeypatch.setattr(trainer, 'wandb', SimpleNamespace(log=log, Image=lambda value: 'image'))
    monkeypatch.setattr(trainer, 'init_wandb', lambda *a, **kw: None)
    monkeypatch.setattr(trainer, 'init_logging', lambda: None)
    monkeypatch.setattr(trainer.jax.config, 'update', lambda *a, **kw: None)
    monkeypatch.setattr(trainer.jax, 'jit', lambda function, **kw: function)
    monkeypatch.setattr(trainer.training_utils, 'array_tree_to_info', lambda _: 'fixture arrays')

    class Progress:
        def __init__(self, iterable, **kw): self.iterable = iterable
        def __iter__(self): return iter(self.iterable)
        def write(self, text): pass
    monkeypatch.setattr(trainer.tqdm, 'tqdm', Progress)
    observation = SimpleNamespace(images={'front': np.zeros((1, 4, 4, 3), dtype=np.float32)})
    batch = (observation, np.zeros((1, 2, 1), dtype=np.float32))
    class Loader:
        def __iter__(self):
            while True: yield batch
    monkeypatch.setattr(trainer._data_loader, 'create_data_loader', lambda *a, **kw: Loader())
    state = SimpleNamespace(step=0, params={'weight': jnp.ones(1)})
    monkeypatch.setattr(trainer, 'init_train_state', lambda *a, **kw: (state, None))
    def update(config, rng, state, batch):
        next_state = SimpleNamespace(step=state.step + 1, params=state.params)
        return next_state, {'loss': jnp.array(float(next_state.step)),
                            'grad_norm': jnp.array(1.0), 'param_norm': jnp.array(2.0)}
    monkeypatch.setattr(trainer, 'train_step', update)

    class Validation:
        def __init__(self, *a): pass
        def evaluate(self, params, step):
            evaluations.append(step)
            return {'val_loss': 0.2, 'task_losses': {'water_plant': 0.1, 'hammer_nail': 0.3}}
    monkeypatch.setattr(validation_module, 'ValidationRunner', Validation)
    manager = SimpleNamespace(wait_until_finished=lambda: None)
    monkeypatch.setattr(trainer._checkpoints, 'initialize_checkpoint_dir', lambda *a, **kw: (manager, False))
    monkeypatch.setattr(trainer._checkpoints, 'save_state',
        lambda manager, state, loader, step: saves.append((step, state.step)))
    config = SimpleNamespace(batch_size=1, fsdp_devices=1, seed=42, checkpoint_dir=tmp_path,
        keep_period=None, overwrite=False, resume=False, wandb_enabled=True,
        validation={'interval': validation_interval}, num_train_steps=total,
        log_interval=log_interval, save_interval=save_interval)

    trainer.main(config)
    assert [step for step, _ in history] == [0, *expected_logs]
    assert saves == [(step, step) for step in expected_saves]
    assert evaluations == [step for step in range(1, total + 1)
                           if step % validation_interval == 0 or step == total]
    records = dict(history)
    for step in evaluations:
        assert records[step]['val/water_plant/loss'] == 0.1
        assert records[step]['val/hammer_nail/loss'] == 0.3
    for step in expected_logs:
        if step % log_interval == 0 or step == total:
            assert 'loss' in records[step]
    # Loss logging averages batches since the preceding training log.
    first_logged = log_interval if log_interval <= total else total
    assert records[first_logged]['loss'] == pytest.approx((1 + first_logged) / 2)
