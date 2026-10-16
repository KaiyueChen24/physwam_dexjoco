"""代理初始化的凭据处理、安装来源与进程归属检查；不访问公网。"""
import gzip
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/setup_mihomo.py'
spec = importlib.util.spec_from_file_location('setup_mihomo', SCRIPT)
mihomo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mihomo)


def test_private_atomic_file_modes_and_replacement(tmp_path):
    target = tmp_path / 'subscription.yaml'
    mihomo.private_write(target, b'old')
    mihomo.private_write(target, b'new')
    assert target.read_bytes() == b'new'
    assert target.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == [target]


def test_offline_binary_decompresses_without_download(tmp_path, monkeypatch):
    archive = tmp_path / 'uploaded.gz'
    archive.write_bytes(gzip.compress(b'fake-mihomo'))
    calls = []
    monkeypatch.setattr(mihomo.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('network accessed'))
    monkeypatch.setattr(mihomo.subprocess, 'run', lambda *a, **k: calls.append((a, k)))
    target = mihomo.install_binary(tmp_path, archive)
    assert target.read_bytes() == b'fake-mihomo'
    assert target.stat().st_mode & 0o777 == 0o700
    assert calls[0][0][0] == [str(target), '-v']


def test_existing_binary_is_reused_without_network(tmp_path, monkeypatch):
    target = tmp_path / 'mihomo'
    target.write_bytes(b'cached')
    target.chmod(0o700)
    monkeypatch.setattr(mihomo.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('network accessed'))
    assert mihomo.install_binary(tmp_path) == target


def test_official_download_checksum_is_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(mihomo.shutil, 'which', lambda _: None)
    monkeypatch.setattr(mihomo.Path, 'home', lambda: tmp_path)
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'tampered download'
    monkeypatch.setattr(mihomo.urllib.request, 'urlopen', lambda *a, **k: Response())
    with pytest.raises(RuntimeError, match='SHA256'):
        mihomo.install_binary(tmp_path)
    assert not (tmp_path / 'mihomo').exists()


@pytest.mark.parametrize('content', [b'', b' ', b'<html>error</html>', b'<!DOCTYPE html>'])
def test_invalid_subscription_rejected(tmp_path, content):
    source = tmp_path / 'sub'
    source.write_bytes(content)
    with pytest.raises(RuntimeError):
        mihomo.subscription_bytes(source)


def test_subscription_download_errors_do_not_expose_credentials(monkeypatch):
    secret_url = 'https://example.invalid/private-token-test-fixture'
    monkeypatch.setattr(mihomo.getpass, 'getpass', lambda *a, **k: secret_url)
    class Opener:
        def open(self, *args, **kwargs):
            raise OSError(secret_url)
    monkeypatch.setattr(mihomo.urllib.request, 'build_opener', lambda *a: Opener())
    # Replace /dev/tty only; no interaction is possible in this unit test.
    import io
    monkeypatch.setattr(mihomo, 'open', lambda *a, **k: io.StringIO(), raising=False)
    with pytest.raises(RuntimeError) as caught:
        mihomo.subscription_bytes()
    assert 'private-token' not in str(caught.value)
    assert '订阅下载失败' in str(caught.value)


def test_generated_config_limits_exposure_and_uses_local_provider():
    config = mihomo.configuration(18888).decode()
    assert 'mixed-port: 18888' in config
    assert 'bind-address: 127.0.0.1' in config
    assert 'allow-lan: false' in config
    assert 'path: ./subscription.yaml' in config
    assert 'external-controller' not in config and 'tun:' not in config


def save_record(state_dir, **changes):
    record = dict(pid=1234, hostname=socket.gethostname(), port=17890, start='original',
                  argv=[str(state_dir / 'mihomo'), '-d', str(state_dir)])
    record.update(changes)
    (state_dir / 'process.json').write_text(json.dumps(record))
    return record


@pytest.mark.parametrize('mismatch', ['hostname', 'start', 'argv'])
def test_stale_or_other_server_process_is_not_owned(tmp_path, monkeypatch, mismatch):
    record = save_record(tmp_path)
    identity = (record['start'], record['argv'])
    if mismatch == 'hostname':
        save_record(tmp_path, hostname='another-server')
    elif mismatch == 'start':
        identity = ('reused-pid', record['argv'])
    else:
        identity = (record['start'], ['/other/process'])
    monkeypatch.setattr(mihomo, 'process_identity', lambda _: identity)
    assert mihomo.running_record(tmp_path) is None


def test_running_proxy_is_reused_and_changes_require_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(mihomo, 'running_record', lambda _: {'port': 17890})
    checks = []
    monkeypatch.setattr(mihomo, 'proxy_check', checks.append)
    mihomo.prepare(tmp_path, 17890)
    assert checks == [17890]
    with pytest.raises(RuntimeError, match='先执行'):
        mihomo.prepare(tmp_path, 18888)
    with pytest.raises(RuntimeError, match='先执行'):
        mihomo.prepare(tmp_path, 17890, subscription_file='new-subscription')


def test_occupied_port_fails_without_installing_or_killing(tmp_path, monkeypatch):
    class Probe:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, *args): raise OSError('in use')
    monkeypatch.setattr(mihomo.socket, 'socket', Probe)
    monkeypatch.setattr(mihomo, 'install_binary', lambda *a: pytest.fail('binary installation attempted'))
    with pytest.raises(RuntimeError, match='占用'):
        mihomo.prepare(tmp_path, 17890)


def test_failed_proxy_check_terminates_only_new_child(tmp_path, monkeypatch):
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, *args): pass
    class Child:
        pid = 321
        terminated = False
        def poll(self): return None
        def terminate(self): self.terminated = True
        def wait(self, **kwargs): pass
    child = Child()
    monkeypatch.setattr(mihomo.socket, 'socket', Socket)
    monkeypatch.setattr(mihomo.socket, 'create_connection', lambda *a, **k: Socket())
    monkeypatch.setattr(mihomo, 'install_binary', lambda *a: tmp_path / 'mihomo')
    monkeypatch.setattr(mihomo, 'subscription_bytes', lambda *a: b'proxies: []')
    monkeypatch.setattr(mihomo.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0))
    calls = []
    def spawn(*args, **kwargs):
        calls.append(kwargs)
        return child
    monkeypatch.setattr(mihomo.subprocess, 'Popen', spawn)
    def fail_check(port): raise RuntimeError('unreachable')
    monkeypatch.setattr(mihomo, 'proxy_check', fail_check)
    with pytest.raises(RuntimeError, match='unreachable'):
        mihomo.prepare(tmp_path, 17890)
    assert child.terminated
    assert calls[0]['start_new_session'] is True
    assert not (tmp_path / 'process.json').exists()


def test_proxy_check_uses_explicit_proxy_and_rejects_failed_egress(monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(mihomo.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='HTTPS'):
        mihomo.proxy_check(18888)
    assert 'http://127.0.0.1:18888' in calls[0][0]
    assert calls[0][1]['stdout'] == subprocess.DEVNULL
