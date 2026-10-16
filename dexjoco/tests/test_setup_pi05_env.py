"""环境安装器的流程检查；替代 Conda/下载命令，不下载依赖、不修改真实环境。"""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/setup_pi05_env.sh"

CONDA_STUB = r'''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
state = pathlib.Path(os.environ['TEST_ENV_STATE'])
record = {'args': args, 'jax_platforms': os.environ.get('JAX_PLATFORMS'),
          'cuda_visible': os.environ.get('CUDA_VISIBLE_DEVICES'),
          'proxy': os.environ.get('https_proxy'), 'hf_home': os.environ.get('HF_HOME'),
          'openpi_cache': os.environ.get('OPENPI_DATA_HOME')}
with open(os.environ['TEST_EVENTS'], 'a') as out:
    out.write(json.dumps(record) + '\n')
if args[:2] == ['info', '--base']:
    print(os.environ['TEST_CONDA_BASE'])
elif args[0] == 'list':
    sys.exit(0 if state.exists() else 1)
elif args[0] in ('create', 'install'):
    assert 'evdev=2.0.0' in args and 'python=3.11' in args
    state.touch()
elif args[0] == 'run':
    pos = args.index('python')
    pyargs = args[pos + 1:]
    if pyargs[:1] == ['-c']:
        code = pyargs[1]
        if 'sys.version_info' in code and os.environ.get('TEST_BAD_PYTHON'):
            print('已有环境必须使用 Python 3.11', file=sys.stderr)
            sys.exit(9)
        if 'print(sys.prefix)' in code:
            print(os.environ['TEST_ENV_PREFIX'])
    elif pyargs[:3] == ['-m', 'pip', 'install']:
        assert state.exists(), 'Conda must run before pip'
        if os.environ.get('TEST_FAIL_PIP'):
            print('pip failure fixture', file=sys.stderr)
            sys.exit(42)
    elif pyargs[:1] == ['-']:
        sys.stdin.read()
else:
    raise AssertionError(args)
'''


@pytest.fixture
def setup_env(tmp_path):
    home = tmp_path / 'home'
    home.mkdir()
    binary_dir = tmp_path / 'bin'
    binary_dir.mkdir()
    stub = binary_dir / 'conda'
    stub.write_text(CONDA_STUB)
    stub.chmod(0o755)
    base = tmp_path / 'conda base'
    base.mkdir()
    env = dict(os.environ, HOME=str(home), CONDA_EXE=str(stub),
               TEST_CONDA_BASE=str(base), TEST_ENV_PREFIX=str(tmp_path / 'env pi05'),
               TEST_ENV_STATE=str(tmp_path / 'env.exists'), TEST_EVENTS=str(tmp_path / 'events.jsonl'),
               PATH=f'{binary_dir}:/usr/bin:/bin')
    return tmp_path, env


def run_installer(setup_env, *args):
    temp, env = setup_env
    result = subprocess.run(['bash', str(SCRIPT), '--cache-dir', str(temp / 'cache'), *args],
                            env=env, cwd=temp, text=True, capture_output=True, timeout=30)
    events_path = Path(env['TEST_EVENTS'])
    events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
    return result, events


def test_dry_run_does_not_call_conda_or_create_cache(setup_env):
    temp, _ = setup_env
    result, events = run_installer(setup_env, '--dry-run', '--proxy', 'http://127.0.0.1:17890')
    assert result.returncode == 0, result.stderr
    assert 'GPU 检查：关闭' in result.stdout
    assert not events and not (temp / 'cache').exists()


@pytest.mark.parametrize('args', [('--name', 'base'), ('--name', 'pi05', '--prefix', '/tmp/test'),
                                  ('--check-gpu', '0,1'), ('--name',), ('--proxy', 'bad-url')])
def test_invalid_options_fail_before_mutation(setup_env, args):
    temp, _ = setup_env
    result, events = run_installer(setup_env, *args)
    assert result.returncode != 0
    assert not events and not (temp / 'cache').exists()


def test_fresh_environment_uses_conda_evdev_and_cpu_checks_by_default(setup_env):
    result, events = run_installer(setup_env, '--proxy', 'http://127.0.0.1:17890')
    assert result.returncode == 0, result.stdout + result.stderr
    assert any(e['args'][0] == 'create' for e in events)
    assert not any(e['args'][0] == 'install' for e in events)
    evdev_check = next(i for i, e in enumerate(events) if 'import evdev._ecodes' in ' '.join(e['args']))
    pip_install = next(i for i, e in enumerate(events) if 'pip' in e['args'] and 'install' in e['args'])
    assert evdev_check < pip_install
    assert any(e['jax_platforms'] == 'cpu' and e['cuda_visible'] == '' for e in events)
    assert not any(e['jax_platforms'] == 'cuda' for e in events)
    assert all(e['proxy'] == 'http://127.0.0.1:17890' for e in events)
    assert 'π0.5 环境配置完成' in result.stdout


def test_existing_wrong_python_is_rejected_before_conda_install(setup_env):
    _, env = setup_env
    Path(env['TEST_ENV_STATE']).touch()
    env['TEST_BAD_PYTHON'] = '1'
    result, events = run_installer(setup_env)
    assert result.returncode != 0
    assert 'Python 3.11' in result.stdout + result.stderr
    assert not any(e['args'][0] in ('create', 'install') for e in events)
    assert not any('pip' in e['args'] for e in events)


def test_explicit_prefix_and_gpu_selection_are_forwarded(setup_env):
    temp, env = setup_env
    prefix = temp / 'env with spaces'
    env['TEST_ENV_PREFIX'] = str(prefix)
    Path(env['TEST_ENV_STATE']).touch()
    result, events = run_installer(setup_env, '--prefix', str(prefix), '--check-gpu', '3')
    assert result.returncode == 0, result.stdout + result.stderr
    for event in events:
        if event['args'][0] != 'info':
            assert event['args'][event['args'].index('-p') + 1] == str(prefix)
    assert any(e['jax_platforms'] == 'cuda' and e['cuda_visible'] == '3' for e in events)


def test_base_prefix_is_rejected_without_package_changes(setup_env):
    _, env = setup_env
    result, events = run_installer(setup_env, '--prefix', env['TEST_CONDA_BASE'])
    assert result.returncode != 0
    assert not any(e['args'][0] in ('create', 'install', 'run') for e in events)


def test_install_failure_stops_before_import_and_gpu_checks(setup_env):
    _, env = setup_env
    env['TEST_FAIL_PIP'] = '1'
    result, events = run_installer(setup_env, '--check-gpu', '0')
    assert result.returncode == 42, result.stdout + result.stderr
    assert '安装失败，阶段：安装 OpenPI' in result.stdout + result.stderr
    assert not any(e['jax_platforms'] in ('cpu', 'cuda') for e in events)
    assert 'π0.5 环境配置完成' not in result.stdout


def test_missing_conda_bootstraps_into_user_home(setup_env):
    temp, env = setup_env
    source = Path(env['CONDA_EXE'])
    copy = temp / 'conda-stub-source'
    copy.write_text(CONDA_STUB)
    copy.chmod(0o755)
    source.unlink()
    env.pop('CONDA_EXE')
    env['TEST_CONDA_BASE'] = str(Path(env['HOME']) / 'miniforge3')
    env['TEST_CONDA_STUB_SOURCE'] = str(copy)
    curl = source.parent / 'curl'
    curl.write_text('''#!/bin/bash
while [[ $# -gt 0 ]]; do
    if [[ "$1" == -o ]]; then output="$2"; break; fi
    shift
done
cat > "$output" <<'INSTALLER'
#!/bin/bash
while [[ $# -gt 0 ]]; do
    if [[ "$1" == -p ]]; then prefix="$2"; break; fi
    shift
done
mkdir -p "$prefix/bin"
cp "$TEST_CONDA_STUB_SOURCE" "$prefix/bin/conda"
INSTALLER
''')
    curl.chmod(0o755)
    result, events = run_installer(setup_env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (Path(env['TEST_CONDA_BASE']) / 'bin/conda').is_file()
    assert any(e['args'][0] == 'create' for e in events)
    assert not (Path(env['HOME']) / '.bashrc').exists()


@pytest.mark.parametrize('args', [('--setup-proxy', '--proxy', 'http://127.0.0.1:1'),
    ('--proxy-port', '17890'), ('--setup-proxy', '--proxy-port', '80'),
    ('--setup-proxy', '--proxy-port', '65536'), ('--setup-proxy', '--proxy-port', 'abc'),
    ('--setup-proxy', '--mihomo-binary', '/missing/mihomo.gz')])
def test_invalid_proxy_setup_does_not_mutate(setup_env, args):
    temp, _ = setup_env
    result, events = run_installer(setup_env, *args)
    assert result.returncode != 0
    assert not events and not (temp / 'cache').exists()


def test_proxy_setup_dry_run_is_noninteractive_and_creates_nothing(setup_env):
    temp, env = setup_env
    result, events = run_installer(setup_env, '--setup-proxy', '--proxy-port', '18888', '--dry-run')
    assert result.returncode == 0, result.stderr
    assert '18888' in result.stdout and '准备 Mihomo' in result.stdout
    assert not events and not (temp / 'cache').exists()
    assert not (Path(env['HOME']) / '.config').exists()


def fake_mihomo_helper(setup_env, failure=False):
    temp, env = setup_env
    wrapper = Path(env['CONDA_EXE']).parent / 'python3'
    wrapper.write_text('''#!/bin/bash
if [[ "$1" == */scripts/setup_mihomo.py ]]; then
    printf '%s\\n' "$@" > "$TEST_HELPER_ARGS"
    if [[ "${TEST_HELPER_FAIL:-}" == 1 ]]; then exit 17; fi
    exit 0
fi
exec /usr/bin/python3 "$@"
''')
    wrapper.chmod(0o755)
    env['TEST_HELPER_ARGS'] = str(temp / 'helper.args')
    if failure:
        env['TEST_HELPER_FAIL'] = '1'
    return Path(env['TEST_HELPER_ARGS'])


def test_proxy_setup_forwards_files_port_and_sets_proxy_before_conda(setup_env):
    temp, _ = setup_env
    helper_args = fake_mihomo_helper(setup_env)
    binary = temp / 'uploaded mihomo.gz'
    subscription = temp / 'subscription.yaml'
    binary.write_bytes(b'test')
    subscription.write_bytes(b'private subscription fixture')
    result, events = run_installer(setup_env, '--setup-proxy', '--proxy-port', '18888',
        '--mihomo-binary', str(binary), '--subscription-file', str(subscription))
    assert result.returncode == 0, result.stdout + result.stderr
    args = helper_args.read_text().splitlines()
    assert args[1:] == ['--port', '18888', '--binary', str(binary), '--subscription-file', str(subscription)]
    assert events and all(event['proxy'] == 'http://127.0.0.1:18888' for event in events)
    assert 'private subscription fixture' not in result.stdout + result.stderr


def test_proxy_failure_stops_before_any_conda_command(setup_env):
    fake_mihomo_helper(setup_env, failure=True)
    result, events = run_installer(setup_env, '--setup-proxy')
    assert result.returncode == 17, result.stdout + result.stderr
    assert '准备 Mihomo 代理' in result.stdout + result.stderr
    assert not events


def test_cache_and_proxy_file_roundtrip_with_shell_special_characters(setup_env):
    temp, env = setup_env
    cache = temp / "aifs cache ' $(touch BAD) `touch BAD2`"
    result, events = run_installer(setup_env, '--cache-dir', str(cache), '--proxy', 'http://127.0.0.1:18888')
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(e['hf_home'] == str(cache / 'huggingface') for e in events)
    assert all(e['openpi_cache'] == str(cache / 'openpi') for e in events)
    envfile = Path(env['HOME']) / '.config/physwam/pi05-env.sh'
    assert envfile.stat().st_mode & 0o777 == 0o600
    sourced = subprocess.run(['bash', '-c',
        'source "$1"; /usr/bin/python3 -c \'import os,json; print(json.dumps(dict(os.environ)))\'',
        'source-test', str(envfile)], env=env, cwd=temp, text=True, capture_output=True, check=True)
    settings = json.loads(sourced.stdout)
    assert settings['HF_HOME'] == str(cache / 'huggingface')
    assert settings['https_proxy'] == 'http://127.0.0.1:18888'
    assert not (temp / 'BAD').exists() and not (temp / 'BAD2').exists()
    assert not (Path(env['HOME']) / '.bashrc').exists()
