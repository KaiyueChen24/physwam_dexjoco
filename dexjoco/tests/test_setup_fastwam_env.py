"""Installation workflow checks using isolated Conda/proxy stubs, without downloads."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/setup_fastwam_env.sh"
CONDA_STUB = r'''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
state = pathlib.Path(os.environ['TEST_ENV_STATE'])
event = {'args': args, 'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'),
         'proxy': os.environ.get('https_proxy'), 'ds_ops': os.environ.get('DS_BUILD_OPS'),
         'hf_home': os.environ.get('HF_HOME'), 'offline': os.environ.get('HF_HUB_OFFLINE')}
with open(os.environ['TEST_EVENTS'], 'a') as out:
    out.write(json.dumps(event) + '\n')
if args[:2] == ['info', '--base']:
    print(os.environ['TEST_CONDA_BASE'])
elif args[0] == 'list':
    sys.exit(0 if state.exists() else 1)
elif args[0] in ('create', 'install'):
    assert 'python=3.10' in args and 'ffmpeg=7' in args and 'cxx-compiler' in args
    assert not any('evdev' in item for item in args)
    state.touch()
elif args[0] == 'run':
    pyargs = args[args.index('python')+1:]
    if pyargs[:1] == ['-c']:
        if 'sys.version_info' in pyargs[1] and os.environ.get('TEST_BAD_PYTHON'):
            print('已有环境必须使用 Python 3.10', file=sys.stderr)
            sys.exit(9)
        if 'print(sys.prefix)' in pyargs[1]:
            print(os.environ['TEST_ENV_PREFIX'])
    elif pyargs[:3] == ['-m', 'pip', 'install']:
        assert state.exists()
        if os.environ.get('TEST_FAIL_PIP'):
            sys.exit(42)
    elif pyargs and pyargs[0].endswith('check_fastwam_env.py'):
        if os.environ.get('TEST_FAIL_CHECK'):
            sys.exit(43)
else:
    raise AssertionError(args)
'''


@pytest.fixture
def setup_env(tmp_path):
    home, binary = tmp_path / "home", tmp_path / "bin"
    home.mkdir()
    binary.mkdir()
    stub = binary / "conda"
    stub.write_text(CONDA_STUB)
    stub.chmod(0o755)
    base = tmp_path / "conda base"
    base.mkdir()
    env = dict(os.environ, HOME=str(home), CONDA_EXE=str(stub), PATH=f"{binary}:/usr/bin:/bin",
               TEST_CONDA_BASE=str(base), TEST_ENV_PREFIX=str(tmp_path / "env fastwam"),
               TEST_ENV_STATE=str(tmp_path / "env.exists"), TEST_EVENTS=str(tmp_path / "events.jsonl"))
    return tmp_path, env


def run_installer(setup_env, *args):
    temp, env = setup_env
    result = subprocess.run(["bash", str(SCRIPT), "--cache-dir", str(temp / "cache"),
                             "--model-dir", str(temp / "wan models"), *args],
                            env=env, cwd=temp, text=True, capture_output=True, timeout=30)
    events_path = Path(env["TEST_EVENTS"])
    events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
    return result, events


def test_dry_run_never_calls_conda_proxy_or_creates_files(setup_env):
    temp, env = setup_env
    result, events = run_installer(setup_env, "--setup-proxy", "--proxy-port", "18888", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "GPU 检查：关闭" in result.stdout and "18888" in result.stdout
    assert not events and not (temp / "cache").exists() and not (temp / "wan models").exists()
    assert not (Path(env["HOME"]) / ".config").exists()


@pytest.mark.parametrize("args", [
    ("--name", "base"), ("--name", "fastwam", "--prefix", "/tmp/env"),
    ("--prefix", "/"), ("--name",), ("--check-gpu", "0,1"),
    ("--model-source", "unknown"), ("--proxy", "bad-url"),
    ("--setup-proxy", "--proxy", "http://127.0.0.1:1"), ("--proxy-port", "17890"),
    ("--setup-proxy", "--proxy-port", "80"), ("--setup-proxy", "--proxy-port", "65536"),
    ("--setup-proxy", "--subscription-file", "/missing/sub.yaml"),
])
def test_invalid_arguments_fail_before_mutation(setup_env, args):
    temp, _ = setup_env
    result, events = run_installer(setup_env, *args)
    assert result.returncode != 0 and not events
    assert not (temp / "cache").exists() and not (temp / "wan models").exists()


def test_install_order_cpu_default_and_independent_settings(setup_env):
    _, env = setup_env
    settings = Path(env["HOME"]) / ".config/physwam"
    settings.mkdir(parents=True)
    (settings / "pi05-env.sh").write_text("pi05 configuration stays intact")
    result, events = run_installer(setup_env, "--proxy", "http://127.0.0.1:18888")
    assert result.returncode == 0, result.stdout + result.stderr
    installs = [e for e in events if "pip" in e["args"] and "install" in e["args"]]
    assert any(e["args"][0] == "create" for e in events)
    assert "torch==2.7.1+cu128" in installs[1]["args"]
    assert "deepspeed==0.18.7" in installs[2]["args"]
    assert "--no-build-isolation" in installs[2]["args"] and installs[2]["ds_ops"] == "0"
    assert "openpi-client" in " ".join(installs[3]["args"])
    assert not any("jax" in " ".join(e["args"]).lower() for e in installs)
    checks = [e for e in events if any(a.endswith("check_fastwam_env.py") for a in e["args"])]
    assert len(checks) == 1 and "cpu" in checks[0]["args"] and checks[0]["offline"] == "1"
    assert all(e["cuda"] == "" for e in events if e["args"][0] == "run")
    assert all(e["proxy"] == "http://127.0.0.1:18888" for e in events)
    assert (settings / "pi05-env.sh").read_text() == "pi05 configuration stays intact"
    assert "FastWAM 环境配置完成" in result.stdout


def test_existing_wrong_python_fails_before_package_changes(setup_env):
    _, env = setup_env
    Path(env["TEST_ENV_STATE"]).touch()
    env["TEST_BAD_PYTHON"] = "1"
    result, events = run_installer(setup_env)
    assert result.returncode != 0
    assert "Python 3.10" in result.stdout + result.stderr
    assert not any(e["args"][0] in ("create", "install") or "pip" in e["args"] for e in events)


def test_base_prefix_is_never_modified(setup_env):
    _, env = setup_env
    result, events = run_installer(setup_env, "--prefix", env["TEST_CONDA_BASE"])
    assert result.returncode != 0
    assert not any(e["args"][0] in ("create", "install", "run") for e in events)


def test_prefix_and_explicit_gpu_check(setup_env):
    temp, env = setup_env
    prefix = temp / "environment with spaces"
    env["TEST_ENV_PREFIX"] = str(prefix)
    Path(env["TEST_ENV_STATE"]).touch()
    result, events = run_installer(setup_env, "--prefix", str(prefix), "--check-gpu", "3")
    assert result.returncode == 0, result.stdout + result.stderr
    for event in events:
        if event["args"][0] != "info":
            assert event["args"][event["args"].index("-p") + 1] == str(prefix)
    gpu = [e for e in events if "gpu" in e["args"]]
    assert len(gpu) == 1 and gpu[0]["cuda"] == "3"
    assert all(e["cuda"] == "" for e in events if e["args"][0] == "run" and "pip" in e["args"])


@pytest.mark.parametrize("failure,status", [("TEST_FAIL_PIP", 42), ("TEST_FAIL_CHECK", 43)])
def test_failure_stops_before_success_or_overwriting_settings(setup_env, failure, status):
    _, env = setup_env
    env[failure] = "1"
    settings = Path(env["HOME"]) / ".config/physwam/fastwam-env.sh"
    settings.parent.mkdir(parents=True)
    settings.write_text("previous working settings")
    result, events = run_installer(setup_env, "--check-gpu", "0")
    assert result.returncode == status, result.stdout + result.stderr
    assert "安装失败，阶段" in result.stdout + result.stderr
    assert "FastWAM 环境配置完成" not in result.stdout
    assert not any("gpu" in e["args"] for e in events)
    assert settings.read_text() == "previous working settings"


def test_persisted_paths_and_model_source_round_trip_safely(setup_env):
    temp, env = setup_env
    cache = temp / "cache ' $(touch BAD) `touch BAD2`"
    model_dir = temp / "Wan models with spaces"
    result, events = run_installer(setup_env, "--cache-dir", str(cache), "--model-dir", str(model_dir),
                                   "--model-source", "modelscope", "--proxy", "http://127.0.0.1:18888")
    assert result.returncode == 0, result.stdout + result.stderr
    settings = Path(env["HOME"]) / ".config/physwam/fastwam-env.sh"
    assert settings.stat().st_mode & 0o777 == 0o600
    command = ['bash', '-c', 'source "$1"; /usr/bin/python3 -c \'import os,json; print(json.dumps(dict(os.environ)))\'',
               'source-test', str(settings)]
    sourced = subprocess.run(command, env=env, cwd=temp, text=True, capture_output=True, check=True)
    values = json.loads(sourced.stdout)
    assert values["HF_HOME"] == str(cache / "huggingface")
    assert values["MODELSCOPE_CACHE"] == str(cache / "modelscope")
    assert values["TORCH_EXTENSIONS_DIR"] == str(cache / "torch_extensions")
    assert values["DIFFSYNTH_MODEL_BASE_PATH"] == str(model_dir)
    assert values["DIFFSYNTH_DOWNLOAD_SOURCE"] == "modelscope"
    assert values["https_proxy"] == "http://127.0.0.1:18888"
    assert not (temp / "BAD").exists() and not (temp / "BAD2").exists()
    assert not (Path(env["HOME"]) / ".bashrc").exists()


def test_proxy_helper_runs_before_conda_and_failure_stops_installation(setup_env):
    temp, env = setup_env
    wrapper = Path(env["CONDA_EXE"]).parent / "python3"
    wrapper.write_text('''#!/bin/bash
if [[ "$1" == */scripts/setup_mihomo.py ]]; then
    printf '%s\\n' "$@" > "$TEST_HELPER_ARGS"
    if [[ -n "${TEST_HELPER_FAIL:-}" ]]; then exit 17; fi
    exit 0
fi
exec /usr/bin/python3 "$@"
''')
    wrapper.chmod(0o755)
    env["TEST_HELPER_ARGS"] = str(temp / "helper.args")
    binary, subscription = temp / "mihomo.gz", temp / "sub.yaml"
    binary.touch()
    subscription.write_text("private subscription fixture")
    result, events = run_installer(setup_env, "--setup-proxy", "--proxy-port", "18888",
                                   "--mihomo-binary", str(binary), "--subscription-file", str(subscription))
    assert result.returncode == 0, result.stdout + result.stderr
    args = Path(env["TEST_HELPER_ARGS"]).read_text().splitlines()
    assert args[1:] == ["--port", "18888", "--binary", str(binary), "--subscription-file", str(subscription)]
    assert all(e["proxy"] == "http://127.0.0.1:18888" for e in events)
    assert "private subscription fixture" not in result.stdout + result.stderr
    Path(env["TEST_EVENTS"]).unlink()
    env["TEST_HELPER_FAIL"] = "1"
    result, events = run_installer(setup_env, "--setup-proxy")
    assert result.returncode == 17 and not events


def test_missing_conda_bootstraps_miniforge(setup_env):
    temp, env = setup_env
    stub = Path(env["CONDA_EXE"])
    source = temp / "conda-stub-source"
    source.write_text(CONDA_STUB)
    source.chmod(0o755)
    stub.unlink()
    env.pop("CONDA_EXE")
    env["TEST_CONDA_BASE"] = str(Path(env["HOME"]) / "miniforge3")
    env["TEST_CONDA_STUB_SOURCE"] = str(source)
    curl = stub.parent / "curl"
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
    assert (Path(env["TEST_CONDA_BASE"]) / "bin/conda").is_file()
    assert any(e["args"][0] == "create" for e in events)
