"""Behavior checks for dependency validation and actual CPU media decoding."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_fastwam_env", ROOT / "scripts/check_fastwam_env.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_requirements_check_rejects_wrong_cuda_wheel_and_respects_markers(monkeypatch):
    installed = {"torch": "2.7.1+cu128", "PyYAML": "6.0.2"}
    monkeypatch.setattr(checker.metadata, "version", installed.__getitem__)
    checker.check_requirements(["# comment", "", "torch==2.7.1+cu128", "PyYAML==6.0.2 # config",
                                'missing==1; python_version < "3.0"'])
    installed["torch"] = "2.7.1+cu126"
    with pytest.raises(RuntimeError, match="依赖版本不符"):
        checker.check_requirements(["torch==2.7.1+cu128"])


def test_cpu_video_and_parquet_round_trip(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT.parent / "fastwam/src"))
    checker.check_video_and_parquet()


def test_manual_environment_accepts_newer_build_tools_and_runtime_patch_versions(monkeypatch):
    installed = {"torch": "2.7.1+cu128", "wheel": "0.48.0", "setuptools": "81.0.0",
                 "pytest": "9.0.0", "PyYAML": "6.0.3", "websockets": "15.0.1", "msgpack": "1.1.2"}
    requested = []
    def version(name):
        requested.append(name)
        return installed[name]
    monkeypatch.setattr(checker.metadata, "version", version)
    checker.check_environment_requirements(["torch==2.7.1+cu128"], ["wheel==0.45.1", "pytest==8.3.5"])
    assert not {"wheel", "setuptools", "pytest"}.intersection(requested)


def test_strict_installer_still_requires_the_pinned_build_tools(monkeypatch):
    installed = {"torch": "2.7.1+cu128", "wheel": "0.48.0"}
    monkeypatch.setattr(checker.metadata, "version", installed.__getitem__)
    with pytest.raises(RuntimeError, match="wheel=0.48.0"):
        checker.check_environment_requirements(["torch==2.7.1+cu128"], ["wheel==0.45.1"], strict_versions=True)


def test_manual_environment_still_rejects_wrong_training_dependencies(monkeypatch):
    monkeypatch.setattr(checker.metadata, "version", lambda name: "2.7.1+cu126")
    with pytest.raises(RuntimeError, match="torch=2.7.1"):
        checker.check_environment_requirements(["torch==2.7.1+cu128"], [])


def test_manual_environment_rejects_unsupported_runtime_api(monkeypatch):
    installed = {"PyYAML": "6.0.3", "websockets": "13.1", "msgpack": "1.1.2"}
    monkeypatch.setattr(checker.metadata, "version", installed.__getitem__)
    with pytest.raises(RuntimeError, match="websockets=13.1"):
        checker.check_environment_requirements([], [])


def test_gpu_check_fails_on_cpu_environment():
    import torch
    if torch.cuda.is_available():
        pytest.skip("requires a CPU-only test process")
    with pytest.raises(AssertionError, match="未检测到可用 CUDA GPU"):
        checker.check_gpu()


def test_current_training_profiles_without_removed_smoke_config():
    configs = checker.check_training_profiles(ROOT)
    assert set(configs) == {"fastwam_single", "fastwam_dual", "fastwam_single_rand_full", "fastwam_dual_rand_full"}
    dual_full = configs["fastwam_dual_rand_full"]
    assert dual_full["framework"]["group"]["data_regime"] == "rand_full"
    assert len(dual_full["data"]["train"]["tasks"]) == 5
    assert dual_full["framework"]["launch"]["num_processes"] == 8
