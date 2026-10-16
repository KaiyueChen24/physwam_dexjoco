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


def test_gpu_check_fails_on_cpu_environment():
    import torch
    if torch.cuda.is_available():
        pytest.skip("requires a CPU-only test process")
    with pytest.raises(AssertionError, match="未检测到可用 CUDA GPU"):
        checker.check_gpu()
