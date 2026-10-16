"""数据吞吐测量必须排除启动暖机，并对真实取批耗时计算统计。"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_pi05_data import measure_batches


def test_benchmark_excludes_warmup_and_reports_tail_latency():
    now = [0.0]
    def stream():
        for cost in [100, 100, 1, 2, 3]:
            now[0] += cost
            yield object()
    result = measure_batches(iter(stream()), warmup=2, batches=3,
                             batch_size=128, clock=lambda: now[0])
    assert result["seconds_per_batch_mean"] == 2
    assert result["seconds_per_batch_p95"] == pytest.approx(2.9)
    assert result["samples_per_second"] == 64
    assert result["measured_batches"] == 3
