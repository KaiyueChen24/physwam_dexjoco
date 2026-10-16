"""[数据/性能] 只测真实训练数据的 CPU 准备吞吐，不加载模型、不占 GPU、不改 run。"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import time

import numpy as np


def measure_batches(iterator, *, warmup, batches, batch_size, clock=time.perf_counter):
    """暖机不计入稳态统计；包含解码、变换、组批和 CPU Tensor 转换。"""
    for _ in range(warmup):
        next(iterator)
    waits = []
    for _ in range(batches):
        started = clock()
        next(iterator)
        waits.append(clock() - started)
    elapsed = sum(waits)
    return {"batch_size": batch_size, "warmup_batches": warmup, "measured_batches": batches,
            "seconds_per_batch_mean": float(np.mean(waits)),
            "seconds_per_batch_p95": float(np.quantile(waits, 0.95)),
            "samples_per_second": batch_size * batches / max(elapsed, 1e-9)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", required=True, help="[数据/来源] 已有 run 目录；复用训练划分、batch 和归一化资产。")
    parser.add_argument("--num-workers", required=True, type=int, help="[数据/CPU] 本次测试的加载进程数；每次运行只测一个值。")
    parser.add_argument("--warmup-batches", type=int, default=4, help="[性能/暖机] 不计时的批次数，默认 4。")
    parser.add_argument("--batches", type=int, default=30, help="[性能/预算] 实测批次数，默认 30；应大于进程数的两倍。")
    args = parser.parse_args()
    if args.num_workers < 0 or args.warmup_batches < 0 or args.batches < 1:
        parser.error("num-workers/warmup-batches 必须非负，batches 必须大于零")

    # 子进程继承这些设置；不修改外部终端环境，不调用模型初始化或统计保存。
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.pop("JAX_PLATFORM_NAME", None)
    from group_common import build_train_config, read_run
    resolved = read_run(Path(args.run_dir).expanduser().resolve())
    config = dataclasses.replace(build_train_config(resolved), num_workers=args.num_workers)
    from openpi.training import data_loader
    loader = data_loader.create_data_loader(config, shuffle=True, framework="pytorch")
    result = measure_batches(iter(loader), warmup=args.warmup_batches,
                             batches=args.batches, batch_size=config.batch_size)
    result.update({"run_dir": resolved["run_dir"], "num_workers": args.num_workers,
                   "scope": "CPU data preparation only; excludes GPU transfer, compute and communication"})
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
