"""Offline environment checks for FastWAM; never constructs or downloads a full model."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def check_requirements(requirements):
    from packaging.requirements import Requirement

    for text in requirements:
        line = text.split("#", 1)[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        actual = metadata.version(requirement.name)
        if not requirement.specifier.contains(actual):
            raise RuntimeError(f"依赖版本不符：{requirement.name}={actual}，需要 {requirement.specifier}")


def check_environment_requirements(requirements, constraints, *, strict_versions=False):
    # The model/training dependencies still follow the upstream project's exact pins.
    check_requirements(requirements)
    if strict_versions:
        check_requirements(constraints)
    else:
        # These are runtime API requirements. Build/test tools (wheel, setuptools,
        # pytest) are installation recipe pins, not prerequisites for policy execution.
        check_requirements(["PyYAML>=6,<7", "websockets>=14,<16", "msgpack>=1.0.5,<2"])


def check_video_and_parquet():
    import av
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from fastwam.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames

    with tempfile.TemporaryDirectory(prefix="fastwam-env-check-") as directory:
        root = Path(directory)
        video = root / "sample.mp4"
        with av.open(str(video), "w") as output:
            stream = output.add_stream("mpeg4", rate=10)
            stream.width = stream.height = 16
            stream.pix_fmt = "yuv420p"
            for value in (0, 80, 160):
                frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), value, np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        decoded = decode_video_frames(video, [0.0, 0.1, 0.2], tolerance_s=0.02, backend="pyav")
        assert decoded.device.type == "cpu" and tuple(decoded.shape) == (3, 3, 16, 16), decoded.shape
        assert torch.isfinite(decoded).all()
        np.testing.assert_allclose(decoded.mean(dim=(1, 2, 3)).numpy() * 255, [0, 80, 160], atol=10)
        path = root / "sample.parquet"
        pq.write_table(pa.table({"episode_index": [0, 1], "value": [1.0, 2.0]}), path)
        assert pq.read_table(path).to_pydict() == {"episode_index": [0, 1], "value": [1.0, 2.0]}


def check_cpu(root):
    import numpy as np
    import torch
    import torchvision
    import accelerate
    import deepspeed
    import torchcodec
    import websockets.asyncio.server
    import fastwam
    from fastwam.runtime import create_fastwam
    from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset as V2Dataset
    from fastwam.datasets.lerobot3.lerobot_dataset import LeRobotDataset as V3Dataset
    from openpi_client import msgpack_numpy

    assert Path(fastwam.__file__).resolve().is_relative_to((root.parent / "fastwam/src").resolve()), fastwam.__file__
    assert torch.version.cuda == "12.8", f"需要 cu128 构建，实际为 {torch.version.cuda}"
    assert not torch.cuda.is_available(), "CPU 检查必须隐藏 GPU"
    # Exercise the same optimizer family on a tiny CPU model without allocating the actual policy.
    layer = torch.nn.Linear(23, 22)
    before = layer.weight.detach().clone()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    loss = layer(torch.ones(2, 23)).square().mean()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss) and not torch.equal(before, layer.weight)
    sample = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_array_equal(msgpack_numpy.unpackb(msgpack_numpy.packb(sample)), sample)
    check_video_and_parquet()
    import yaml
    # Check both real CLI profiles without accessing data, creating runs or initializing a model.
    for profile in ("fastwam_smoke", "fastwam_single"):
        command = [sys.executable, str(root / "scripts/train_fastwam_group.py"), "--experiment",
                   f"configs/experiments/{profile}.yaml", "--run-name", "environment_check", "--dry-run"]
        result = subprocess.run(command, cwd=root, check=True, text=True, capture_output=True, timeout=60)
        config = yaml.safe_load(result.stdout)
        assert config["framework"]["model"] == "fastwam"
        print(f"配置检查通过：{profile}，{config['framework']['launch']}")
    return {"cpu_check_passed": True, "checks": ["imports", "cpu_optimizer", "msgpack_numpy",
            "pyav_decode", "parquet", "train_cli_profiles"], "torch_cuda_build": torch.version.cuda}


def check_gpu():
    import torch

    assert torch.cuda.is_available(), "未检测到可用 CUDA GPU；请检查驱动和可见卡设置"
    assert torch.cuda.device_count() == 1, "GPU 自检只应暴露 --check-gpu 指定的一张卡"
    assert torch.cuda.is_bf16_supported(), "当前配方要求 GPU 支持 BF16"
    assert torch.distributed.is_nccl_available(), "当前 PyTorch 缺少 NCCL 支持"
    device = torch.device("cuda:0")
    layer = torch.nn.Linear(16, 16, device=device, dtype=torch.bfloat16)
    before = layer.weight.detach().clone()
    loss = layer(torch.ones(4, 16, device=device, dtype=torch.bfloat16)).float().square().mean()
    loss.backward()
    with torch.no_grad():
        layer.weight.add_(layer.weight.grad, alpha=-0.1)
    torch.cuda.synchronize()
    assert torch.isfinite(loss).item() and not torch.equal(before, layer.weight)
    name = torch.cuda.get_device_name(0)
    print("指定 GPU 的 BF16 前向/反向检查通过：", name)
    return {"gpu_check_passed": True, "gpu_name": name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("cpu", "gpu"))
    parser.add_argument("--dexjoco-root", required=True, type=Path)
    parser.add_argument("--constraints", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--strict-versions", action="store_true",
                        help="额外严格核对安装约束中的构建/测试工具版本；一键安装器使用此选项")
    args = parser.parse_args()
    root = args.dexjoco_root.resolve()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("FastWAM 安装器需要 Python 3.10")
    # The editable project's own metadata is the source of truth for all upstream version pins.
    requirements = metadata.requires("fastwam")
    if not requirements:
        raise RuntimeError("缺少 FastWAM 安装依赖元数据，请重新执行安装器")
    check_environment_requirements(requirements, args.constraints.read_text().splitlines(),
                                   strict_versions=args.strict_versions)
    sys.path.insert(0, str(root.parent / "fastwam"))
    if args.mode == "cpu":
        report = {"python": sys.version, "source_dir": str(root.parent), **check_cpu(root),
                  "strict_versions": args.strict_versions,
                  "packages": {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]}}
        print("FastWAM CPU 环境检查通过")
    else:
        report = json.loads(args.report.read_text())
        report.update(check_gpu())
        report["strict_versions"] = args.strict_versions
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
