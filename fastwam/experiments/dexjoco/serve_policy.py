"""OpenPI wire-compatible service without importing the OpenPI model/JAX stack."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time


async def serve(policy, host, port):
    from openpi_client import msgpack_numpy
    from websockets.asyncio.server import serve as websocket_serve

    async def handler(socket):
        packer = msgpack_numpy.Packer()
        await socket.send(packer.pack(policy.metadata))
        async for message in socket:
            try:
                obs = msgpack_numpy.unpackb(message)
                start = time.perf_counter()
                result = policy.infer(obs)
                result["server_timing"] = {"infer_ms": (time.perf_counter() - start) * 1000}
                await socket.send(packer.pack(result))
            except Exception as exc:
                await socket.send(f"{type(exc).__name__}: {exc}")
                await socket.close(code=1011, reason="Policy inference failed")
                return
    async with websocket_serve(handler, host, port, compression=None, max_size=64 * 1024 * 1024):
        print(f"FastWAM listening on ws://{host}:{port}", flush=True)
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help='[配置与实验] 已有实验目录，从中读取配置快照和资产。')
    parser.add_argument("--checkpoint", required=True, help='[权重与结果保存] 完整 FastWAM 策略 .pt 文件，不是初始化 backbone。')
    parser.add_argument("--dexjoco-root", required=True, help='[配置与实验] DexJoCo 源码目录，用于定位客户端传输包。')
    parser.add_argument("--gpus", default="0", help='[GPU 与并行] 可见 GPU 编号，逗号分隔，例如 0,1；不是 GPU 数量。')
    parser.add_argument("--host", default="127.0.0.1", help='[部署与连接] 服务监听地址；127.0.0.1 为本机，0.0.0.0 监听所有接口。')
    parser.add_argument("--port", type=int, default=8001, help='[部署与连接] 策略服务端口。')
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        parser.error("Invalid port")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(Path(__file__).resolve().parents[2] / "checkpoints"))
    # Only the small client transport package is shared, not OpenPI's training dependencies.
    sys.path.insert(0, str(Path(args.dexjoco_root).resolve() / "openpi/packages/openpi-client/src"))
    from .policy import DexJoCoFastWAMPolicy
    policy = DexJoCoFastWAMPolicy(args.run_dir, args.checkpoint)
    print(json.dumps(policy.metadata, indent=2), flush=True)
    asyncio.run(serve(policy, args.host, args.port))


if __name__ == "__main__":
    main()
