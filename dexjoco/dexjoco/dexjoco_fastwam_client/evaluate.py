"""Synchronous closed-loop evaluator with explicit episode/step alignment."""
import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import yaml


def validate_metadata(actual, expected):
    for key in ("model", "group", "run_name", "config_signature", "checkpoint_sha256", "stats_sha256", "action_dim",
                "state_dim", "action_horizon", "camera_order", "control_hz", "image_size"):
        if key not in expected or actual.get(key) != expected[key]:
            raise ValueError(f"Server mismatch: {key}")
    if actual["model"] != "fastwam":
        raise ValueError("Expected FastWAM policy")


def execute_episode(env, infer, *, task, episode_id, horizon, action_dim, execute_steps, max_steps,
                    on_step=None, on_infer=None, seed=0):
    """Pure control loop, injectable for tests; no asynchronous buffers survive reset."""
    env.reset()
    step = 0
    while not env.is_done and step < max_steps:
        request = env.get_obs()
        request.update(task=task, episode_id=episode_id, step_id=step, seed=seed)
        result = infer(request)
        if result.get("episode_id") != episode_id or result.get("step_id") != step:
            raise ValueError("Stale inference response from another episode/step")
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (horizon, action_dim) or not np.isfinite(actions).all():
            raise ValueError("Invalid action chunk")
        if on_infer:
            on_infer(result, step)
        for action in actions[:min(execute_steps, max_steps - step)]:
            env.step(action)
            step += 1
            if on_step:
                on_step()
            if env.is_done:
                break
    return {"episode_id": episode_id, "steps": step, "success": bool(env.is_success),
            "terminated": bool(env.is_done), "step_limit_reached": step >= max_steps and not env.is_done}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help='[闭环评测] 本次评测使用的环境 YAML 路径。')
    parser.add_argument("--metadata", required=True, help='[闭环评测] 预期的服务端元数据 JSON，用于核对模型和 checkpoint 身份。')
    parser.add_argument("--output", required=True, help='[权重与结果保存] 本次评测视频、逐回合结果和延迟记录的输出目录。')
    parser.add_argument("--host", default="127.0.0.1", help='[部署与连接] 策略服务地址。')
    parser.add_argument("--port", type=int, default=8001, help='[部署与连接] 策略服务端口。')
    parser.add_argument("--seed", type=int, default=0, help='[闭环评测] 本次评测的环境随机种子，不修改训练 seed。')
    parser.add_argument("--episodes", type=int, default=50, help='[闭环评测] 每个任务、每个 seed 的评测回合数。')
    parser.add_argument("--execute-steps", type=int, default=8, help='[闭环评测] 每次预测后执行多少步再观察；必须不大于 action_horizon。')
    parser.add_argument("--max-steps", type=int, default=3000, help='[闭环评测] 每个回合最多执行的环境动作步数，不是训练更新步数。')
    parser.add_argument("--rand-full", action="store_true", help='[闭环评测] 启用环境的完整视觉/场景随机化；不改变训练数据。')
    parser.add_argument("--no-video", action="store_true",
                        help="[评测/录制] 关闭 MP4 录制；保留策略所需相机图像和指标输出。")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    expected = json.loads(Path(args.metadata).read_text())
    horizon = expected["action_horizon"]
    if min(args.episodes, args.execute_steps, args.max_steps) <= 0 or args.execute_steps > horizon:
        parser.error("Episodes/execute-steps/max-steps must be positive; execute-steps <= horizon")
    os.environ.setdefault("MUJOCO_GL", "egl")
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect
    if not args.no_video:
        import imageio.v2 as imageio
    from .env import DexJoCoFastWAMEnv

    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        raise FileExistsError("Evaluation output already contains files; use a new attempt directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    env = DexJoCoFastWAMEnv(env_name=config["env_name"], camera_mapping=config["camera_mapping"],
                           seed=args.seed, rand_full=args.rand_full, randomize_dynamics=False,
                           dual_arm=config["robot_type"] == "dual_arm", prompt=config["prompt"],
                           render_mode="rgb_array", password=config.get("password"))
    successes = 0
    try:
        with connect(f"ws://{args.host}:{args.port}", open_timeout=15, compression=None, max_size=64 * 1024 * 1024) as socket:
            actual = msgpack_numpy.unpackb(socket.recv(timeout=15))
            validate_metadata(actual, expected)
            env.start()
            sim_hz = getattr(env.env, "hz", None)
            if sim_hz is not None and abs(float(sim_hz) - actual["control_hz"]) > 1e-6:
                raise ValueError("Simulator step frequency differs from training data fps")
            packer = msgpack_numpy.Packer()
            for episode in range(args.episodes):
                writers = {key: imageio.get_writer(target / f"episode_{episode:03d}_{key}.mp4",
                                                 fps=actual["control_hz"])
                           for key in config["camera_mapping"].values()} if not args.no_video else {}
                def record():
                    for key, frame in env.get_raw_images().items():
                        writers[key].append_data(frame)
                def infer(obs):
                    start = time.perf_counter()
                    socket.send(packer.pack(obs))
                    value = socket.recv(timeout=180)
                    if isinstance(value, str):
                        raise RuntimeError(value)
                    result = msgpack_numpy.unpackb(value)
                    result["roundtrip_ms"] = (time.perf_counter() - start) * 1000
                    return result
                def log_infer(result, step):
                    entry = {"episode_id": episode, "timestamp": step,
                             "roundtrip_ms": result["roundtrip_ms"],
                             "server_infer_ms": result.get("server_timing", {}).get("infer_ms")}
                    with (target / "latency.jsonl").open("a") as stream:
                        stream.write(json.dumps(entry) + "\n")
                try:
                    result = execute_episode(env, infer, task=config["env_name"], episode_id=episode,
                                             horizon=horizon, action_dim=actual["action_dim"],
                                             execute_steps=args.execute_steps, max_steps=args.max_steps,
                                             on_step=None if args.no_video else record,
                                             on_infer=log_infer, seed=args.seed)
                finally:
                    for writer in writers.values():
                        writer.close()
                successes += int(result["success"])
                with (target / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(result) + "\n")
                print(json.dumps(result), flush=True)
        (target / f"success_rate_{successes}_{args.episodes}.txt").write_text(str(successes / args.episodes))
    finally:
        env.close()


if __name__ == "__main__":
    main()
