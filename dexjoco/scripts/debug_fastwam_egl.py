"""Replay a recorded FastWAM evaluation child under GDB using EGL."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]


def build_command(result_path, destination, egl_device_id, debugger):
    result = json.loads(result_path.read_text())
    command = result.get("command")
    if (not isinstance(command, list) or len(command) < 3
            or any(not isinstance(value, str) for value in command)
            or command[1:3] != ["-m", "dexjoco_fastwam_client.evaluate"]):
        raise ValueError("result.json must contain a FastWAM evaluation child command")
    command = list(command)
    for flag in ("--config", "--metadata", "--output"):
        if command.count(flag) != 1 or command.index(flag) == len(command) - 1:
            raise ValueError(f"Recorded command has an invalid {flag}")
    for flag in ("--config", "--metadata"):
        original = Path(command[command.index(flag) + 1])
        if not original.is_absolute():
            original = ROOT / original
        if not original.is_file():
            raise FileNotFoundError(f"Missing saved evaluation asset: {original}")
        command[command.index(flag) + 1] = str(original.resolve())
    command[command.index("--output") + 1] = str(destination / "metrics")
    native = [debugger, "-q", "-nx", "--batch",
              "-ex", "set pagination off", "-ex", "set debuginfod enabled off",
              "-ex", "set disable-randomization off", "-ex", "set print thread-events off",
              "-ex", "run", "-ex", "bt", "-ex", "thread apply all bt 30",
              "-ex", "info sharedlibrary", "--args", *command]
    env = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
           "MUJOCO_EGL_DEVICE_ID": str(egl_device_id),
           "PYTHONFAULTHANDLER": "1", "PYTHONUNBUFFERED": "1"}
    env["PYTHONPATH"] = str(ROOT / "dexjoco") + os.pathsep + env.get("PYTHONPATH", "")
    return native, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path,
                        help="失败任务的 result.json；读取其原始评测子进程命令和服务参数。")
    parser.add_argument("--egl-device-id", type=int, required=True,
                        help="EGL 枚举的设备编号；不自动套用 CUDA_VISIBLE_DEVICES 的逻辑编号。")
    parser.add_argument("--dry-run", action="store_true", help="打印诊断计划，不启动 GDB 或写文件。")
    args = parser.parse_args()
    if args.egl_device_id < 0:
        parser.error("egl-device-id must be nonnegative")
    result_path = args.result.expanduser().resolve()
    destination = result_path.parent / "egl_debug" / uuid.uuid4().hex[:10]
    debugger = shutil.which("gdb")
    if not debugger and not args.dry_run:
        parser.error("找不到 gdb，请安装或加载服务器提供的 GDB。")
    command, env = build_command(result_path, destination, args.egl_device_id, debugger or "gdb")
    plan = {"source_result": str(result_path), "output": str(destination),
            "command": command, "gl_backend": "egl", "egl_device_id": args.egl_device_id,
            "note": "Native debugging only; this does not produce an official evaluation result."}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "debug_plan.json").write_text(json.dumps(plan, indent=2))
    log_path = destination / "gdb.log"
    print("GDB log:", log_path, flush=True)
    print("另一个终端可执行：tail -f", log_path, flush=True)
    with log_path.open("w") as log:
        # Debug the actual evaluator, rather than its parent launcher.
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    (destination / "debugger_exit.json").write_text(json.dumps({"returncode": process.returncode}))
    print("诊断结束；请提供 gdb.log 中 received signal 附近的原生栈。", flush=True)
    print("GDB 的返回码不代表评测成功率。", flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
