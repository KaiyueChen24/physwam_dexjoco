#!/usr/bin/env python3
"""Query the nine Linux servers using OpenSSH and one shared password."""

import argparse
import concurrent.futures
import getpass
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import warnings


HOST = "112.192.23.232"
USER = "9qTymlC6XNreTg7J"
PORTS = tuple(range(61010, 61019))

# Linux counters are sampled for one second. MemAvailable includes reclaimable
# cache; guest counters must not be counted again in the CPU total.
PROBE = r"""
export LC_ALL=C
printf '主机: %s\n' "$(hostname)"
printf '运行: '; uptime
cpu_before=$(awk '/^cpu / {t=0; for(i=2;i<=9;i++)t+=$i; print t, $5+$6}' /proc/stat)
sleep 1
awk -v before="$cpu_before" '
  /^cpu / {
    split(before,b," "); t=0; for(i=2;i<=9;i++)t+=$i;
    dt=t-b[1]; idle=$5+$6-b[2];
    if(dt>0) printf "CPU: %.1f%% (1 秒采样)\n", 100*(dt-idle)/dt;
    else print "CPU: N/A";
  }' /proc/stat
printf '逻辑 CPU: '; getconf _NPROCESSORS_ONLN
awk '
  /^MemTotal:/ {total=$2}
  /^MemAvailable:/ {available=$2; found=1}
  /^SwapTotal:/ {swap=$2}
  /^SwapFree:/ {free_swap=$2}
  END {
    if(total && found)
      printf "内存: %.1f / %.1f GiB (%.1f%%)\n", (total-available)/1048576, total/1048576, 100*(total-available)/total;
    else print "内存: N/A (缺少 MemAvailable)";
    printf "Swap: %.1f / %.1f GiB\n", (swap-free_swap)/1048576, swap/1048576;
  }' /proc/meminfo
printf '\n磁盘（指定路径所在文件系统）:\n'
df -hP -- __DISK_PATHS__ 2>&1
printf '\nGPU（序号, 型号, 利用率 %%, 已用显存 MiB, 总显存 MiB, 温度 °C）:\n'
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>&1
    gpu_status=$?
    if [ "$gpu_status" -ne 0 ]; then
        printf 'GPU 查询失败（退出码 %s）\n' "$gpu_status"
    fi
    if [ '__PROCESSES__' = 'yes' ]; then
        printf '\nGPU 计算进程（GPU UUID, PID, 进程名, 显存 MiB；空白表示未报告进程）:\n'
        printf 'GPU 序号与 UUID 对照:\n'
        nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>&1
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>&1
    fi
else
    printf '未找到 nvidia-smi（无法判断是否有 GPU）\n'
fi
printf '\nCPU 占用最高的 5 个进程（%%CPU 为进程生命周期平均值）:\n'
ps -eo user,pid,pcpu,pmem,comm --sort=-pcpu | head -n 6
printf '\n__PHYSWAM_PROBE_COMPLETE__\n'
"""


def positive_number(value):
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("必须是有限的正数")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=HOST, help="公网地址")
    parser.add_argument("--user", default=USER, help="SSH 用户名")
    parser.add_argument("--ports", type=int, nargs="+", default=PORTS,
                        help="要查询的端口，默认 61010–61018")
    parser.add_argument("--timeout", type=positive_number, default=25,
                        help="每台机器的总超时秒数，默认 25")
    parser.add_argument("--interval", type=positive_number,
                        help="每轮查询结束后等待多少秒再刷新，Ctrl+C 退出")
    parser.add_argument("--processes", action="store_true",
                        help="显示 GPU 计算进程及 GPU UUID 对照")
    parser.add_argument("--disk-path", action="append", default=[],
                        help="额外检查的磁盘路径，可重复指定；默认检查 / 和 /dev/shm")
    args = parser.parse_args()
    if any(not 1 <= port <= 65535 for port in args.ports):
        parser.error("端口必须在 1–65535 之间")
    if args.host.startswith("-") or args.user.startswith("-"):
        parser.error("主机名和用户名不能以 - 开头")
    args.ports = list(dict.fromkeys(args.ports))
    return args


def query_server(port, args, env, probe):
    command = [
        "ssh", "-T", "-p", str(port), "-l", args.user,
        "-o", "ConnectTimeout=10",
        "-o", "ConnectionAttempts=1",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "PreferredAuthentications=password,keyboard-interactive",
        "-o", "PubkeyAuthentication=no",
        "-o", "NumberOfPasswordPrompts=1",
        "-o", "BatchMode=no",
        "-o", "ControlMaster=no", "-o", "ControlPath=none",
        "-o", "ClearAllForwardings=yes", "-o", "ForwardAgent=no",
        "-o", "ForwardX11=no", "-o", "PermitLocalCommand=no",
        args.host, "sh -s",
    ]
    try:
        result = subprocess.run(command, input=probe, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=env, timeout=args.timeout, start_new_session=True)
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        return False, f"超时（{args.timeout:g} 秒），已收集的信息：\n{partial}"
    except OSError as exc:
        return False, f"无法启动 SSH: {exc}"
    marker = "__PHYSWAM_PROBE_COMPLETE__"
    complete = marker in result.stdout
    output = result.stdout.replace(marker, "").strip()
    if result.returncode or not complete:
        output += f"\n查询失败或未完成（SSH 退出码 {result.returncode}）"
    if result.stderr.strip():
        output += "\nSSH: " + result.stderr.strip()
    return result.returncode == 0 and complete, output.strip()


def main():
    args = parse_args()
    if not shutil.which("ssh"):
        print("未找到 ssh，请先安装 OpenSSH 客户端。", file=sys.stderr)
        return 2
    try:
        # Refuse getpass's fallback to echoed input when no terminal exists.
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            password = getpass.getpass("请输入所有服务器共用的 SSH 密码（隐藏输入）: ")
    except (getpass.GetPassWarning, EOFError):
        print("请在交互式终端运行，以便隐藏输入密码。", file=sys.stderr)
        return 2
    if not password:
        print("密码不能为空。", file=sys.stderr)
        return 2

    paths = ["/", "/dev/shm", *args.disk_path]
    probe = PROBE.replace("__DISK_PATHS__", shlex.join(paths))
    probe = probe.replace("__PROCESSES__", "yes" if args.processes else "no")
    # This private temporary file contains only code, never the password.
    # Only SSH children receive the password environment variable.
    with tempfile.TemporaryDirectory(prefix="physwam-ssh-") as directory:
        askpass = Path(directory) / "askpass"
        askpass.write_text('#!/bin/sh\nprintf \'%s\\n\' "$PHYSWAM_SSH_PASSWORD"\n')
        askpass.chmod(0o700)
        env = dict(os.environ, SSH_ASKPASS=str(askpass),
                   SSH_ASKPASS_REQUIRE="force", DISPLAY=os.environ.get("DISPLAY") or ":0",
                   PHYSWAM_SSH_PASSWORD=password)
        del password
        while True:
            print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S %Z')} | 查询 {len(args.ports)} 台服务器 ===",
                  flush=True)
            results = {}
            # Completion messages appear immediately; details stay in port order.
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(9, len(args.ports))) as pool:
                futures = {pool.submit(query_server, port, args, env, probe): port
                           for port in args.ports}
                for future in concurrent.futures.as_completed(futures):
                    port = futures[future]
                    results[port] = future.result()
                    print(f"  :{port} {'完成' if results[port][0] else '失败'}", flush=True)
            for port in args.ports:
                number = f"第 {port - 61009} 台 | " if port in PORTS else ""
                internal = f" | 10.254.30.{port - 60889}:24680" if port in PORTS and args.host == HOST else ""
                print(f"\n{'=' * 72}\n{number}{args.host}:{port}{internal}\n{'-' * 72}")
                print(results[port][1])
            failures = sum(not success for success, _ in results.values())
            print(f"\n查询完成: {len(results) - failures} 成功 / {failures} 失败", flush=True)
            if args.interval is None:
                return 1 if failures else 0
            print(f"{args.interval:g} 秒后开始下一轮；Ctrl+C 退出。", flush=True)
            time.sleep(args.interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已停止。", file=sys.stderr)
        sys.exit(130)
