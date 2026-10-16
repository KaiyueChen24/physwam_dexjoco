#!/usr/bin/env python3
"""为当前服务器准备仅监听回环地址的 Mihomo；仅依赖 Python 标准库和 curl。"""
import argparse
import fcntl
import getpass
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

VERSION = "v1.19.31"
ASSET = f"mihomo-linux-amd64-compatible-{VERSION}.gz"
DOWNLOAD_URL = f"https://github.com/MetaCubeX/mihomo/releases/download/{VERSION}/{ASSET}"
SHA256 = "04cf9f09671704f839ddbee2e93069dc831a4123a75281e725d1d96ab9ac1afc"


def private_write(path, content, mode=0o600):
    """原子替换，临时文件从创建起就仅本人可读写。"""
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".setup-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def install_binary(state_dir, supplied=None):
    target = state_dir / "mihomo"
    if supplied:
        content = Path(supplied).expanduser().read_bytes()
    elif target.is_file() and os.access(target, os.X_OK):
        return target
    else:
        existing = shutil.which("mihomo")
        if not existing and os.access(Path.home() / ".local/bin/mihomo", os.X_OK):
            existing = str(Path.home() / ".local/bin/mihomo")
        if existing:
            content = Path(existing).read_bytes()
        else:
            print(f"下载官方 Mihomo {VERSION}（Linux amd64 compatible）…", flush=True)
            try:
                with urllib.request.urlopen(DOWNLOAD_URL, timeout=60) as response:
                    content = response.read()
            except Exception:
                raise RuntimeError("Mihomo 下载失败；请先上传官方 .gz，再传入 --mihomo-binary 路径") from None
            if hashlib.sha256(content).hexdigest() != SHA256:
                raise RuntimeError("Mihomo 下载文件 SHA256 不符，已停止")
    if content.startswith(b"\x1f\x8b"):
        content = gzip.decompress(content)
    private_write(target, content, mode=0o700)
    subprocess.run([str(target), "-v"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return target


def subscription_bytes(supplied=None):
    if supplied:
        content = Path(supplied).expanduser().read_bytes()
    else:
        # stream 只用于输出提示；getpass 自行从控制终端读取并关闭回显。
        # TTY 不可 seek，不能使用 Python 文本 r+ 模式，否则会误报为无交互终端。
        try:
            with open("/dev/tty", "w") as tty:
                url = getpass.getpass("粘贴 Clash/Mihomo 订阅链接后回车（不显示）：", stream=tty).strip()
        except OSError:
            raise RuntimeError("无交互终端；请通过 --subscription-file 提供已下载的订阅文件") from None
        if urllib.parse.urlsplit(url).scheme != "https":
            raise RuntimeError("订阅链接需要使用 https://")
        request = urllib.request.Request(url, headers={"User-Agent": "clash.meta"})
        try:
            # 下载订阅时代理可能尚未启动，使用直连；不打印包含密钥的异常信息。
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=60) as response:
                content = response.read()
        except Exception:
            raise RuntimeError("订阅下载失败；可在本地下载后上传，并用 --subscription-file 指定") from None
    if not content.strip():
        raise RuntimeError("订阅文件为空")
    if content.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        raise RuntimeError("下载结果是 HTML 页面，请提供 Clash/Mihomo 订阅")
    return content


def configuration(port):
    # 使用本地 provider 仅提取订阅节点，避免导入订阅中的监听/TUN/控制 API 设置。
    return f"""# 由 setup_mihomo.py 管理；类别：网络/本机代理。
mixed-port: {port}
bind-address: 127.0.0.1
allow-lan: false
mode: rule
log-level: warning
ipv6: false
proxy-providers:
  subscription:
    type: file
    path: ./subscription.yaml
    health-check:
      enable: true
      url: https://www.gstatic.com/generate_204
      interval: 300
proxy-groups:
  - name: AUTO
    type: url-test
    use: [subscription]
    url: https://www.gstatic.com/generate_204
    interval: 300
rules:
  - MATCH,AUTO
""".encode()


def process_identity(pid):
    """启动时间与参数共同验证 PID，防止共享 HOME 或 PID 重用导致误停其他进程。"""
    try:
        proc = Path("/proc") / str(pid)
        start = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
        argv = proc.joinpath("cmdline").read_bytes().split(b"\0")[:-1]
        return start, [os.fsdecode(arg) for arg in argv]
    except (OSError, IndexError):
        return None


def running_record(state_dir):
    try:
        record = json.loads((state_dir / "process.json").read_text())
        expected_argv = [str(state_dir / "mihomo"), "-d", str(state_dir)]
        identity = process_identity(record["pid"])
        if (record["hostname"] == socket.gethostname() and record["argv"] == expected_argv
                and identity == (record["start"], expected_argv)):
            return record
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def proxy_check(port):
    # 同时检查监听和实际 HTTPS 出口；不能仅凭端口存在认定代理可用。
    result = subprocess.run([
        "curl", "--noproxy", "", "--proxy", f"http://127.0.0.1:{port}",
        "-fIsS", "--connect-timeout", "5", "--max-time", "20", "https://github.com",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError("代理未通过 GitHub HTTPS 连通性检查；请检查订阅有效性和节点网络")


def prepare(state_dir, port, binary=None, subscription_file=None):
    record = running_record(state_dir)
    if record:
        if record["port"] != port or binary or subscription_file:
            raise RuntimeError("本脚本的 Mihomo 已运行；更新端口、订阅或二进制前，请先执行 setup_mihomo.py --stop")
        proxy_check(port)
        print(f"复用已验证的 Mihomo：http://127.0.0.1:{port}")
        return
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            raise RuntimeError("代理端口已被占用；请使用 --proxy 复用已有代理，或更换 --proxy-port") from None
    executable = install_binary(state_dir, binary)
    subscription = state_dir / "subscription.yaml"
    if subscription_file or not subscription.is_file():
        private_write(subscription, subscription_bytes(subscription_file))
    private_write(state_dir / "config.yaml", configuration(port))
    log_path = state_dir / "mihomo.log"
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(log_fd, "ab") as log:
        test = subprocess.run([str(executable), "-t", "-d", str(state_dir)], stdout=log, stderr=log)
        if test.returncode:
            raise RuntimeError(f"Mihomo 配置检查失败；日志保存在 {log_path}（可能含节点信息，请勿整份公开）")
        argv = [str(executable), "-d", str(state_dir)]
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 close_fds=True, start_new_session=True)
    try:
        for _ in range(100):
            if child.poll() is not None:
                raise RuntimeError(f"Mihomo 启动失败；请在服务器检查 {log_path}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("Mihomo 未在 10 秒内监听端口")
        proxy_check(port)
        identity = process_identity(child.pid)
        if identity is None or identity[1] != argv:
            raise RuntimeError("无法确认 Mihomo 进程身份")
        private_write(state_dir / "process.json", json.dumps({
            "pid": child.pid, "hostname": socket.gethostname(), "port": port,
            "start": identity[0], "argv": argv,
        }).encode())
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise
    print(f"Mihomo 已在后台启动并通过检查：http://127.0.0.1:{port}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=17890, help="[网络/端口] 本机混合代理端口，默认 17890")
    parser.add_argument("--binary", help="[网络/安装] 已上传的 Linux amd64 Mihomo .gz 或二进制路径")
    parser.add_argument("--subscription-file", help="[网络/订阅] 已下载的 Clash/Mihomo 订阅路径；否则现场隐藏输入链接")
    parser.add_argument("--stop", action="store_true", help="[网络/进程] 仅停止本脚本启动且身份匹配的 Mihomo")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("端口范围应为 1024–65535")
    os.umask(0o077)
    state_dir = Path.home() / ".config/physwam/mihomo"
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_dir.chmod(0o700)
    with (state_dir / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stop:
            record = running_record(state_dir)
            if record:
                os.kill(record["pid"], signal.SIGTERM)
                print("已向本脚本的 Mihomo 发送停止信号")
            else:
                print("没有身份匹配的 Mihomo 进程，无需停止")
        else:
            prepare(state_dir, args.port, args.binary, args.subscription_file)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # 不打印下载异常/traceback，防止订阅凭据出现在安装日志中。
        message = str(error) if isinstance(error, RuntimeError) else type(error).__name__
        print(f"Mihomo 准备失败：{message}", file=sys.stderr)
        sys.exit(1)
