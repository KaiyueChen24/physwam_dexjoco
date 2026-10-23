# PhysWAM

DexJoCo 仿真、π0.5 基线与 FastWAM 训练／推理统一工作区。
Git 仓库根目录为本目录；两个子目录是同一仓库中的源码组件。

```text
physwam/
├── dexjoco/       # 仿真、任务配置、实验启动器与评测
└── fastwam/       # FastWAM 模型、训练器与 DexJoCo 适配
```

- [FastWAM 接入与训练评测](dexjoco/docs/fastwam_integration.md)
- [π0.5 分组训练](dexjoco/docs/group_training.md)
- [配置索引](dexjoco/docs/configuration_guide.md)
- [DexJoCo 原项目说明](dexjoco/README.md)
- [FastWAM 原项目说明](fastwam/README.md)

## 一键配置 FastWAM 环境

从工作区根目录执行：

```bash
bash dexjoco/scripts/setup_fastwam_env.sh
```

默认建立或复用 Python 3.10 的 `fastwam` 环境，缺少 Conda 时安装 Miniforge。
安装 FFmpeg 7、编译工具、PyTorch 2.7.1 CUDA 12.8、DeepSpeed 及 FastWAM 的固定版本依赖，
并安装策略服务使用的轻量 `openpi-client`。安装后执行 pip check、CPU 优化器更新、
真实小视频解码、Parquet/MessagePack 往返和两份 FastWAM 配方的 dry-run 检查。
默认隐藏 GPU，不下载模型权重或数据；此环境与 π0.5 和仿真环境分别安装。

新服务器需要安装/启动代理并使用 AIFS 缓存时，路径换成该机已挂载、可写的个人目录：

```bash
bash dexjoco/scripts/setup_fastwam_env.sh --name fastwam \
  --setup-proxy --proxy-port 17890 \
  --cache-dir /aifs/chenkaiyue/.cache/physwam \
  --model-dir /aifs/chenkaiyue/models/fastwam
```

已有代理时用 `--proxy http://127.0.0.1:17890` 替代 `--setup-proxy --proxy-port 17890`。
代理离线二进制、订阅文件参数与 π0.5 安装器相同，复用 `setup_mihomo.py`。
`--model-dir` 仅指定 Wan 组件目录；ActionDiT 初始化骨干仍在 `fastwam/checkpoints/`。
每次新开终端，按安装器打印的 Conda source 命令加载 Conda，再执行：

```bash
conda activate fastwam
source "$HOME/.config/physwam/fastwam-env.sh"
```

先查看计划可加 `--dry-run`；有空闲 GPU 时可加 `--check-gpu 0` 验证单卡 BF16 前向/反向。
该检查不替代四卡 smoke 训练。安装后按 [FastWAM 接入说明](dexjoco/docs/fastwam_integration.md#数据与首次短跑)
准备数据并执行 prepare/inspect/initialize/train。

## 一键配置 π0.5 环境

克隆后从工作区根目录执行：

```bash
bash dexjoco/scripts/setup_pi05_env.sh
```

默认建立或复用 Python 3.11 的 `pi05` 环境，未找到 Conda 时安装 Miniforge。
脚本使用固定的关键依赖版本，先安装 Conda 预编译 `evdev`，再安装 OpenPI 与 LeRobot，
最后执行 `pip check`、模块导入、序列化和 JAX CPU 运算检查。
安装完成后，按终端输出的 `source ...` 和 `conda activate ...` 命令加载环境。

新服务器需要 Mihomo 和 AIFS 缓存时（路径换成本机已挂载、可写的个人目录）：

```bash
bash dexjoco/scripts/setup_pi05_env.sh --name pi05 \
  --setup-proxy --proxy-port 17890 \
  --cache-dir /aifs/chenkaiyue/.cache/physwam
```

首次运行会隐藏输入订阅链接；可用 `--subscription-file /路径/订阅.yaml` 代替。
GitHub 无法直连时，先上传官方 Mihomo `.gz`，再追加 `--mihomo-binary "$HOME/mihomo.gz"`。
已有代理时，将 `--setup-proxy --proxy-port 17890` 换成 `--proxy http://127.0.0.1:17890`。
代理后台运行，仅监听本机；订阅保存在 `~/.config/physwam/mihomo/`，不写进仓库。
安装后及每次新开终端，激活环境并加载本机代理和缓存：

```bash
conda activate pi05  # 首次先执行安装器打印的 Conda source 命令
source "$HOME/.config/physwam/pi05-env.sh"
```

`--cache-dir` 包括 pip、Conda、Hugging Face、OpenPI 缓存和日志；
下载脚本仍写入 `dexjoco/datasets/`、`dexjoco/checkpoints/`，不会因缓存参数自动搬迁。
详细参数、代理重启/停止见 [中文配置索引](dexjoco/docs/configuration_guide.md#环境安装参数)。
默认检查只用 CPU；获得空闲 GPU 后可加 `--check-gpu 0`。
环境安装覆盖 π0.5 训练与策略服务；数据、权重、NVIDIA 驱动和仿真环境分别准备。

## 使用

### 批量查看服务器使用情况

在本机交互式终端运行（仅依赖 Python 3.8+ 和 OpenSSH 8.4+，远端为 Linux）：

```bash
python3 scripts/server_status.py
```

隐藏输入一次共用 SSH 密码后，并发查询 `112.192.23.232:61010–61018` 共 9 台服务器，
显示 CPU、负载、内存、Swap、根目录与 `/dev/shm` 空间、各 GPU 利用率和显存、CPU 占用最高的进程。
密码不写入脚本或文件，不作为命令行参数传递；临时密码提示程序在退出时删除。
首次连接自动记录新主机密钥到本机 `known_hosts`，已有主机密钥发生变化时拒绝连接。
GPU 指标依赖远端 `nvidia-smi`，未安装或驱动错误会显示提示。

```bash
# 每轮结束后等待 30 秒刷新，额外显示 GPU 计算进程；Ctrl+C 退出
python3 scripts/server_status.py --interval 30 --processes

# 只查第 1、2、9 台，额外查看 /aifs 所在文件系统，每台最多等待 40 秒
python3 scripts/server_status.py --ports 61010 61011 61018 --disk-path /aifs --timeout 40
```

单台超时或认证失败不会影响其他服务器；一次性查询有失败时退出码为 1。
CPU 使用率采样 1 秒，进程列表的 `%CPU` 是进程生命周期平均值，内存已用量按 `MemTotal - MemAvailable` 计算。

### 运行实验

现有命令仍从各组件目录运行，两个 Python 环境独立安装：

```bash
cd dexjoco
python scripts/train_fastwam_group.py \
  --experiment configs/experiments/fastwam_smoke.yaml \
  --run-name fastwam_water_smoke --dry-run
```

在工作区根目录统一执行 `git status`、`git add`、`git commit` 与 `git push`。
`origin` 继续指向用户的 physwam_dexjoco 仓库；`upstream` 是 DexJoCo 上游，
`fastwam-upstream` 是 FastWAM 上游。上游目前采用根目录布局，更新时需按对应子目录导入，
不要直接将上游根目录覆盖到工作区根目录。

合并提交保留两边历史，并纳入合并时的 FastWAM 本地改动。
FastWAM 初始上游版本见 [UPSTREAM_REVISION](fastwam/UPSTREAM_REVISION)。

## 第三方组件

子目录的 LICENSE 与第三方版权说明继续适用于各自代码。
Git 子模块配置统一位于工作区根目录的 `.gitmodules`；例如数据转换器的
Diffusion Policy 子模块路径为 `dexjoco/dexjoco-data-converter/src/3rd/diffusion_policy`。
