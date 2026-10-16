# π0.5 单臂／双臂多任务训练与部署

参数类别、GPU 设置、部署与 eval 选项见 [中文配置索引](configuration_guide.md)。

除环境安装命令外，下面的命令从 `dexjoco/` 目录执行。脚本内部处理 OpenPI 的工作目录要求。
单臂六任务共享一个 policy，双臂五任务共享另一个 policy；第一版采用 rand_obj。

## 一键安装训练环境

以下命令从 PhysWAM 工作区根目录（含 dexjoco/ 和 fastwam/）执行：

```bash
bash dexjoco/scripts/setup_pi05_env.sh --name pi05
# 新服务器需要安装/启动 Mihomo、指定本机 AIFS 缓存时：
bash dexjoco/scripts/setup_pi05_env.sh --name pi05 --setup-proxy \
  --proxy-port 17890 --cache-dir /aifs/chenkaiyue/.cache/physwam
# 已有代理则用 --proxy http://127.0.0.1:17890 替代 --setup-proxy --proxy-port 17890
```

两条示例选择一条即可。安装器检测现有 Conda，缺失时安装 Miniforge；安装 Python 3.11、
FFmpeg 7、Conda 预编译 evdev，以及受 `configs/environments/pi05_constraints.txt` 约束的依赖。
可重复执行以补齐依赖或重建 editable 源码关联；已有环境若不是 Python 3.11，会报错并要求另选环境名。
默认只做 CPU 检查，包括 LeRobot/OpenPI 导入、NumPy 序列化和 JAX 运算；不要求空闲 GPU。
有已分配的空闲卡时，可追加 `--check-gpu 0` 检查该卡。

安装完成后复制脚本打印的 Conda 初始化、激活命令，再加载本机设置：

```bash
conda activate pi05
source "$HOME/.config/physwam/pi05-env.sh"
cd dexjoco
```

`--cache-dir /aifs/chenkaiyue/.cache/physwam` 同时指定包缓存、HF/OpenPI 缓存与安装日志；
更换服务器时重新传入该机可写路径。它不移动 `datasets/`、`checkpoints/` 或训练输出。
`--setup-proxy` 首次隐藏输入订阅链接，也可传 `--subscription-file`；
无法直连 GitHub 时先上传 Mihomo，并追加 `--mihomo-binary "$HOME/mihomo.gz"`。
代理凭据和环境设置保存在用户目录，后台代理仅监听回环地址，无需 sudo。
代理进程重启、更新及全部参数见 [环境安装参数](configuration_guide.md#环境安装参数)。
服务器仍需 NVIDIA 驱动及已挂载的存储；数据、权重、FastWAM 和仿真环境另行配置。

## 配置与数据契约

- `configs/task_groups/single_arm.yaml`：六个单臂任务、状态 23、数据动作 22、模型动作 32。
- `configs/task_groups/dual_arm.yaml`：五个双臂任务、状态 46、数据／模型动作 44。
- 每个任务配置语言指令、原始相机映射和可选 `episodes`（用于划分的候选 episode ID 列表）。
- `episodes: null` 将全部示范作为划分候选；若提供 ID 列表，先限制候选，再按 validation_ratio 划分。验证关闭时这些候选全部用于训练。
- 当前默认每任务留出 10% 完整 episode，仅计算 loss；训练与验证不交叉，归一化仅使用划分后的训练 episode。
- `pi05_smoke.yaml`：四卡 FSDP、真实数据与 base 权重、全量微调、全局 batch 4、20 步短跑。
- `pi05_full.yaml`：默认单臂全部六任务联合训练，八卡 FSDP、全量微调、全局 batch 256、100,000 步；CPU 加载进程 16。
  多任务步数不是论文复现保证，请根据验证成绩和训练预算调整。
- 两份配方均显式设置 `finetune_mode: full`，使用标准 `gemma_2b` 与 `gemma_300m`，视觉、语言、动作模块全部更新。
- `--group`、`--gpus` 覆盖 YAML；正式配置默认 `single_arm`、`0,1,2,3,4,5,6,7`，smoke 仍使用四卡。组内六个任务各以 1/6 概率采样，共享一个模型。
- 全局 batch 必须能被可见 GPU 数整除；`fsdp_devices` 必须整除可见 GPU 数。正式默认 8 表示一个八卡分片组；1 表示不分片的数据并行。
- 当前任务均衡采样支持单进程 JAX 的单机多卡，不支持多节点或 PyTorch DDP。
- 各任务先通过自己的 LeRobot reader 构造动作 chunk，再统一相机字段；chunk 不跨 episode。
- `task_balanced` 使每个任务获得相同采样概率；`frame_uniform` 按所有帧均匀采样。
- 统计用固定 seed、按训练混合分布抽取低维帧，无需视频解码。样本数和方法写入 provenance。
- 单臂保留官方 32 维模型 padding，输出裁剪到 22；双臂沿用官方 44 维动作头。

## 留出验证集（仅计算 loss）

`pi05_full.yaml` 和 `pi05_smoke.yaml` 已接入以下参数：

```yaml
# 【验证 / validation：只计算 loss，不反向传播】
validation_ratio: 0.1          # [数据/划分] 每任务留出完整 episode 的比例；0 关闭
validation_seed: 42            # [验证/随机性] 固定 episode 划分与验证噪声
validation_interval: 1000      # [验证/频率] 每多少次更新计算一次；训练最后一步也计算
validation_batch_size: 8       # [验证/GPU] π0.5 所有 GPU 合计的 batch，须能被可见卡数整除
validation_max_batches: null   # [验证/预算] null 遍历全部留出帧；整数限制每任务的验证批次数
```

全量短跑使用验证间隔 10、全局验证 batch 4，每任务最多 2 批；正式配方按 validation_max_batches 控制每任务预算。
真实数据按每任务的 episode 数向上取整，至少留一条验证和一条训练示范；不足两条时报错。
例如每任务 100 条 episode，默认 90 条训练、10 条验证；episode 长短不一时帧数比例不保证恰为 9:1。
同一任务在两个模型中采用相同的候选 episode、比例和种子时，得到相同划分。

先运行已有的 `--stage prepare` 检查数据并保存划分；`--stage stats` 或 `train` 只使用训练部分计算统计。
`--dry-run` 不访问真实数据，因此不展开实际 episode ID。完整数据划分在 prepare/stats/train 阶段确定。
同一个 run 的比例、种子或候选数据不能改变；恢复会核对配置与 `data_split.json`。已有无验证的实验需要使用新的 run-name。

验证逐任务遍历留出帧，动作 chunk 保持在各自 episode 内，复用训练归一化，关闭训练模式与数据增强。
尾批次补齐只为满足 JAX 分片形状，重复项不计入 loss。默认全量验证不丢最后不足一个 batch 的样本。
`task_balanced` 汇总为各任务平均 loss 的算术平均；`frame_uniform` 按实际验证样本数加权。
如果设置批次上限，则每任务使用固定随机帧子集，记录 `full_validation: false`，不会冒充全量验证。

运行产物：

- `runs/<name>/data_split.json`：候选、训练和验证 episode ID，比例、种子与取整方式。
- `resolved_config.yaml`：划分后的训练组和独立 validation_group。
- `validation_loss.jsonl`：每个验证步的总 loss、逐任务 loss、已验证/可用样本数和是否全量。
- `train.log`：`[validation] ... val_loss=...`；启用 W&B 时还记录 `val/loss`、`val/<task>/loss`。

验证不调用梯度或优化器更新，不消耗训练使用的噪声随机流，不运行模拟器。
本功能仅记录 loss，不自动早停或选择最佳 checkpoint；动作成功率仍由独立闭环评测衡量。

## 四卡全量微调 smoke test

`pi05_smoke.yaml` 使用与正式训练一致的全量更新范围，以四卡 FSDP、较小 batch 和步数检查链路；正式配置使用八卡 FSDP。
默认单臂六任务共享一个 π0.5 模型；实际峰值显存与编译耗时需要在服务器短跑确认。
先准备所选任务组的全部数据和 base 权重；每个任务至少有两条候选 episode，才能划分训练和验证。

```bash
conda activate pi05
cd ~/physwam/dexjoco

# 查看配置；不访问数据、不加载模型、不创建 run
python scripts/smoke_train_pi05.py --dry-run

# 需要时下载单臂数据和 base 权重，并按数据说明完成解压
python scripts/prepare_group_data.py --group single_arm

# 四张 H100：真实单臂六任务、20 次更新、两次验证，随后保存/恢复并再更新一步
python scripts/smoke_train_pi05.py --run-name pi05_single_full_smoke
```

用 `--dataset-root /服务器/数据父目录` 指定外部数据盘；初始化目录可用 `--checkpoint /路径/params` 覆盖。
若后续需要双臂，另行准备五任务和 44 维 base 权重，用独立 run 检查：

```bash
python scripts/prepare_group_data.py --group dual_arm
python scripts/smoke_train_pi05.py --group dual_arm \
  --run-name pi05_dual_full_smoke --gpus 0,1,2,3
```

脚本检查各任务数据和图像、训练指标有限、参数更新、留出集 loss、保存、恢复状态完全一致以及恢复后继续一步。
结果位于 `runs/<name>/smoke_result.json`、`validation_loss.jsonl` 和 `train.log`。
专用 smoke 在第 20、21 步保存；`save_interval: 10` 仅在通过 train_group.py 使用该配置时生效。
`validation_max_batches: 2` 只检查每任务最多两批（当前全局 batch 4，即最多 8 帧），结果注明是否全量；512 帧的归一化估计也仅供短跑。
正式训练请用独立 run-name 和 pi05_full.yaml，重新计算正式统计；短跑不是任务收敛或成功率保证。

## 服务器准备与训练

服务器复制同一份代码，用上面的一键脚本建立 pi05 训练环境，并另行建立 dexjoco 仿真环境。
命令中的工作目录按服务器实际位置调整；相对数据路径会自动依据仓库位置解析。
保留 `openpi-client` 的 NumPy 兼容修补；不要把仿真环境的 NumPy 1.26.4 安装进 pi05。

```bash
conda activate pi05
cd ~/physwam/dexjoco

# 下载 rand_obj 的六个单臂任务和 pi05_base 权重，记录实际 Hugging Face revision
python scripts/prepare_group_data.py --group single_arm

# 确认：finetune_mode=full、六个单臂任务、gpus=0,1,2,3,4,5,6,7、fsdp_devices=8
python scripts/train_group.py --dry-run

# 检查真实数据、相机解码及指令
python scripts/train_group.py --stage prepare --run-name single_full

# 分组计算统计（train 阶段在统计不存在时也会自动计算）
python scripts/train_group.py --stage stats --run-name single_full

# 正式全量微调：默认八卡 FSDP、全局 batch 256（每卡 32 条样本）
python scripts/train_group.py --run-name single_full
```

这份正式配方占用八张卡；启动前确认卡均可用，并已获得对应资源。若指定其他卡号，prepare/stats/train 应保持同一配置。
改为两卡时必须同时设 `--gpus 0,1 --fsdp-devices 2`，并根据实际显存调整 batch；不能只改可见卡数。
若后续训练双臂，先准备数据/权重，再用独立 run；以下命令在单臂八卡训练结束或停止后运行：

```bash
python scripts/prepare_group_data.py --group dual_arm
python scripts/train_group.py --group dual_arm --run-name dual_full
```

恢复完整训练：

```bash
python scripts/train_group.py --run-name single_full --resume
```

运行记录包括 `resolved_config.yaml`、`data_manifest.json`、`environment.txt`、`train.log`。
初始化模型、采样方式、数据、batch 和优化器设置在同一 run 内不能静默改变。
旧 LoRA run 的快照没有 `finetune_mode` 时仍按 LoRA 构建，服务与恢复保持原结构；全量训练必须使用新的 run-name。
支持恢复模型、优化器、随机步计数；上游 reader/sampler 迭代游标未持久化，因此不保证数据顺序逐帧复现。
checkpoint 位于 `runs/<run-name>/checkpoints/pi05_<group>_multitask/<run-name>/<step>/`。

## 八卡训练使用 tmux

以下命令在 H100 服务器执行；tmux 让 SSH 断开后训练继续运行。
没有 tmux 时用 `sudo apt-get install tmux`，没有安装权限则可在 pi05 环境使用 `conda install -c conda-forge tmux`。
先同步代码，检查 `nvidia-smi` 中八卡均可用；已有训练不要重复启动。

```bash
tmux new-session -s pi05_8gpu
```

进入 tmux 后运行（按实际服务器路径调整）：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate pi05
source "$HOME/.config/physwam/pi05-env.sh"
cd /aifs/chenkaiyue/physwam/dexjoco
export WANDB_MODE=online
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORMS JAX_PLATFORM_NAME WANDB_RUN_ID WANDB_RESUME
RUN="pi05_single_full_8gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs "runs/$RUN"
export WANDB_DIR="$PWD/runs/$RUN"
set -o pipefail
python -u scripts/train_group.py \
  --experiment configs/experiments/pi05_full.yaml \
  --group single_arm --gpus 0,1,2,3,4,5,6,7 --fsdp-devices 8 \
  --batch-size 256 --num-workers 16 --run-name "$RUN" \
  2>&1 | tee "logs/$RUN.launch.log"
```

这是新 run，从 base 权重开始，不加 `--resume`；不需要 nohup 或 `&`。
先按 `Ctrl+B`，松开后按 `D` 可退出查看而保留训练；`tmux attach -t pi05_8gpu` 返回会话。
在训练窗口直接按 `Ctrl+C` 会停止前台训练；tmux 不会在进程被终止时自动保存 checkpoint。
`tmux ls` 查看会话。batch 为 256，学习率、100000 步预算、验证和保存间隔均保持原设置；
相对 batch 128，固定步数下的训练采样总量翻倍；如需对照相同样本预算，应另行配置步数和预热。
八卡的提速幅度需实测，数据供应不足时增加 GPU 不能消除等待。

## 数据加载与 GPU 空闲排查

`pi05_full.yaml` 的 `num_workers` 设为 16，控制 CPU 视频解码、图像变换和组批的进程数。
这只是起点，不保证 8 或 16 一定更快；进程过多也可能造成 CPU、内存和共享存储争用。
保持待测 run 的 batch、模型和数据不变，先对 4/8/16 做对照。查看当前进程可用的 CPU 逻辑核数：

```bash
nproc
```

在暂停训练后测试，避免争用 CPU/I/O 污染对比。以下命令只读取已有 run 的训练划分、batch 和归一化资产，
不加载 base 权重或模型，不启动 W&B，不占用 GPU，也不覆盖原 run；需先有 `norm_stats.json`。
单次脚本结束会退出加载进程。示例中的 run-name 替换为实际 batch 256 实验：

```bash
RUN=pi05_single_full_8gpu_实际时间戳
for workers in 4 8 16; do
  python scripts/benchmark_pi05_data.py --run-dir "runs/$RUN" \
    --num-workers "$workers" --warmup-batches 8 --batches 64
done
```

比较 `samples_per_second`（越高越好）、`seconds_per_batch_mean` 和 `seconds_per_batch_p95`（越低越好）。
测试包括视频解码、训练数据变换、组批和 CPU Tensor 转换，不包括 GPU 搬运、模型计算和多卡通信。
第一次运行可能遇到冷文件缓存；对最好的候选重复一次确认，不能把 CPU 测试吞吐当成 GPU 训练吞吐。

同步本次代码后，可以在已有 checkpoint 的原 run 恢复时只调整 workers，无需从 base 重训：

```bash
python scripts/train_group.py --experiment configs/experiments/pi05_full.yaml \
  --run-name "$RUN" --gpus 0,1,2,3,4,5,6,7 --fsdp-devices 8 \
  --batch-size 256 --num-workers 16 --resume
```

此例假定原 run 也是八卡 FSDP=8；不能将原四卡 FSDP=4 run 直接改为八卡恢复。
先确认上一进程已停止且 checkpoint 完整写入；恢复会从最近保存点开始，未保存的更新不能恢复。
加载进程数变更写入 `runtime_changes.jsonl`，当前配置更新到 `resolved_config.yaml`。
batch、学习率、模型、数据划分等设置仍不允许在原 run 内更改；reader/sampler 游标仍不保证逐帧复现。

训练循环用 CPU 循环步数编号，初始化/恢复时才读取 GPU 步数，不再每步读回标量。
W&B 与控制台的性能指标随 `log_interval` 输出：

- `perf/samples_per_second`：当前训练窗口的全局样本吞吐；窗口末等待训练指标完成。
- `perf/data_wait_seconds_per_step`：主进程取下一批的平均耗时，包括数据等待与数组分片提交；
  部分耗时可能与 GPU 计算重叠，不能直接当作 GPU 空闲时间。

第一个窗口包含首次训练 JIT 编译；性能窗口不计入显式验证和 checkpoint 保存调用，后台保存 I/O 的影响仍可能出现。
若增加 workers 后吞吐没有改善，下一步应采集 CPU/I/O 和 JAX profiler，定位解码、存储或通信瓶颈，
不要仅凭功耗波形改视频编码、训练样本或切换解码后端。

## W&B 登录、GPU 与 loss 监控

正式配方 `pi05_full.yaml` 默认 `wandb_enabled: true`、`wandb_project: physwam_dexjoco`。
`wandb_project` 由分组入口传入 OpenPI；未提供该字段的历史配方仍使用 `openpi` 项目。
所属个人/团队沿用登录账号，也可在终端用 `WANDB_ENTITY` 指定有写权限的团队。
专用 smoke 脚本用于本地链路检查，不调用正式训练器的 W&B 初始化。

先在服务器 pi05 环境登录；API key 仅在交互提示中输入，不写进 YAML、Git 或命令行：

```bash
conda activate pi05
source "$HOME/.config/physwam/pi05-env.sh"
export WANDB_MODE=online
wandb login
```

云端 API key 可在 https://wandb.ai/authorize 获取。
使用当前节点已经配置的代理；`WANDB_DIR` 可指向 AIFS 上的可写目录以保存本地同步文件。
训练启动后控制台会打印 run 链接，run 名称来自 `--run-name`。

| 内容 | W&B 指标/位置 | 当前正式配方的频率 |
|---|---|---|
| GPU 利用率、显存、温度、功耗 | System 面板，按 GPU 展示；具体项目取决于 NVML/驱动可见性 | W&B 按时间周期采样，与训练 log_interval 独立 |
| 总训练 loss | `loss` | 每 log_interval 次更新记录上一段训练 batch 的平均值，最后一步也记录 |
| 梯度与矩阵参数范数 | `grad_norm`、`param_norm` | 与总训练 loss 同步 |
| 总验证 loss | `val/loss` | 每 validation_interval 次更新及训练结束时 |
| 各任务验证 loss | `val/water_plant/loss` 等六条曲线 | 与总验证 loss 同步 |
| 输入相机样本 | `camera_views` | step 0，最多五条样本的相机拼图 |
| 实验配置 | run Config | 初始化时记录 |

训练器通过 W&B Settings 将 GPU 监控限制到 `--gpus` 指定的数字编号；
例如 π0.5 使用 0,1,2,3 时，不将另一组 FastWAM 使用的 4,5,6,7 纳入该 run。
GPU 系统指标通常以时间为横轴，loss 以已完成的优化器更新次数为横轴。
这不是逐任务训练 loss；当前需求采用逐任务验证 loss 衡量每个任务的留出集表现。

正式配置 `validation_ratio: 0.1` 按完整 episode 留出，`validation_max_batches: 50` 限制每任务最多 50 批；设为 null 才遍历全部留出帧。
同一模型依次计算六任务的验证 loss，不更新模型，最终 `val/loss` 为六个任务平均 loss 的等权平均。
训练预算由 num_train_steps 指定；例如 100000 步、验证间隔 1000 时，验证发生在 1000、2000、…、100000，共 100 次。
它计算的是验证 loss；DexJoCo 闭环成功率由独立评测脚本计算，不在训练期间自动运行。
完整验证可能花费较长时间，期间训练更新暂停，GPU 利用率与训练阶段可能不同。

训练、验证及 checkpoint 都按已完成的更新次数编号；同一更新上的训练与验证指标合并为一次 W&B 提交。
每 save_interval 次更新保存 checkpoint；keep_period 只保留实际存在且编号为其倍数的保存点，此外保留最新一个。
例如 save_interval=4000、keep_period=10000 时，长期保留点为 20000、40000 等；不会补存 10000、30000 等未触发保存的步数。
W&B run ID 保存在 checkpoint 根目录的 `wandb_id.txt`，配合原 run 的 `--resume` 继续记录。
这些指标不会自动把大型 checkpoint 上传为 W&B Artifact；模型文件仍保存在 AIFS 的 run 目录。

## 部署与批量评测

部署指 DexJoCo 仿真中的闭环推理。可以将整个 `runs/<run-name>/` 复制回本地，
包含最终配置、checkpoint、assets 和统计；服务不需要训练数据。

终端 A（pi05 环境）：

```bash
python scripts/serve_group.py --run-dir runs/single_full --gpus 0 --port 8000
```

默认选该 run 最新完成的 checkpoint，可用 `--checkpoint <完整step目录>` 指定验证选出的版本。
默认监听 127.0.0.1；客户端和服务不在同一机器时明确设置 `--host` 和网络访问。
双臂换成 `runs/dual_full`，另一个服务可以使用端口 8001。

终端 B（dexjoco 环境）：

```bash
conda activate dexjoco
cd ~/physwam/dexjoco
python scripts/evaluate_group.py --run-dir runs/single_full \
  --port 8000 --seeds 0 1 2 --episodes 50
python scripts/summarize_results.py \
  --evaluation-dir runs/single_full/evaluation/rand_obj
```

这表示每个任务、每个 seed 50 次，即单臂 6×3×50 次。
先做两次闭环检查可加 `--tasks water_plant --seeds 0 --episodes 2 --evaluation-name smoke_eval`。
评测核对服务器的任务组、run、checkpoint 与 action horizon。当前评测器需要 horizon 30。
任务的环境、相机和 prompt 使用训练时保存的任务组配置。

结果包含逐任务视频、日志、成功次数、`latency.jsonl`，以及 JSON/CSV 汇总。
耗时统计分别记录 WebSocket 往返和服务端推理时间；汇总排除每个任务 job 的第一个 chunk，
原始记录仍保留编译／预热时间。它们不等同于整个机器人控制循环耗时。

失败的 job 写入结果并使批量命令失败；重跑用 `--retry-failed`。
已完成的 job 自动跳过。改变模型或评测设置时用新的 evaluation-name，避免混合结果。
有缺失 job 时不报告任务平均值；仅评测任务子集时汇总标记 `all_group_tasks_evaluated: false`。
历史框架自测生成的权重不支持用于当前部署与评测。
验证 checkpoint 与正式测试请使用不同 evaluation-name 和不同 seeds。

## 验证代码

```bash
conda activate pi05
cd ~/physwam/dexjoco
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_group_framework.py tests/test_pi05_finetuning.py
```

测试覆盖组别维度、相机映射、指令保留、任务均衡采样、episode 选择、完整结果判定，以及全量/LoRA 模型构造与实际训练参数过滤器。
ROS 的 pytest 自动插件可能与 Conda Python 不兼容，因此关闭无关插件自动加载。

FastWAM 使用同一任务组和独立训练／推理入口，单臂动作直接使用 22 维。
完整框架、环境与命令见 [FastWAM 接入说明](fastwam_integration.md)。
本页的 train_group.py、serve_group.py、evaluate_group.py 继续专用于 π0.5。
