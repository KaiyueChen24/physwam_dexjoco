# π0.5 单臂／双臂多任务训练与部署

参数类别、GPU 设置、部署与 eval 选项见 [中文配置索引](configuration_guide.md)。

所有下面的命令从仓库根目录执行。脚本内部处理 OpenPI 的工作目录要求。
单臂六任务共享一个 policy，双臂五任务共享另一个 policy；第一版采用 rand_obj。

## 配置与数据契约

- `configs/task_groups/single_arm.yaml`：六个单臂任务、状态 23、数据动作 22、模型动作 32。
- `configs/task_groups/dual_arm.yaml`：五个双臂任务、状态 46、数据／模型动作 44。
- 每个任务配置语言指令、原始相机映射和可选 `episodes`（用于划分的候选 episode ID 列表）。
- `episodes: null` 将全部示范作为划分候选；若提供 ID 列表，先限制候选，再按 validation_ratio 划分。验证关闭时这些候选全部用于训练。
- 当前默认每任务留出 10% 完整 episode，仅计算 loss；训练与验证不交叉，归一化仅使用划分后的训练 episode。
- `local_smoke.yaml`：合成数据、batch 1、20 步、关闭 W&B 上传。
- `h100_lora.yaml`：真实数据、LoRA、全局 batch 32、60,000 步起点。
  多任务步数不是论文复现保证，请根据验证成绩和训练预算调整。
- 全局 batch 需整除可见 GPU 数量。`fsdp_devices=1` 为单机数据并行；更大的合法值开启参数分片。
- 当前任务均衡采样支持单进程 JAX 的单机多卡，不支持多节点或 PyTorch DDP。
- 各任务先通过自己的 LeRobot reader 构造动作 chunk，再统一相机字段；chunk 不跨 episode。
- `task_balanced` 使每个任务获得相同采样概率；`frame_uniform` 按所有帧均匀采样。
- 统计用固定 seed、按训练混合分布抽取低维帧，无需视频解码。样本数和方法写入 provenance。
- 单臂保留官方 32 维模型 padding，输出裁剪到 22；双臂沿用官方 44 维动作头。

## 留出验证集（仅计算 loss）

`h100_lora.yaml` 和 `local_smoke.yaml` 已接入以下参数：

```yaml
# 【验证 / validation：只计算 loss，不反向传播】
validation_ratio: 0.1          # [数据/划分] 每任务留出完整 episode 的比例；0 关闭
validation_seed: 42            # [验证/随机性] 固定 episode 划分与验证噪声
validation_interval: 1000      # [验证/频率] 每多少次更新计算一次；训练最后一步也计算
validation_batch_size: 8       # [验证/GPU] π0.5 所有 GPU 合计的 batch，须能被可见卡数整除
validation_max_batches: null   # [验证/预算] null 遍历全部留出帧；整数限制每任务的验证批次数
```

本地 smoke 的验证间隔为 10、验证 batch 为 1；合成 episode 仅检查隔离与 loss 流程。
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

## 本地 smoke test

```bash
conda activate openpi
cd ~/physwam/dexjoco

# 查看配置，不创建模型、不下载数据、不创建运行目录
python scripts/smoke_train_pi05.py --group single_arm --dry-run

# 完整 π0.5 LoRA：使用已经下载的 water_plant checkpoint 作为链路检查初始化
python scripts/smoke_train_pi05.py --group single_arm \
  --run-name local_pi05_single_smoke --gpus 0
```

脚本检验：所有指标有限、可训练参数更新、checkpoint 保存、恢复状态完全一致、
恢复后继续一步。结果写入 `runs/<run-name>/smoke_result.json`，日志写入 `train.log`。
采用合成数据的权重不能用于报告任务性能；服务器正式训练从 base 重新初始化。
已有 run 的 checkpoint 不会自动覆盖，每次重跑请换 run-name。

双臂 smoke 需要 44 维初始化权重，可在需要时只下载权重：

```bash
python scripts/prepare_group_data.py --group dual_arm --weights-only
python scripts/smoke_train_pi05.py --group dual_arm \
  --run-name local_pi05_dual_smoke --gpus 0
```

用于代码验证的较小模型模式：

```bash
python scripts/smoke_train_pi05.py --group single_arm --tiny --steps 2 \
  --run-name framework_single_test --gpus 0
python scripts/smoke_train_pi05.py --group dual_arm --tiny --steps 2 \
  --run-name framework_dual_test --gpus 0
```

`--tiny` 使用 dummy 语言／动作模型并冻结 PaliGemma，仍调用同一个训练步和保存恢复逻辑。
它仅验证框架，不证明完整 π0.5 的显存可用性。脚本会打印实际可见 GPU。

## 服务器准备与训练

服务器复制同一份代码，按仓库原有方法建立 openpi 和 dexjoco 两个环境。
命令中的工作目录按服务器实际位置调整；相对数据路径会自动依据仓库位置解析。
保留 `openpi-client` 的 NumPy 兼容修补；不要把仿真环境的 NumPy 1.26.4 安装进 openpi。

```bash
conda activate openpi
cd ~/physwam/dexjoco

# 下载 rand_obj 的 11 个任务和两种 base 权重，记录实际 Hugging Face revision
python scripts/prepare_group_data.py --group all

# 检查真实数据、相机解码及指令
python scripts/train_group.py --group single_arm --stage prepare \
  --run-name single_baseline --gpus 0,1
python scripts/train_group.py --group dual_arm --stage prepare \
  --run-name dual_baseline --gpus 2,3

# 分组计算统计（train 阶段在统计不存在时也会自动计算）
python scripts/train_group.py --group single_arm --stage stats \
  --run-name single_baseline --gpus 0,1
python scripts/train_group.py --group dual_arm --stage stats \
  --run-name dual_baseline --gpus 2,3

# 四张卡时，在两个终端分别执行；只有两张卡时依次使用 0,1
python scripts/train_group.py --group single_arm \
  --run-name single_baseline --gpus 0,1
python scripts/train_group.py --group dual_arm \
  --run-name dual_baseline --gpus 2,3
```

先进行真实数据短跑时，使用独立 run-name，例如：

```bash
python scripts/train_group.py --group single_arm --run-name real_single_smoke \
  --gpus 0,1 --batch-size 2 --steps 20 --num-workers 0
```

恢复完整训练：

```bash
python scripts/train_group.py --group single_arm --run-name single_baseline \
  --gpus 0,1 --resume
```

运行记录包括 `resolved_config.yaml`、`data_manifest.json`、`environment.txt`、`train.log`。
初始化模型、采样方式、数据、batch 和优化器设置在同一 run 内不能静默改变。
支持恢复模型、优化器、随机步计数；上游 reader/sampler 迭代游标未持久化，因此不保证数据顺序逐帧复现。
checkpoint 位于 `runs/<run-name>/checkpoints/pi05_<group>_multitask/<run-name>/<step>/`。

## 部署与批量评测

部署指 DexJoCo 仿真中的闭环推理。可以将整个 `runs/<run-name>/` 复制回本地，
包含最终配置、checkpoint、assets 和统计；服务不需要训练数据。

终端 A（openpi 环境）：

```bash
python scripts/serve_group.py --run-dir runs/single_baseline --gpus 0 --port 8000
```

默认选该 run 最新完成的 checkpoint，可用 `--checkpoint <完整step目录>` 指定验证选出的版本。
默认监听 127.0.0.1；客户端和服务不在同一机器时明确设置 `--host` 和网络访问。
双臂换成 `runs/dual_baseline`，另一个服务可以使用端口 8001。

终端 B（dexjoco 环境）：

```bash
conda activate dexjoco
cd ~/physwam/dexjoco
python scripts/evaluate_group.py --run-dir runs/single_baseline \
  --port 8000 --seeds 0 1 2 --episodes 50
python scripts/summarize_results.py \
  --evaluation-dir runs/single_baseline/evaluation/rand_obj
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
合成／tiny checkpoint 必须显式允许才能服务和评测，汇总会标记不适合 benchmark。
验证 checkpoint 与正式测试请使用不同 evaluation-name 和不同 seeds。

## 验证代码

```bash
conda activate openpi
cd ~/physwam/dexjoco
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_group_framework.py
```

测试覆盖组别维度、相机映射、指令保留、任务均衡采样、episode 选择、完整结果判定。
ROS 的 pytest 自动插件可能与 Conda Python 不兼容，因此关闭无关插件自动加载。

FastWAM 使用同一任务组和独立训练／推理入口，单臂动作直接使用 22 维。
完整框架、环境与命令见 [FastWAM 接入说明](fastwam_integration.md)。
本页的 train_group.py、serve_group.py、evaluate_group.py 继续专用于 π0.5。
