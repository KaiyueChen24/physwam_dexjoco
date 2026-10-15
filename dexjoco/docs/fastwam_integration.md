# FastWAM / DexJoCo 框架

参数类别、GPU 设置、部署与 eval 选项见 [中文配置索引](configuration_guide.md)。

## 目录与职责

本工作区包含同级 `dexjoco/` 与 `fastwam/`。现有 π0.5 路径继续使用原脚本。
FastWAM 官方模型、注意力、flow matching 损失、训练器保持原样；适配代码位于
`fastwam/experiments/dexjoco/`。本地框架测试不等同于完成模型训练或复现论文成绩。

| 位置 | 职责 |
|---|---|
| `configs/task_groups/*.yaml` | 复用现有任务成员、指令、相机映射、训练 episode 选择 |
| `configs/experiments/fastwam_baseline.yaml` | FastWAM 的 batch、步数、视频采样与推理参数 |
| `scripts/train_fastwam_group.py` | 解析任务组，检查数据，写运行快照，启动官方训练 |
| `fastwam/configs/data/dexjoco_*.yaml` | 单／双臂维度与序列适配器配置 |
| `fastwam/experiments/dexjoco/data.py` | 检查本地 LeRobot 元数据，计算所选训练帧的精确 min/max |
| `fastwam/experiments/dexjoco/dataset.py` | 按任务调用官方 v2/v3 reader，构造视频／动作序列 |
| `fastwam/experiments/dexjoco/transforms.py` | 训练／推理共用图像处理和归一化 |
| `fastwam/experiments/dexjoco/policy.py` | 严格加载完整策略，调用官方 infer_action，反归一化 |
| `fastwam/experiments/dexjoco/serve_policy.py` | 仅复用 openpi-client 的 NumPy/MessagePack 传输协议 |
| `dexjoco/dexjoco_fastwam_client/` | 原始图像观测、既有动作转换、同步闭环评测 |
| `scripts/evaluate_fastwam_group.py` | 批量评测、运行身份检查、失败重试和结果记录 |

训练时每个组默认联合训练一个策略；`--mode per_task` 则依次训练各任务。
FastWAM 单臂动作维度为 **22**、状态 **23**；双臂动作 **44**、状态 **46**。
现有任务组的 `model_action_dim: 32` 属于 π0.5，FastWAM 解析器不会继承它。
组文件的相机值是 LeRobot 特征后缀，例如 `observation.images.front`；
策略侧固定采用单臂 `[base, wrist]`、双臂 `[base, wrist_left, wrist_right]` 顺序。
如果实际数据使用其他后缀，先修正任务组映射。

## 环境准备

在独立 FastWAM 环境安装官方依赖；不把 Torch/CUDA 或 FastWAM 的 NumPy 版本装进现有仿真环境。
下面的命令从 `~/physwam/fastwam` 执行。上游 revision 记录在 `UPSTREAM_REVISION`。

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
python -m pip install -U pip
python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install pyyaml 'websockets>=14,<16' msgpack pytest
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
```

GPU/CUDA 驱动必须与官方安装要求匹配。默认 batch=1 是每个设备的 batch，
不是显存容量保证。有效全局 batch 还要乘 GPU 数量与梯度累积次数。
`fastwam` 环境使用 on-the-fly T5 编码，第一版不需要预生成文本缓存。
服务端按任务缓存文本条件，但每次观测仍重新编码视觉条件。

仿真端使用现有 `dexjoco` 环境，需 `openpi-client`、`websockets`、`msgpack` 等现有传输依赖。
源码运行脚本会把新 package 加入 PYTHONPATH；需要控制台入口时重新执行
`python -m pip install -e ./dexjoco`，之后可运行 `dexjoco-fastwam-eval --help`。

## 先查看配置（无需数据、权重、GPU）

在 `~/physwam/dexjoco` 执行：

```bash
python scripts/train_fastwam_group.py --group single_arm --tasks water_plant \
  --run-name fastwam_water_smoke --dry-run
python scripts/train_fastwam_group.py --group dual_arm \
  --run-name fastwam_dual_baseline --dry-run
```

`--dry-run` 不创建 run、不下载模型、不启动训练。可用 `--fastwam-root` 指定非同级目录，
`--dataset-root` 指定数据目录。相对数据路径按 DexJoCo 仓库根目录解析。

## 数据与最小训练

先下载并解压 LeRobot 数据。可复用已有 `prepare_group_data.py --group single_arm --data-only`；
该下载脚本的环境与解压说明沿用 `group_training.md`，不会下载 FastWAM 权重。
数据适配支持官方本地 LeRobot v2/v3 视频布局，要求 action/state 是已转换好的 rotvec 动作与机器人状态，
不接受特权环境状态或 π0.5 的 padding。所有任务的数据 fps 必须匹配 `control_hz`。
默认 horizon=32、video_stride=4，训练视频为 9 帧；不是 33 个视频帧。
未来查询由官方 reader 在 episode 内截断并生成 padding mask，绝对目标动作在末尾复制。

第一阶段只选择 water_plant：

```bash
conda activate fastwam
cd ~/physwam/dexjoco

# 只读取元数据和低维 parquet，计算所选训练 episodes 的统计量
python scripts/train_fastwam_group.py --group single_arm --tasks water_plant \
  --run-name fastwam_water_smoke --steps 20 --stage prepare

# 解码每个任务的首个序列，输出形状及首尾图像，检查相机和时间对应
python scripts/train_fastwam_group.py --group single_arm --tasks water_plant \
  --run-name fastwam_water_smoke --steps 20 --stage inspect

# 使用官方预处理脚本从 Wan 构造 ActionDiT backbone（可能下载大模型）
python scripts/train_fastwam_group.py --group single_arm --tasks water_plant \
  --run-name fastwam_water_smoke --steps 20 --stage initialize --gpus 0

# 使用官方 Accelerate / DeepSpeed 训练器
python scripts/train_fastwam_group.py --group single_arm --tasks water_plant \
  --run-name fastwam_water_smoke --steps 20 --stage train --gpus 0
```

同一个 run 的参数必须保持一致，因此上面四个命令都保留 `--steps 20`。
初始化 backbone 由官方明确排除机器人动作输入／输出层，可以供两种动作维度共享。
已存在的初始化不会被覆盖。resume 必须明确指定同一 run 的完整 state 目录：

```bash
python scripts/train_fastwam_group.py --group single_arm --tasks water_plant \
  --run-name fastwam_water_smoke --steps 20 --stage train --gpus 0 \
  --resume runs/fastwam_water_smoke/checkpoints/state/step_000010
```

这是恢复示例，目录必须实际存在；它会继续原先的训练步数目标。
初始化权重不是策略 checkpoint，完整部署必须使用训练生成的 mot + proprio_encoder 权重。
官方生成式视频评测仍关闭（eval_every=0）；独立留出集通过 validation_loss 路径只计算原始训练目标的 loss。
闭环成功率仍通过独立 seeds／任务环境评测，不能把验证 loss 当作成功率。

## 留出验证集（仅计算 loss）

`fastwam_baseline.yaml` 默认开启验证：

```yaml
# 【验证 / validation】
validation_ratio: 0.1          # [数据/划分] 每任务留出的完整 episode 比例；0 关闭
validation_seed: 42            # [验证/随机性] 固定划分与验证噪声
validation_interval: 1000      # [验证/频率] 每多少次优化器更新验证一次；最后一步也验证
validation_batch_size: 1       # [验证/GPU] 每张 GPU 的验证 batch，不进行梯度累积
validation_max_batches: null   # [验证/预算] null 遍历全量；整数限制每任务的全局验证批次数
```

先从任务组 `episodes` 指定的候选（null 为全部）中划分，验证集成员不会进入训练 reader 或训练统计。
每任务验证条数为 ceil(N×比例)，限制在 1 到 N−1；不足两条时需补充数据或关闭验证。
划分算法与 π0.5 共用 `scripts/dataset_split.py`，相同候选、比例、种子得到相同 episode 成员。
`--stage prepare` 保存 `data_split.json` 并构造独立的 `data.train` / `data.val`，后者复用训练集的 min/max。
`--dry-run` 只展示实验设置，不读取元数据或展开 episode ID。

训练器以 `torch.no_grad()` 调用官方 `training_loss`，记录总 loss 及视频/动作分量，不调用视频生成、推理服务或模拟器。
验证临时切换到 eval 模式，结束后恢复各模块模式和 Python/NumPy/Torch 的随机状态，不执行 backward 或 optimizer.step。
所有 rank 对齐验证调用，按有效样本数汇总，尾批次的占位样本不计入统计，只由主进程写入结果。
任务均衡模式汇总各任务平均 loss；frame_uniform 模式按实际样本数加权。

查看 `runs/<name>/validation_loss.jsonl`、`train.log`，或启用 W&B 后的 `val/loss`、`val/<task>/loss`。
默认遍历完整验证集；设置批次上限时按任务固定抽帧，并记录是否全量。验证比例是 episode 比例，不是精确帧比例。
恢复必须沿用原划分和统计；已有无验证的训练不能改配置后继续作为独立验证，应新建 run。

## 多任务训练

```bash
# 单臂六任务共享一个策略
python scripts/train_fastwam_group.py --group single_arm \
  --run-name fastwam_single_baseline --stage prepare
python scripts/train_fastwam_group.py --group single_arm \
  --run-name fastwam_single_baseline --stage train --gpus 0,1

# 双臂五任务共享另一个策略
python scripts/train_fastwam_group.py --group dual_arm \
  --run-name fastwam_dual_baseline --stage prepare
python scripts/train_fastwam_group.py --group dual_arm \
  --run-name fastwam_dual_baseline --stage train --gpus 0,1

# 需要逐任务独立模型时显式选择
python scripts/train_fastwam_group.py --group single_arm --mode per_task \
  --run-name fastwam_independent --stage prepare
```

任务均衡通过等长的虚拟任务分区实现：每个任务重复较短数据、循环覆盖自身训练帧，
再交给官方 ResumableEpochSampler 打乱与分布式分发。一个完整虚拟 epoch 各任务样本数相同，
任意短前缀不保证严格均衡。`frame_uniform` 则按所有任务帧连接，不重复小任务。
统计量使用全部所选训练帧的精确 min/max，因此不存在不同采样权重下均值估计的歧义。
`episodes` 有选择时 reader 与统计量均遵守选择；请让同一原始示范的随机化版本留在同一划分。

## 服务与闭环评测

终端 A，FastWAM 环境，从 FastWAM 根目录启动：

```bash
conda activate fastwam
cd ~/physwam/fastwam
python -m experiments.dexjoco.serve_policy \
  --dexjoco-root ../dexjoco --run-dir ../dexjoco/runs/fastwam_water_smoke \
  --checkpoint ../dexjoco/runs/fastwam_water_smoke/checkpoints/weights/step_000020.pt \
  --gpus 0 --port 8001
```

需保留 Wan 的 VAE、T5 等基础组件，只有 policy checkpoint 不构成全部模型资产。
run 可复制到其他机器；服务使用其快照配置与统计量，不访问训练数据路径。

终端 B，现有 DexJoCo 环境：

```bash
conda activate dexjoco
cd ~/physwam/dexjoco
python scripts/evaluate_fastwam_group.py --run-dir runs/fastwam_water_smoke \
  --tasks water_plant --episodes 2 --seeds 0 --evaluation-name smoke_eval --port 8001
python scripts/summarize_results.py \
  --evaluation-dir runs/fastwam_water_smoke/evaluation/smoke_eval
```

默认同步执行：预测 32 步，执行 8 步再规划。推理期间仿真不 step，
该结果验证闭环功能，不证明实际机器人实时控制能力。完整异步执行、延迟补偿和重叠融合尚未实现。
请求携带 task、prompt、episode_id、step_id；响应必须回显 episode_id、step_id。
重置后不保留动作缓存。服务返回物理单位下的 rotvec 动作，环境只做一次四元数转换。
`click_mouse` 沿用官方评测的 30 步初始姿态对齐（在第一个策略观测前）。

正式评测更换 evaluation-name、seeds 和 episode 数；`--rand-full` 启用视觉随机化。
物体随机化沿用任务原环境，rand_full 不会自动改变训练数据。
同一 evaluation-name 固定 checkpoint SHA256、配置签名、任务、步数和随机化设置，
已有完成 job 跳过；失败 job 需 `--retry-failed`。缺失 job 不报告完整任务均值。
控制 hz 按示范标注及环境 step 接口定义；无需修改 MuJoCo 内部 physics_dt。

## 运行产物与验证

```
runs/<name>/
  fastwam_config.yaml          # 传给官方 trainer 的完整配置
  resolved_config.yaml         # 划分后的任务与实验快照
  data_split.json              # 候选、训练与验证 episode 清单
  validation_loss.jsonl         # 逐步总 loss、逐任务 loss 与验证样本数
  dataset_manifest.json        # metadata hash、parquet size/mtime、episode 选择
  dataset_stats.json           # 精确 min/max 与统计来源
  versions.json                # 两个仓库版本、工作区状态、Python
  inspection/                  # 每个任务首尾相机拼接与序列形状
  checkpoints/weights/         # 官方策略 .pt
  checkpoints/state/           # 官方优化器、调度器、随机状态及进度
  train.log
  evaluation/<name>/           # 计划、逐 job 结果、视频、逐 episode、延迟及汇总
```

数据清单记录 metadata hash 和 parquet 的 size/mtime，不对全部视频做内容哈希。
当前自动按每任务比例划分验证 episode，仅计算 loss；不自动选择最佳 checkpoint。
FastWAM 的完整模型 GPU 训练与显存适配仍需在目标服务器验证。

测试只需 CPU。完整 FastWAM 环境安装后，从 DexJoCo 根目录执行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_fastwam_framework.py
```

测试包含单／双臂维度、配置解析、官方归一化一致性、所选 episode 统计隔离、
临时 MP4/Parquet 测试夹具的官方 v3 reader 边界检查、任务均衡、模拟模型推理后处理、
同步重规划与旧 episode 响应拒绝。它们不替代 GPU 训练与真实任务性能评测。
