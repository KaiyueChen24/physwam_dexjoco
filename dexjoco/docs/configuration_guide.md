# 训练、GPU、部署与评测参数索引

本文件说明当前 π0.5 与 FastWAM 分组框架的配置来源。配置字段和命令行选项使用原名，中文注释说明类别、单位与适用阶段。
参数旁的中文标记是说明，不是新增配置键；不能把未接入的键写进 YAML 并期待生效。

## 类别与修改位置

| 类别 | 用途 | 修改位置 |
|---|---|---|
| 配置 / config | 格式版本、模型选择、实验名、源码路径 | 实验 YAML 与命令行 |
| 模型 / model | 动作预测长度、文本长度、模型分支与维度 | 实验 YAML、任务组与模型模板 |
| 数据 / data | 任务、数据目录、相机、episode、视频解码和采样 | 任务组 YAML 与实验 YAML |
| 归一化 / normalization | 状态、动作统计的计算和保存 | π0.5 实验 YAML；FastWAM 适配器使用所选训练帧精确 min/max |
| 训练 / train | 更新步数、batch、学习率、优化器、梯度累积 | 实验 YAML；部分默认值位于 Python 或上游模板 |
| GPU / gpu | 可见显卡、参数分片、混合精度、梯度检查点 | `--gpus`、π0.5 实验 YAML、FastWAM 实验 YAML 与 Accelerate/DeepSpeed |
| 日志与保存 / logging & checkpoint | 指标记录、W&B、权重保存和恢复 | 实验 YAML 与命令行 |
| 部署 / serve | 服务权重、监听地址、端口 | 服务入口命令行 |
| 推理 / inference | 动作采样迭代、噪声调度 | FastWAM 实验 YAML，部署时读取已保存快照 |
| 留出验证 / validation | 按 episode 留出数据，仅前向计算 loss | 实验 YAML 中 validation_*；划分写入 run |
| 评测 / eval | 任务环境、回合数、种子、执行步数、超时、重试 | 环境 YAML 与评测入口命令行 |

主要配置已逐字段标注：

- `configs/experiments/h100_lora.yaml`：π0.5 正式训练。
- `configs/experiments/pi05_smoke.yaml`：单张 H100 80GB、真实数据的 π0.5 短跑。
- `configs/experiments/fastwam_baseline.yaml`：FastWAM 训练与推理起始配方。
- `configs/task_groups/{single_arm,dual_arm}.yaml`：两组共享的任务与数据契约。
- `configs/{rand_obj,rand_full,multi_task,ipad_reasoning}/*.yaml`：仿真环境与语言指令。
- `openpi/config.yaml`：原始 OpenPI/DexJoCo 配置入口，不替代分组实验 YAML。
- 同级 `fastwam/configs/data/dexjoco_{single,dual}.yaml`：数据适配模板。
- 同级 `fastwam/configs/{train.yaml,model/fastwam.yaml}`：FastWAM 上游默认训练/模型设置。
- 同级 `fastwam/scripts/accelerate_configs/accelerate_zero1_ds.yaml`：当前 FastWAM 启动器使用的 GPU/DeepSpeed 设置。

## 读取优先级与运行快照

π0.5：支持的命令行覆盖项优先于实验 YAML；`dataset_root: null` 回退到任务组路径。实验 YAML 中的 `sampling` 会覆盖组内默认值。
FastWAM：支持的命令行覆盖项优先于实验 YAML，实验配置再覆盖上游训练和模型模板；采样方式读取实验 YAML。
FastWAM 适配器还会显式设置部分字段，例如 `eval_every=0`（生成式视频评测关闭；留出 loss 验证使用独立路径）。这些固定设置在构造器中标有中文类别。

训练 run 中保存的配置是该次实验采用的值。服务/评测以 run 快照为准，修改源 YAML 不会自动改变已有 run。
π0.5 的 `resolved_config.yaml` 记录组和实验设置，但不会展开所有 Python 默认值。
FastWAM 的 `fastwam_config.yaml` 包含传给官方训练器的配置，并保留 Hydra 插值；`resolved_config.yaml` 保存组与实验快照。

## GPU 与 batch 的区别

| 参数 | π0.5 | FastWAM |
|---|---|---|
| `--gpus 0,1` | 让 JAX 看到两张卡 | 启动两个 Accelerate 训练进程 |
| `batch_size` | 所有卡合计的全局 batch | 每张卡的 batch |
| 梯度累积 | 当前分组入口未暴露该选项 | `gradient_accumulation_steps` |
| 有效全局 batch | `batch_size` | `batch_size × GPU 数 × gradient_accumulation_steps` |
| 参数分片 | `fsdp_devices`，1 为不分片的数据并行 | 当前使用 Accelerate + DeepSpeed ZeRO-1 模板 |
| 减少激活显存 | 沿用模型内部实现 | `gradient_checkpointing` |

π0.5 的全局 batch 必须能被可见 GPU 数整除；可见 GPU 数必须能被 `fsdp_devices` 整除。
`num_workers` 是 CPU 数据加载工作进程数；`stats_batch_size` 是 π0.5 低维统计的 CPU 批次，均不表示 GPU 数。

## 训练入口的命令行参数

执行各脚本的 `--help` 可以看到中文类别与说明。

| 入口 | 关键参数与用途 |
|---|---|
| `scripts/train_group.py` | `--group`、`--experiment`、`--run-name`、`--gpus`；`--stage prepare/stats/train`；`--resume` 恢复最近状态 |
| `scripts/smoke_train_pi05.py` | 默认读取 H100 真实数据 pi05_smoke；脚本自身测试恢复，不接受 --resume |
| `scripts/train_fastwam_group.py` | `--stage prepare/inspect/initialize/train`；`--tasks` 可选子集；`--mode joint/per_task`；`--resume` 必须指向完整训练状态目录 |
| `scripts/prepare_group_data.py` | `--group`、`--data-only`、`--weights-only`、`--data-revision`、`--model-revision`；权重下载针对 π0.5 |

## 部署与闭环评测参数

部署参数由服务入口命令行设置，不从训练 YAML 读取。

| 参数 | 类别 | 用途 |
|---|---|---|
| `--run-dir` | 部署/实验 | 使用该 run 的结构、归一化与任务快照 |
| `--checkpoint` | 部署/权重 | π0.5 为完整 checkpoint 目录；FastWAM 为训练完成的策略 `.pt`，不是初始化 backbone |
| `--gpus` | GPU/部署 | 部署进程可见的显卡编号 |
| `--host` | 部署/连接 | 服务端为监听地址；评测端为要连接的服务器地址 |
| `--port` | 部署/连接 | π0.5 默认 8000，FastWAM 默认 8001 |

评测使用 `scripts/evaluate_group.py` 或 `scripts/evaluate_fastwam_group.py`：

| 参数 | 类别 | 用途与当前默认值 |
|---|---|---|
| `--tasks` | 评测/任务 | 选择训练快照中的任务，省略则全部评测 |
| `--group` | 评测/身份 | 仅 π0.5：可选，核对与训练组一致 |
| `--seeds` | 评测/随机性 | 环境种子列表；π0.5 默认 0、1、2，FastWAM 默认 0；与训练 seed 独立 |
| `--episodes` | 评测/预算 | 每个任务、每个 seed 的回合数，默认 50 |
| `--evaluation-name` | 评测/保存 | 结果子目录名，默认 rand_obj；名称不会自动开启随机化 |
| `--timeout` | 评测/超时 | 每个任务与 seed 组合的进程超时，π0.5 默认 7200 秒、FastWAM 默认 86400 秒 |
| `--retry-failed` | 评测/重试 | 重跑已有失败 job；已完成 job 继续跳过 |
| `--dry-run` | 评测/计划 | 只打印任务计划，不连接服务或运行仿真 |
| `--max-steps` | 评测/预算 | FastWAM 每个回合的环境步数上限，默认 3000；不是训练更新数 |
| `--execute-steps` | 评测/重规划 | FastWAM 每次最多执行的预测动作步数，省略时读取 run 的 execute_steps |
| `--rand-full` | 评测/随机化 | FastWAM 批量入口开启完整视觉随机化；不改变训练数据 |

π0.5 批量入口当前没有 `--rand-full`、`--randomize-dynamics` 或 `--replan-ratio` 覆盖项。
直接使用 `dexjoco-openpi-eval` 时，这三个参数分别控制完整视觉随机化、动力学随机化与异步重规划阈值比例。
它的 `--render-mode` 控制窗口/图像模式，`--record-pressed-digits` 控制密码记录，`--metrics-output` 指定耗时 JSONL。
`--pad-state-dim46` 仅供相应兼容模型使用；正常单臂分组模型不需要该选项。
这些字段已在函数签名处添加中文注释，可查看 `dexjoco-openpi-eval --help`。

FastWAM 实验 YAML 中三个运行参数的区别：

- `action_horizon`：预测多少个动作时间步。
- `execute_steps`：闭环中执行多少步后再观察/规划。
- `num_inference_steps`：生成每个动作块时迭代采样多少次；`sigma_shift` 调整采样时间调度。

## Python 与上游模板中的默认值

π0.5 的 `scripts/group_common.py::build_train_config` 固定使用语言侧 `gemma_2b_lora`（rank=16、alpha=16）和动作侧 `gemma_300m_lora`（rank=32、alpha=32）。
AdamW 默认 β1=0.9、β2=0.95、eps=1e-8、weight_decay=1e-10，梯度裁剪为 1；EMA 关闭。
冻结策略沿用模型过滤器，视觉编码器仍可训练，不是只更新 LoRA。默认模型计算类型为 bfloat16，不意味着优化器状态均为 BF16。

FastWAM 的 `experiments/dexjoco/config.py::build_config` 从官方 `configs/train.yaml` 与 `configs/model/fastwam.yaml` 继承基础设置，再覆盖任务维度与实验参数。
当前官方训练器使用 AdamW（β1=0.9、β2=0.95）、总步数 5% 的学习率预热及模板指定的调度类型；梯度裁剪读取 `max_grad_norm`，默认 1。
`eval_every=0` 关闭内置训练集视频评测；`load_text_encoder=true` 在线编码文本；`compile_training_denoise=false` 关闭该路径的训练编译。
`num_epochs=100000` 仅在 max_steps 缺失时用于估算总步数；当前配方有 max_steps，因此由 max_steps 决定结束。

当前 DeepSpeed JSON 是标准 JSON，不能插入注释；字段解释如下：

| JSON 字段 | 类别与用途 |
|---|---|
| `train_batch_size` | 训练/批次：全局有效 batch，auto 由 Accelerate 解析 |
| `train_micro_batch_size_per_gpu` | 训练/批次：每卡微批次，auto 由加载器与 Accelerate 推导 |
| `gradient_accumulation_steps` | 训练/累积：auto 从 Accelerator 的训练设置取得 |
| `zero_optimization.stage` | GPU/分片：1 为 ZeRO-1，主要分片优化器状态 |
| `zero_optimization.offload_optimizer.device` | GPU/卸载：none 表示不将优化器卸载到 CPU/NVMe |
| `zero_optimization.offload_param.device` | GPU/卸载：none 表示不卸载参数 |
| `zero_optimization.overlap_comm` | GPU/通信：是否让通信与计算重叠 |
| `zero_optimization.contiguous_gradients` | GPU/梯度缓冲：是否将梯度放入连续缓冲区 |
| `zero_optimization.reduce_bucket_size` | GPU/通信：归约桶的元素数上限，不是训练 batch |
| `zero_optimization.allgather_bucket_size` | GPU/通信：聚合桶的元素数上限，不是训练 batch |

不要直接修改模型隐藏维度、层数、注意力头数等结构参数来适配显存，它们与预训练权重形状有关。
新增可调参数时，需要同时在读取器/构造器接入，并补齐本文件与对应 YAML 的中文类别和单位说明。

## 留出集 loss 参数（validation，与模拟器 eval 分开）

| 参数 | 类别 | 含义 |
|---|---|---|
| `validation_ratio` | 数据/划分 | 每任务候选 episode 的留出比例，默认 0.1；0 关闭；按完整 episode 划分 |
| `validation_seed` | 验证/随机性 | 固定划分与验证噪声，默认 42；两种模型同任务、同候选时采用相同划分 |
| `validation_interval` | 验证/频率 | 每多少次优化器更新验证一次，正式训练默认 1000，最后一次更新也验证 |
| `validation_batch_size` | 验证/GPU | π0.5 为全局 batch（默认 8）；FastWAM 为每卡 batch（默认 1），不做梯度累积 |
| `validation_max_batches` | 验证/预算 | null 遍历全部；整数限制每任务的批次数，固定抽帧并标注是否全量 |

任务组的 `episodes` 现在表示划分候选：null 使用全部示范，列表先限制候选，再留出验证部分。
真实数据不足两条候选 episode 时无法同时保留训练和验证，会报错；不会偷偷把训练数据用于验证。
划分和样本数见 `data_split.json`；逐步验证指标见 `validation_loss.jsonl`。归一化始终只使用划分后的训练数据。
验证逐任务统计，按 task_balanced 或 frame_uniform 汇总；验证 loss 不作为模拟器成功率。

当前 FastWAM 仍设置 `eval_every=0`，其含义是关闭生成式视频评测，并非关闭本次新接入的留出 loss 验证。
更改比例或种子会改变数据划分，必须使用新的 run-name，不能把已经参与训练的数据改称独立验证集。
