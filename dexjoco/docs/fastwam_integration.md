# FastWAM / DexJoCo 框架

参数类别、GPU 设置、部署与 eval 选项见 [中文配置索引](configuration_guide.md)。

## 目录与职责

本工作区包含同级 `dexjoco/` 与 `fastwam/`。现有 π0.5 路径继续使用原脚本。
FastWAM 官方模型、注意力、flow matching 损失、训练器保持原样；适配代码位于
`fastwam/experiments/dexjoco/`。本地框架测试不等同于完成模型训练或复现论文成绩。

| 位置 | 职责 |
|---|---|
| `configs/task_groups/*.yaml` | 复用现有任务成员、指令、相机映射、训练 episode 选择 |
| `configs/experiments/fastwam_single.yaml` | 八卡单臂六任务联合训练配方，默认 30,000 步 |
| `configs/experiments/fastwam_smoke.yaml` | 四卡真实数据短跑，20 步，每 10 步保存/验证 |
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

从 PhysWAM 工作区根目录（含 dexjoco/ 和 fastwam/）执行一键安装器。
它建立独立 FastWAM 环境；上游 revision 记录在 `fastwam/UPSTREAM_REVISION`。

```bash
bash dexjoco/scripts/setup_fastwam_env.sh

# 新服务器需要代理及 AIFS 缓存时，换成该机可写路径：
bash dexjoco/scripts/setup_fastwam_env.sh --setup-proxy --proxy-port 17890 \
  --cache-dir /aifs/chenkaiyue/.cache/physwam \
  --model-dir /aifs/chenkaiyue/models/fastwam
```

两个安装命令选择一个即可。已有代理用 `--proxy URL` 替代 --setup-proxy；
也支持 --prefix、--mihomo-binary、--subscription-file、--dry-run 和 --check-gpu INDEX。
代理管理复用 `setup_mihomo.py`，具体参数见 [环境安装参数](configuration_guide.md#环境安装参数)。
安装器检测 Conda，缺失时安装 Miniforge；创建/复用 Python 3.10 环境，安装 FFmpeg 7、
编译工具、PyTorch 2.7.1+cu128、DeepSpeed 0.18.7、FastWAM 和轻量 openpi-client。
FastWAM 的依赖版本来自 `fastwam/pyproject.toml`；附加工具受
`configs/environments/fastwam_constraints.txt` 约束，完整版本写入安装日志旁的 versions.json。
Torch 先安装；DeepSpeed 在该环境中构建，安装阶段隐藏 GPU、关闭预编译自定义扩展。
默认检查 pip 依赖、CPU 参数更新、小视频解码、Parquet/消息传输和训练配方解析，不需要空闲 GPU。
`--check-gpu 0` 另验证指定单卡 BF16 前向/反向及 NCCL 可用性，不启动多卡通信或完整训练。

手动安装的环境可直接运行 `scripts/check_fastwam_env.py`。默认按项目依赖核对训练库，
对 PyYAML、websockets、msgpack 检查运行接口兼容范围，不要求 wheel、setuptools、pytest
与安装配方逐版本一致。一键安装器会加 `--strict-versions`，额外核对完整安装约束。
例如 wheel 0.48.0 不会单独使手动环境自检失败；训练依赖和实际视频/GPU 运算仍需通过检查。

按脚本输出加载 Conda，然后在每个新终端激活并加载本机设置：

```bash
conda activate fastwam
source "$HOME/.config/physwam/fastwam-env.sh"
```

该文件独立于 pi05-env.sh，记录代理、HF/ModelScope/扩展缓存及 Wan 模型目录，
默认模型下载源为 huggingface，可用 --model-source modelscope 改变。
--model-dir 控制 Wan 组件位置，默认 `fastwam/checkpoints/`；生成的 ActionDiT 骨干仍位于
`fastwam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`。
安装器不下载数据/模型、不初始化骨干，不执行训练；后续 initialize/train 会按需下载组件。
安装阶段不修改 NVIDIA 驱动或系统 CUDA；完整模型的显存和 DeepSpeed 运行链路需服务器 smoke 验证。
默认 batch=1 是每个设备的 batch，
不是显存容量保证。有效全局 batch 还要乘 GPU 数量与梯度累积次数。
`fastwam` 环境使用 on-the-fly T5 编码，第一版不需要预生成文本缓存。
服务端按任务缓存文本条件，但每次观测仍重新编码视觉条件。

仿真端使用现有 `dexjoco` 环境，需 `openpi-client`、`websockets`、`msgpack` 等现有传输依赖。
源码运行脚本会把新 package 加入 PYTHONPATH；需要控制台入口时重新执行
`python -m pip install -e ./dexjoco`，之后可运行 `dexjoco-fastwam-eval --help`。

## 训练配方与启动优先级

`fastwam_single.yaml` 默认单臂全部 6 个任务联合训练一个 policy，
使用八卡 "0,1,2,3,4,5,6,7"，训练前需释放全部卡上的旧进程。
`fastwam_smoke.yaml` 默认单臂 water_plant，`fastwam_dual.yaml` 默认双臂全部任务；
这两个配方默认四卡 "0,1,2,3"。
有效全局 batch=每卡 batch×卡数×梯度累积次数；以实际 YAML 和 dry-run 输出为准。
保持 BF16、梯度检查点、ZeRO-1，未启用 CPU offload。
这是按每卡 80GB 的 H100 规划的起点，显存与吞吐仍需服务器实测。
`--group`、`--tasks`、`--gpus` 可覆盖 YAML；`tasks: null` 才会选择整个任务组。

| 配方 | 更新步数 | 保存间隔 | 验证间隔 | 每任务验证批次上限 |
|---|---:|---:|---:|---:|
| fastwam_single.yaml（默认，单臂全部任务，8 卡） | 30000 | 1000 | 1000 | 50 个全局批次 |
| fastwam_smoke.yaml | 20 | 10 | 10 | 2 个全局批次 |
| fastwam_dual.yaml（双臂全部任务） | 20000 | 1000 | 1000 | 50 个全局批次 |

smoke 使用同一个官方训练器与真实数据；没有 dummy 模型，也不是 π0.5 的 smoke 脚本。
它不会自动检查参数差分或自动恢复后再多训一步。保存、验证的结果应结合日志检查，恢复需要显式命令。
正式训练请使用新的 run-name，不在同一 run 内切换配方或修改预算。

### W&B 运行名称

可直接在 `configs/experiments/fastwam_single.yaml` 中设置：

```yaml
wandb_enabled: true
wandb_name: fastwam_single_all_8h100
```

`wandb_name` 覆盖基础 `fastwam/configs/train.yaml` 的 `wandb.name`；
设为 `null` 时跟随该次启动的 `--run-name`（per_task 模式包含任务后缀）。
省略字段的旧配方保持基础配置名称，兼容旧 run；名称必须是非空字符串或 null。
名称不改变输出目录，生成后的值保存在 `runs/<run-name>/fastwam_config.yaml` 的 `wandb.name`。
已 prepare 的 run 固定配置快照，修改该字段后应使用新 run-name；不会重命名正在运行的 W&B 记录。

## 数据与首次短跑

以下命令均从 `~/physwam/dexjoco` 执行，在独立 fastwam 环境中运行：

```bash
conda activate fastwam
cd ~/physwam/dexjoco

FASTWAM_ARGS=(
  --experiment configs/experiments/fastwam_smoke.yaml
  --run-name fastwam_water_smoke
)

# 不读取数据、不创建 run、不下载权重；确认只有 water_plant、4 个进程、全局 batch 4
python scripts/train_fastwam_group.py "${FASTWAM_ARGS[@]}" --dry-run

# 下载/准备真实数据后：划分 episodes，检查元数据，计算仅含训练集的统计
python scripts/train_fastwam_group.py "${FASTWAM_ARGS[@]}" --stage prepare

# 每任务解码一个真实序列，输出首尾画面和张量形状
python scripts/train_fastwam_group.py "${FASTWAM_ARGS[@]}" --stage inspect

# 首次准备一次：只使用首个可见 GPU；骨干已存在时手动跳过此命令，入口会拒绝覆盖
python scripts/train_fastwam_group.py "${FASTWAM_ARGS[@]}" --stage initialize

# 启动四个训练进程，训练 20 步
python scripts/train_fastwam_group.py "${FASTWAM_ARGS[@]}" --stage train
```

需要使用另外四张卡时，在每个命令追加 `--gpus 4,5,6,7`。同一 run 参数保持一致。
训练日志为 `runs/fastwam_water_smoke/train.log`；检查 loss 有限、验证记录及第 10/20 步权重和状态目录。
初始化会按需下载 Wan 组件；初始化骨干不是训练后的策略 checkpoint。

数据预期位于 `datasets/dexjoco_lerobot_datasets/water_plant/`，或使用 `--dataset-root` 覆盖父目录。
可复用 `prepare_group_data.py --group single_arm --data-only` 下载，但该命令选择整个单臂组，不只 water_plant。
适配器支持官方本地 LeRobot v2/v3 视频布局；单臂 action/state 必须为 22/23 维，且不含特权状态或 π0.5 padding。
视频与动作 fps 必须匹配 control_hz。horizon=32、video_stride=4 对应 9 个视频帧。

检查恢复时，显式恢复该 run 已保存的第 10 步完整状态，继续到原目标第 20 步：

```bash
python scripts/train_fastwam_group.py "${FASTWAM_ARGS[@]}" --stage train \
  --resume runs/fastwam_water_smoke/checkpoints/state/step_000010
```

目录必须实际存在。smoke checkpoint 只用于检查加载和闭环，不作为正式性能结果。
训练完成后，从下文的服务和评测入口进行闭环检查。

## 八卡单臂全部任务联合训练

`fastwam_single.yaml` 的 `tasks: null` 选择 water_plant、hammer_nail、click_mouse、
pick_bucket、pinch_tongs、fold_glasses 六个任务，由语言指令区分，共享一个 policy。
保持双相机、state/action 23/22 维、32 步动作和 9 帧视频；任务数量增加不增加单个样本的相机数。
`task_balanced` 让各任务在虚拟数据集中占相同数量的索引，并不保证每个 batch 的任务数完全相等。

起始参数为每卡 batch 16 × 8 卡 × 梯度累积 1 = 全局 batch 128，
学习率保持 1e-4，max_steps=30000，对应固定 5% 预热为 1500 次优化器更新。
训练预算不是单任务步数的机械倍增，应结合每任务留出集 loss 与闭环成功率选择 checkpoint。
每卡 4 个数据 worker，共 32 个；实际 CPU、视频解码与存储吞吐仍需服务器检查。
若后续提高每卡 batch，应同步比较全局 batch、每步耗时和样本吞吐；改参数后使用新的 run-name。

先同步配置到服务器，确认 GPU 0–7 均已释放旧任务：

```bash
source /data/9qTymlC6XNreTg7J/home/miniforge3/etc/profile.d/conda.sh
conda activate fastwam
cd /aifs/chenkaiyue/physwam/dexjoco

FASTWAM_ARGS=(
  --experiment configs/experiments/fastwam_single.yaml
  --run-name fastwam_single_all_8h100_v1
)
python scripts/train_fastwam_group.py "${FASTWAM_ARGS[@]}" --dry-run
python scripts/train_fastwam_group.py "${FASTWAM_ARGS[@]}" --stage prepare
python scripts/train_fastwam_group.py "${FASTWAM_ARGS[@]}" --stage inspect
python scripts/train_fastwam_group.py "${FASTWAM_ARGS[@]}" --stage train
```

不要附加 `--tasks water_plant`，它会覆盖 YAML 并退回单任务。
prepare 会为六个任务重新固定训练/验证 episodes 并计算统一训练集归一化统计；
inspect 应输出六条任务记录，video/action/proprio 分别为 [3,9,224,448]、[32,22]、[32,23]。
已有 `fastwam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` 可复用，
文件存在时跳过 initialize；本配方从基础初始化开始，不加载旧 water_plant 策略 checkpoint。
`--resume` 只用于继续本 run 的完整训练状态，不能用来切换任务集合或归一化统计。

训练日志为 `runs/fastwam_single_all_8h100_v1/train.log`。
每 1000 步保存、验证并启用 W&B；优先检查首批优化器更新的显存峰值、有限 loss 和实际吞吐。
可先以本配方另建短跑 run（例如增加 `--steps 20`、换 run-name），通过后再新建正式 run；
短跑的预算与学习率预热不同，不能把它的 loss 当作正式训练表现。

## 单任务正式训练

```bash
# 默认配方为全部任务；通过 --tasks 显式选择 water_plant，其他参数沿用配方
python scripts/train_fastwam_group.py --tasks water_plant --run-name fastwam_water_single --stage prepare
python scripts/train_fastwam_group.py --tasks water_plant --run-name fastwam_water_single --stage train

# 改为其他单臂任务（prepare/inspect/train 都保持相同参数）
python scripts/train_fastwam_group.py --tasks hammer_nail \
  --run-name fastwam_hammer_single --stage prepare
```

官方生成式视频评测保持关闭（eval_every=0）；独立留出集只计算 training_loss。
验证 loss 不等于闭环成功率。

## 留出验证集（仅计算 loss）

`fastwam_single.yaml` 默认开启验证：

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

## 扩展到多任务

单臂配方默认是单任务。如需组内联合训练，可在自己的实验 YAML 中将 `tasks` 改为 null，
或者使用 `--tasks` 显式列出成员。例如双臂任务子集：

```bash
python scripts/train_fastwam_group.py --group dual_arm \
  --tasks bimanual_assembly bimanual_hanoi \
  --run-name fastwam_dual_subset --stage prepare
```

正式训练的所有阶段重复相同任务与实验参数。`--mode per_task` 会为所选任务分别创建 run。
任务均衡通过等长虚拟任务分区和官方 sampler 实现；frame_uniform 按所有实际帧采样。

## 双臂全部任务共享一个 policy

`configs/experiments/fastwam_dual.yaml` 设置 `group: dual_arm`、`tasks: null`，
读取任务组当前的 5 个任务：`bimanual_microwave_cook`、`bimanual_unlock_ipad`、
`bimanual_hanoi`、`bimanual_assembly`、`bimanual_photograph`。
采用 `task_balanced` 采样，各任务通过语言指令区分；joint 模式只生成一个 run 和一套共享权重。

状态 46 维、动作 44 维；三相机按 `base/wrist_left/wrist_right` 顺序横向拼接为 224×672。
每卡 batch=1、梯度累积=8，四卡有效全局 batch=32；每进程 2 个 CPU 数据 worker，启用 W&B。
每 1000 次更新计算各任务留出 loss，每任务最多 50 个全局批次，指标会标注是否全量。
20000 步是五任务共享的更新预算；三相机显存和多 worker 吞吐仍需服务器短跑验证。

从 DexJoCo 根目录启动，所有阶段保持相同参数：

```bash
FASTWAM_DUAL_ARGS=(
  --experiment configs/experiments/fastwam_dual.yaml
  --run-name fastwam_dual_all
  --gpus 4,5,6,7
  --mode joint
)
python scripts/train_fastwam_group.py "${FASTWAM_DUAL_ARGS[@]}" --dry-run
python scripts/train_fastwam_group.py "${FASTWAM_DUAL_ARGS[@]}" --stage prepare
python scripts/train_fastwam_group.py "${FASTWAM_DUAL_ARGS[@]}" --stage inspect
python scripts/train_fastwam_group.py "${FASTWAM_DUAL_ARGS[@]}" --stage train
```

准备全部五个任务的数据，并提前登录 W&B。已有 ActionDiT 初始化骨干可复用；
若尚未生成，在 train 前用相同参数执行一次 `--stage initialize`。
首次短跑可为上述参数统一追加 `--steps 20` 并改用独立的 run-name，
正式训练切换回新的 run-name。不要传 `--mode per_task`，否则会拆为五个独立 policy。

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
同一 evaluation-name 固定 checkpoint SHA256、配置签名、任务、步数、随机化设置、渲染后端和视频录制设置，
已有完成 job 跳过；失败 job 需 `--retry-failed`。缺失 job 不报告完整任务均值。
控制 hz 按示范标注及环境 step 接口定义；无需修改 MuJoCo 内部 physics_dt。

### EGL 原生崩溃时隔离渲染后端

如果日志先输出某回合 `success: true`，随后在 `mujoco_rendering.py` 的
`mjr_readPixels` 路径出现 `Fatal Python error: Aborted`，该回合已成功，
整个评测仍未完成，不能据此计算完整成功率。WebSocket 线程出现在栈中本身不能证明连接故障。

批量入口支持 `--gl-backend osmesa`，将仿真图像改为 CPU 软件渲染，
策略服务继续使用原有 GPU。需先同步评测脚本，保留正在运行的推理服务。
在服务器的 `dexjoco` 环境先确认 OSMesa 可用：

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python -c \
  'import mujoco; c=mujoco.GLContext(64,64); c.make_current(); c.free(); print("OSMesa OK")'
```

若缺少 OSMesa 库，需要安装与该环境兼容的 OSMesa 运行库后再评测。
以下示例连接已经加载 v5 的 step 4000 权重、监听 8002 端口的服务：

```bash
conda activate dexjoco
cd /aifs/chenkaiyue/physwam/dexjoco
python scripts/evaluate_fastwam_group.py \
  --run-dir runs/fastwam_water_single_4h100_v5 \
  --tasks water_plant --host 127.0.0.1 --port 8002 \
  --seeds 0 --episodes 2 --gl-backend osmesa \
  --evaluation-name ckpt_004000_osmesa
python scripts/summarize_results.py \
  --evaluation-dir runs/fastwam_water_single_4h100_v5/evaluation/ckpt_004000_osmesa
```

软件渲染通常较慢；该方式用于隔离 EGL 路径，不能确认底层驱动或上下文故障的具体原因。
切换后端必须使用新的 `evaluation-name`，避免覆盖原来的部分结果。
`plan.json`、`result.json` 和汇总均记录后端；汇总拒绝混合 EGL 与 OSMesa 结果。
原有未记录后端的 FastWAM 结果按 EGL 处理。

### 保留 EGL，隔离视频录制路径

如果简单 EGL 场景以及真实任务的保持姿态测试均通过，而完整策略评测仍崩溃，
可以只关闭 MP4 录制进行对照。先同步批量入口、客户端 `evaluate.py` 和汇总脚本，
推理服务不需重启：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=3 \
PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 \
python scripts/evaluate_fastwam_group.py \
  --run-dir runs/fastwam_water_single_4h100_v5 \
  --tasks water_plant --host 127.0.0.1 --port 8002 \
  --seeds 0 --episodes 2 --gl-backend egl --no-video \
  --evaluation-name ckpt_004000_egl_novideo
```

`--no-video` 不创建视频写入器，也不启动录制用的 FFmpeg 进程；
双相机策略观测仍正常渲染，模型、动作执行、随机种子和成功判定不变。
输出目录仍叫 `videos/`，其中保留 `episodes.jsonl`、`latency.jsonl` 和成功计数文件，
但没有 MP4。计划、结果与汇总记录 `record_video: false`；旧版结果默认视为已录制。
更换录制设置必须换评测名，汇总拒绝与计划不一致的录制设置。

单次关闭录制后通过只表明故障在该次运行中未复现，不能直接证明 FFmpeg 是根因；
策略动作导致的场景变化、时序变化及间歇性驱动故障仍需进一步区分。
如果关闭录制仍崩溃，应获取崩溃进程的原生调用栈继续定位。

## 验证尾批次与 VAE CUDA Graph

如果训练正常到第 1000 步，第一次验证在 VAE 编码的 `cudagraph_trees.py` 报
`The size of tensor a (8) must match the size of tensor b (6)`，需要同步
`fastwam/src/fastwam/models/wan22/fastwam.py` 的 VAE 输入形状保护。
训练固定形状仍使用静态 CUDA Graph；与首次捕获的形状、stride、精度或设备不一致的输入
使用相同 VAE 的普通前向，避免验证 batch 8、尾批次 6、缺样本 rank 的占位 batch 1
进入已有静态捕获。训练尾批次也受保护，恢复原形状时继续复用原捕获。

该修复不丢弃样本、不补齐有效样本、不改归一化和验证加权计数，不需要降低
`validation_batch_size: 8` 或关闭验证；普通 VAE 前向的耗时需服务器实测。
本地测试包含真实微型 Wan VAE 的数值对照、训练→验证→尾批次→训练、
完整验证计数和 RNG/梯度保留；CUDA 环境还会运行实际 CUDA Graph 后端测试。

```bash
# FastWAM 环境，DexJoCo 根目录；不下载模型权重
OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/test_fastwam_vae_batching.py
```

先用新的 run-name、`--steps 20 --stage train` 检查最终验证能够完成，
再创建正式 run；不要直接等到第 1000 步才确认验证路径。
代码修复不改变 run 快照，已有完整训练状态可按原配置 `--resume`；
当前保存顺序在验证之后，第 1000 步验证失败不能证明已保存该步 checkpoint。
恢复前需检查 `checkpoints/state/step_*/trainer_state.json` 及完整状态资产，
没有有效状态时需新建 run 重训，日志中的 step 不能用于恢复。

## 梯度检查点与启动兼容性

DexJoCo 配方使用 `gradient_checkpointing: true`，生成的模型配置保持
`compile_training_denoise: false`。MoT 的 `forward_joint_core` 在训练且启用梯度时，
以非重入检查点逐层重计算完整 video/action 联合层，包括 QKV、混合注意力、
交叉注意力和 FFN；推理与只前向验证不启用重计算。目前训练编译与该检查点路径
不能同时开启，冲突时会提示关闭 `compile_training_denoise`。

如果旧版启动报 `Wan MoT gradient checkpointing is not supported by the compiled tensor core`，
需同步 `fastwam/src/fastwam/models/wan22/mot.py` 的检查点实现。
仅删除异常检查或关闭实验 YAML 中的显存优化，并不能验证这条训练路径正确。
该错误发生在模型构造阶段；同步修复后可沿用原 run 配置、统计量和已完成的 ActionDiT
初始化，直接重试 `--stage train`，不需重新 `initialize`。

以下 CPU 回归测试不下载权重，覆盖真实微型 Wan/Action DiT 的多层前向、输入和参数梯度、
反向重计算、激活保存量、完整训练 loss 与优化器更新：

```bash
# 在 DexJoCo 根目录、FastWAM 环境执行
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/test_fastwam_checkpointing.py
```

这些检查不替代目标服务器的四卡 BF16/DeepSpeed 训练与显存验证。

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
