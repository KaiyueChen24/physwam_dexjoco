# FLORA × DexJoCo：离线推理与评测

使用 NVIDIA 已发布的权重给 DexJoCo 视频做评分、动作切分与动作描述。这里不训练模型，不改变 FastWAM / π0.5 的代码或环境。

**本目录包含真实官方推理源码，并非根据论文重新实现模型。** 官方代码发布在 Hugging Face 的模型仓库 `batch_utils/` 中。来源已固定到 commit 并记录 SHA-256；许可证、第三方声明和官方原始 README 均保留。

- [代码架构与数据契约](docs/architecture.md)
- [官方代码审查、固定版本和论文差异](docs/upstream_audit.md)
- [官方项目](https://research.nvidia.com/labs/gear/flora/)
- [Rewarder 官方代码](https://huggingface.co/nvidia/Flora-Rewarder/tree/031299e46e452fc0cedb4e0975eb9fa95fc37783/batch_utils)
- [TempSeg → Captioner 官方代码](https://huggingface.co/nvidia/Flora-Captioner/tree/ad744f46335aec671f5caac09ac97d817fbdb114/batch_utils)

## 目录

```text
flora/
├── configs/                    # 单臂、双臂、FastWAM rollout 配置
├── physwam_flora/
│   ├── cli.py                  # prepare / inspect / run / summarize
│   ├── datasets.py             # LeRobot v2/v3、FastWAM 录像、手工 manifest
│   ├── media.py                # CPU 视频检查、v3 episode 提取
│   ├── jobs.py                 # 官方 JSONL 格式与相机顺序；隔离真值标签
│   ├── runner.py               # 进程/环境隔离、固定 revision、日志与来源记录
│   ├── evaluation.py           # 结果评分指标、dense 解析失败统计
│   └── common.py               # 配置、路径、JSON、源码完整性检查
├── vendor/
│   ├── data_engine/            # 原样保留官方 TempSeg → Captioner 实现
│   ├── rewarder/               # 原样保留官方 terminal / dense / rubric 实现
│   └── sources.lock.json       # 上游 commit 与每个文件的 SHA-256
├── scripts/setup_env.sh        # 两套独立 Python 3.12 环境
├── scripts/download_models.py  # 预览模型 revision；--execute 才下载
├── tests/                      # CPU 测试，包含真实合成视频及官方预处理
└── docs/
```

## 当前完成范围

- 已接通官方四个入口：`terminal`、`dense`、`rubric`、`caption`。
- `caption` 表示完整 TempSeg → Captioner 流水线，采用官方窗口、拼接、重试、提示词、采样和精度处理。
- 支持 LeRobot v2 每 episode 视频、v3 共享视频的元数据裁切，以及本工作区 FastWAM 评测录像。
- 支持手工 manifest 接入其他策略录像；不假定 π0.5 的文件名与 FastWAM 相同。
- 只做离线分析；无 RL、策略训练或实时机器人控制。
- 已通过 CPU 数据流和官方预处理验证。尚未安装完整 GPU 推理环境、下载模型权重或运行 H100 模型推理，因此没有真实 DexJoCo 标注效果、吞吐或峰值显存结果。

## 环境与资源

官方公开代码验证配置为 Python 3.12。Rewarder 使用 `vllm==0.24.0`，时序标注使用 `torch==2.11.0` / `transformers==5.8.1`，分别安装，避免与已有策略环境冲突。

从 **physwam 根目录**执行。下面先预览安装命令；去掉 `--dry-run` 才安装依赖：

```bash
bash flora/scripts/setup_env.sh rewarder --dry-run
bash flora/scripts/setup_env.sh captioner --dry-run

bash flora/scripts/setup_env.sh rewarder
# 确定需要动作标注时再安装：
bash flora/scripts/setup_env.sh captioner
```

可用 `FLORA_PYTHON=/path/to/python3.12` 指定解释器。安装器不下载模型，不修改已有 Conda 环境，不自动安装驱动。

官方新发布的 `batch_utils/README.md` 明确说明：terminal/dense 用单张 H100 80GB；rubric 默认 TP=2；TempSeg/Captioner 在 80GB GPU 上先后运行。此次本地未复测这些资源结论。初跑 `max_num_seqs: 1`；rubric 长上下文更重。Captioner 发布资产包含 FP32 权重，官方建议至少 150GB 模型下载/缓存空间；加 Rewarder 和视频中间产物，建议预留 300GB 以上。

配置里的解释器路径默认指向上述 `.venv-*`。控制层也可以只安装 `pip install -e './flora[data]'`，再把 `runtime.*_python` 指向已有的、满足官方依赖的 Python 3.12 环境。

## 第一步：确认数据并生成 manifest

先修改 `flora/configs/dexjoco_single.yaml` 或 `dexjoco_dual.yaml` 的任务、数据根目录和 episode 选择。所有 YAML 中的路径均相对 **physwam 根目录**；命令行文件路径按正常 shell 当前目录解析。

```bash
flora/.venv-rewarder/bin/python -m physwam_flora verify-sources

flora/.venv-rewarder/bin/python -m physwam_flora prepare \
  --config flora/configs/dexjoco_single.yaml \
  --output flora/runs/pick_bucket_inputs
```

输出 `episodes.jsonl` 和 `prepare.json`。LeRobot v2 引用已有视频，不复制；v3 根据每相机的 chunk/file/from/to 元数据提取 episode，额外写 `media/`。v3 使用原始帧率逐帧导出 H.264 CRF 0，检查帧数与时间连续性，记录原文件和偏移，不把共享文件内其他 episode 送入模型。导出的原始文件仍保持不变。

任务指令复用 `dexjoco/configs/task_groups/*.yaml`。专家演示的成功标签默认为 `null`，不因目录名或来源推断成功。默认只取每任务前 5 条，方便检查接入；正式评测请显式选择具有代表性的成功和失败样本。

对于 FastWAM 评测，使用 `dexjoco_rollout.yaml`，填写 `source.root` 指向某次 attempt 或其 `videos/`，从 `episodes.jsonl` 获取真实成功标签。若有同目录的 `environment.yaml`，以评测快照的任务指令和相机映射为准。`--no-video` 运行没有 MP4，prepare 会明确报错，不把日志伪装成视觉输入。

## 第二步：纯 CPU 输入检查

```bash
flora/.venv-rewarder/bin/python -m physwam_flora inspect \
  --manifest flora/runs/pick_bucket_inputs/episodes.jsonl --stage terminal
```

检查可解码性、帧率、帧数和视角同步。`--stage caption` 额外检查官方最大 120 秒、7200 帧、单文件 256 MiB、单边 4096 像素、至少 4 FPS 等限制。不自动降采样、裁短整任务或给截断片段继承整段成功标签。

单臂：Rewarder 使用 `base,wrist`；Captioner 默认只用 `base`，因为官方不支持两视角动作标注。双臂：Rewarder 顺序是 `wrist_left,base,wrist_right`；Captioner 以三个命名参数传入主相机与左右腕相机，由官方实现布局。`profile: dex` 是官方输入配置，不代表 DexJoCo 已经被验证或用于训练。

## 第三步：预览并运行结果评分

```bash
flora/.venv-rewarder/bin/python -m physwam_flora run \
  --config flora/configs/dexjoco_single.yaml \
  --manifest flora/runs/pick_bucket_inputs/episodes.jsonl \
  --stage terminal --gpus 0 \
  --output flora/runs/pick_bucket_terminal --dry-run

# 去掉 --dry-run，真正调用官方模型；首次会下载固定版本权重。
flora/.venv-rewarder/bin/python -m physwam_flora run \
  --config flora/configs/dexjoco_single.yaml \
  --manifest flora/runs/pick_bucket_inputs/episodes.jsonl \
  --stage terminal --gpus 0 \
  --output flora/runs/pick_bucket_terminal
```

`--dry-run` 不写文件、不解码、不联网、不加载 GPU。另有 `--native-check`：执行上游的 CPU 输入检查；Rewarder 可能联网下载 tokenizer/processor，但不下载权重。两者不能同时使用。

运行输出包括：

```text
pick_bucket_terminal/
├── jobs.jsonl         # 传给模型的任务、视频；没有成功真值
├── labels.jsonl       # 只供事后评估的成功真值
├── inputs.json        # 解码与同步检查结果
├── run.json           # 命令、配置、输入 manifest 摘要、上游 revision
├── run.log            # 官方进程完整日志
├── status.json        # complete / failed / interrupted
└── terminal.jsonl     # 官方原始结构化结果
```

所有 prepare/run 输出目录必须是新目录；失败时保留日志和已完成结果，不静默覆盖，也不自动把部分结果视为完整评测。

## 其他阶段

使用同一 `run` 命令，更换 `--stage` 和输出目录即可：

| stage | 官方模块 | 结果 |
|---|---|---|
| terminal | `batch_utils.terminal_reward` | 1～5 分、分数 token 概率、expected_score、解释 |
| dense | `batch_utils.dense_reward` | 时段质量、原始回答、invalid_clips |
| rubric | `batch_utils.rubric_reward` | 自定义步骤分数、采样与 medoid 结果 |
| caption | `batch_utils`（data_engine） | `annotations/<id>.json`，含时间戳、动作描述与审计记录 |

`rubric` 需要 `--gpus 0,1`，并在 `rubric.files` 中配置 **任务名 → JSON 文件**。JSON 为官方 `action`/`grading` 格式的步骤列表，等级包含 `label`/`points`，各步骤最高分之和必须为 1。框架不自动生成任务成功标准；格式见 [上游 rubric 示例](vendor/rewarder/batch_utils/examples/rubric_jobs.jsonl)。

`caption` 需要安装 captioner 环境；控制命令可以从 rewarder 环境启动，实际模型进程会使用 `runtime.captioner_python`。两个上游包都叫 `batch_utils`，必须由 runner 分进程调用，不要在同一个 Python 进程里同时导入。

## 权重缓存与预下载

无需训练；每个模型 revision 均固定。可以先预览再显式下载：

```bash
flora/.venv-rewarder/bin/python flora/scripts/download_models.py rewarder

HF_HOME="$PWD/flora/models/huggingface" \
  flora/.venv-rewarder/bin/python flora/scripts/download_models.py rewarder --execute

HF_HOME="$PWD/flora/models/huggingface" FLORA_MODEL_CACHE="$PWD/flora/models/data_engine" \
  flora/.venv-captioner/bin/python flora/scripts/download_models.py captioner --execute
```

后一个命令包括 TempSeg adapter、Qwen3.5-9B base 和 Captioner。必要时用 Hugging Face 登录或环境变量提供访问凭据，不把 token 写进配置。官方 README 中仍有旧的“private repositories”文字；本次源码审查通过公开 NVIDIA 仓库进行，不依赖内部 GR00T checkout。

## 结果评估

有独立环境成功标签时：

```bash
flora/.venv-rewarder/bin/python -m physwam_flora summarize \
  --predictions flora/runs/pick_bucket_terminal/terminal.jsonl \
  --labels flora/runs/pick_bucket_terminal/labels.jsonl \
  --threshold 5 --output flora/runs/pick_bucket_terminal/metrics.json
```

报告混淆矩阵、balanced accuracy、失败召回率、基于整数分数的 AUROC、缺失预测和未标注数量。阈值 5 是本地评估约定，不是官方成功概率校准结果；只有一个类别时，balanced accuracy/AUROC 输出 null。专家数据没有成功真值时不伪造准确率。

过程质量先检查解析失败：

```bash
flora/.venv-rewarder/bin/python -m physwam_flora summarize \
  --kind dense --predictions flora/runs/pick_bucket_dense/dense.jsonl
```

官方把无法解析的 dense 窗口回退为 normal，必须同时检查 `invalid_clips`。定位错误时间、解释是否正确仍需人工或额外事件真值。Dense 是利用后续画面的离线判断。

## CPU 测试

在安装 `[data]` 的解释器中，从 physwam 根目录运行：

```bash
PYTHONPATH=flora python -m unittest discover -s flora/tests -v
```

覆盖真实合成视频、v2/v3 数据适配、v3 帧边界、双臂视角顺序、缺失录像、真值隔离、固定 revision、官方 Captioner 输入验证、官方 Rewarder 预处理/输出解析和评估指标。没有 PyAV / PyArrow 时相应测试会跳过；不能把跳过视为 GPU 推理通过。
