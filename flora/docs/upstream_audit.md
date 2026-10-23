# 官方代码审查记录

审查日期：2026-10-23。依据是 NVIDIA Hugging Face 公开仓库中已发布的实际源码；没有依赖私有 GitLab、社区实现或仅按论文推测的接口。

## 来源与版本

| 组件 | 仓库 | 本地代码 revision |
|---|---|---|
| Data Engine（TempSeg → Captioner） | [nvidia/Flora-Captioner](https://huggingface.co/nvidia/Flora-Captioner/tree/ad744f46335aec671f5caac09ac97d817fbdb114) | `ad744f46335aec671f5caac09ac97d817fbdb114` |
| 配对 TempSeg 源码 | [nvidia/Flora-TempSeg](https://huggingface.co/nvidia/Flora-TempSeg/tree/b3f079b160a23f78026b72b80fb7a0e05c57eba8) | `b3f079b160a23f78026b72b80fb7a0e05c57eba8` |
| Rewarder | [nvidia/Flora-Rewarder](https://huggingface.co/nvidia/Flora-Rewarder/tree/031299e46e452fc0cedb4e0975eb9fa95fc37783) | `031299e46e452fc0cedb4e0975eb9fa95fc37783` |

Captioner 与 TempSeg 两个仓库的 `batch_utils/` 在本次下载中逐文件比较完全相同，因此只在 `vendor/data_engine` 保留一份运行时。66 个上游文件连同 SHA-256 记录在 `vendor/sources.lock.json`，不包括权重。源文件保持原样，保留 LICENSE.pdf 与第三方许可证/声明。

**源码 revision 与模型 revision 分开。** 官方 `data_engine/batch_utils/models.json` 固定的是：

- TempSeg adapter：`114a1076f70eb53f4a58ac045505db3087e79347`。
- Qwen3.5-9B base：`c202236235762e1c871ad0ccb60c8ee5ba337b9a`。
- Captioner：`fef0bbfe11829ca7f3539d58da7a6a2c2f109dc6`。
- 本地 Rewarder 包装器给上游 `--revision` 传入源码锁定版本 `031299e46e452fc0cedb4e0975eb9fa95fc37783`，该快照包含公开权重与处理器。

允许通过 `rewarder.model_path` 使用本地权重目录；该目录身份由使用者负责，run.json 会记录实际路径，不能把它自动称作已校验的官方固定权重。Data Engine 保留官方模型 manifest 与 FP32 精确值检查。

## 本地代码的编写依据与取舍

这套工作区代码是对固定版本官方推理程序的 DexJoCo 接入，不是重新实现模型、训练算法或论文实验。官方实现保存在 `vendor/`；新写的 `physwam_flora/` 负责输入转换、运行编排和结果对照。

| 本地模块 | 直接依据 | 本地新增的职责或约束 |
|---|---|---|
| `datasets.py` | `dexjoco/configs/task_groups/*.yaml` 的任务指令/相机映射，LeRobot 元数据，FastWAM 的录像与评测记录格式 | 将 episode 转成统一清单；v3 共享视频提取单条片段；示范数据不自动赋成功标签 |
| `media.py` | 官方 Data Engine `api.py` / `media.py` 的文件、时长及同步契约，Rewarder 按帧索引处理时间的方式 | 模型加载前检查视频；本地统一要求 CFR 是适配器额外约束，不是所有官方模式都要求 CFR |
| `jobs.py` | 官方 Rewarder `engine.py` / `core.py` 的 JSONL 字段，Data Engine `cli.py` / `api.py` 的输入字段 | 将主视角、腕视角映射到官方接口，单独存放成功标签，显式读取任务评分表 |
| `runner.py` | 官方三个 `*_reward.py` CLI，以及 Data Engine `python -m batch_utils` 入口 | 独立解释器、工作目录与 GPU 可见性，固定版本，保存命令和配置，检查输出完整性 |
| `evaluation.py` | 官方 terminal 输出分数和 dense 的 `valid` 字段，以及本地真实成功标签 | 计算混淆矩阵、AUROC 等离线指标；这些不是官方论文评测脚本，阈值须明确记录 |
| `scripts/` | 两套官方 `requirements.txt`、模型清单与仓库 revision | 安装隔离环境、下载指定版本；不安装策略训练环境 |

配置 YAML 是**本地包装器的格式**，不是官方提供的 DexJoCo 配置。下面的区别影响如何理解“复现”：

| 设置 | 当前本地值/行为 | 固定版官方依据与区别 |
|---|---|---|
| `rewarder.max_num_seqs` | `1`，初跑控制并发 | `engine.py` 公共默认为 `8` |
| `rewarder.gpu_memory_utilization` | 各阶段统一 `0.85` | 公共默认 `0.88`；rubric 入口覆盖为 `0.85` |
| `rewarder.with_rationale` | `true`，便于检查评分依据 | 官方 CLI 默认不开启；开启会改变提示词和输出格式，不只是多保存一个字段 |
| `caption.include_audit` | `true`，保留时序与输入记录 | 官方 CLI 默认不开启；控制是否保留结果中的 `pipeline` 字段 |
| `caption.profile` / `views` | `dex`；单臂主视角、双臂三视角 | 官方支持这些模式；选择它们适配 DexJoCo 是本地假设，尚未验证域迁移效果 |
| `source.limit_per_task` | 示范 `5`、rollout `20`，按编号升序取前若干条 | 本地试跑选择，无官方对应参数；不构成随机抽样或完整 benchmark |
| TP / rubric 候选数 | terminal/dense `1`、rubric `2`；候选数 `9` | 沿用对应官方入口默认值 |

采样帧数、上下文长度、生成温度等未在本地 YAML 暴露的参数沿用固定版上游实现。不能随意向 YAML 加键就认为已修改官方行为，应先检查 `runner.py` 是否读取并传参。

`verify-sources` 检查本地 vendor 文件是否与 `sources.lock.json` 的哈希一致；它不验证用户自选权重，也不证明 GPU 推理正确。配置注释说明的是代码当前行为，不代表已在新服务器上实测成功。

## Rewarder 调用链

入口：`terminal_reward.py` / `dense_reward.py` / `rubric_reward.py`。

`engine.py`：CLI、JSONL 读取、`resolve_model`、processor、vLLM `LLM` 实例化。BF16，Gated DeltaNet prefill 使用 Triton，开启 prefix caching，明确传 tensor_parallel_size。`--dry-run` 只取 processor 文件，但可能联网。

`core.py`：PyAV 探测/解码、相机 collage 规划、视频索引元数据、原始提示词、结构化输出 grammar、评分解析、dense 窗口和拼接、rubric medoid。

| 阶段 | 代码中的关键实现 | 本地处理 |
|---|---|---|
| terminal | `terminal_frame_indices` / `terminal_messages` / `parse_terminal` / `score_probabilities` | 直接复用，整段最多 64 帧；保留 probabilities 与 expected_score |
| dense | `dense_windows` / `dense_messages` / `parse_dense` / `stitch_dense` | 直接复用，64 帧/4 FPS、32 个 group、尾部裁去 4 秒；报告 invalid_clips |
| rubric | `rubric_messages` / `parse_rubric` / `medoid` | 直接复用，任务自定义 rubric，由配置显式提供 |

terminal/dense 公开 CLI 默认 16,384 上下文，默认 TP=1。三视角 collage 顺序 left/center/right。`progress` 是 expected_score 的线性缩放，不是成功概率。

官方 `parse_dense` 对解析失败窗口给出 normal 回退，并设置 valid=false。不能只读均值就判定轨迹良好；本地 summarize 单独报告这类窗口。

## TempSeg → Captioner 调用链

入口：`infer.py` → `batch_utils.cli.main` → `DataEnginePipeline.annotate` → `pipeline.annotate`。

- `api.py`：公开输入契约与 `inspect_inputs`，CPU 检查不加载权重。
- `media.py`：PTS 探测、三相机同步检查、逐帧解码与 profile-specific composer。
- `pipeline.py`：先 `segment`，后 `caption_single` / `caption_three`；保留独立并发事件。
- `bootstrap.py`：校验 vendor manifest，再加载原生 production adapter。
- `vendor/production_adapter/...`：实际窗口调度、片段重建、边界重试、时序契约。
- `models.py`：固定权重下载、完整 model manifest 校验、BF16 基础权重与 FP32 GDN 保留。
- `vendor/tempseg_fp32_loader.py`：校验 496 个 LoRA tensor 的 SHA/数值与 FP32 精度，base+adapter 放置于 `cuda:0`，生成时 BF16 autocast。
- Captioner 使用 `device_map='balanced'`，并检查所有参数驻留 CUDA；不能把多卡可见误解为所有模型均采用 tensor parallel。

本地只构造官方 jobs 字段，不替换提示词、计时单位、profile 或拼接规则。单臂默认主视角；双臂固定 main/left_wrist/right_wrist。实际布局由官方 Dex/YAM 等 profile 处理。

## 论文/模型卡与发布代码的差异

前期介绍依据论文和模型卡。本次搭建采用上述固定 commit 的**可执行公开代码**，存在以下必须保留的差异记录：

| 项目 | 论文/模型卡描述 | 公开代码当前默认 |
|---|---|---|
| rubric 采样 | 1 FPS、最多约 82 帧/相机 | `RUBRIC_FPS=2.0`、`RUBRIC_MAX_FRAMES=166` |
| rubric 上下文 | 模型卡提及 32,768 tokens | CLI `max_model_len=98304`、输出最多 20,480 tokens |
| rubric 生成 | 逐步评分/推理 | 温度 1、默认 9 个样本、medoid 汇总，TP=2 |
| 官方硬件说明 | 模型卡只列 H100 | 新 batch_utils README 明确 terminal/dense 单张、rubric 两张 H100 80GB |
| Captioner 权重 | 27B 模型规模 | README 说明发布的是 FP32 资产，加载时有混合精度约束，下载空间不能按 BF16 简单估算 |

因此，默认运行属于“固定公开代码的推理复现”，不能宣称逐项复现论文评测配置或分数。调整 rubric 参数前应另建配置并保留差异记录。本地没有运行 H100，官方硬件声明尚未在这里复测。

上游 README 的部分 private-weights 文案和 HF 自动生成的通用 pipeline 示例，不作为本地推理入口；以实际 `batch_utils` 源码、固定 models.json 和输入契约为准。

## 验证边界

已完成源码 SHA-256 与上游 CODE_MANIFEST 一致性检查、Python 编译、安装脚本语法/命令预览、真实合成视频的 CPU 测试。直接运行了官方 Captioner 的 CPU 输入检查和官方 Rewarder 的解码、采样、输出解析及 dense 拼接函数。

未完成：GPU 依赖安装、权重下载、模型前向、vLLM kernel 编译、真实 DexJoCo 录像评测、吞吐/显存测量。运行 GPU smoke 后再记录硬件、驱动、依赖版本与具体数据，以免把框架测试当作模型效果验证。
