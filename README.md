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

## 使用

现有命令仍从各组件目录运行，两个 Python 环境独立安装：

```bash
cd dexjoco
python scripts/train_fastwam_group.py --group single_arm --tasks water_plant \
  --run-name fastwam_water_smoke --steps 20 --dry-run
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
