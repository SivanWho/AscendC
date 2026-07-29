# 昇腾 AI 创新大赛——算子挑战赛 S9 工作区

本仓库用于 S9 赛季的资料归档、算子实现、正确性回归、性能实验与最终提交。赛事信息按 2026-07-19 核验，主赛官方页面的报名与提交截止时间为 **2026-09-15 12:00**，指定 **CANN 社区版 8.5.0**、Ascend C 和昇腾云上算力。

## 目录

```text
workspace/
├── raceS9/                 # 5 道赛题的实际开发目录
│   ├── Concat/
│   ├── Greater/
│   ├── IndexAdd/
│   ├── Transpose/
│   └── SquareSumV1/
├── utils/
│   ├── repos/              # 官方/社区高认可度仓库（各自保留独立 Git 历史）
│   ├── competitions/       # S6、S7、S8 官方题包
│   ├── official-materials/ # S9 题包、打包脚本和环境手册
│   └── skills/             # 面向本次比赛的 Skill 导航
└── docs/                   # 工作流、资源索引、实验模板、提交检查表
```

每个赛题目录统一包含：

- `op_host/`：Host 侧注册、shape 推导和 Tiling。
- `op_kernel/`：Ascend C Kernel。
- `tests/official/`：官方题包公开调用样例的原样副本。
- `benchmarks/`：本地基准、profiling 脚本与汇总结果。
- `notes/`：设计决策、失败方案和性能记录。

## 先做什么

1. 阅读 [docs/WORKFLOW.md](docs/WORKFLOW.md) 和 [raceS9/README.md](raceS9/README.md)。
2. 按官方环境手册准备 EulerOS 2.10、CANN 8.5.0 与 NPU 环境。
3. 选择一题，在其 `tests/official/` 下跑通公开样例；公开样例只用于通路验证，不能代表隐藏用例。
4. 先实现完全泛化的正确版本，再建立 profiling 基线和实验日志。
5. 每次优化只改变一个主要变量，并同时回归精度、非对齐、广播/归约、极小与极大 shape。

可在 PowerShell 中运行结构自检：

```powershell
.\scripts\validate_workspace.ps1
```

提交 ZIP 生成后，用下面的命令把 commit、哈希、文件大小和平台成绩写入对应赛题日志：

```powershell
.\scripts\record_submission.ps1 -Operator Concat -ZipPath .\Concat.zip -PlatformScore 559.628
```

## 关键规则摘要

- 1–3 人组队，每名选手只能加入一支队伍。
- 精度通过是性能计分前提；所有验证用例的 AI Core 耗时求和，按耗时升序排名。
- 官方给出的常规精度标准：fp16 为 1e-3，fp32 为 1e-4，int8/int32 无误差；特殊算子可能单独审视。
- Tiling 必须泛化，不得针对已公布用例硬编码，否则可按 0 分处理。
- 可多次提交，但规则页写明“以最后提交为准”；上榜源码、`custom_*.run` 包及最终提交内容必须一致。
- 最终包只应含 `op_host/`、`op_kernel/` 和 `custom_*.run`，并使用主办方提供的 `zip_op.sh`。

## 已发现的官方信息差异

- 主赛页面截止时间是 2026-09-15 12:00；GCC 浙江专项页面写的是 2026-09-07 12:00。两者可能是主赛与专项通道的不同截止时间，应分别确认。
- 主赛“大赛信息”模块写年度积分榜前 50 名晋级，“评分规则”模块仍写前 30 名并提示规则将升级。此项不要自行假定，以赛事群/官方最新通知为准。

## 权威入口

- [S9 主赛官方页面](https://www.hiascend.com/developer/contests/details/41ffbad2024e4ccfa43520c57ffa7b9e?channelCode=S22)
- [GCC 浙江专项页面](https://www.gccorg.com/article/51/23.html)
- [S9 非官方历史榜单追踪](https://ascend.nan2inf.com/?contest=s9)
- [Ascend C 算子编程指南](https://www.hiascend.com/document/redirect/CannCommunityOpdevAscendC)
- [CANNBot Skills](https://gitcode.com/cann/cannbot-skills/tree/master)

> 规则、时间、用例与榜单都会变化。提交前务必重新检查官方页面和赛事群通知。
