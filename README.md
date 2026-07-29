# AscendC

华为昇腾算子挑战赛 S9 的本地工作区归档，包含赛题源码、官方样例、
benchmark/优化版本、测试脚本、提交包、调优记录、相关开源资料镜像以及
AscendC 调优 skill。

## 目录

- `workspace/raceS9/`：各赛题的主要工作区、实现记录、测试与提交归档。
- `workspace/raceS9/submissions/20260729_upload_ready/`：当前可上传判题器的
  `Concat.zip`、`Greater.zip`、`IndexAdd.zip` 和 `SquareSumV1.zip`。
- `workspace/utils/repos/`：赛事和 AscendC 相关开源仓库的源码快照。
- `workspace/utils/skills/`：比赛期间收集的 AscendC/CANN 技巧。
- `development_history/`：各阶段源码、官方测试材料、远程构建脚本、回归与
  profiling 记录。
- `skills/optimize-ascendc-operators/`：本次比赛中沉淀的 Codex AscendC
  分析、profiling、调优和打包工作流。

## 当前版本摘要

- Concat：对超过 UB 的连续段进行分片并行，合成场景 AICore 中位耗时由
  6.54 us 降到 4.88 us。
- SquareSumV1：补齐 `ReduceSum` 后 Vector 到 Scalar 的硬件事件同步，
  20/20 压力测试通过。
- IndexAdd：提供按 index 顺序更新的安全版本，避免输出维度较大时的任务
  数量爆炸，并保持重复 index 的顺序语义。
- Greater：保存当前五个判题 Case 均通过的版本。

详细结果见：

- `development_history/s9_delivery_20260729/PROFILE_AND_TUNING_NOTES.md`
- `development_history/s9_delivery_20260729/SUBMISSION_MANIFEST.md`

## 注意

这是私有比赛工作仓库。`utils/repos` 和 `development_history` 中包含来自
其他开源项目或官方样例的源码快照，其版权和许可证仍归各原项目所有。
提交比赛前应以 `SUBMISSION_MANIFEST.md` 中记录的 SHA-256 和包结构校验结果
为准。

