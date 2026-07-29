# utils

`utils` 保存不直接进入 S9 最终提交、但用于学习、复现和调优的资源。

- `repos/`：高相关开源仓库，各自保留 `.git`，便于查看来源和更新历史。
- `competitions/`：S6、S7、S8 赛事题包与公开调用样例。
- `official-materials/`：S9 官方题包、环境手册和打包脚本。
- `skills/`：本次 S9 使用的 CANN/AscendC 学习资料和工作流。

新增的 `repos/s7` 来自 [ascend-124/s7](https://github.com/ascend-124/s7)，固定检查版本为提交 `4fc4c56b093cb9e53d6ca649cd694542761b7d26`。针对 Concat 的可复用经验与实测结论记录在 `repos/s7-notes.md`。

第三方仓库默认不并入参赛代码；只把经过官方测试和隐藏几何回归验证的通用技巧移植到 `raceS9`。
