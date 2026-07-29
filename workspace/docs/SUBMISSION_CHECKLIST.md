# 提交检查表

## 正确性

- [ ] 所有官方 dtype 均覆盖。
- [ ] 极小/普通/极大 shape 均通过。
- [ ] 非 32 字节对齐、尾块和零长度边界已测。
- [ ] 负轴、广播、重复 index、任意 permutation、多轴归约等题目特有边界已测。
- [ ] NaN/Inf/-Inf 与整数精确行为符合 PyTorch Golden。
- [ ] 连续重复运行无偶发错误、越界或未初始化数据。

## 性能

- [ ] 在固定环境中完成预热和多次测量。
- [ ] 记录 AI Core 耗时而不是只记录端到端耗时。
- [ ] 记录核数、UB tile、buffer 数和 Tiling key。
- [ ] 新版本相对已知最优版本没有关键 case 明显退化。
- [ ] 小 shape 和大 shape 都检查过 Double Buffer/核数的收益。

## 泛化与代码质量

- [ ] 没有按公开 shape 或 case 编号硬编码。
- [ ] Tiling 对 rank、轴、dtype、非对齐和尾核均有通用处理。
- [ ] Host 与 Kernel 对 TilingData 的布局和单位一致。
- [ ] 错误路径、资源释放和 workspace 大小已检查。
- [ ] 已运行 CANNBot 的代码审查/精度调试清单并人工复核结果。

## 打包

- [ ] 工作区对应一个明确 Git commit。
- [ ] `custom_*.run` 由该 commit 编译。
- [ ] 源码与最后一次上榜版本一致。
- [ ] 包内只有 `op_host/`、`op_kernel/`、`custom_*.run`。
- [ ] 使用官方 `zip_op.sh`。
- [ ] 在临时目录解压并复核层级。
- [ ] 已记录 ZIP SHA-256、提交时间和成绩。
- [ ] 已确认本次“最后提交”不会覆盖当前更优有效成绩。
