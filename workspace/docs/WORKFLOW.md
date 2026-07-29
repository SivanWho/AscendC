# S9 参赛工作流

目标不是“写出一个能过公开样例的 Kernel”，而是建立一条可重复的 **规格 → 泛化实现 → 正确性矩阵 → Profiling → 单变量优化 → 提交归档** 流程。

## 0. 冻结规则与环境

每周或收到官方通知后更新一次规则快照，记录：

- CANN、驱动、固件、OS、芯片型号和 PyTorch/torch_npu 版本。
- 官方截止时间、打包规则、精度阈值与榜单规则。
- 当前提交包 SHA-256、Git commit、提交时间和平台成绩。

不要在同一轮实验中升级 CANN 或更换机器；环境变化会使耗时失去可比性。

## 1. 规格拆解

对每题写出一张约束矩阵：dtype、rank、shape、属性、负轴、空维、非 32 字节对齐、特殊值、广播/重复 index/多轴归约等。官方 Excel 是接口规格，`tests/official` 只是一个公开冒烟样例。

先用 PyTorch 参考算子生成 Golden，再为边界场景建立自有测试。禁止把公开 shape 写死进 Tiling 或 Kernel。

## 2. 建立正确性基线

先做最简单、完全泛化、易审计的版本：

1. Host 侧规范化属性（负轴、dims、axis）并验证输出 shape。
2. 按 32 字节对齐设计尾块处理，明确 GM/UB 的合法访问边界。
3. 单核或保守多核版本先通过完整自测。
4. 比较不同 dtype 的误差与特殊值行为。
5. 保存一个“已知正确但可能较慢”的 commit，后续用于二分回退。

## 3. 建立性能基线

同一环境下至少记录：

- shape、dtype、属性与数据分布。
- AI Core 执行时间的中位数、P10/P90、预热次数与重复次数。
- 使用核数、每核数据量、UB tile、buffer 数、Tiling key。
- `PipeUtilization`、搬运与 Vector 指令占比、拖尾核现象。

可从以下命令形态开始，按 CANN 8.5.0 的工具帮助调整参数：

```bash
msprof --output=./profiling --ai-core=on \
  --aic-metrics=PipeUtilization <your_test_command>
```

## 4. 优化阶梯

每一步只验证一个主要假设，并在 [EXPERIMENT_LOG_TEMPLATE.md](EXPERIMENT_LOG_TEMPLATE.md) 中记录。

1. **核间切分**：充分利用物理核，同时避免尾核拖尾和严重负载不均。
2. **核内 Tiling**：根据 UB 容量、输入/输出 buffer 数和 dtype 计算 tile；不把公开 shape 当常量。
3. **对齐与尾块**：主路径走对齐 DMA，尾块使用合法的 padding/掩码方案。
4. **流水并行**：评估 CopyIn/Compute/CopyOut 重叠；Double Buffer 对大数据常有利，但可能劣化小 shape。
5. **搬运优化**：合并连续搬运、减少小块 DMA、复用地址计算，关注 stride 和广播带来的非连续访问。
6. **API 与数据类型**：选择适合的高阶 API，避免无意义 Cast；对 fp16/bfloat16 的归约评估 fp32 累加。
7. **多路径泛化**：用 Tiling key 区分算法族（例如连续/广播、last-axis/非 last-axis、small/large），判断依据必须来自通用属性。
8. **小 shape 优化**：降低启动、TilingData 搬运、队列与同步开销；不要默认 Double Buffer 总是更快。

## 5. 回归门禁

任何性能改动合入前必须通过：

- 所有 dtype 与 rank。
- 非 32 字节对齐和尾块。
- 极小、普通、极大 shape。
- 题目特有边界：空 Concat 分片、Greater 广播和 NaN、IndexAdd 重复索引、Transpose 任意合法排列、SquareSumV1 负轴/多轴/keep_dims。
- 重复执行，排除偶发错误和未初始化内存。
- 与基线对比，确认不是只优化一个 case 而大幅劣化其他 case。

## 6. 提交闭环

1. 提交前保持工作区干净，记录 commit。
2. 编译 `custom_*.run`，确认它来自同一 commit。
3. 只保留 `op_host/`、`op_kernel/` 和 `custom_*.run`。
4. 使用 `utils/official-materials/zip_op.sh` 打包。
5. 解压到临时目录复核结构、文件权限和源码一致性。
6. 保存 ZIP 的 SHA-256、提交时间、平台成绩和日志。
7. 因平台按“最后提交”处理，提交前确认新包不会覆盖更好的有效版本。

仓库提供 `scripts/record_submission.ps1`，用于把上述关键信息追加到对应赛题的 `notes/submissions.md`。

## 7. 每周节奏建议

- 周一：同步规则、榜单和仓库，选定本周一个瓶颈。
- 周二至周四：单变量实验，日终保存最优有效 commit。
- 周五：全矩阵回归、长时间稳定性测试、整理 profiling。
- 周末：提交候选包并复核榜单；不要在截止日前才首次走完打包流程。
