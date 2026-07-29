# Concat V4：S7 经验驱动的数据量自适应并行

## 结论

V4 在 V3 通用二维 DMA 的基础上，只修改 Host 的行批次并行度选择。CANN 8.5 / Ascend 910B4 上，官方脚本三次独立运行分别为 8.70、8.92、8.84 μs，中位数 8.84 μs；V3 为 10.17 μs，稳定提升约 13.1%。Profiler CSV 的三次中位数分别为 8.7605、9.11、8.89 μs，中位数 8.89 μs。

这仍是单个公开用例的 30 次调用中位数，不等同于排行榜全部隐藏用例的总耗时。

## 从 S7 学到的内容

检查仓库 [ascend-124/s7](https://github.com/ascend-124/s7) 的提交 `4fc4c56b093cb9e53d6ca649cd694542761b7d26` 后，重点验证了两类做法：

1. `SegmentReduceGrad` 根据数据量阈值计算 Block Dim，说明小工作量不应机械占满所有 AIV。
2. `Fmax` 和 `SegmentReduceGrad` 在循环外分配硬事件 ID、循环内复用、结束后释放。

第二项在当前 Concat 上实测变慢，因此没有保留。真正有效的是第一项背后的粒度控制原则。

## V4 tiling

Host 计算总输入字节数，并以 16 KiB/目标核估算并行需求：

```text
input_bytes = outer * output_axis * inner * element_bytes
target_cores = clamp(ceil(input_bytes / 16 KiB), 1, available_cores)
chunks_per_input = max(1, ceil(target_cores / input_count))
rows_per_task = min(rows_for_ub, ceil(outer / chunks_per_input))
```

公开用例只有 64 KiB 输入和 9 个切片。V3 为覆盖 40 核把每个输入继续拆行，形成 45 个任务；V4 形成 9 个任务，每个输入一次二维 DMA 批次，Block Dim 为 9。这样降低了任务循环、地址计算、DMA 同步和小数据多核调度开销。

大输入仍会根据总字节数增加 `chunks_per_input`，并受 UB 容量约束；因此该策略不是公开 shape 特化。

## 被拒绝的实验

- 循环外复用 `MTE2_MTE3`/`MTE3_MTE2` 事件 ID：10.3605 μs，慢于 V3。
- 把 Block Dim 直接限制为估算的 4 核：11.710 μs，并行不足。
- 单行批次时跳过 task 除法/取模：8.82 μs，未稳定优于保留版本。

这些版本的 profiler 均保存在 `benchmarks/`，避免以后重复走弯路。

## 正确性

全新安装目录下，`tests/regression_v3.py` 的 8 组逐元素精确比较全部通过，覆盖四种 dtype、首/中/末轴、负轴、非对齐、零长度、64 输入和大于 UB 的回退路径；随后原样官方测试准确性通过。
