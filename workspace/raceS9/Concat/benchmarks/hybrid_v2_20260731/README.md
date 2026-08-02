# Concat Hybrid V2：shape roofline 与 MTE 记录

测试日期：2026-07-31  
设备：Ascend 910B4（dav-2201）  
软件：CANN 8.5.0  
算子版本：`s9_concat_hybrid_20260730/install_v2`

## 测试口径

- 14 组代表性 shape 全部通过逐元素精度校验。
- 官方测试扩展每次调用会发起 30 次 Concat kernel。统计第一组 30 次中的后 20 次，丢弃前 10 次预热。
- `Task Duration` 取 20 次的中位数，同时在 CSV 中保留 min/median/mean/max。
- Concat 的最低 GM 流量按 `输入读取 + 输出写入 = 2 × output_bytes` 计算。
- GM 参考峰值采用 1.6 TB/s；这是公开硬件参考值，不是本次运行实测到的 HBM 带宽。
- AIV 频率由 profiler 的 `aiv_total_cycles / aiv_time / blockDim` 反推，稳定在约 1.65 GHz。
- 本地 MTE 模型按每方向、每核 64 B/cycle。该数值是分析模型，不作为 910B 官方保证参数。
- 显式串行路径的 MTE 下界按读写总字节计算；双队列路径按 MTE2/MTE3 完全重叠的理想状态计算。
- `theoretical_floor = max(GM参考下界, MTE参考下界)`。它不包含 kernel 启动、Scalar 地址计算、事件同步、非连续 DMA 和尾块损失。

官方参考：

- https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0036.html
- https://www.hiascend.com/hardware/accelerator-card

## 结果总表

| Case | BlockDim | 中位数 us | 理论下界 us | 绝对差距 us | 倍数 | 有效搬运 GB/s | Scalar | MTE2 | MTE3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| official | 9 | 8.610 | 0.138 | 8.472 | 62.43x | 15.22 | 47.9% | 44.4% | 11.2% |
| tile16 | 40 | 11.960 | 2.623 | 9.337 | 4.56x | 350.89 | 37.2% | 56.4% | 10.3% |
| tile32 | 40 | 12.411 | 2.622 | 9.788 | 4.73x | 338.06 | 36.9% | 54.4% | 12.5% |
| tile64 | 32 | 10.780 | 2.622 | 8.158 | 4.11x | 389.14 | 43.5% | 53.3% | 9.4% |
| over64 | 32 | 16.290 | 5.243 | 11.047 | 3.11x | 514.97 | 27.4% | 71.4% | 13.0% |
| dim0_large | 40 | 18.630 | 5.898 | 12.732 | 3.16x | 506.56 | 26.8% | 61.1% | 27.7% |
| dim0_unaligned | 40 | 18.451 | 5.898 | 12.552 | 3.13x | 511.49 | 26.9% | 62.5% | 27.2% |
| dim0_fp32 | 40 | 18.631 | 5.898 | 12.732 | 3.16x | 506.54 | 26.9% | 61.7% | 27.6% |
| dim0_int8 | 40 | 18.281 | 5.898 | 12.382 | 3.10x | 516.24 | 27.6% | 62.7% | 26.0% |
| segment128k | 2 | 5.680 | 1.241 | 4.439 | 4.58x | 92.30 | 60.8% | 34.1% | 19.3% |
| segment256k | 8 | 5.731 | 0.655 | 5.075 | 8.74x | 182.98 | 70.2% | 25.3% | 10.7% |
| segment512k | 16 | 7.530 | 1.311 | 6.219 | 5.74x | 278.51 | 64.8% | 31.6% | 9.0% |
| many64 | 40 | 14.050 | 0.041 | 14.009 | 345.72x | 4.63 | 36.0% | 21.8% | 45.6% |
| many128 | 40 | 13.081 | 0.041 | 13.040 | 320.60x | 4.99 | 40.1% | 26.3% | 40.3% |

完整 shape、dtype、dim、四种耗时统计以及原始 MTE 时间见 `hybrid_v2_shape_roofline.csv`。

## 哪些 shape 问题最大

1. **大量极小输入（many64/many128）是最大相对瓶颈。** 有效数据只有约 65 KB，但耗时 13–14 us，距离纯搬运下界超过 300 倍。此处不是 HBM 带宽不足，而是每输入一次任务/地址计算、短 DMA 描述符及 MTE3 写回同步的固定成本。
2. **dim0 大块与 over64 是最大绝对带宽优化空间。** 它们比参考下界多 11–13 us；有效搬运约 507–516 GB/s。MTE2 占比 61%–71%，下一轮应重点检查读侧 DMA 分块、队列重叠和跨核负载。
3. **官方 shape 仍是固定开销主导。** 只有 128 KB 最低 GM 流量，8.61 us 中 Scalar 约 47.9%、MTE2 约 44.4%。因此继续增大 tile 或追求 HBM 峰值基本不会解决排行榜上的约 100 us 总差距；应减少小段任务、事件和指令描述符数量。
4. **非对齐尾块没有形成明显退化。** `dim0_unaligned` 比对齐的 `dim0_large` 略快，差值在运行波动内，当前尾块泛化路径没有显著异常。
5. **相同总字节的 fp16/fp32/int8 接近。** 四组 dim0 结果都在 18.3–18.6 us，说明性能主要由字节搬运和调度决定，而非 dtype。

## MTE 同步事件与响应

### Path 0：显式单缓冲

`GM --MTE2--> UB -> MTE2_MTE3 event -> UB --MTE3--> GM -> MTE3_MTE2 event -> 下一轮`

- 使用 `HardEvent::MTE2_MTE3`：MTE2 完成后唤醒 MTE3。
- 使用 `HardEvent::MTE3_MTE2`：MTE3 完成后允许 MTE2 复用 UB。
- 该路径在 tile16/tile32/official/many 输入场景中严格串行。
- profiler 标准 `op_summary` 不提供单个 `SetFlag/WaitFlag` 的响应延迟；本报告记录的是同步关系以及各流水单元的累计时间/占比。

### Path 1/2：`TQueBind<..., 2>` 双队列

`Alloc -> MTE2 -> EnQue -> DeQue -> MTE3 -> Free`

- `EnQue/DeQue` 管理 MTE2 生产与 MTE3 消费之间的依赖。
- `FreeTensor` 之后 buffer 才能被下一次 MTE2 复用。
- dim0 场景中 Scalar、MTE2、MTE3 占比之和超过 100%，说明 profiler 已观察到流水重叠；各 ratio 不能简单相加当作总耗时。
- `over64` 的 MTE2 比例达到约 71.4%，是当前最明确的读侧搬运受限样例。

如果需要精确到每条 `SetFlag/WaitFlag` 的等待区间，下一步应导出 msprof timeline 并在 MindStudio 时间线中查看 MTE2/MTE3 队列空洞；`op_summary.csv` 本身只能给流水累计值，不能反推出单事件延迟。
