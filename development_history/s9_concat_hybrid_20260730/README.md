# S9 Concat：S7 启发的自适应三路径优化

日期：2026-07-30  
目标：保留短 2-D DMA 的低启动开销，同时让自然任务数不足的大连续段使用尽量多的 AIV。

## 基线

判题器结果：

| Case | 时间（us） |
|---|---:|
| 1 | 12.560 |
| 2 | 34.808 |
| 3 | 20.692 |
| 4 | 111.516 |
| 5 | 425.860 |
| 合计 | 605.436 |

留档源码位于相邻目录 `s9_concat_hybrid_20260730_before/Concat`，核心文件 SHA-256：

- `op_host/concat.cpp`: `56cb21829997dae9318d7d207e3e483bdb12635face0a6d34c61ca7271732f45`
- `op_kernel/concat.cpp`: `cde3f575f18ae655a12d2923713a334c36f69e6debad5016564c304af5166175`

## V1：短单缓冲 + 长段跨核 tile

- tiling key 0：短段单 Buffer，一次 2-D DMA。
- tiling key 1：超过 64 KiB 的 segment 拆成 `(input, outer row, tile)` 独立任务。
- 使用 `GetCoreNumAiv()`，kernel 显式声明 `KERNEL_TYPE_AIV_ONLY`。
- 1000 个确定性输入契约用例全部通过。
- 官方公开样例三次为 `8.529 / 8.529 / 8.4395 us`，中位数 `8.529 us`。

性能矩阵：

| 几何 | 旧双缓冲（us / blockDim） | V1（us / blockDim） |
|---|---:|---:|
| segment 128 KiB | 9.399 / 2 | 9.619 / 4 |
| segment 256 KiB | 11.239 / 2 | 10.419 / 8 |
| segment 512 KiB | 14.819 / 2 | 11.979 / 16 |
| over64，outer=16 | 24.158 / 32 | 25.038 / 40 |
| dim0 fp16，最少 GM 9 MiB | 43.417 / 2 | 25.678 / 40 |
| dim0 fp32，最少 GM 9 MiB | 43.616 / 2 | 29.498 / 40 |
| dim0 int8，最少 GM 9 MiB | 46.796 / 2 | 25.578 / 40 |

结论：跨核 tile 对自然任务不足的大段非常有效，但 128 KiB 和已有足够 outer
任务的几何会因过度切分略微倒退。

## V2：自适应三路径

- tiling key 0：segment 不超过 64 KiB，单 Buffer 2-D DMA。
- tiling key 1：segment 至少为 `4 * tileBytes`，且
  `outer * nonEmptyInputCount < availableAiv`，跨核 tile。
- tiling key 2：其余长段保留原来的“每 input/outer task 双缓冲”路径。

V2 已完成：

- CANN 8.5 / Ascend 910B 编译成功；
- 四种 dtype 均生成 tiling key 0/1/2 kernel entry；
- 1000/1000 输入契约测试通过；
- official、128/256/512 KiB、over64、dim0 fp16/fp32/int8 profiling
  均完成且精度通过。

远端在 profiling 全部完成后断开 SSH，V2 的逐项 CSV 精确时间尚待连接恢复后抄回。
在读取 CSV、重复官方测试和执行官方 `zip_op.sh` 之前，不将 V2 标记为最终提交版。
