# S9 算子首版实现说明

本版目标是先建立一条可编译、可验证、可继续优化的 Ascend C 实现基线，目标 SoC 为赛题指定的 `ascend910b`（CANN 8.5.0）。Host tiling 通过 `PlatformAscendC` 获取可用 AI Core 数和 UB 容量，没有写死具体 910B 板卡子型号参数。

| 算子 | V1 路径 | 已覆盖语义 | 下一轮重点 |
| --- | --- | --- | --- |
| Concat | 多核、UB 分块、非对齐 `DataCopyPad` | 动态输入列表、负 dim、零长度切片、最多 128 输入 | 合并小切片、双缓冲 |
| Greater | 多核通用索引 | NumPy/PyTorch 广播、NaN/Inf、5 种 dtype、bool 输出 | 同 shape 与连续广播向量化 |
| Transpose | 多核通用置换映射 | 1–8 维、负维编号、任意合法 perm | 2D/末两维交换的搬运转置快路 |
| SquareSumV1 | 输出元素多核 | 多轴/负轴、keep_dims、FP32 累加 | 连续归约的向量平方与 ReduceSum |
| IndexAdd | 单核确定性基线 | 负 dim、重复 index、负 index、5 种 dtype | 按 outer/row 并行；分 dtype 使用原子或分桶 |

## 重要边界

- `IndexAdd` 首版刻意使用单核，避免重复 index 引发写冲突；它是正确性基线，不是最终性能版本。
- `Greater`、`Transpose` 和 `SquareSumV1` 的通用慢路径使用标量 GM 访问，便于先覆盖隐藏 shape，后续必须增加 UB/vector 快路。
- 当前 Windows 机器没有 CANN/ccec 和 910B 设备，源码仍需在比赛环境执行 `build.sh`、安装自定义算子包并运行官方用例；未在本机声称已完成设备编译或精度验收。

## 建议验证顺序

1. 分别生成五个自定义算子工程，将本目录对应的 `op_host`、`op_kernel` 覆盖进去。
2. 使用 CANN 8.5.0、`ASCEND_COMPUTE_UNIT=ascend910b` 编译。
3. 先跑各目录 `tests/official` 公测，再补齐 README 中的边界矩阵。
4. 使用 msProf 记录 AICore 时间、核利用率、MTE/Vector 指标后再进行 V2 优化。
