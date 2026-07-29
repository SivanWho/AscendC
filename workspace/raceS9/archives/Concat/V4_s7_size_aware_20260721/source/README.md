# Concat

## 赛题语义

- 参考语义：`torch.cat(inputs, dim)`。
- 输入：动态 `Tensor[]`，ND 格式，支持 `float32`、`float16`、`int32`、`int8`。
- 属性：`dim: int`，支持正轴和负轴。
- 公开用例：把 `float16 [128,256]` 沿末轴随机切成最大长度 64 的多段，允许零长度张量，再拼回原张量。
- 官方扩展连续执行 30 次 `aclnnConcat`，`get_time.py` 取对应算子耗时中位数。

## 当前实现（V4）

V4 保留 V3 的通用多行二维 DMA，并借鉴 S7 仓库的按数据规模选核经验：Host 以约 16 KiB 输入数据/目标核估算需要的并行批次数，避免小张量为了占满 40 个 AIV 而拆出过多任务。

每个任务处理某个输入的若干 outer 行；片段能放进 UB 时采用二维 GM→UB→GM 搬运，单行片段超过 64 KiB 时自动回退到逐行分块路径。策略只依赖 dtype、shape、UB 容量和硬件核数，不包含公开 shape 或 split 序列判断。

详细结果见 [IMPLEMENTATION_V4.md](IMPLEMENTATION_V4.md)。

## 验证状态

- [x] CANN 8.5 / Ascend 910B4 编译并全新安装成功。
- [x] 8/8 扩展回归通过：四种 dtype、多种轴、非对齐、空切片、64 输入和大于 UB 的回退路径。
- [x] 官方公开用例准确性通过。
- [x] 三次独立官方运行：8.70、8.92、8.84 μs；中位数 8.84 μs。

## 文件说明

- `op_host/`：算子定义、shape/dtype 推导、tiling 和标准 `aclnnConcat` ABI 适配。
- `op_kernel/`：AscendC 核函数。
- `tests/official/`：官方测试工程原样留存。
- `tests/regression_v3.py`：扩展正确性回归（继续适用于 V4）。
- `benchmarks/`：保留版本与被拒绝实验的 `msprof` 数据。
