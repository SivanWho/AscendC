# Concat

## 赛题语义

- 参考语义：`torch.cat(inputs, dim)`。
- 输入：动态 `Tensor[]`，ND 格式，支持 `float32`、`float16`、`int32`、`int8`。
- 属性：`dim: int`，支持正轴和负轴。
- 公开用例：把 `float16 [128,256]` 沿末轴随机切成最大长度 64 的多段，允许零长度张量，再拼回原张量。
- 官方扩展连续执行 30 次 `aclnnConcat`，`get_time.py` 取对应算子耗时中位数。

## 当前工作树（V11 实验）

V11 从 V9 稳定实现出发，加入 `outer × virtualAxis` 二维 tile 实验路径。每个任务把多个物理输入在若干 outer 行上的交集拼入行式 UB tile，再用一次二维 MTE3 写回。

只有输入段32B对齐、输出行大于UB且旧路径明显过度切任务时才启用；其余情况走 V9 fallback。该版本目前是实验代码，还没有替换榜单稳定版本。

详细算法、API单位、失败修复和性能数据见 [VIRTUAL_AXIS_2D_20260803.md](VIRTUAL_AXIS_2D_20260803.md)。

## 验证状态

- [x] CANN 8.5 / Ascend 910B4 编译并全新安装成功。
- [x] 固定种子 1000/1000 泛化回归通过，覆盖四种 dtype、rank 1--6、正负轴、空切片、非对齐和大于 UB 的路径。
- [x] 官方公开非对齐用例走 fallback，准确性和性能通过，实测 8.6095 μs。
- [x] 9 输入高度不均匀用例中位数从 39.260 降至 37.361 μs，改善 4.84%。
- [x] 64 输入高度不均匀用例中位数从 45.541 降至 41.041 μs，改善 9.88%。
- [ ] 尚未生成比赛提交包；需要扩大 shape 矩阵后才能决定是否提交。

## 文件说明

- `op_host/`：算子定义、shape/dtype 推导、tiling 和标准 `aclnnConcat` ABI 适配。
- `op_kernel/`：AscendC 核函数。
- `tests/official/`：官方测试工程原样留存。
- `tests/concat_profile_matrix.py`：包含本轮9输入和64输入对照用例。
- `benchmarks/`：保留版本与被拒绝实验的 `msprof` 数据。
- `submissions/20260801_output_contiguous/Concat.zip`：历史 V10 包，不代表当前 V11 实验代码。
