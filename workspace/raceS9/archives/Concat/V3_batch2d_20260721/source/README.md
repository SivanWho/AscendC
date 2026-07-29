# Concat

## 赛题语义

- 参考语义：`torch.cat(inputs, dim)`。
- 输入：动态 `Tensor[]`，ND 格式，支持 `float32`、`float16`、`int32`、`int8`。
- 属性：`dim: int`，支持正轴和负轴。
- 公开用例：把 `float16 [128, 256]` 沿末轴随机切成最大长度为 64 的多段，切分中允许零长度张量，再拼回原张量。
- 官方扩展连续执行 30 次 `aclnnConcat`，`get_time.py` 取对应算子耗时的中位数。

## 当前实现（V3）

Host 把任意轴的拼接统一看成 `outer × axis × inner`。每项任务处理某个输入的多行连续片段；只要片段能放进 UB，就用一次二维 GM→UB 和一次二维 UB→GM 搬运多行。每任务行数由 UB 容量、最大输入片段和可用 Vector Core 数共同决定，不匹配或硬编码公开 shape。

单行片段超过 64 KiB 时自动回退到逐行分块搬运。输出不重叠，因而不需要原子操作。动态输入通过 `ListTensorDesc` 解包，最多支持 128 项，并跳过零长度片段。

详细设计、测试结果和性能对比见 [IMPLEMENTATION_V3.md](IMPLEMENTATION_V3.md)。

## 验证状态

- [x] CANN 8.5 / Ascend 910B4 编译并全新安装成功。
- [x] 官方公开用例正确性通过。
- [x] 公开用例 30 次中位数：10.17 μs。
- [x] 8 组扩展回归全部通过：四种 dtype、首/中/末轴、负轴、非对齐、零长度、64 输入和大于 UB 的回退路径。

## 文件说明

- `op_host/`：算子定义、shape/dtype 推导、tiling 和标准 `aclnnConcat` ABI 适配。
- `op_kernel/`：AscendC 核函数。
- `tests/official/`：官方测试工程原样留存。
- `tests/regression_v3.py`：扩展正确性回归。
- `benchmarks/`：不同版本的 `msprof` 导出数据。
