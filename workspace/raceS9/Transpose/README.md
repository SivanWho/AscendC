# Transpose

## 官方定义

- 参考语义：`torch.permute(inputs, dims)`。
- 输入：ND tensor，支持 `float32`、`float16`、`int32`、`int8`。
- 属性：`dims: list_int`，必须构成输入维度的合法排列。
- 输出：按 `dims` 重排后的 ND tensor。
- 维度范围：`N,N2 ∈ [1,10000]`，`N3,N4,N5 ∈ [1,1000]`；各维可能非 32 整数倍。
- 公开样例：float16 `[128,256]`，`dims=(1,0)`。
- PyTorch 参考：[torch.permute](https://docs.pytorch.org/docs/2.5/generated/torch.permute.html#torch-permute)。

## 正确性矩阵

- identity、2D 转置、相邻轴交换、首尾轴交换和高维任意排列。
- 含 size=1 的维度、非对齐维度、极小/极大 tensor。
- 所有 dtype、rank 与输出 shape。
- 输入/输出地址映射覆盖首元素、尾元素和每个维度边界。

## 优化假设（需 Profiling 证明）

identity 可直接连续拷贝；2D 或等价的连续块转置可采用 UB 分块，兼顾读写合并；一般 permutation 需要通用 stride 映射。Host 侧应折叠连续且相对顺序不变的维度，减少有效 rank 和 Kernel 除模次数。

对大 tensor 关注多核分块和 GM 读写合并，对小 tensor 关注通用索引、同步和过度分核开销。分块尺寸应由 dtype、UB 和排列特征计算，不按公开 `[128,256]` 固定。

## 完成标准

- [ ] 所有合法 permutation 和高 rank 回归通过。
- [ ] identity/2D/通用路径均有测试。
- [ ] tile 边缘和非对齐写回无遗漏、无越界。
- [ ] 对输入连续读取与输出连续写入的取舍有 profiling 证据。
