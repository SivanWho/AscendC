# IndexAdd

## 官方定义

- 参考语义：`torch.index_add(self, dim, index, source)`，默认加法系数为 1。
- `self/source`：ND tensor，支持 `float32`、`bfloat16`、`float16`、`int32`、`int8`。
- `index`：一维 `(M)`、`int32`，`M ∈ [1,8000]`。
- `source`：除 `dim` 维长度为 `M` 外，其余维度与 `self` 对应维相同。
- `dim: int`，默认 0；输出 shape 与 `self` 相同。
- 维度范围：`N,N2 ∈ [1,10000]`，`N3,N4 ∈ [1,1000]`，可能非 32 整数倍。
- 公开样例：int8 `self=[32,128]`、`index=[120]`、`source=[120,128]`、`dim=0`；index 随机生成，因而可能重复。
- PyTorch 参考：[torch.index_add](https://docs.pytorch.org/docs/2.5/generated/torch.index_add.html#torch-index-add)。

## 正确性矩阵

- `dim` 位于首/中/末维，含负轴。
- index 唯一、重复、高冲突、乱序以及边界值。
- `M=1`、较大 M，所有 dtype 和非对齐 inner size。
- int8/int32 的精确结果与溢出行为按 PyTorch Golden 核对。
- 输出必须先包含 `self`，再累加全部 source；不能遗漏重复 index 的贡献。

## 优化假设（需 Profiling 证明）

先将 `self` 拷贝到输出，再按 index 累加 source。低冲突与高冲突可能需要不同策略：直接 scatter/原子路径简单但可能争用；分块、局部聚合或排序可减少冲突但有额外开销。是否可用原子 API、支持哪些 dtype 必须以 CANN 8.5.0 文档和真实编译结果为准。

可将 `dim` 外的连续区域压成 inner block，以块为单位搬运和累加；Host 侧预计算 stride，避免 Kernel 中重复做高成本索引换算。

## 完成标准

- [ ] 重复 index 和高冲突压力测试通过。
- [ ] 全 dtype 的累加精度/整数行为通过。
- [ ] 不同 dim 与非对齐 inner block 通过。
- [ ] 低冲突/高冲突方案有可重复 profiling，而非凭感觉选择。
