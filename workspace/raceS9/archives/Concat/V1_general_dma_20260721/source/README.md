# Concat

## 官方定义

- 参考语义：`torch.cat(inputs, dim)`。
- 输入：`inputs`，`tensor_list`，ND；支持 `float32`、`float16`、`int32`、`int8`。
- 属性：`dim: int`，默认 0；输出 shape 由 `dim` 决定。
- 维度范围：`N,N2 ∈ [1,10000]`，`N3,N4 ∈ [1,1000]`；各维可能不是 32 的整数倍。
- 公开样例：float16 `[128,256]`，`dim=-1`，随机切成最大长度 64 的多段；生成器允许产生长度为 0 的分片。
- PyTorch 参考：[torch.cat](https://docs.pytorch.org/docs/2.5/generated/torch.cat.html#torch.cat)。

## 正确性矩阵

- 正轴与负轴，首轴/中间轴/末轴。
- 单输入、多输入、大量小输入，含 0 长度分片。
- 拼接维长度和非拼接维长度均包含非对齐值。
- 极小与极大 tensor，四种 dtype。
- Host 侧输出 shape、输入列表元数据和 Kernel 偏移必须一致。

## 优化假设（需 Profiling 证明）

Concat 通常偏 memory-bound。可把张量视为 `outer × concat_axis × inner`，优先让连续的 `inner` 数据走大块 DMA；末轴、首轴和一般轴可使用不同的泛化 Tiling key。输入很多且分片很小时，应关注 DMA 次数、地址计算和队列开销；大块场景再评估多核负载与 Double Buffer。

不要假定所有输入非空、数量固定或 `dim=-1`，也不要按公开 split 序列硬编码。

## 完成标准

- [ ] 全 dtype、轴、输入数量、空分片与非对齐通过。
- [ ] 大块和大量小分片分别有基线。
- [ ] 核间按输出数据量负载均衡。
- [ ] 每条特殊路径都由通用属性选择并有回归测试。
