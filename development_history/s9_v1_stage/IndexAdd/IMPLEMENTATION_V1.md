# IndexAdd V1

首版使用一个 AI Core：先复制 `self`，再按 source 扁平索引映射到目标位置累加。这样在 index 重复时结果确定，不依赖不适用于全部 dtype 的原子加。V2 将按 outer/inner 分桶，安全地利用 910B 多核。
