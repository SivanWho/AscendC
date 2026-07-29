# Greater V2：对齐分核与向量比较

V1 的逐元素 GM 访问使 profiler 标量占比约 97.5%，官方中位数为 105.244us。

V2 快路径由一般属性选择：FP16、两输入同 shape、总元素数为 256 的倍数。Host 将输出
划分为 32 字节对齐的连续区间，每核独占一个区间。Kernel 通过 DMA 将两输入搬入 UB，
用 `Compare(CMPMODE::GT)` 生成掩码，通过 `Select` 展开成 FP16 0/1，再 Cast 为 bool/int8
并整块写回。广播、非对齐和其他 dtype 保留 V1 正确性回退。

公开 `[32,64]` 用例使用 8 个 AIV：

- V1 median：105.244us
- 最终 ZIP 复验 median：2.981us
- 最终 ZIP profiler min/mean/max：2.760/3.391/6.000us
- 加速：35.30×

官方精度、多次随机特殊值、广播以及 FP16/FP32/BF16/INT32/INT8 回归全部通过。
