# SquareSumV1 V2：补齐行布局与向量归约

V1 虽然融合平方和求和，但仍逐元素访问 GM 并用标量累加，profiler 标量占比约 99.5%，
官方中位数为 137.565us。

V2 快路径按一般属性选择：FP16、只归约连续末轴、补齐宽度不超过 64、行数不超过 255，
且所有临时数据可放入 UB。通过一次二维 `DataCopyPad` 把各行补齐到 32 字节边界，
Cast 到 FP32 后向量平方，再用带补齐行 stride 的 `WholeReduceSum<float, true>` 求每行和，
最后 Cast 回 FP16，并只写回逻辑输出字节。其余情况保留 V1 回退。

公开 `[123,31]`、axis=-1 用例：

- V1 median：137.565us
- 最终 ZIP 复验 median：5.641us
- 最终 ZIP profiler min/mean/max：5.100/5.652/7.601us
- 加速：24.39×

官方精度以及 `7x17`、`5x64`、`255x1`、标量输出、多轴、FP32、BF16 回归全部通过。
