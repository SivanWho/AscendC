# Greater V1

Host 侧将两个输入右对齐并生成广播 stride（被广播的维 stride 为 0）；Kernel 按输出元素跨核分配。浮点比较直接使用 `>`，因此 NaN 比较为 false，正负无穷按 IEEE 语义处理。V2 应为同 shape 和连续广播增加 UB 向量路径。
