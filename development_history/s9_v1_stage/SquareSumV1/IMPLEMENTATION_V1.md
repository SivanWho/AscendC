# SquareSumV1 V1

归约轴在 Host 规范化为 mask，Kernel 将独立输出元素分配给不同 AI Core，并以 FP32 累加平方值后转换回输入 dtype。支持多轴、负轴与 `keep_dims`（后者只影响输出 shape，不影响扁平结果顺序）。
