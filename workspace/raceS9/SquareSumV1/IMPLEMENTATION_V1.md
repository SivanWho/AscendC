# SquareSumV1 V1 实现记录

状态：已在 Ascend 910B4 上完成全量编译、安装和官方公开用例验证。

关键决策：

1. Host 将负 axis 规范化为非负轴并生成归约 mask，同时计算输入 shape/stride。
2. Kernel 直接执行平方与求和，不生成平方中间 tensor；FP16/BF16 使用 FP32 累加。
3. keep_dims 只影响输出 shape，不改变扁平输出元素的顺序。
4. V1 强制单核，避免跨核步进式 half/BF16 标量写回覆盖相邻输出。
5. 官方 PyTorch C++ extension 使用 GCC 10.3 编译，避免 GCC 7 版本不足和 Bisheng ABI 不兼容。

当前版本用于建立可提交、可运行、具备一般 shape/axis 泛化能力的正确性基线；V2 再做
对齐连续分核与 UB 向量归约。
