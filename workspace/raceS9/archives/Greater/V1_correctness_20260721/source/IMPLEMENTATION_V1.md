# Greater V1 实现记录

状态：已在 Ascend 910B4 上完成全量编译、安装和官方公开用例验证。

关键决策：

1. 内部名使用 `GreaterCustom`，用 ACLNN 适配层保留官方 `aclnnGreater` ABI，避免和 CANN 内置同名算子冲突。
2. Host 预计算广播 shape/stride，广播维 stride 设为 0。
3. FP16/BF16 采用位级 IEEE 比较；任一输入为 NaN 时输出 false，`+0` 与 `-0` 相等。
4. V1 强制单核。此前跨核步进式 `SetValue<int8_t>` 在 2048 元素随机测试中约产生 500 个错误，根因是相邻字节的宽粒度 GM 写回互相覆盖。
5. 官方 C++ extension 必须用 `/home/ma-user/gcc/bin/g++`（GCC 10.3）编译；Bisheng 编译出的 PyTorch bridge 会在 tensor layout 处发生 ABI 异常。

当前版本以泛化正确性为目标，性能优化留给 V2 的对齐连续分核和 UB 向量化。
