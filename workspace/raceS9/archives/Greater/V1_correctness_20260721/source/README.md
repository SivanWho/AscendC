# Greater

## 赛题接口

- 参考语义：`torch.gt(self, other)`。
- 输入：两个 dtype 相同的 ND tensor，支持 `float32`、`bfloat16`、`float16`、`int32`、`int8`。
- 输出：广播后 shape 的 `bool` tensor。
- 必须覆盖一般广播以及 `Inf`、`-Inf`、`NaN`、`+0/-0`。
- 官方公开用例：两个 `float16 [32, 64]` tensor，随机插入特殊值。
- 官方调用 ABI：`aclnnGreater(self, other, out)`。

## 当前 V1

为避免与 CANN 内置 `Greater` 注册冲突，算子内部注册为 `GreaterCustom`，并由
`op_host/aclnn_greater_compat.cpp` 导出裁判要求的 `aclnnGreater*` 接口。

Host 侧完成右对齐广播、输出 shape 和输入 stride 计算；Kernel 支持五种 dtype。
FP16/BF16 使用 IEEE 位级比较，正确处理 NaN、无穷和正负零。V1 暂用单核标量
实现，以规避多个 AI Core 对 bool 单字节 GM 写回时的覆盖竞争。

## 目录

- `op_host/`：注册、shape/type 推导、tiling 和 ACLNN 兼容入口。
- `op_kernel/greater_custom.cpp`：AscendC kernel。
- `tests/official/`：官方原始公开测试。
- `tests/diagnose_official_bridge.py`：特殊值、多 dtype、广播回归。
- `custom_opp_euleros_aarch64.run`：与当前源码对应的 Ascend 910B 编译包。

## 已验证环境与结果

- CANN 8.5.0，Ascend 910B4。
- 官方 extension 使用 GCC 10.3 编译；系统 GCC 7 不满足 PyTorch 2.5 要求。
- 官方公开用例：精度通过，`time_use = 105244000`（官方脚本原始计数，30 次调用）。
- 扩展回归：FP16/FP32/BF16/INT32/INT8、广播及特殊值全部零 mismatch。
- run 包 SHA256：`2a82bc9b1db8917f30d8e018a4c1e9be4b825786bae9bcd1b7fa508a8d1ccf9c`。

## 下一版重点

将输出按 32 字节对齐的连续区间分配给多个核，并为同 shape、标量广播和末维广播
增加 UB 向量快路径。不要恢复跨核步进式 bool 单字节写回。
