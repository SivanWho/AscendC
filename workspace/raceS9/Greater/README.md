# Greater

## 赛题接口

- 参考语义：`torch.gt(self, other)`。
- 输入：两个 dtype 相同的 ND tensor，支持 `float32`、`bfloat16`、`float16`、`int32`、`int8`。
- 输出：广播后 shape 的 `bool` tensor。
- 必须覆盖一般广播以及 `Inf`、`-Inf`、`NaN`、`+0/-0`。
- 官方公开用例：两个 `float16 [32, 64]` tensor，随机插入特殊值。
- 官方调用 ABI：`aclnnGreater(self, other, out)`。

## 当前 V2

为避免与 CANN 内置 `Greater` 注册冲突，算子内部注册为 `GreaterCustom`，并由
`op_host/aclnn_greater_compat.cpp` 导出裁判要求的 `aclnnGreater*` 接口。

Host 侧完成右对齐广播、输出 shape 和输入 stride 计算；Kernel 支持五种 dtype。
FP16/BF16 使用 IEEE 位级比较，正确处理 NaN、无穷和正负零。V2 为同 shape、
对齐的 FP16 输入增加 UB 向量比较和连续分核快路径，其他 dtype/广播形状使用 V1 回退。

## 目录

- `op_host/`：注册、shape/type 推导、tiling 和 ACLNN 兼容入口。
- `op_kernel/greater_custom.cpp`：AscendC kernel。
- `tests/official/`：官方原始公开测试。
- `tests/diagnose_official_bridge.py`：特殊值、多 dtype、广播回归。
- `custom_opp_euleros_aarch64.run`：与当前源码对应的 Ascend 910B 编译包。

## 已验证环境与结果

- CANN 8.5.0，Ascend 910B4。
- 官方 extension 使用 GCC 10.3 编译；系统 GCC 7 不满足 PyTorch 2.5 要求。
- 官方公开用例：精度通过；最终 ZIP 解压复验中位数 `2.981us`，V1 为 `105.244us`，加速 `35.30×`。
- 扩展回归：FP16/FP32/BF16/INT32/INT8、广播及特殊值全部零 mismatch。
- run 包 SHA256：`d3f59b6e47c20eb6001c295d7d4e84810b7ecff46e11340fe5f7e8030417a4e0`。

## 后续重点

继续为 FP32/BF16 和常见广播模式增加向量路径，并在不同数据量上搜索最优核数。
