# SquareSumV1

## 赛题接口

- 参考语义：`torch.sum(torch.square(input), dim=axis, keepdim=keep_dims)`。
- 输入：ND tensor，支持 `float16`、`bfloat16`、`float32`。
- 属性：`axis: list_int`，`keep_dims: bool`。
- 输出：dtype 与输入一致，shape 由 axis 和 keep_dims 决定。
- 官方公开用例：`float16 [123, 31]`，`axis=-1`，`keep_dims=true`。
- 官方调用 ABI：`aclnnSquareSumV1(input, axis, keep_dims, out)`。

## 当前 V1

Host 侧规范化负轴、构造归约 mask、计算输入 stride、输出元素数与归约长度。
Kernel 将平方和融合，FP16/BF16 输入均使用 FP32 中间累加，最后转换回输入 dtype。
支持单轴、多轴、负轴、非末轴、全归约以及 keep_dims 两种取值。

V1 暂用单核标量实现，以规避多个 AI Core 对 FP16/BF16 标量 GM 写回时的相邻元素
覆盖竞争。它是正确性基线，不是最终排名版本。

## 目录

- `op_host/`：注册、shape/type 推导和 tiling。
- `op_kernel/square_sum_v1.cpp`：AscendC kernel。
- `tests/official/`：官方原始公开测试。
- `tests/diagnose_official_bridge.py`：多 dtype、多轴和 keep_dims 回归。
- `custom_opp_euleros_aarch64.run`：与当前源码对应的 Ascend 910B 编译包。

## 已验证环境与结果

- CANN 8.5.0，Ascend 910B4。
- 官方 extension 使用 GCC 10.3 编译。
- 官方公开用例：精度通过，`time_use = 137565500`（官方脚本原始计数，30 次调用）。
- 扩展回归：FP16/FP32/BF16，末轴/多轴/负轴/全归约及 keep_dims 均通过。
- run 包 SHA256：`9f7782f065ea8494d1299c28ae849991aeacb2781e2f522c60c6b32dd1e16ef6`。

## 下一版重点

按输出连续区间对齐分核；为连续末轴归约增加 UB 搬运、向量平方和分层归约；对非连续
轴保留泛化路径。长归约再评估分核部分和、workspace 与二次归约。
