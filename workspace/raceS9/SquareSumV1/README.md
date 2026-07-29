# SquareSumV1

## 赛题接口

- 参考语义：`torch.sum(torch.square(input), dim=axis, keepdim=keep_dims)`。
- 输入：ND tensor，支持 `float16`、`bfloat16`、`float32`。
- 属性：`axis: list_int`，`keep_dims: bool`。
- 输出：dtype 与输入一致，shape 由 axis 和 keep_dims 决定。
- 官方公开用例：`float16 [123, 31]`，`axis=-1`，`keep_dims=true`。
- 官方调用 ABI：`aclnnSquareSumV1(input, axis, keep_dims, out)`。

## 当前 V2

Host 侧规范化负轴、构造归约 mask、计算输入 stride、输出元素数与归约长度。
Kernel 将平方和融合，FP16/BF16 输入均使用 FP32 中间累加，最后转换回输入 dtype。
支持单轴、多轴、负轴、非末轴、全归约以及 keep_dims 两种取值。

V2 对 FP16 单末轴短归约增加 UB 快路径：二维 `DataCopyPad` 补零、FP32 向量平方、
`WholeReduceSum` 和一次逻辑长度写回。多轴、其他 dtype 或超出 UB/指令限制的形状
继续使用 V1 泛化回退。

## 目录

- `op_host/`：注册、shape/type 推导和 tiling。
- `op_kernel/square_sum_v1.cpp`：AscendC kernel。
- `tests/official/`：官方原始公开测试。
- `tests/diagnose_official_bridge.py`：多 dtype、多轴和 keep_dims 回归。
- `custom_opp_euleros_aarch64.run`：与当前源码对应的 Ascend 910B 编译包。

## 已验证环境与结果

- CANN 8.5.0，Ascend 910B4。
- 官方 extension 使用 GCC 10.3 编译。
- 官方公开用例：精度通过；最终 ZIP 解压复验中位数 `5.641us`，V1 为 `137.565us`，加速 `24.39×`。
- 扩展回归：FP16/FP32/BF16，末轴/多轴/负轴/全归约及 keep_dims 均通过。
- run 包 SHA256：`63d823a874dea00919ea8bd1f69e9c12c94e8e299fb8494571075da010ad5510`。

## 后续重点

扩展到大于 255 行的分块/分核末轴归约，以及较长归约的分层求和和 workspace 路径。
