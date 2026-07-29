# Power算子

## 1.功能描述

逐元素计算：

```
y = exp(power * log(x * scale + shift))
```

支持的输入数据类型：`fp16` / `bf16` / `fp32`（fp16、bf16在kernel内cast为fp32计算后回写）。

## 2.算子属性

| 属性 | 类型 | 必选 | 默认值 | 含义 |
|---|---|---|---|---|
| `power` | float | 是 | -      | 幂指数 |
| `scale` | float | 否 | `1.0`  | 输入线性缩放因子 |
| `shift` | float | 否 | `0.0`  | 输入线性平移量 |

## 3.实现要点

- 使用 **Elementwise模板**（`ElewiseBaseTiling` + `ElementwiseSchWithScalar` + `DAGSch`）。
- 在tiling层完成所有`scale / power / shift`标量计算与分支决策，最终通过`culType`枚举值
  + dtype编码出`tilingKey`，在kernel中实例化对应的DAG，避免kernel内运行时分支：

| culType | 计算 |
|---|---|
| `ALL_ZEROS` | `y = 0` |
| `BROADCAST_SCALAR` | `y = bcastVal` (host端预计算`pow(shift, power)`、异常值`+inf`/`NaN`，或`power=0`时的`1.0`) |
| `LINEAR` (`power=1`) | `y = x*scale + shift` （fused MulAdd语义：`Duplicate(shift) + Axpy(scale)`） |
| `SQUARE` (`power=2`) | `base = x*scale + shift; y = base * base` |
| `CUBE` (`power=3`) | `base = x*scale + shift; y = base^3` |
| `GENERIC_POW_POS` | 通用，`power>0`，零底数映射为0 |
| `GENERIC_POW_NEG` | 通用，`power<0`，零底数映射为`+inf` |

通用分支在kernel内通过`Compare + Select`合并三种取值（正/负/零底数）：

```
absBase   = |base|
logAbs    = log(absBase)
rawExp    = exp(power * logAbs)
posVal    = rawExp                  // base > 0
negVal    = rawExp * negScalar      // negScalar = ±1 (整数power)或NaN (非整数power)
zeroVal   = 0 / +inf                // 由tilingKey区分
tmp       = base > 0 ? posVal : negVal
y         = base == 0 ? zeroVal : tmp
```

`isclose`判等方法参考`math/is_close`：`|a-b| <= atol + rtol*|b|`，`atol=1e-8`，`rtol=1e-5`。

## 4.不支持范围

- 不支持广播；输入与输出shape一致。
- 输入dtype仅支持`fp16/bf16/fp32`；其它类型在tiling层会直接报错。
- 本算子目前仅生成ascend950平台的二进制，不包含aclnn接口模块。

## 5.目录结构

```
power/
├── CMakeLists.txt
├── README.md
├── docs/DESIGN.md
├── op_graph/power_proto.h
├── op_host/
│   ├── power_def.cpp
│   ├── power_infershape.cpp
│   ├── arch35/
│   │   ├── power_tiling_arch35.h
│   │   └── power_tiling_arch35.cpp
│   └── config/ascend950/{power_binary.json, power_simplified_key.ini}
└── op_kernel/
    ├── power_apt.cpp
    └── arch35/{power_struct.h, power_dag.h}
```
