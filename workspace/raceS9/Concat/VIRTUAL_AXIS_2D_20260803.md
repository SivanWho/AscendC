# Concat 虚拟 axis 二维分块实验（2026-08-03）

## 状态

这是基于 V9 稳定路径增加的实验性 tiling key 3，尚未直接作为比赛提交包。
原有 input-centric 路径、长段并行路径和长链双缓冲路径均保留为 fallback。

环境：Ascend 910B4（dav-2201）、CANN 8.5、40 个可用 AIV。

## 已确认的 API 事实

CANN 8.5 的 `DataCopyPad` 使用 `DataCopyExtParams` 时：

- `blockLen` 的单位是字节；
- GM 侧 `srcStride`/`dstStride` 的单位是字节；
- UB（VECIN/VECOUT）侧 stride 的单位是 32B data block；
- `blockCount` 上限为 4095；
- `blockLen` 在 dav-c220 实现中为21位，上限 2097151 字节。

官方说明：<https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0265.html>

首次实现曾把 GM→UB 的 `dstStride` 当成字节，导致其放大32倍并触发
`The write address of the MTE instruction is out of range`。修正为
`(tileRowBytes-copyBytes)/32` 后，两组用例和综合回归均通过。

## Host tiling

新增字段：

- `virtualTileBytes`：virtualAxis 每块的逻辑字节宽度，本实验为32 KiB；
- `outerPerVirtualTile`：一个 tile 包含的 outer 行数；
- `virtualTileCount`：每行沿 virtualAxis 的 tile 数；
- `outerTileCount`：outer 方向的 tile 数。

任务编号对应规则网格：

```text
outerTileId   = taskId / virtualTileCount
virtualTileId = taskId % virtualTileCount
```

启用条件均为形状的一般属性：输入数为9--128、每个物理输入段为32B对齐、
输出行大于64 KiB、单段不进入长段路径、原路径任务数至少为核数的4倍，
且二维网格能提供足够并行度。不满足时继续使用 V9 fallback。

## Kernel 映射

每个 tile 对应：

```text
[firstOuter, firstOuter + rowCount)
×
[virtualBeginBytes, virtualEndBytes)
```

Kernel 遍历 `axisPrefixes/axisSizes`，只处理与 virtual 区间相交的输入。
每个交集产生一次二维 MTE2：

```text
srcStride(GM, bytes)  = inputRowBytes - copyBytes
dstStride(UB, blocks) = (tileRowBytes - copyBytes) / 32
```

所有 MTE2 完成后，以一次二维 MTE3 写出该 tile 的全部 outer 行。
不同任务拥有完全不重叠的输出矩形，不使用原子操作或动态抢任务。

## 两组目标用例

### 9 输入

- shape：9个 `[128, axis_i]` FP16；
- `axis_i = [16,32,64,128,256,512,2048,8192,32768]`；
- 最大/最小长度相差2048倍；
- output shape：`[128,44016]`；
- 最小 GM 读写流量：22,536,192 bytes；
- 旧任务数1152，新二维任务数192。

### 64 输入

- shape：64个 `[64, axis_i]` FP16；
- `axis_i = 16 * 2^(i mod 10)`；
- 最大/最小长度相差512倍；
- output shape：`[64,98448]`；
- 最小 GM 读写流量：25,202,688 bytes；
- 旧任务数1024，新二维任务数224。

## 性能结果

测试脚本外层运行30轮，官方扩展每轮内部调用30次 Concat，因此每个版本、每个
shape 含900个 AICore `Task Duration` 样本。原始 CSV 与汇总表保存在同名 benchmark 目录。

| 用例 | V9中位数/us | Virtual2D中位数/us | 改善 |
|---|---:|---:|---:|
| 9输入高度不均匀 | 39.260 | 37.361 | 4.84% |
| 64输入高度不均匀 | 45.541 | 41.041 | 9.88% |

64输入场景中 Scalar ratio 从0.236增至0.308，但任务数和 MTE3 比例下降，
总时间仍改善。这说明 prefix 遍历有成本，但此处负载均衡与写回合并的收益更大。

## 正确性与回退

- 两个目标用例均与 `torch.cat` 逐元素完全一致；
- 官方9输入非对齐用例走旧路径并通过，官方耗时8.6095 us；
- 固定种子综合回归1000/1000通过，检查43,357,777字节输出；
- 不对齐、输出行不超过UB、超过128输入以及长段路径不会启用本实验路径。

## 尚未证明的事项

两组合成用例的改善不能证明私有判题总分一定改善。上一版一维连续输出路径曾在
合成数据上变快、私榜却退化，因此本路径在打包提交前还需要增加 tile 16/32/64 KiB
对比以及更多不同 outer、inner、输入分布的矩阵测试。
