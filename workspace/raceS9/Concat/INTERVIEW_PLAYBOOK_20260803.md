# AscendC Concat 面试作战手册

> 使用原则：这里的数字都有本地 CSV、Profiler 或比赛记录支撑。只有当你能解释实验方法、
> 变量控制和结论边界时，才用第一人称。故事可以讲得有张力，但不要把类比讲成硬件事实。

## 一句话定位

我不是只写了一个 Concat Kernel，而是围绕它搭了一套小型的性能工程闭环：先建立硬件
搬运下界和命令链成本模型，再用 shape 矩阵定位控制开销、负载不均和带宽利用率问题，
最后用有严格 fallback 的 Virtual2D chunk 调度解决长尾输入，并保留所有失败实验。

## 两分钟主叙事

Concat 没有数学计算，本质是把多个物理上离散的输入重排到连续输出。最初的实现以
input 为任务单位：每个任务处理某个输入的一组 outer 行。它简单、泛化，但输入大小高度
不均时，任务粒度也高度不均，而且每个输入都要独立写回，MTE3 命令数量较多。

我后来把所有输入沿 concat axis 拼成一条逻辑 `virtualAxis`，用 `axisPrefixes` 保存每个
物理输入在逻辑空间中的区间。它像一个很轻量的虚拟页表：逻辑地址先落到一个输入区间，
再换算成该输入内的物理偏移。这个说法是调度类比，不是 MMU，也不会逐元素查表。

调度器借鉴 chunked processing，把 `outer × virtualAxis` 切成 UB 能容纳的二维 tile。
每个 task 只遍历与自己的 virtual 区间相交的少量输入，用若干次二维 MTE2 把碎片拼到
同一块 UB，再尽量用一次二维 MTE3 写回连续输出。不同 task 的输出矩形完全不重叠，
不需要原子操作。

我没有用两个合成 shape 的提升宣称算法普遍更快，而是保留原来的 V9 路径作为 fallback。
Virtual2D 只在输入数、32B 对齐、输出行宽、任务膨胀程度和二维并行度都满足一般条件时
启用。9 输入高度不均匀场景从 39.260 降到 37.361 us，64 输入从 45.541 降到
41.041 us；与此同时任务数分别从 1152 降到 192、从 1024 降到 224。

## 宏观设计怎么讲

### 虚拟页表不是噱头

映射关系是：

```text
axisPrefixes[i] <= virtualOffset < axisPrefixes[i + 1]
    => 第 i 个输入
physicalOffset = virtualOffset - axisPrefixes[i]
```

真正的价值不是“查地址”，而是把调度单位从物理输入改成规则的二维输出区间。旧任务的
大小由输入决定；新任务的大小主要由 UB tile 决定，所以面对最大/最小相差 2048 倍的
输入仍能得到比较均匀的工作量。

### Chunked 的含义

不是把 tensor 随意切碎，而是把输出看成 `outer × rowBytes` 的二维平面：

```text
outerTileId   = taskId / virtualTileCount
virtualTileId = taskId % virtualTileCount
```

一个 task 对应一个不重叠矩形：

```text
[firstOuter, firstOuter + rowCount)
× [virtualBeginBytes, virtualEndBytes)
```

这样能同时控制 UB 占用、单 task 工作量和可并行 task 数量。

## 最有说服力的真实数据

### 1. 先测当前环境，而不是背芯片宣传参数

在 Ascend 910B4 上使用 `ascend-dmi --bw -t d2d` 连续测 25 次，每次复制
20.97 GB：

| 指标 | GB/s |
|---|---:|
| min | 753.337 |
| median | 762.392 |
| mean | 761.531 |
| max | 764.467 |
| stddev | 2.725 |

我把它称为“当前环境实测 D2D payload 带宽”，不称为 HBM 标称峰值。Concat 的乐观
搬运下界使用：

```text
T_floor = output_payload_bytes / 762.392 GB/s
```

D2D 工具报告的是一次 copy 的 payload/time，硬件内部已经包含读和写，所以不能再把
payload 乘二后除以同一个 D2D 数字。

### 2. Shape 病理分类

| shape | payload | 实测 | 搬运下界 | 实测/下界 | 诊断 |
|---|---:|---:|---:|---:|---|
| 官方9输入小例 | 64 KiB | 8.6095 us | 0.086 us | 约100× | launch、tiling、Scalar与命令固定成本 |
| Virtual2D 9输入 | 11,268,096 B | 37.361 us | 14.780 us | 2.53× | 搬运与调度混合受限 |
| Virtual2D 64输入 | 12,601,344 B | 41.041 us | 16.529 us | 2.48× | 描述符遍历和负载不均仍有空间 |

由此得到的不是“所有 shape 都双缓冲”，而是三段式策略：极小 shape 优先减少任务和命令；
中等 shape 优先二维 DMA 与负载均衡；长 tile 链才考虑 MTE2/MTE3 重叠。

### 3. 单核命令链 cycle 微实验

受控实验固定一个 AIV、一条 MTE2、一次 `MTE2_MTE3` 同步、一条 MTE3 和一次 UB
复用同步，只改变 payload。每个点 30 个样本：

| payload | task/us | AIV cycles | Scalar | MTE2 | MTE3 |
|---:|---:|---:|---:|---:|---:|
| 32 B | 3.46 | 4,840 | 4,216 | 766 | 187 |
| 128 B | 3.44 | 4,774 | 4,169 | 773 | 192 |
| 2 KiB | 3.54 | 4,938 | 4,176 | 863 | 238 |
| 8 KiB | 3.57 | 4,986 | 4,175 | 883 | 265 |
| 32 KiB | 3.84 | 5,420 | 4,178 | 1,143 | 474 |
| 64 KiB | 4.30 | 6,201 | 4,166 | 1,615 | 764 |

对七个点拟合：

```text
AIV cycles = 4823.7 + 0.020474 × payload_bytes
R² = 0.99047
```

斜率对应约 48.84 payload B/cycle。在 profiler 推导的约 1.65 GHz 下，相当于单核
约 80.59 GB/s payload。4824 cycle 是整条命令链固定成本，不是某一条 MTE 指令延迟。

### 4. 真实的测量气泡

第一次把 32B 放在整个程序的首个测量点，得到 5.04 us；后续 128B--2KiB 却只有
约 3.5 us。与其编造“小包反常”，我增加了一个不会和正式点混淆的 96B warm-up，
32B 随即变成 3.46 us。冷启动 CSV 和预热 CSV 都保留。

这能体现的工程习惯是：发现反直觉数据时，先怀疑测量协议、频率状态和缓存状态，
再讨论硬件机制。

## 六个“极客自驱”案例

### 案例一：用异常反查文档单位

Virtual2D 第一版上板报：

```text
The write address of the MTE instruction is out of range
```

代码中把 GM→UB 的 `dstStride` 当成字节。查 CANN 8.5 文档和本机 dav-c220 接口后确认：

- `blockLen`：字节；
- GM 侧 stride：字节；
- UB 侧 stride：32B data block。

把 `(tileRowBytes-copyBytes)` 改成除以 32 后，目标 shape 和 1000 例回归通过。这个案例
不要说成“手册错了”；准确说法是“同一个结构体字段在 GM 和 UB 侧单位不同，接口很容易
被误读，我用运行时越界和文档交叉验证定位了问题”。

### 案例二：同步错误不是随机精度误差

最早复用单个 UB buffer 时只使用了同流水 barrier，出现陈旧数据。MTE2 与 MTE3 属于
不同流水，最终使用 `MTE2_MTE3` 建立生产到消费依赖，再用 `MTE3_MTE2` 保证写回完成后
才能复用 UB。同步正确版为 17.4705 us；二维 DMA 批行后约 10.17 us。

### 案例三：核越多可能越慢

官方小例只有 64 KiB payload。旧策略为占满 40 个 AIV 生成 45 个任务；按约
16 KiB/目标核限制并行度后，自然变成9个输入任务。三次独立中位数为 8.70、8.92、
8.84 us，相对 10.17 us 改善约13%。强制4核和复用 event ID 都实测后被拒绝。

### 案例四：双缓冲不是默认答案

约9 MiB最小 GM 流量的长链场景：

| 路径 | 时间/us |
|---|---:|
| 单缓冲 | 51.646 |
| 双缓冲16 KiB | 52.296 |
| 双缓冲32 KiB | 37.397 |
| 双缓冲64 KiB | 31.438 |

16 KiB 因 queue 和 DMA 命令成本反而变慢；只有 tile 足够大、每核有连续 tile 链时，
MTE2/MTE3 重叠才能覆盖额外成本。

### 案例五：主动保存失败版本

一维连续输出实验在部分合成大 shape 上提升28%--41%，但私榜总耗时从605.436退化到
641.920 us，Case5 单项增加26.464 us。于是没有继续为好看的本地曲线辩护，而是撤销
该路径，把碎片化 MTE2、Scalar 遍历和 fallback 条件纳入下一版设计。

### 案例六：把打包也当 ABI

曾经的 `-1 us` 并不全是 Kernel 问题，还包括 run 包名称、算子符号、ZIP 目录和 Unix
可执行位不符合裁判契约。最终自动检查：正斜杠 entry、显式目录、Unix origin、run
模式0750、源码/run 哈希一致，并从 ZIP 全新解压安装后再跑官方 harness。

## 测试体系怎么讲

固定种子1000例覆盖：

- fp16、fp32、int32、int8；
- rank 1--6；
- 第一、中间、最后和负 axis；
- 零长度输入；
- 1/2/4/32B边界与非对齐长度；
- UB临界值、超过UB的 fallback；
- 9/64/128/256输入；
- 高度不均匀输入分布。

1000/1000通过，与 CPU `torch.cat` 逐元素比较，共检查 43,357,777 字节输出。性能数据
和精度数据分开保存；性能用中位数，不用单次最好值。

## 高频追问与回答

### 你真的测到了 MTE 指令 latency 吗？

没有把 profiler 的 pipeline counter 等同于单条硬件指令 latency。我测的是受控命令链：
固定两条 DMA 和两次同步，只改变 payload，再用斜率和截距区分固定成本与数据相关成本。

### 你知道 MTE 指令队列深度吗？

没有可靠公开证据，所以不会从 Scalar 停顿反推出“队列一定是8”。我只能说队列饱和、
同步和描述符数量可能造成 back-pressure，并用 task 数、tile 大小和 pipeline counter 做
受控对照。

### 762 GB/s 是芯片峰值吗？

不是。它是这台机器上 DMI D2D 的实测 payload copy 带宽，用作同环境下的乐观基线。
我不会把它等同于厂商标称 HBM 带宽。

### 为什么 Virtual2D 只提升不到10%？

因为它解决的是高度不均匀输入下的任务膨胀和 MTE3 碎片，不会消除 launch、tiling、
prefix 遍历和所有 MTE2。64输入任务数下降78.1%，但时间只下降9.88%，恰好说明任务数
不是唯一瓶颈，Scalar ratio 甚至从0.236升到0.308。

### 为什么不二分查 prefix table？

输入上限小，而且一个 tile 往往与连续少量输入相交。单调 cursor 能利用访问顺序；
二分会增加每个 tile 的分支和随机 metadata 访问。对超过内联数组的输入还要考虑 tiling
buffer 容量，不能只看算法复杂度。

### 如何避免针对公开 Case 特化？

启用条件只依赖 dtype、对齐、UB容量、输入数、任务膨胀和二维并行度，不匹配某个公开
shape 的精确维度或数值。未证明受益的 shape 一律走稳定 fallback。

### 如果再做一周，你会做什么？

第一，补齐16/32/64 KiB Virtual2D tile 的全矩阵；第二，把每个 tile 的输入交集数纳入
成本模型；第三，自动生成 `actual/floor` 热力图，用它区分固定成本区、调度区和带宽区；
第四，在不扩大正确性风险的前提下，对跨多个小输入的 tile 尝试批量描述符或更紧凑的
metadata布局。

## 绝对不要说的四句话

1. “我测出910B的MTE队列深度就是8。”
2. “4824 cycle就是一条DataCopy的延迟。”
3. “虚拟页表是我在NPU上实现了MMU。”
4. “合成shape快10%，所以私榜一定快10%。”

把它们分别替换为：受控命令链、固定成本拟合、逻辑区间映射、带 fallback 的实验路径。

## 证据路径

- `INTERVIEW_EVIDENCE_20260803.md`
- `benchmarks/interview_evidence_20260803/dmi_d2d_raw.csv`
- `benchmarks/interview_evidence_20260803/mte_chain_cycle_summary.csv`
- `benchmarks/interview_evidence_20260803/virtual2d_roofline.csv`
- `benchmarks/mte_cycle_probe_910b4_cann8.5_20260803/`
- `benchmarks/v11_virtual2d_910b4_cann8.5_20260803/`
- `tests/concat_mte_cycle_probe.py`
- `tests/official_input_1000/results/remote_20260803_virtual2d/`
