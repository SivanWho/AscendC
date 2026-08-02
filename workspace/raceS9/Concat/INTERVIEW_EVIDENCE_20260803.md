# AscendC Concat 面试证据包（2026-08-03）

## 使用边界

下面只使用已经实现、运行或记录的数据。可以把表达讲得有冲击力，但不要声称：

- 自己测到了未公开的 MTE 指令队列深度；
- `msprof` pipeline time 等于单条硬件指令 latency；
- 合成 shape 的提升已经等价于私榜提升；
- “虚拟页表”是硬件 MMU 页表。本项目中它是区间映射类比。

## 90秒项目介绍

我做的是 AscendC Concat 搬运算子。宏观上，我把多个物理上不连续的输入抽象成一条
virtual axis，用 prefix table 表示每个输入的逻辑区间，作用类似一个轻量的“虚拟页表”。
调度器再借鉴 chunked prefill，把 `outer × virtualAxis` 切成 UB 可容纳的二维 chunk。
Kernel 不逐元素查表，而是遍历与 chunk 相交的少量输入区间，用多次二维 MTE2 拼入 UB，
再尽量用一次二维 MTE3 写回。

工程上我没有把一次合成数据的加速当结论。我做了1000例精确回归、官方 harness、
两组输入数量和分布都不同的性能矩阵，还保留所有失败 profiler。9输入不均匀场景从
39.260降到37.361 us，64输入从45.541降到41.041 us；对应任务数从1152降到192、
从1024降到224。当前版本仍标记为实验路径，因为上一版曾出现本地快、私榜反而慢的反例。

## 实测硬件基线

### D2D带宽

命令：

```bash
ascend-dmi --bw -t d2d -d 0 --fmt json
```

当前 Ascend 910B4 共25个20.97 GB样本：

- 最小：753.337 GB/s；
- 中位数：762.392 GB/s；
- 均值：761.531 GB/s；
- 最大：764.467 GB/s；
- 标准差：2.725 GB/s。

这个值是工具报告的“复制payload/time”。对于纯 Concat，理论下界使用：

```text
T_floor = output_payload_bytes / measured_d2d_payload_bandwidth
```

它是忽略 launch、tiling、地址计算和小传输效率损失的乐观下界，不是芯片标称峰值。

### Virtual2D roofline

| 场景 | 版本 | 中位数/us | 等效单核cycle | payload GB/s | 下界/us | 实际/下界 |
|---|---|---:|---:|---:|---:|---:|
| 9输入不均匀 | V9 | 39.260 | 60,789 | 287.0 | 14.780 | 2.66× |
| 9输入不均匀 | Virtual2D | 37.361 | 56,730 | 301.6 | 14.780 | 2.53× |
| 64输入不均匀 | V9 | 45.541 | 68,819 | 276.7 | 16.529 | 2.76× |
| 64输入不均匀 | Virtual2D | 41.041 | 62,252 | 307.0 | 16.529 | 2.48× |

`aiv_total_cycles / blockDim / aiv_time` 稳定得到约1650 MHz。Virtual2D让9输入的
等效cycle下降6.7%，64输入下降9.5%。这与 task duration 的4.84%和9.88%改善方向一致。

注意：aggregate traffic GB/s 把一次读和一次写都计入，不能直接除以 DMI 的payload
带宽；与 DMI 对比时使用 payload GB/s。

## 单核 MTE 命令链微实验

实验固定为：一个输入、一个 outer row、一个 AIV、一条 MTE2、`MTE2_MTE3`同步、
一条 MTE3和`MTE3_MTE2`复用同步，只改变payload 32B--64KiB。每个点30个样本。

第一次把32B放在首个测量位置时得到5.04 us，而后面的128B--2KiB只有约3.5 us。
这不是32B更慢，而是冷启动/频率爬升污染。加入独立96B warm-up 后，32B变为3.46 us。

| payload | task/us | AIV cycle | Scalar cycle | MTE2 cycle | MTE3 cycle |
|---:|---:|---:|---:|---:|---:|
| 32B | 3.46 | 4,840 | 4,216 | 766 | 187 |
| 128B | 3.44 | 4,774 | 4,169 | 773 | 192 |
| 2KiB | 3.54 | 4,938 | 4,176 | 863 | 238 |
| 8KiB | 3.57 | 4,986 | 4,175 | 883 | 265 |
| 32KiB | 3.84 | 5,420 | 4,178 | 1,143 | 474 |
| 64KiB | 4.30 | 6,201 | 4,166 | 1,615 | 764 |

对7个正式点做线性拟合：

```text
AIV cycles ≈ 4823.7 + 0.020474 × payload_bytes
R² = 0.99047
```

解释：这条单核命令链约有4824 cycle固定成本；稳态斜率对应约48.84 payload
Bytes/cycle，即约80.59 GB/s payload、161.18 GB/s聚合读写流量。不能把4824 cycle
说成“单条MTE latency”，它包含 Scalar调度、两次事件和两条DMA命令。

## 哪些 shape 真正有问题

用实测D2D下界观察：

- 64/128个极小输入只有约32 KiB payload，但耗时约13--14 us；理论搬运只需约0.043 us。
  这类 shape 比带宽下界慢300倍以上，问题是描述符、Scalar、DMA命令和同步，而不是HBM。
- 官方64 KiB小样例实测约8.6 us，带宽下界约0.086 us，同样属于固定成本主导。
- 11--12 MiB的两组 Virtual2D shape 距下界约2.5倍，开始真正进入数据搬运主导区间。
- 因此调度策略必须随 shape 分区，不能用一个“全核、固定tile、统一双缓冲”覆盖全部输入。

## 可以讲的极客式案例

### 1. 先怀疑测量系统

32B首轮结果异常后，没有解释成硬件特性，而是增加可识别的96B warm-up并重测，
把5.04 us修正为3.46 us。原始冷启动和预热后CSV都保留。

### 2. 用硬件异常反查API单位

Virtual2D首次上板报 `The write address of the MTE instruction is out of range`。
检查 CANN 8.5 接口和 dav-c220 本地实现后确认：GM stride单位为Byte，UB stride单位为
32B data block。把 `(rowStride-copyBytes)` 修正为除以32后，两个目标用例和1000例回归通过。

### 3. 同步不能靠直觉

早期单UB实现只用了同流水 barrier，出现陈旧UB数据。MTE2和MTE3是不同流水，最终使用
`MTE2_MTE3`和`MTE3_MTE2` HardEvent建立生产、消费和复用依赖。正确版当时为17.4705 us，
后续二维DMA批行再降到约10.17 us。

### 4. 核越多不一定越快

公开样例只有64 KiB。旧策略为了占满40个AIV生成45个任务；按约16 KiB/目标核限制并行度，
自然变成9个输入任务，三次独立中位数8.70、8.92、8.84 us，比10.17 us改善约13%。
强制4核和复用event ID都实测后拒绝。

### 5. 双缓冲存在甜区

9 MiB长链场景中：单缓冲51.646 us；16 KiB双缓冲52.296 us，反而更慢；32 KiB为
37.397 us；64 KiB为31.438 us。说明重叠收益必须覆盖队列和命令开销，不能只因为
“双缓冲听起来高级”就启用。

### 6. 主动保留反例，避免合成数据过拟合

一维连续输出版本在本地若干大shape上提升28%--41%，但私榜总耗时从605.436退化到
641.920 us，Case5单项增加26.464 us。由此撤销该路径，并把MTE2碎片化和Scalar遍历
纳入后续启用条件。这比只展示成功版本更能证明工程判断力。

### 7. 测试库不是“随机跑几个”

固定种子1000例覆盖fp16/fp32/int32/int8、rank 1--6、正负axis、零长度输入、
1/2/4/32B边界、非对齐、UB边界及超过UB的传输；逐元素精确比较，共检查
43,357,777字节输出。另有官方harness、9/64/128/256输入和长段tile矩阵。

### 8. 把交付格式当ABI的一部分

曾遇到 `-1 us` 不是Kernel错误，而是run包名称、算子符号或ZIP结构不符合裁判契约。
最终流程固定为Linux官方脚本打包、检查Unix可执行位和正斜杠entry、从ZIP全新解压安装、
再跑官方harness并核对SHA256。

## 高频追问的安全回答

**问：为什么叫虚拟页表？**

答：是类比，不是MMU。`axisPrefixes[i:i+1]`把逻辑virtualAxis区间映射到第i个物理输入；
翻译粒度是输入区间，不是元素，所以不会产生逐元素查表。

**问：为什么不是所有shape都启用？**

答：跨输入tile会减少MTE3，但可能增加碎片化MTE2和Scalar遍历。当前只在32B对齐、
输出行大于UB、旧任务数明显过多且二维网格并行度足够时启用，其余走V9 fallback。

**问：你测到MTE队列深度了吗？**

答：没有可靠公开依据，我不把pipeline停顿反推成精确队列深度。我测的是整条命令链的
cycle和MTE2/MTE3 active time，并用shape、tile和命令数做受控对照。

**问：理论最少时间怎么算？**

答：先用当前卡 `ascend-dmi d2d` 的762.392 GB/s中位数作为payload copy带宽，
再用 `output_bytes/BW` 得到乐观下界。小shape受固定成本支配，大shape再看实际/下界倍数、
MTE比例和核间负载。这个下界不是性能承诺，只是诊断坐标。

**问：项目最重要的失败是什么？**

答：本地合成数据提升不等于私榜提升。一维连续输出版本私榜退化6%，迫使我把
“能加速某个shape”和“可泛化的调度策略”分开，并建立保留负结果和严格fallback的流程。

## 证据索引

- `benchmarks/interview_evidence_20260803/dmi_d2d_raw.csv`
- `benchmarks/interview_evidence_20260803/mte_chain_cycle_summary.csv`
- `benchmarks/interview_evidence_20260803/virtual2d_roofline.csv`
- `benchmarks/mte_cycle_probe_910b4_cann8.5_20260803/` 原始 profiler CSV
- `benchmarks/v11_virtual2d_910b4_cann8.5_20260803/` Virtual2D 原始 profiler CSV
- `tests/concat_mte_cycle_probe.py` 可复现微实验
- `tests/official_input_1000/results/remote_20260803_virtual2d/` 1000例结果
