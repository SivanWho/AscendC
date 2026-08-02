# Concat 双缓冲调优记录（2026-07-30）

## 判题反馈与方向修正

上一轮“多输入描述符游标”版本在真实判题中继续倒退约 2 μs。该结果否决了
“Case5 主要由超过 128 个输入的元数据扫描造成”的假设。失败包已归档为：

```text
submissions/archive/20260730_many_input_v3_judge_regressed_2us/Concat.zip
```

本轮从 607.8225 μs 的稳定版重新开始，不继承该版本的按输入数限核策略。

## 实现

Concat 的数据路径没有 Vector 计算：

```text
GM --MTE2--> UB --MTE3--> GM
```

稳定版为单个 UB buffer，并在每个 tile 后执行 MTE2→MTE3、MTE3→MTE2
事件等待。长 tile 链因此被强制串行。

最终候选使用：

```cpp
TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQue_;
pipe_.InitBuffer(copyQue_, 2, 64 * 1024);
```

每个 tile 执行：

```text
Alloc → DataCopyPad(GM→UB) → EnQue
      → DeQue → DataCopyPad(UB→GM) → Free
```

队列负责 buffer 生命周期和跨流水同步。两块 64KB UB 允许下一 tile 的
MTE2 与上一 tile 的 MTE3 重叠，不使用固定周期等待。

该写法与华为 CANN 8.5 的
[纯搬运算子复用 VECIN/VECOUT 最佳实践](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0027.html)
以及
[TQueBind EnQue 接口说明](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0140.html)
一致。

## Tile 搜索

同一张 Ascend 910B4、CANN 8.5，连续 dim0 大段用例：

| 版本 | Tile | AI Core 中位时间（μs） | 相对稳定版 |
|---|---:|---:|---:|
| 稳定版单 buffer | 64KB | 51.646 | 基线 |
| 双 buffer | 16KB | 52.296 | -1.3% |
| 双 buffer | 32KB | 37.397 | +27.6% |
| 双 buffer | 64KB | 31.438 | +39.1% |

16KB 会产生过多 DMA/队列描述符，启动开销抵消流水收益；64KB 最优。

## 甜区

每个输入的连续段逐渐增大时：

| 单输入连续段 | 稳定版（μs） | 双缓冲（μs） | 提升 |
|---:|---:|---:|---:|
| 128KB | 6.850 | 6.570 | 4.1% |
| 256KB | 9.240 | 7.909 | 14.4% |
| 512KB | 14.339 | 10.659 | 25.7% |
| 数 MiB | 51.646 | 31.438 | 39.1% |

这说明收益由 tile 链长度决定，不依赖某个公开 shape。

## dtype 与非对齐验证

约 9 MiB 最小 GM 流量：

| 用例 | 稳定版（μs） | 双缓冲（μs） | 提升 |
|---|---:|---:|---:|
| fp16，对齐 | 51.646 | 31.438 | 39.1% |
| fp16，非对齐 | 54.106 | 32.967 | 39.1% |
| fp32 | 51.766 | 32.128 | 37.9% |
| int8 | 54.335 | 32.907 | 39.4% |

最终精确构建重新测得：

```text
official geometry: 8.560 μs
dim0 large:         31.587 μs
```

稳定版大段的最小 GM 有效吞吐为约 182.73 GB/s，最终版为约
298.77 GB/s。这里只称“基于最小读写字节数的有效吞吐”，不等同于 HBM
峰值利用率。

Profiler 中稳定版大段的 MTE2、MTE3、scalar 比例总和约为 1；双缓冲版
明显大于 1，说明不同流水发生了时间重叠。

## 常规路径

公开 geometry 三次稳定版中位数：

```text
8.680 / 8.670 / 8.619 μs
```

双缓冲候选三次：

```text
8.709 / 8.759 / 8.619 μs
```

差异约 0.04 μs。16/32/64KB 边界、64/128 个小输入等用例均处于约
±2% 波动范围。双缓冲的主要价值是少核、长 tile 链，不是小块路径。

## 新增健壮性修复

host 侧新增：

- 所有输入 dtype 必须相同；
- 所有非 concat 轴维度必须相同；
- 不支持 dtype 返回失败，不再默认映射到 fp16；
- rank、负维度、负 shape、乘法和求和溢出检查；
- `rowTaskCount` 转为 uint32_t 前检查；
- InferShape 同步检查 rank、dim 和非 concat 轴。

这些检查发生在 host/tiling 阶段，不增加 AI Core 执行时间。

## 测试

- 23 个确定性合法用例全部精确通过；
- 2 个非法输入（混合 dtype、非 concat 轴不一致）均正确拒绝；
- 50 个固定 seed 的随机压力用例全部精确通过；
- 覆盖 fp16/fp32/int32/int8、rank 1–6、空 Tensor、负 dim、
  31/32/33B 边界、4094/4095/4096 行、16/32/64KB 边界、超 UB、
  大连续段、非对齐、多输入；
- 使用主办方原始 `zip_op.sh`；
- ZIP 全新解压、安装后，快速矩阵和非对齐大段均通过。

## 提交包

```text
Concat.zip SHA-256:
ad1be5133d7e2b816972d25d1504399531dd31322f9b48e8cd624e90cd779b78

构建 .run SHA-256:
fe1aedf65ba790629375bfe074c8b1c819d17dcc955526289cab90d04ce773e5
```

比赛 ZIP 内的同一 run 构建产物按要求命名为：

```text
custom_opp_ubuntu_aarch64.run
```

