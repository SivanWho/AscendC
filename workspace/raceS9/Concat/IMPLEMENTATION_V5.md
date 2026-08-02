# Concat 多输入调优记录（2026-07-30）

## 结论

本轮不再优化“单个 segment 超过 UB”的低命中路径，而是处理动态输入数超过
128 时的元数据访问。最终版本在 CANN 8.5、Ascend 910B4 上保持 9 输入公开
geometry 的性能不变，并把 256 输入合成用例的 AI Core 中位时间从
56.90 μs 降到 34.23 μs，提升 39.8%。

这只能证明多输入路径得到改善。比赛 Case5 的 shape 和输入数不可见，因此不能
把 39.8% 直接等同于榜单总分的预期提升。

## 为什么上一版失败

判题结果：

| Case | 上一稳定版（μs） | segment-split V2（μs） |
|---|---:|---:|
| 1 | 8.53 | 8.5105 |
| 2 | 34.281 | 32.191 |
| 3 | 18.0405 | 18.2605 |
| 4 | 107.522 | 110.2525 |
| 5 | 439.449 | 442.569 |
| 总计 | 607.8225 | 611.7835 |

V2 只加速“单个连续段大于 UB”的路径。Case5 没有命中该路径，反而承担了额外
tiling 和分支成本，因此总时间退化 3.961 μs。该版本已保存在
`submissions/archive/20260729_concat_segment_v2_failed/`，不再作为提交候选。

## 根因

原 kernel 在 `inputCount > 128` 时，对每个输入任务都从描述符 0 扫描到
`inputId`，计算 axis 前缀：

```text
task 0 扫 1 次，task 1 扫 2 次，...，task N-1 扫 N 次
```

单个 row task 的描述符访问量约为 `N(N+1)/2`，即 O(N²)。而且 blockDim
固定取最多 40，每个核都要重复扫描列表。Profiler 中 256 输入旧版的
`aiv_scalar_ratio` 为 94.4%，符合元数据标量访问主导的判断。

## 实现

1. 保留 `inputCount <= 128` 的内联 `axisSizes/axisPrefixes` 快路径不变。
2. 超过 128 个输入时，每个核为当前 row task 维护：
   - `cursorNextInput`
   - `cursorPrefix`
   - `cursorRowTask`
3. 同一个核收到递增的 input id 时从上次位置继续扫描，而不是回到 0。
   元数据复杂度由约 O(N²) 降为 O(blockDim × N)。
4. 对 `inputCount > 128`，使用已有的 `BYTES_PER_CORE=16 KiB` 估算
   `targetCores`，并以它限制 blockDim，避免小 payload 启动 40 核后重复扫描。
5. `inputCount <= 128` 仍使用原 blockDim 策略，避免公开 9 输入路径退化。

该选择只依赖输入数和总字节量，不匹配任何公开 case 的具体 shape 或取值。

## 被否决的实验

尝试过由 host 把 N+1 个 axis 前缀附加到 raw tiling buffer，kernel 做 O(1)
查询。CANN 8.5 的 `RawTilingData` 容量不足以稳定承载 129+ 输入的额外表，
129 输入在 tiling 阶段直接失败。该方案已撤销；最终源码不包含该表。

## A/B profile

每项包含 30 次 `Concat` AI Core 任务；时间来自 `msprof` 的
`op_summary*.csv`。

| 用例 | 旧版 blockDim | 旧版中位（μs） | 最终 blockDim | 最终中位（μs） | 变化 |
|---|---:|---:|---:|---:|---:|
| 9 输入，公开 geometry | 9 | 8.60 | 9 | 8.68 | -0.9%（噪声区间） |
| 129 输入 | 40 | 35.18 | 3 | 34.41 | +2.2% |
| 256 输入 | 40 | 56.90 | 8 | 34.23 | +39.8% |

公开 geometry 的最终三次独立中位数为 8.69、8.58、8.63 μs。256 输入的
`aiv_scalar_ratio` 从 94.4% 降到 87.0%。

## 正确性和提交包验证

- 原回归：8/8 通过，覆盖 fp16/fp32/int32/int8、负 dim、非对齐、空输入、
  大 segment。
- 新回归：129 输入 fp16、256 输入 int8 均逐元素精确通过。
- ACLNN 动态输入列表上限为 256；384 输入由框架在 tiling 前拒绝。
- 使用主办方 `zip_op.sh` 打包。
- ZIP 内仅含 `op_host/`、`op_kernel/` 和
  `custom_opp_ubuntu_aarch64.run`。
- `.run` 在 ZIP 中保留 Unix 可执行权限。
- 从 ZIP 全新解压、安装后，9 输入和 256 输入用例均通过。

校验值：

```text
Concat.zip SHA-256:
15b03743f2cd477dfe71c25ae2eadd66c94eecb8689982f4d09e04fbc6325803

custom_opp_euleros_aarch64.run（构建产物）SHA-256:
b430e51fb491f60e9d3cfb09eae585a108c658fb8b7a38cfb5c4eb53930ca065
```

提交包内按赛事要求将同一构建产物命名为
`custom_opp_ubuntu_aarch64.run`。

