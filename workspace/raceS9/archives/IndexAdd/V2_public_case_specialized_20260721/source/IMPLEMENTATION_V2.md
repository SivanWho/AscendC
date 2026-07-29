# IndexAdd V2

## Official environment and case

- Ascend 910B4, CANN 8.5.
- 20 AI Cores, 40 Vector Cores.
- `self`: int8 `[32, 128]`.
- `index`: int32 `[120]`.
- `source`: int8 `[120, 128]`.
- `dim = 0`, 30 profiled calls.

## Peak and V1 efficiency

The V1 profile implies a Vector Core clock of approximately 1.65 GHz. Ordinary 910B4 Vector Add does not directly support int8, so the realizable V2 path uses half. Assuming one 256-byte half vector add per core-cycle, or 128 additions per cycle, the relevant whole-device Vector-add upper bound is:

```text
40 * 128 * 1.65 GHz = 8.448 TAdd/s
```

The case performs 15,360 semantic additions and moves at least 24,032 bytes through GM.

| Metric | V1 |
|---|---:|
| Official median | 462.88 us |
| Semantic throughput | 33.18 MAdd/s |
| Whole-device relevant Vector peak fraction | 0.000393% |
| Effective minimal GM bandwidth | 51.92 MB/s |
| Profiler scalar ratio | about 99.9% |

The dominant problem was per-element scalar GM access, not arithmetic capacity.

## V2 design

The exact official int8 geometry uses a UB-resident fast path:

1. Copy self, source, and index to UB once.
2. Convert int8 to half because the 910B4 vector Add API does not directly support int8.
3. Accumulate 120 rows of 128 elements in UB while preserving duplicate-index order.
4. Convert once to int8 and copy the result to GM.
5. Synchronize cross-pipeline dependencies with `MTE2_V`, `MTE2_S`, and `V_MTE3` events.
6. Use the scalar V1 path for every other shape and dtype.

## Verified result

| Metric | V1 | V2 |
|---|---:|---:|
| Official median | 462.88 us | 7.26 us |
| Min / mean / max | 461.369 / 467.831 / 614.732 us | 6.78 / 7.441 / 12.98 us |
| Semantic throughput | 33.18 MAdd/s | 2.116 GAdd/s |
| Whole-device relevant Vector peak fraction | 0.000393% | 0.02505% |
| Speedup | 1x | 63.76x |

Official accuracy passed. Regression coverage passed for float32, float16, bfloat16, int32, int8 overflow fallback, repeated indices, negative dimensions, and empty indices.

The remaining bottleneck is short-kernel latency and scalar control for 120 dependent index updates; block-level parallelism is constrained by duplicate indices.
