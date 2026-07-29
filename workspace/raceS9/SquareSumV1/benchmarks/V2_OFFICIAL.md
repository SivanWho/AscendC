# SquareSumV1 V2 官方 profiling

- 设备：Ascend 910B4，CANN 8.5.0
- 输入：FP16 `[123,31]`，axis=-1，keep_dims=true
- Block Dim：1
- 最终 ZIP 解压复验 median：5.641us
- profiler min/mean/max：5.100/5.652/7.601us
- V1 median：137.565us
- speedup：24.39×

语义工作量约为 3813 次平方和 3690 次加法，共 7503 ops；语义吞吐约 1.330 GOPS。
最小 GM 流量约 7872 bytes，对应 1.396 GB/s 有效带宽。

若按约 800MHz、每 AIV 每周期 64 个 FP32 元素操作估算，单个已使用 AIV 的相关峰值
约 51.2 GOPS，语义效率约 2.61%。小输入仍明显受启动、Cast 和 DMA 固定开销限制。
