# Greater V2 官方 profiling

- 设备：Ascend 910B4，CANN 8.5.0
- 输入：两个 FP16 `[32,64]` tensor
- Block Dim：8
- 最终 ZIP 解压复验 median：2.981us
- profiler min/mean/max：2.760/3.391/6.000us
- V1 median：105.244us
- speedup：35.30×

语义吞吐：2048 comparisons / 2.981us = 0.687 GOPS。
最小 GM 流量约 10,240 bytes，对应 3.435 GB/s 有效带宽。

若按 profiler 推导的约 800MHz、每 AIV 每周期 128 个 FP16 比较估算，8 个已使用 AIV
的峰值约 819.2 GOPS，语义效率约 0.084%。该小用例仍主要受启动和流水建立开销限制。
