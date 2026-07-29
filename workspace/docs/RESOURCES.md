# 资源与技巧索引

## 必读官方资源

- [S9 官方页面](https://www.hiascend.com/developer/contests/details/41ffbad2024e4ccfa43520c57ffa7b9e?channelCode=S22)：规则、题包、算力、榜单和提交入口的唯一主入口。
- [Ascend C 算子编程指南](https://www.hiascend.com/document/redirect/CannCommunityOpdevAscendC)：Host、Kernel、Tiling、编程范式与 API。
- [Ascend C 最佳实践：获取性能数据](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0008.html)：Profiling 与流水图。
- [Tiling 核间负载均衡](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0037.html)：多核切分、L2 与拖尾。
- [限制 TilingData 结构大小](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0018.html)：小 shape 下的 Host→Kernel 元数据开销。
- [Double Buffer 原理](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0090.html)：流水收益与适用条件。

## 高价值实践文章

- [Ascend C 性能优化技巧 04：Tiling 优化](https://www.hiascend.com/forum/thread-02104162803169201072-1-1.html)：多核、L2 Cache 切分和负载均衡。
- [Ascend C 性能优化技巧 05：API 使用优化](https://www.hiascend.com/developer/techArticles/20241107-1)：高阶 API 与调用方式的性能影响。
- [Erf 算子训练营实践](https://www.hiascend.com/forum/thread-0269202456643546167-1-1.html)：用 shape 阈值区分是否开启 Double Buffer，说明小 shape 不能机械套用流水模板。
- [为什么需要算子调优工具](https://www.hiascend.com/developer/blog/details/0259195794730207015)：采集→定位→优化→验证闭环。
- [2024 冠军赛赛题解析](https://www.hiascend.com/forum/thread-0250192265977266478-1-1.html)：往届真实赛题的拆解与性能优化视角。
- [四大获奖团队代表分享](https://www.bilibili.com/video/BV1Bf421D7RC/)：参赛经验和团队协作。

博客可提供思路，但版本、芯片和接口可能不同。实现前应回到 CANN 8.5.0 官方文档核对 API 和硬件约束。

## 本地仓库怎么用

- `utils/repos/cannbot-skills`：先读 `ops/npu-arch`、`ascendc-tiling-design`、`ascendc-performance-best-practices`、`ops-profiling`、`ascendc-precision-debug`、`ascendc-code-review`。
- `utils/repos/cann-learning-hub`：系统教程、Notebook、`skills/ascendc-ops-project` 和 `skills/cannjudge-submit`。
- `utils/repos/ascend-samples-operator/operator/ascendc`：官方 Ascend C 样例，优先寻找与当前数据搬运/归约模式相近的实现。
- `utils/repos/cann-ops`：生产级算子工程结构、Host/Tiling/Kernel 和测试模式。
- `utils/repos/asc-tools`：VS Code 开发、调试和调优工具链。

## 通用经验提炼

1. **公开样例是通路样例，不是测试集。** S9 Excel 明确覆盖多 dtype、宽 shape 范围与非对齐；评分规则也明确测试数据随机生成。
2. **先区分瓶颈。** Memory-bound 算子优先减少搬运和地址计算；归约/散射算子还需考虑冲突、同步和中间精度。
3. **小 shape 与大 shape 分开建模。** 大 shape 看流水和带宽，小 shape 看启动、同步、TilingData 和过度分核。
4. **泛化的多路径是允许的，case 硬编码不是。** 路径选择应基于 dtype、rank、连续性、广播形态、轴位置或数据规模等通用属性。
5. **成绩必须可复现。** 每次平台提交都应能追溯到 commit、二进制哈希、环境和本地 profiling。
