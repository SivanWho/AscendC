# 本地开源仓库清单

以下仓库均以浅克隆方式拉取，`ascend-samples-operator` 仅稀疏检出 `operator/`。提交哈希记录于 2026-07-19。

| 本地目录 | 来源 | 当前提交 | 用途 |
|---|---|---|---|
| `cannbot-skills` | https://gitcode.com/cann/cannbot-skills.git | `b0937a528e29` | Ascend C 设计、Tiling、性能、精度、调试与审查 Skills |
| `cann-learning-hub` | https://gitcode.com/cann/cann-learning-hub.git | `4d45198698e3` | CANN 8.5+ 学习路径、教程、Notebook 与竞赛 Skills |
| `ascend-samples-operator` | https://gitee.com/ascend/samples.git | `6511a5f4a45a` | 官方 Ascend C 自定义算子样例 |
| `cann-ops` | https://gitee.com/ascend/cann-ops.git | `35fcd12e27bc` | 官方生产级算子实现、工程结构与测试参考 |
| `asc-tools` | https://gitcode.com/cann/asc-tools.git | `2cb003aa5a03` | 一站式 Ascend C 开发、调试和调优工具 |

使用前阅读每个仓库的 LICENSE、README 和版本配套说明。参考代码不能替代独立实现；比赛明确禁止队伍之间私下共享代码并会检查相似代码。

## 更新方式

本机全局 Git 当前配置了一个可能未启动的本地代理。若代理不可用，可只对单条命令临时禁用，不修改全局设置：

```powershell
git -c http.proxy= -c https.proxy= -C .\cannbot-skills pull --ff-only
```

更新后同步修改上表提交哈希，并重新核对 CANN 8.5.0 兼容性。
