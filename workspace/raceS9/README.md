# 华为算子挑战赛 S9 工作区

本目录按五道赛题分别维护源码、官方测试、实验记录和提交包。提交时只使用
`submissions/` 中由主办方 `zip_op.sh` 在 Linux 环境生成的 ZIP。

| 赛题 | 参考语义 | 当前状态 |
|---|---|---|
| [IndexAdd](IndexAdd/README.md) | `torch.index_add` | 已有可提交版本 |
| [Concat](Concat/README.md) | `torch.cat` | 已有可提交版本 |
| [Greater](Greater/README.md) | `torch.gt` | V2，官方用例 2.981us，较 V1 加速 35.30× |
| [SquareSumV1](SquareSumV1/README.md) | `sum(square(x))` | V2，官方用例 5.641us，较 V1 加速 24.39× |
| [Transpose](Transpose/README.md) | `torch.permute` | 待继续实现和验证 |

## 目录约定

每题目录中的 `op_host/`、`op_kernel/` 和 `custom_opp_euleros_aarch64.run`
必须来自同一版源码。`tests/official/` 保存官方公开例程，`benchmarks/` 保存性能数据，
`notes/` 保存设计和失败方案。

`submissions/` 只存放最终上传 ZIP。ZIP 内必须只有一个 `<OpName>_zip/` 根目录，
其下包含 `op_host/`、`op_kernel/` 和可执行的 `custom_*.run`，不得加入 README、测试、
缓存或其他无关文件。

## 当前验证环境

- CANN 8.5.0
- Ascend 910B4
- 算子工程使用官方 CANN 构建链
- PyTorch 官方 extension 使用 GCC 10.3；系统 GCC 7 版本不足

隐藏用例会覆盖多 dtype、不同取值范围、不同 shape 和边界情况。所有 tiling 必须保持
泛化，不能针对公开用例硬编码。
