# S9 官方提交通道调研（2026-08-03）

## 结论

S9 目前没有公开的 Git、GitHub Actions、GitCode CI、CLI 或 REST API 判题提交通道。
比赛期间的性能判分仍通过昇腾社区比赛页上传 ZIP；获奖优秀作品则在赛后按主办方要求，向指定 GitCode 仓库提交代码并合入 PR。两者不能混用。

## 证据链

1. S9 官方赛事页把“作品提交”定义在 2026-06-15 至 2026-09-15 12:00 的比赛阶段，同时要求 CANN 8.5.0 和云上 Ascend 环境。
2. 同一页面仅在“参赛要求”和“重要说明”中提到 GitCode，原文语义是“获奖优秀作品需要在 GitCode 指定代码仓进行开源且成功合入 PR”，不是用 GitCode 触发实时榜单判分。
3. 赛题提交说明要求源码 `op_host/`、`op_kernel/` 与 `custom_*.run` 同目录，并必须使用主办方 `zip_op.sh` 生成 ZIP 后上传。判题错误也以 `Zip_Check`、`Incorrect op name`、`Run failed` 等网页结果返回。
4. 昇腾社区通用作品提交页也只公开 ZIP 文件上传控件，没有公开 Git URL 或 API token 流程。

官方入口：

- S9 赛事页：<https://www.hiascend.com/developer/contests/details/41ffbad2024e4ccfa43520c57ffa7b9e>
- CANN 8.5.0：<https://www.hiascend.com/developer/download/community/result?module=pt+cann&pt=7.1.0&cann=8.5.0>

## 现在能自动化到什么程度

可以安全自动化：

1. Git 记录源码、实验矩阵、Profiler 摘要和提交包哈希；
2. 在 910B/CANN 8.5 环境编译 `custom_opp_*.run`；
3. 安装 run 包并执行官方样例、1000 例回归和必要的 msprof；
4. 调用官方 `zip_op.sh` 生成最终 ZIP；
5. 验证 ZIP 根目录、必需文件名、run 包算子名、SHA-256 和压缩包大小；
6. 打开官方网页并选择已验证的 ZIP。

不应伪装成官方能力：

- Git push 不会触发 S9 判题；
- 不应抓取登录 Cookie、私有接口或构造未公开请求绕过网页；
- 未看到主办方公开 API 文档前，不把浏览器内部网络请求当作稳定接口；
- 最终“提交”会覆盖上一次成绩，属于外部状态变更，应在人确认题目和 ZIP 后执行。

## 推荐工作流

```text
feature branch
  -> compile on CANN 8.5 / 910B
  -> fresh install run package
  -> official smoke test
  -> 1000-case regression
  -> selected msprof matrix
  -> official zip_op.sh
  -> verify structure + SHA-256
  -> push GitHub evidence commit
  -> human-confirmed web upload
  -> record judge result and commit hash
```

提交记录至少保存：题目、分支、commit、ZIP SHA-256、run SHA-256、构建环境、回归结果、网页判题结果和时间。这样即使没有官方 Git 提交通道，也能让每次榜单成绩准确回溯到唯一源码。
