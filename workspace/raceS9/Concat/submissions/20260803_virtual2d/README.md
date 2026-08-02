# Concat Virtual2D 提交包（2026-08-03）

- 提交文件：`Concat.zip`
- 目标环境：Ascend 910B4 / CANN 8.5.0 / EulerOS aarch64
- 打包方式：主办方原始 `zip_op.sh`，在远端 Linux 环境执行
- ZIP SHA-256：`d5459fefd06a49826e1077f0ddf2f8de9c011a214c1378bb0f30065cdfc0c5f3`
- run SHA-256：`6b32132ad077a88badd837e9232cfe3d20549d2dee8eda171b1ad066ef7623d7`

## 源码一致性

- `op_host/concat.cpp`：`88ab93efcde16794f4e43dc02446eef659e8d2cd14e6287818ec7d787c7186c2`
- `op_host/concat_tiling.h`：`636ccbc80318c72e823c4902f067879a7e3e6a6e28c682fb916e1179b00eb0ac`
- `op_kernel/concat.cpp`：`5446acbd61ebd7be24d1f962c80c928ec1d380124b0a37ce461ed2140c975ee2`

以上源码先同步至构建目录再清洁编译；ZIP 内源码哈希与本地源码一致，ZIP 内 run
哈希与清洁编译产物一致。

## 验证

- 清洁编译成功，run 自解压包生成成功。
- 使用官方脚本生成 ZIP。
- 从 ZIP 全新解压并安装至独立目录成功。
- 解压安装后，随官方例程提供的 Case1 精度验证通过。
- 本地官方例程只定义了 Case1；Case2--Case5 会触发 `KeyError`，因此没有把脚本误报的
  `passed` 当作有效结果。
- 当前算法此前已通过固定种子 1000 例回归；本次相对该验证版本只新增注释，仍重新编译
  以满足源码与 run 逐字对应的比赛要求。

## ZIP 契约检查

- 恰好9个条目，无额外文件；
- 根目录为 `Concat_zip/`；
- `op_host/`、`op_kernel/` 和目录条目完整；
- 路径全部使用正斜杠；
- ZIP 为 Unix origin；
- `.run` 文件模式为 `0750`，保留可执行位。
