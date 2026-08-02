# 官方输入契约 1000 例测试

本目录不包含、也不修改提交算子的 `op_host/` 和 `op_kernel/`。测试扩展只通过
官方同样的 `aclnnConcat(inputs, dim, result)` ABI 进行单次调用，移除了官方性能
扩展中专用于计时的 30 次循环和 `aclnnMul` 干扰。

固定种子为 `20260730`，用例数固定为 1000：

- 支持 dtype：fp16、fp32、int32、int8；
- ND rank：1–6；正/负 dim；
- 1–256 个动态输入，含空 slice；
- 31/32/33B 对齐边界、4094/4095/4096 行、16/32/64KB 与超 UB 段；
- 小范围与宽范围随机值；
- 其余用例由固定 seed 随机生成，约束为合法 ND concat 输入且单例输出不超过 8MiB。

运行前应 source CANN 环境与目标 `.run` 安装目录的 `set_env.bash`，然后：

```bash
python3 setup.py build_ext --inplace
python3 run_1000.py
```

可用 `--start`、`--stop` 做失败重现或分片运行；`results/manifest_1000.json` 保留
完整可复现清单，CSV 记录每一个已完成用例。
