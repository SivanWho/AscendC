# Concat V1

动态 `Tensor[]` 由 `ListTensorDesc` 解包；任务按 `(outer, input_id)` 切给 910B AI Core。每个连续片段通过 UB 分块并使用 `DataCopyPad` 精确处理非 32 字节对齐尾块。支持负 `dim`、零长度输入切片和最多 128 个输入。
