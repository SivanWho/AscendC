import torch
import torch_npu
import custom_ops_lib


def run_case(name, self_cpu, index_cpu, source_cpu, dim):
    expected = torch.index_add(
        self_cpu, dim, index_cpu.to(torch.int64), source_cpu
    )
    actual = custom_ops_lib.custom_op(
        self_cpu.npu(), index_cpu.npu(), source_cpu.npu(), dim
    ).cpu()
    if expected.dtype in (torch.int8, torch.int32):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif expected.dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    print(f"PASS {name}: shape={tuple(self_cpu.shape)}, dim={dim}")


torch.manual_seed(20260723)

run_case(
    "official-int8",
    torch.randint(-50, 50, (32, 128), dtype=torch.int8),
    torch.randint(0, 32, (120,), dtype=torch.int32),
    torch.randint(-10, 10, (120, 128), dtype=torch.int8),
    0,
)
run_case(
    "int8-overflow",
    torch.full((32, 128), 120, dtype=torch.int8),
    torch.zeros(120, dtype=torch.int32),
    torch.full((120, 128), 127, dtype=torch.int8),
    0,
)
run_case(
    "middle-dim",
    torch.randn(3, 19, 7, dtype=torch.float32),
    torch.randint(0, 19, (25,), dtype=torch.int32),
    torch.randn(3, 25, 7, dtype=torch.float32),
    1,
)
run_case(
    "inner-over-ub-tile",
    torch.randn(3, 9001, dtype=torch.float16),
    torch.tensor([2, 0, 2, 1, 0], dtype=torch.int32),
    torch.randn(5, 9001, dtype=torch.float16),
    0,
)
run_case(
    "more-than-8000-indices",
    torch.full((1, 1), 120, dtype=torch.int8),
    torch.zeros(8001, dtype=torch.int32),
    torch.full((8001, 1), 127, dtype=torch.int8),
    0,
)
run_case(
    "empty-index",
    torch.randn(2, 3, 5, dtype=torch.float32),
    torch.empty(0, dtype=torch.int32),
    torch.empty(2, 0, 5, dtype=torch.float32),
    1,
)
run_case(
    "zero-sized-self-dimension",
    torch.empty(0, 5, dtype=torch.float32),
    torch.empty(0, dtype=torch.int32),
    torch.empty(0, 5, dtype=torch.float32),
    0,
)

print("INDEX_ADD_REGRESSION_PASS")
