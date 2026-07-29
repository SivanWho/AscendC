import torch
import torch_npu

import index_add_custom_test


def run_case(name, self_cpu, index_cpu, source_cpu, dim):
    expected = self_cpu.index_add(dim, index_cpu.to(torch.int64), source_cpu)
    actual = index_add_custom_test.index_add(
        self_cpu.npu(), index_cpu.npu(), source_cpu.npu(), dim
    )
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    if expected.dtype in (torch.int8, torch.int32):
        torch.testing.assert_close(actual_cpu, expected, rtol=0, atol=0)
    elif expected.dtype == torch.float32:
        torch.testing.assert_close(actual_cpu, expected, rtol=1e-4, atol=1e-4)
    else:
        torch.testing.assert_close(actual_cpu, expected, rtol=1e-3, atol=1e-3)
    print(f"PASS {name}: shape={tuple(self_cpu.shape)}, dim={dim}")


torch.manual_seed(20260721)

run_case(
    "official_geometry_int8",
    torch.randint(-50, 50, (32, 128), dtype=torch.int8),
    torch.randint(0, 32, (120,), dtype=torch.int32),
    torch.randint(-10, 10, (120, 128), dtype=torch.int8),
    0,
)

run_case(
    "official_geometry_int8_strong_overflow",
    torch.full((32, 128), 120, dtype=torch.int8),
    torch.zeros(120, dtype=torch.int32),
    torch.full((120, 128), 127, dtype=torch.int8),
    0,
)

run_case(
    "int8_non_aligned",
    torch.randint(-128, 128, (17, 37), dtype=torch.int8),
    torch.randint(0, 17, (53,), dtype=torch.int32),
    torch.randint(-128, 128, (53, 37), dtype=torch.int8),
    0,
)

run_case(
    "int8_middle_dim_outer_groups",
    torch.randint(-128, 128, (3, 19, 7), dtype=torch.int8),
    torch.randint(0, 19, (25,), dtype=torch.int32),
    torch.randint(-128, 128, (3, 25, 7), dtype=torch.int8),
    1,
)

run_case(
    "int8_max_index_count",
    torch.randint(-128, 128, (4, 33), dtype=torch.int8),
    torch.randint(0, 4, (8000,), dtype=torch.int32),
    torch.randint(-128, 128, (8000, 33), dtype=torch.int8),
    0,
)

run_case(
    "float32_inner_larger_than_tile",
    torch.randn(3, 9001, dtype=torch.float32),
    torch.tensor([2, 0, 2, 1, 0], dtype=torch.int32),
    torch.randn(5, 9001, dtype=torch.float32),
    0,
)

run_case(
    "float16_negative_dim",
    torch.randn(2, 5, 11, dtype=torch.float16),
    torch.tensor([4, 1, 4, 0, 3, 1, 2], dtype=torch.int32),
    torch.randn(2, 5, 7, dtype=torch.float16),
    -1,
)

run_case(
    "bfloat16_high_collision",
    torch.randn(2, 7, 13, dtype=torch.bfloat16),
    torch.zeros(101, dtype=torch.int32),
    torch.randn(2, 101, 13, dtype=torch.bfloat16),
    1,
)

run_case(
    "int32_overflow",
    torch.full((3, 65), 2**31 - 8, dtype=torch.int32),
    torch.tensor([1, 1, 1, 2], dtype=torch.int32),
    torch.full((4, 65), 2**30, dtype=torch.int32),
    0,
)

run_case(
    "empty_index",
    torch.randn(2, 3, 5, dtype=torch.float32),
    torch.empty(0, dtype=torch.int32),
    torch.empty(2, 0, 5, dtype=torch.float32),
    1,
)

print("V3_ALL_REGRESSION_TESTS_PASS")
