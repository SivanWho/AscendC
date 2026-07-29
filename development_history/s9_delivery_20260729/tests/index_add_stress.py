import sys

import torch
import torch_npu
import custom_ops_lib


CASES = [
    ((32, 128), 0, 120, torch.int8, "official-int8"),
    ((17, 31), 0, 9, torch.float16, "fp16-dim0"),
    ((4, 11, 7), 1, 13, torch.float32, "fp32-middle"),
    ((3, 5, 8191), -1, 23, torch.int32, "int32-last-large-inner"),
    ((2, 9, 257), 1, 17, torch.bfloat16, "bf16-middle"),
    ((2, 5, 7, 9), 2, 0, torch.float16, "empty-index"),
    ((2, 3, 4, 5, 6, 7), 3, 19, torch.float16, "rank6-middle"),
    ((1, 2, 20000), 1, 7, torch.float32, "multi-tile-inner"),
    ((300, 4), 0, 9000, torch.float16, "index-over-ub-cache"),
]


def make_values(shape, dtype):
    if dtype in (torch.int8, torch.int32):
        low, high = (-10, 11) if dtype == torch.int8 else (-1000, 1001)
        return torch.randint(low, high, shape, dtype=dtype)
    return torch.empty(shape, dtype=torch.float32).uniform_(-10, 10).to(dtype)


def run_case(case_id):
    shape, dim, index_count, dtype, name = CASES[case_id]
    canonical_dim = dim % len(shape)
    torch.manual_seed(2026072900 + case_id)
    self_cpu = make_values(shape, dtype)
    index_cpu = torch.randint(
        0, shape[canonical_dim], (index_count,), dtype=torch.int32
    )
    source_shape = list(shape)
    source_shape[canonical_dim] = index_count
    source_cpu = make_values(source_shape, dtype)
    expected = torch.index_add(self_cpu, dim, index_cpu, source_cpu)
    actual = custom_ops_lib.custom_op(
        self_cpu.npu(), index_cpu.npu(), source_cpu.npu(), dim
    ).cpu()
    if dtype in (torch.int8, torch.int32):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    print(
        f"PASS {case_id} {name}: shape={shape} dim={dim} "
        f"index_count={index_count} dtype={dtype}"
    )


if __name__ == "__main__":
    for item in sys.argv[1:]:
        run_case(int(item))
