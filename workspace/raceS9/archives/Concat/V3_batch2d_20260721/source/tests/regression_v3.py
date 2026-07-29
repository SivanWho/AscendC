import torch
import torch_npu
import custom_ops_lib


def run_case(name, tensors, dim):
    expected = torch.cat(tensors, dim=dim)
    npu_inputs = [tensor.npu() for tensor in tensors]
    actual = custom_ops_lib.custom_op(npu_inputs, dim, list(expected.shape)).cpu()
    if not torch.equal(actual, expected):
        mismatch = int((actual != expected).sum().item())
        raise AssertionError(f"{name}: {mismatch} mismatched elements")
    print(f"PASS {name}: shape={tuple(expected.shape)}, dtype={expected.dtype}, dim={dim}")


torch.manual_seed(20260721)

cases = [
    (
        "official_geometry_fp16",
        list(torch.split(torch.randn(128, 256, dtype=torch.float16), [27, 40, 63, 24, 50, 26, 19, 2, 5], -1)),
        -1,
    ),
    (
        "fp32_dim0_nonaligned",
        [torch.randn(3, 7), torch.randn(5, 7), torch.randn(1, 7)],
        0,
    ),
    (
        "int32_middle_dim",
        [
            torch.randint(-10000, 10000, (2, 3, 5), dtype=torch.int32),
            torch.randint(-10000, 10000, (2, 7, 5), dtype=torch.int32),
            torch.randint(-10000, 10000, (2, 1, 5), dtype=torch.int32),
        ],
        1,
    ),
    (
        "int8_last_dim_nonaligned",
        [
            torch.randint(-128, 128, (17, 13), dtype=torch.int8),
            torch.randint(-128, 128, (17, 29), dtype=torch.int8),
            torch.randint(-128, 128, (17, 1), dtype=torch.int8),
        ],
        -1,
    ),
    (
        "fp16_rank4_negative_dim",
        [torch.randn(2, 3, 4, 11, dtype=torch.float16), torch.randn(2, 3, 7, 11, dtype=torch.float16)],
        -2,
    ),
    (
        "zero_sized_slices",
        [torch.randn(9, 0, dtype=torch.float16), torch.randn(9, 5, dtype=torch.float16), torch.randn(9, 0, dtype=torch.float16)],
        1,
    ),
    (
        "large_segment_fallback",
        [torch.randn(2, 40000, dtype=torch.float16), torch.randn(1, 40000, dtype=torch.float16)],
        0,
    ),
    (
        "many_inputs_with_empty",
        [torch.randint(-128, 128, (4, i % 4), dtype=torch.int8) for i in range(64)],
        1,
    ),
]

for case in cases:
    run_case(*case)

print(f"ALL PASS: {len(cases)}/{len(cases)}")
