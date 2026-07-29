import torch
import torch_npu
import custom_ops_lib


def run_case(name, tensors, dim):
    expected = torch.cat(tensors, dim=dim)
    actual = custom_ops_lib.custom_op(
        [tensor.npu() for tensor in tensors], dim, list(expected.shape)
    ).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(
        f"PASS {name}: inputs={len(tensors)}, "
        f"shape={tuple(expected.shape)}, dtype={expected.dtype}, dim={dim}"
    )


torch.manual_seed(20260723)

run_case(
    "official-fp16",
    list(
        torch.split(
            torch.randn(128, 256, dtype=torch.float16),
            [27, 40, 63, 24, 50, 26, 19, 2, 5],
            -1,
        )
    ),
    -1,
)
run_case(
    "large-outer-narrow-int8",
    [
        torch.randint(-128, 128, (8192, 1), dtype=torch.int8),
        torch.randint(-128, 128, (8192, 2), dtype=torch.int8),
        torch.randint(-128, 128, (8192, 1), dtype=torch.int8),
    ],
    1,
)
run_case(
    "large-outer-narrow-fp16",
    [
        torch.randn(65536, 1, dtype=torch.float16),
        torch.randn(65536, 1, dtype=torch.float16),
    ],
    1,
)
run_case(
    "more-than-128-inputs",
    [torch.randint(-128, 128, (4, 1), dtype=torch.int8) for _ in range(129)],
    1,
)
run_case(
    "over-ub-segment-fallback",
    [torch.randn(2, 40000), torch.randn(1, 40000)],
    0,
)
run_case(
    "empty-concat-slices",
    [
        torch.randn(9, 0, dtype=torch.float16),
        torch.randn(9, 5, dtype=torch.float16),
        torch.randn(9, 0, dtype=torch.float16),
    ],
    1,
)
run_case(
    "zero-outer-dimension",
    [torch.empty(0, 3), torch.empty(0, 2)],
    1,
)
run_case(
    "all-empty-concat-axis",
    [torch.empty(5, 0, dtype=torch.int32), torch.empty(5, 0, dtype=torch.int32)],
    1,
)

print("CONCAT_REGRESSION_PASS")
