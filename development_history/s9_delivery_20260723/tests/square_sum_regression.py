import torch
import torch_npu
import custom_ops_lib


CASES = [
    ((123, 31), torch.float16, (-1,), True, "public-fast"),
    ((7, 17), torch.float16, (-1,), False, "fp16-padded-fast"),
    ((257, 257), torch.float16, (-1,), False, "fp16-long-suffix"),
    ((64, 128), torch.float32, (-1,), True, "fp32-suffix"),
    ((4, 3, 5, 7), torch.float32, (2, 3), False, "two-axis-suffix"),
    ((16, 17, 18), torch.float32, (0, 2), True, "noncontiguous"),
    ((8, 9, 10), torch.bfloat16, (-1,), True, "bf16-suffix"),
    ((31, 17, 9), torch.float32, (), False, "empty-axis-reduce-all"),
    ((4, 3, 5), torch.float32, (-1, 2, 1), True, "duplicate-axis"),
    ((7, 0), torch.float16, (-1,), True, "zero-reduced-last-dim"),
    ((0, 31), torch.float16, (-1,), True, "zero-output-count"),
    ((), torch.float32, (), False, "scalar-empty-axis"),
]


def canonical_axes(rank, axes):
    if not axes:
        return tuple(range(rank))
    result = []
    for axis in axes:
        axis = axis + rank if axis < 0 else axis
        if axis not in result:
            result.append(axis)
    return tuple(result)


for index, (shape, dtype, axes, keep_dims, label) in enumerate(CASES):
    torch.manual_seed(20260723 + index)
    source = torch.empty(shape, dtype=torch.float32).uniform_(-1.0, 1.0).to(dtype)
    expected = torch.sum(
        torch.square(source),
        dim=canonical_axes(source.dim(), axes),
        keepdim=keep_dims,
    )
    actual = custom_ops_lib.custom_op(
        source.npu(), axes, keep_dims, list(expected.shape)
    ).cpu()
    if dtype == torch.float32:
        rtol, atol = 1.0e-4, 1.0e-4
    else:
        rtol, atol = 1.0e-2, 1.0e-2
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    print(
        f"PASS {label}: shape={shape}, dtype={dtype}, "
        f"axes={axes}, output={tuple(actual.shape)}"
    )

print("SQUARE_SUM_V1_REGRESSION_PASS")
