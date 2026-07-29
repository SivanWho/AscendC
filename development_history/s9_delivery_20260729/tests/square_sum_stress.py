import sys

import torch
import torch_npu
import custom_ops_lib


CASES = [
    ((2, 3, 4, 5, 6, 7), torch.float16, (-1,), True, "rank6-last-fp16"),
    ((2, 3, 4, 5, 6, 7), torch.float32, (2, 3), False, "rank6-middle-contiguous"),
    ((2, 3, 4, 5, 6, 7), torch.float32, (0, 2, 4), True, "rank6-noncontiguous"),
    ((2, 3, 4, 5, 6, 7), torch.bfloat16, (), False, "rank6-reduce-all-bf16"),
    ((2024, 3000), torch.float16, (-1,), False, "large-last-fp16"),
    ((2024, 3000), torch.float32, (0,), True, "large-first-fp32"),
    ((2024, 3000), torch.float32, (), False, "large-reduce-all-fp32"),
    ((4,), torch.float16, (0,), True, "vector4-fp16"),
    ((4,), torch.float32, (0,), False, "vector4-fp32"),
    ((255, 64), torch.float16, (-1,), False, "fast-max-255x64"),
    ((256, 64), torch.float16, (-1,), False, "fast-row-boundary-256x64"),
    ((255, 65), torch.float16, (-1,), False, "fast-col-boundary-255x65"),
    ((17, 4095), torch.float16, (-1,), False, "suffix-4095"),
    ((17, 4096), torch.float16, (-1,), False, "suffix-4096"),
    ((17, 4097), torch.float16, (-1,), False, "suffix-4097"),
    ((3, 4096, 17), torch.float32, (1,), False, "contiguous-middle-4096"),
    ((3, 4097, 17), torch.float32, (1,), False, "contiguous-middle-4097"),
    ((4097, 17), torch.bfloat16, (0,), False, "first-axis-4097-bf16"),
    ((1, 1, 1, 1, 1, 1, 1, 1), torch.float32, tuple(range(8)), True, "rank8-all"),
    ((3, 5, 7), torch.float16, (0, 2), False, "generic-fp16"),
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


def run_case(index):
    shape, dtype, axes, keep_dims, label = CASES[index]
    torch.manual_seed(2026072900 + index)
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
        f"PASS {index} {label}: shape={shape}, dtype={dtype}, "
        f"axes={axes}, output={tuple(actual.shape)}"
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\n".join(f"{i}: {case[4]}" for i, case in enumerate(CASES)))
    else:
        for argument in sys.argv[1:]:
            run_case(int(argument))
