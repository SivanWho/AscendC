import sys

import torch
import torch_npu
import custom_ops_lib


CASES = [
    ((123, 31), torch.float16, (-1,), True, "public-fast"),
    ((257, 257), torch.float16, (-1,), False, "fp16-long-suffix"),
    ((64, 128), torch.float32, (-1,), True, "fp32-suffix"),
    ((4, 3, 5, 7), torch.float32, (2, 3), False, "fp32-two-axis-suffix"),
    ((16, 17, 18), torch.float32, (0, 2), True, "fp32-noncontiguous"),
    ((32, 33, 34), torch.float16, (1,), False, "fp16-noncontiguous"),
    ((8, 9, 10), torch.bfloat16, (-1,), True, "bf16-suffix"),
    ((31, 17, 9), torch.float32, (), False, "empty-axis-reduce-all"),
    ((4, 3, 5), torch.float32, (-1, 2, 1), True, "duplicate-axis"),
]


def canonical_axes(rank, axes):
    if not axes:
        return tuple(range(rank))
    normalized = []
    for axis in axes:
        axis = axis + rank if axis < 0 else axis
        if axis not in normalized:
            normalized.append(axis)
    return tuple(normalized)


def main(index):
    shape, dtype, axes, keep_dims, label = CASES[index]
    torch.manual_seed(20260722 + index)
    source = torch.empty(shape, dtype=torch.float32).uniform_(-1.0, 1.0).to(dtype)
    golden_axes = canonical_axes(source.dim(), axes)
    golden = torch.sum(torch.square(source), dim=golden_axes, keepdim=keep_dims)
    actual = custom_ops_lib.custom_op(source.npu(), axes, keep_dims, list(golden.shape)).cpu()

    if dtype == torch.float32:
        rtol, atol = 1.0e-4, 1.0e-4
    else:
        rtol, atol = 1.0e-2, 1.0e-2
    close = torch.isclose(actual, golden, rtol=rtol, atol=atol)
    if not torch.all(close):
        indices = torch.nonzero(~close, as_tuple=False).flatten()
        print("MISMATCH_INDICES", indices[:128].tolist())
        print("ACTUAL", actual.flatten()[indices[:32]].tolist())
        print("GOLDEN", golden.flatten()[indices[:32]].tolist())
    torch.testing.assert_close(actual, golden, rtol=rtol, atol=atol)
    print(f"PASS {index}: {label}, shape={shape}, dtype={dtype}, axes={axes}, output={tuple(actual.shape)}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
