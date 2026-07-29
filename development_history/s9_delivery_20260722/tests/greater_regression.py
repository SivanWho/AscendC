import sys

import torch
import torch_npu
import custom_ops_lib


CASES = [
    ((32, 64), (32, 64), torch.float16, "fp16-fast"),
    ((17, 33), (17, 33), torch.float16, "fp16-tail"),
    ((64, 64), (64, 64), torch.float32, "fp32"),
    ((17, 33), (17, 33), torch.int32, "int32"),
    ((5, 7), (5, 7), torch.int8, "int8"),
    ((8, 16), (8, 16), torch.bfloat16, "bf16"),
    ((4, 1, 7), (1, 5, 1), torch.float16, "broadcast-two-sides"),
    ((2, 3, 4), (), torch.float32, "broadcast-scalar"),
]


def make_tensor(shape, dtype, seed):
    torch.manual_seed(seed)
    if dtype in (torch.int32, torch.int8):
        return torch.randint(-20, 21, shape, dtype=dtype)
    result = torch.empty(shape, dtype=torch.float32).uniform_(-20.0, 20.0).to(dtype)
    if result.numel() >= 8:
        flat = result.flatten()
        flat[0] = float("inf")
        flat[1] = float("-inf")
        flat[2] = float("nan")
        flat[3] = 0.0
        flat[4] = -0.0
    return result


def main(index):
    shape_a, shape_b, dtype, label = CASES[index]
    a = make_tensor(shape_a, dtype, 1000 + index)
    b = make_tensor(shape_b, dtype, 2000 + index)
    golden = torch.gt(a, b)
    actual = custom_ops_lib.custom_op(a.npu(), b.npu()).cpu()
    if not torch.equal(actual, golden):
        mismatches = torch.nonzero(actual != golden, as_tuple=False)
        raise AssertionError(f"{label}: {mismatches.shape[0]} mismatches; first={mismatches[:16].tolist()}")
    print(f"PASS {index}: {label}, a={shape_a}, b={shape_b}, dtype={dtype}, output={tuple(actual.shape)}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
