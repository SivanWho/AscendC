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
    ((), (), torch.float32, "scalar-scalar"),
    ((0, 3), (1, 3), torch.float16, "empty-output"),
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


for index, (shape_a, shape_b, dtype, label) in enumerate(CASES):
    a = make_tensor(shape_a, dtype, 1000 + index)
    b = make_tensor(shape_b, dtype, 2000 + index)
    expected = torch.gt(a, b)
    actual = custom_ops_lib.custom_op(a.npu(), b.npu()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(
        f"PASS {label}: a={shape_a}, b={shape_b}, "
        f"dtype={dtype}, output={tuple(actual.shape)}"
    )

print("GREATER_REGRESSION_PASS")
