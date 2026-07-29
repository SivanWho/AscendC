import torch
import torch_npu

import custom_ops_lib


torch.npu.config.allow_internal_format = False


def main():
    x1 = torch.tensor(
        [-3.0, -2.0, -1.0, -0.0, 0.0, 1.0, 2.0, 3.0,
         float("-inf"), float("inf"), float("nan")],
        dtype=torch.float16,
    )
    x2 = torch.tensor(
        [-2.0, -3.0, -1.0, 0.0, -0.0, 0.0, 3.0, 2.0,
         -100.0, 100.0, 0.0],
        dtype=torch.float16,
    )
    expected = torch.gt(x1, x2)
    actual = custom_ops_lib.custom_op(x1.npu(), x2.npu()).cpu()
    print("x1      =", x1)
    print("x2      =", x2)
    print("expected=", expected)
    print("actual  =", actual)
    print("mismatch=", torch.nonzero(actual != expected).flatten())

    for attempt in range(3):
        a = (torch.rand((32, 64), dtype=torch.float16) * 2000 - 1000)
        b = (torch.rand((32, 64), dtype=torch.float16) * 2000 - 1000)
        expected_random = torch.gt(a, b)
        actual_random = custom_ops_lib.custom_op(a.npu(), b.npu()).cpu()
        mismatch = torch.nonzero(actual_random != expected_random)
        print(
            f"random attempt {attempt}: mismatch={mismatch.shape[0]}, "
            f"true_expected={expected_random.sum().item()}, "
            f"true_actual={actual_random.sum().item()}"
        )
        if mismatch.numel():
            coords = mismatch[:10]
            print("first mismatch coords=", coords)
            print("expected values=", expected_random[coords[:, 0], coords[:, 1]])
            print("actual values=", actual_random[coords[:, 0], coords[:, 1]])

    cases = [
        ("fp16_same", torch.randn(17, 33, dtype=torch.float16), torch.randn(17, 33, dtype=torch.float16)),
        ("fp32_broadcast", torch.randn(3, 1, 5), torch.randn(1, 4, 1)),
        ("int32_broadcast", torch.arange(24, dtype=torch.int32).reshape(2, 3, 4), torch.tensor([[[7]]], dtype=torch.int32)),
        ("int8_broadcast", torch.arange(-8, 8, dtype=torch.int8).reshape(2, 8), torch.tensor([0], dtype=torch.int8)),
        ("bf16_broadcast", torch.randn(2, 1, 7).to(torch.bfloat16), torch.randn(1, 5, 1).to(torch.bfloat16)),
    ]
    failures = []
    for name, a, b in cases:
        expected_case = torch.gt(a, b)
        actual_case = custom_ops_lib.custom_op(a.npu(), b.npu()).cpu()
        mismatch_count = int((actual_case != expected_case).sum().item())
        print(f"{name}: shape={tuple(actual_case.shape)}, mismatch={mismatch_count}")
        if mismatch_count:
            failures.append(name)
    if failures:
        raise AssertionError(f"Greater regression failed: {failures}")


if __name__ == "__main__":
    main()
