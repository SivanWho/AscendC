import torch
import torch_npu

import custom_ops_lib


torch.npu.config.allow_internal_format = False


def main():
    torch.manual_seed(20260721)
    cases = [
        ("fp16_last_keep", torch.randn(123, 31, dtype=torch.float16), (-1,), True),
        ("fp32_multi", torch.randn(2, 3, 4, dtype=torch.float32), (0, 2), False),
        ("bf16_middle_keep", torch.randn(3, 5, 7).to(torch.bfloat16), (1,), True),
        ("fp16_tail_multi", torch.randn(2, 3, 4, 5, dtype=torch.float16), (-1, -2), True),
        ("fp32_full", torch.randn(4, dtype=torch.float32), (0,), False),
        ("fp16_fast_7x17", torch.randn(7, 17, dtype=torch.float16), (-1,), False),
        ("fp16_fast_5x64", torch.randn(5, 64, dtype=torch.float16), (-1,), True),
        ("fp16_fast_255x1", torch.randn(255, 1, dtype=torch.float16), (-1,), False),
        ("fp16_fast_scalar_out", torch.randn(31, dtype=torch.float16), (-1,), False),
    ]
    failures = []
    for name, x, axis, keep_dims in cases:
        expected = torch.sum(torch.square(x), axis, keepdim=keep_dims)
        actual = custom_ops_lib.custom_op(
            x.npu(), axis, keep_dims, list(expected.shape)
        ).cpu()
        if x.dtype == torch.float32:
            ok = torch.allclose(actual, expected, rtol=1e-4, atol=1e-4, equal_nan=True)
        else:
            ok = torch.allclose(actual, expected, rtol=1e-2, atol=1e-2, equal_nan=True)
        max_abs = float((actual.float() - expected.float()).abs().max().item())
        print(
            f"{name}: output_shape={tuple(actual.shape)}, "
            f"max_abs={max_abs:.6g}, pass={ok}"
        )
        if not ok:
            failures.append(name)
    if failures:
        raise AssertionError(f"SquareSumV1 regression failed: {failures}")


if __name__ == "__main__":
    main()
