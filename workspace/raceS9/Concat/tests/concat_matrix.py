import argparse
import traceback

import torch
import torch_npu

import custom_ops_lib


def make_tensor(shape, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    if dtype in (torch.int8, torch.int32):
        low, high = (-128, 128) if dtype == torch.int8 else (-100000, 100000)
        return torch.randint(low, high, shape, dtype=dtype, generator=generator)
    return torch.randn(shape, dtype=dtype, generator=generator)


def run_case(name, shapes, dim, dtype):
    cpu_inputs = [
        make_tensor(shape, dtype, 20260730 + index)
        for index, shape in enumerate(shapes)
    ]
    expected = torch.cat(cpu_inputs, dim=dim)
    npu_inputs = [tensor.npu() for tensor in cpu_inputs]
    actual = custom_ops_lib.custom_op(
        npu_inputs, dim, list(expected.shape)
    ).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    moved_bytes = 2 * expected.numel() * expected.element_size()
    print(
        f"PASS {name}: inputs={len(shapes)} dim={dim} "
        f"shape={tuple(expected.shape)} dtype={dtype} "
        f"min_gm_bytes={moved_bytes}"
    )


def run_invalid(name, tensors, dim):
    try:
        output_shape = list(torch.cat(tensors, dim=dim).shape)
    except Exception:
        output_shape = [1]
    try:
        custom_ops_lib.custom_op(
            [tensor.npu() for tensor in tensors], dim, output_shape
        )
        torch.npu.synchronize()
    except Exception:
        print(f"PASS {name}: rejected invalid input")
        return
    raise AssertionError(f"{name}: invalid input was accepted")


def cases():
    result = [
        ("official_fp16", [(128, n) for n in (27, 40, 63, 24, 50, 26, 19, 2, 5)], -1, torch.float16),
        ("fp32_dim0", [(3, 7), (5, 7), (1, 7)], 0, torch.float32),
        ("int32_middle", [(2, 3, 5), (2, 7, 5), (2, 1, 5)], 1, torch.int32),
        ("int8_last", [(17, 13), (17, 29), (17, 1)], -1, torch.int8),
        ("rank6_middle", [(2, 3, 4, 5, 6, 7), (2, 3, 1, 5, 6, 7)], 2, torch.float16),
        ("empty_edges", [(9, 0), (9, 5), (9, 0)], 1, torch.float16),
        ("byte_31_32_33", [(11, 31), (11, 32), (11, 33)], 1, torch.int8),
        ("half_31_32_33_bytes", [(13, 15), (13, 16), (13, 17)], 1, torch.float16),
        ("float_28_32_36_bytes", [(7, 7), (7, 8), (7, 9)], 1, torch.float32),
        ("inner_stride_nonaligned", [(7, 3, 257), (7, 5, 257)], 1, torch.float16),
        ("rows_4094", [(4094, 3), (4094, 5)], 1, torch.float16),
        ("rows_4095", [(4095, 3), (4095, 5)], 1, torch.float16),
        ("rows_4096", [(4096, 3), (4096, 5)], 1, torch.float16),
        ("tile_16k_boundary", [(3, 8192), (3, 17)], 1, torch.float16),
        ("tile_32k_boundary", [(3, 16384), (3, 17)], 1, torch.float16),
        ("tile_64k_boundary", [(3, 32768), (3, 17)], 1, torch.float16),
        ("over_64k_segment", [(3, 40000), (3, 40003)], 1, torch.float16),
        ("few_large_dim0", [(2, 131072), (3, 131072)], 0, torch.float16),
        ("few_large_strided", [(8, 65537), (8, 65539)], 1, torch.float16),
        ("many_64_with_empty", [(4, i % 5) for i in range(64)], 1, torch.int8),
        ("many_128", [(8, 1 + i % 3) for i in range(128)], 1, torch.float16),
        ("many_129", [(8, 1 + i % 3) for i in range(129)], 1, torch.float16),
        ("many_256", [(4, 1 + i % 3) for i in range(256)], 1, torch.int8),
    ]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    selected = cases()[:10] if args.quick else cases()
    failures = []
    for case in selected:
        try:
            run_case(*case)
        except Exception:
            failures.append(case[0])
            traceback.print_exc()

    invalid_tests = [
        (
            "mixed_dtype",
            [torch.randn(2, 3, dtype=torch.float16), torch.randn(2, 4, dtype=torch.float32)],
            1,
        ),
        (
            "mismatched_nonconcat_axis",
            [torch.randn(2, 3), torch.randn(4, 5)],
            1,
        ),
    ]
    for invalid in invalid_tests:
        try:
            run_invalid(*invalid)
        except Exception:
            failures.append(invalid[0])
            traceback.print_exc()

    if failures:
        raise SystemExit(f"FAILED: {failures}")
    print(f"ALL PASS: valid={len(selected)} invalid={len(invalid_tests)}")


if __name__ == "__main__":
    main()
