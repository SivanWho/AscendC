import argparse
import time

import torch
import torch_npu
import custom_ops_lib


def synchronize():
    torch.npu.synchronize()


def make_case(name):
    torch.manual_seed(20260729)
    if name == "official":
        source = torch.randn(128, 256, dtype=torch.float16)
        sizes = [27, 40, 63, 24, 50, 26, 19, 2, 5]
        return list(torch.split(source, sizes, -1)), -1
    if name == "large_segment":
        return [
            torch.randn(2, 40000, dtype=torch.float16),
            torch.randn(1, 40000, dtype=torch.float16),
        ], 0
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("official", "large_segment"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    cpu_inputs, dim = make_case(args.case)
    expected = torch.cat(cpu_inputs, dim=dim)
    inputs = [item.npu() for item in cpu_inputs]
    output_shape = list(expected.shape)

    for _ in range(args.warmup):
        output = custom_ops_lib.custom_op(inputs, dim, output_shape)
    synchronize()

    start = time.perf_counter()
    for _ in range(args.repeat):
        output = custom_ops_lib.custom_op(inputs, dim, output_shape)
    synchronize()
    elapsed_us = (time.perf_counter() - start) * 1.0e6 / args.repeat

    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    print(
        f"PASS case={args.case} repeats={args.repeat} "
        f"host_average_us={elapsed_us:.3f}"
    )


if __name__ == "__main__":
    main()
