import argparse

import torch
import torch_npu

import custom_ops_lib


def make_case(name):
    torch.manual_seed(20260730)
    if name == "official":
        source = torch.randn(128, 256, dtype=torch.float16)
        return list(torch.split(source, [27, 40, 63, 24, 50, 26, 19, 2, 5], -1)), -1
    if name == "tile16":
        return [torch.randn(64, 8192, dtype=torch.float16), torch.randn(64, 8201, dtype=torch.float16)], 1
    if name == "tile32":
        return [torch.randn(32, 16384, dtype=torch.float16), torch.randn(32, 16393, dtype=torch.float16)], 1
    if name == "tile64":
        return [torch.randn(16, 32768, dtype=torch.float16), torch.randn(16, 32777, dtype=torch.float16)], 1
    if name == "over64":
        return [torch.randn(16, 65537, dtype=torch.float16), torch.randn(16, 65539, dtype=torch.float16)], 1
    if name == "dim0_large":
        return [torch.randn(4, 262144, dtype=torch.float16), torch.randn(5, 262144, dtype=torch.float16)], 0
    if name == "dim0_unaligned":
        return [torch.randn(4, 262145, dtype=torch.float16), torch.randn(5, 262145, dtype=torch.float16)], 0
    if name == "dim0_fp32":
        return [torch.randn(4, 131072, dtype=torch.float32), torch.randn(5, 131072, dtype=torch.float32)], 0
    if name == "dim0_int8":
        return [
            torch.randint(-128, 128, (4, 524289), dtype=torch.int8),
            torch.randint(-128, 128, (5, 524289), dtype=torch.int8),
        ], 0
    if name == "segment128k":
        return [torch.randn(1, 65536, dtype=torch.float16), torch.randn(1, 65536, dtype=torch.float16)], 0
    if name == "segment256k":
        return [torch.randn(1, 131072, dtype=torch.float16), torch.randn(1, 131072, dtype=torch.float16)], 0
    if name == "segment512k":
        return [torch.randn(1, 262144, dtype=torch.float16), torch.randn(1, 262144, dtype=torch.float16)], 0
    if name == "many64":
        return [torch.randn(128, 1 + index % 3, dtype=torch.float16) for index in range(64)], 1
    if name == "many128":
        return [torch.randn(64, 1 + index % 3, dtype=torch.float16) for index in range(128)], 1
    raise ValueError(name)


parser = argparse.ArgumentParser()
parser.add_argument(
    "case",
    choices=(
        "official", "tile16", "tile32", "tile64", "over64",
        "dim0_large", "dim0_unaligned", "dim0_fp32", "dim0_int8",
        "segment128k", "segment256k", "segment512k",
        "many64", "many128",
    ),
)
args = parser.parse_args()

cpu_inputs, dim = make_case(args.case)
expected = torch.cat(cpu_inputs, dim=dim)
npu_inputs = [tensor.npu() for tensor in cpu_inputs]
actual = custom_ops_lib.custom_op(npu_inputs, dim, list(expected.shape))
torch.npu.synchronize()
torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
print(
    f"PASS case={args.case} inputs={len(cpu_inputs)} "
    f"shape={tuple(expected.shape)} min_gm_bytes={2 * expected.numel() * expected.element_size()}"
)
