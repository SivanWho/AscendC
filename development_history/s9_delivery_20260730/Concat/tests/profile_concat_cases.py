import argparse

import torch
import torch_npu

import custom_ops_lib


def make_case(name):
    torch.manual_seed(20260730)
    if name == "official":
        source = torch.randn(128, 256, dtype=torch.float16)
        return list(
            torch.split(source, [27, 40, 63, 24, 50, 26, 19, 2, 5], -1)
        ), -1
    if name == "many129":
        return [
            torch.randn(128, index % 4, dtype=torch.float16)
            for index in range(129)
        ], 1
    if name == "many256":
        return [
            torch.randn(128, 1 + index % 3, dtype=torch.float16)
            for index in range(256)
        ], 1
    raise ValueError(name)


parser = argparse.ArgumentParser()
parser.add_argument("case", choices=("official", "many129", "many256"))
parser.add_argument("--rounds", type=int, default=30)
args = parser.parse_args()

cpu_inputs, dim = make_case(args.case)
expected = torch.cat(cpu_inputs, dim=dim)
npu_inputs = [tensor.npu() for tensor in cpu_inputs]

result = None
for _ in range(args.rounds):
    result = custom_ops_lib.custom_op(
        npu_inputs, dim, list(expected.shape)
    )
torch.npu.synchronize()
torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
print(
    f"PASS case={args.case} rounds={args.rounds} "
    f"inputs={len(cpu_inputs)} shape={tuple(expected.shape)}"
)
