#!/usr/bin/env python3
"""Targeted correctness/performance cases for whole-row Virtual2D."""

import argparse

import torch
import torch_npu

import custom_ops_lib


def make_case(name):
    torch.manual_seed(20260803)
    if name == "whole_32x64":
        return [torch.randn(2048, 32, dtype=torch.float16) for _ in range(32)], -1
    if name == "whole_64x64":
        return [torch.randn(512, 32, dtype=torch.float16) for _ in range(64)], -1
    if name == "whole_128x32":
        return [torch.randn(512, 16, dtype=torch.float16) for _ in range(128)], -1
    if name == "whole_32x256":
        return [torch.randn(512, 128, dtype=torch.float16) for _ in range(32)], -1
    if name == "whole_48_mixed":
        widths = [16, 32, 64] * 16
        return [torch.randn(512, width, dtype=torch.float16) for width in widths], -1
    if name == "whole_below1m":
        return [torch.randn(128, 16, dtype=torch.float16) for _ in range(64)], -1
    raise ValueError(name)


parser = argparse.ArgumentParser()
parser.add_argument(
    "case",
    choices=(
        "whole_32x64",
        "whole_64x64",
        "whole_128x32",
        "whole_32x256",
        "whole_48_mixed",
        "whole_below1m",
    ),
)
parser.add_argument("--rounds", type=int, default=1)
args = parser.parse_args()

cpu_inputs, dim = make_case(args.case)
expected = torch.cat(cpu_inputs, dim=dim)
npu_inputs = [tensor.npu() for tensor in cpu_inputs]
actual = None
for _ in range(args.rounds):
    actual = custom_ops_lib.custom_op(npu_inputs, dim, list(expected.shape))
torch.npu.synchronize()
torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
print(
    f"PASS case={args.case} rounds={args.rounds} inputs={len(cpu_inputs)} "
    f"shape={tuple(expected.shape)} min_gm_bytes={2 * expected.numel() * expected.element_size()}"
)
