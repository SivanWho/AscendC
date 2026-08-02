import random

import torch
import torch_npu

import custom_ops_lib


SEED = 20260730
random.seed(SEED)
torch.manual_seed(SEED)

DTYPES = (torch.float16, torch.float32, torch.int32, torch.int8)


def make_tensor(shape, dtype):
    if dtype == torch.int8:
        return torch.randint(-128, 128, shape, dtype=dtype)
    if dtype == torch.int32:
        return torch.randint(-(2**30), 2**30, shape, dtype=dtype)
    return torch.randn(shape, dtype=dtype)


def run_one(index):
    rank = random.randint(1, 6)
    dim = random.randrange(rank)
    dtype = random.choice(DTYPES)
    input_count = random.randint(2, 16)

    base = [random.randint(1, 8) for _ in range(rank)]
    if index % 10 == 0:
        # Force a long, non-aligned tile chain without matching a published case.
        rank = 2
        dim = 0
        base = [1, 32769 + index * 257]
        input_count = random.randint(2, 5)

    shapes = []
    for input_id in range(input_count):
        shape = list(base)
        shape[dim] = random.choice((0, 1, 2, 3, 7, 17))
        if index % 10 == 0:
            shape[dim] = random.randint(1, 8)
        shapes.append(tuple(shape))

    if all(shape[dim] == 0 for shape in shapes):
        shape = list(shapes[-1])
        shape[dim] = 1
        shapes[-1] = tuple(shape)

    cpu_inputs = [make_tensor(shape, dtype) for shape in shapes]
    expected = torch.cat(cpu_inputs, dim=dim)
    npu_inputs = [tensor.npu() for tensor in cpu_inputs]
    actual = custom_ops_lib.custom_op(
        npu_inputs, dim, list(expected.shape)
    ).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(
        f"PASS {index:03d}: dtype={dtype} rank={rank} dim={dim} "
        f"inputs={input_count} output={tuple(expected.shape)}"
    )


for case_id in range(50):
    run_one(case_id)

print(f"ALL PASS: 50/50 seed={SEED}")
