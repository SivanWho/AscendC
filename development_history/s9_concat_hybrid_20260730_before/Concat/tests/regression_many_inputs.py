import torch
import torch_npu

import custom_ops_lib


def run_case(name, input_count, rows, widths, dtype):
    torch.manual_seed(20260730 + input_count)
    tensors = []
    for index in range(input_count):
        width = widths[index % len(widths)]
        if dtype == torch.int8:
            tensor = torch.randint(-128, 128, (rows, width), dtype=dtype)
        else:
            tensor = torch.randn(rows, width, dtype=dtype)
        tensors.append(tensor)

    expected = torch.cat(tensors, dim=1)
    npu_inputs = [tensor.npu() for tensor in tensors]
    actual = custom_ops_lib.custom_op(
        npu_inputs, 1, list(expected.shape)
    ).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(
        f"PASS {name}: inputs={input_count}, shape={tuple(expected.shape)}, "
        f"dtype={dtype}"
    )


run_case("prefix_table_129_fp16", 129, 128, [0, 1, 2, 3], torch.float16)
run_case("prefix_table_256_int8", 256, 64, [0, 1, 3, 5], torch.int8)
print("ALL PASS: 2/2 (ACLNN dynamic-input contract limit: 256)")
