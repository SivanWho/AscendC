import torch
import torch_npu

import custom_ops_lib


# One input, one outer row and a segment no larger than 64 KiB select the
# single-core, single-buffer path.  The official extension invokes Concat 30
# times, giving 30 AICore samples for each payload size in one profiler run.
PAYLOAD_BYTES = (32, 128, 512, 2048, 8192, 32768, 65536)


# Use a shape that is not part of the measurement matrix, so its profiler rows
# can be removed unambiguously.  This avoids attributing device frequency
# ramp-up and first-launch effects to the 32-byte transfer.
warmup_elements = 96 // 2
warmup_cpu = torch.arange(warmup_elements, dtype=torch.float16).reshape(
    1, warmup_elements
)
warmup_npu = warmup_cpu.npu()
warmup_actual = custom_ops_lib.custom_op(
    [warmup_npu], 1, [1, warmup_elements]
)
torch.npu.synchronize()
torch.testing.assert_close(warmup_actual.cpu(), warmup_cpu, rtol=0, atol=0)
print("PASS warmup_payload_bytes=96")


for payload_bytes in PAYLOAD_BYTES:
    elements = payload_bytes // 2
    cpu_input = torch.arange(elements, dtype=torch.float16).reshape(1, elements)
    npu_input = cpu_input.npu()
    actual = custom_ops_lib.custom_op([npu_input], 1, [1, elements])
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), cpu_input, rtol=0, atol=0)
    print(f"PASS payload_bytes={payload_bytes} shape=(1,{elements})")
