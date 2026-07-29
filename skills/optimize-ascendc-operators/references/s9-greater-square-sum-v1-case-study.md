# S9 Greater and SquareSumV1 case study

## Environment and harness

- Ascend 910B4, CANN 8.5.0.
- The official PyTorch extension repeats the custom ACLNN call 30 times and reports the median of calls 11–30.
- Compile the PyTorch extension with GCC 10.3 in this image. GCC 7 is too old for PyTorch 2.5, while a Bisheng-built bridge produced `Unknown layout` because of C++ ABI incompatibility.

## Greater

The correctness baseline performed scalar `GetValue`/`SetValue` and used one core after proving that cross-core strided bool writes corrupt neighboring bytes. Its official steady-state median was about 105.244 us and profiler scalar ratio was about 97.5%.

The optimized path is selected only when:

- both FP16 inputs have the same contiguous shape;
- the logical element count and each per-core range are multiples of 256 elements;
- each core owns one contiguous 32-byte-aligned output interval.

Load both inputs into UB, call `Compare(..., CMPMODE::GT, ...)`, expand the predicate mask with `Select` into FP16 0/1 values, cast to `int8_t`, then DMA the contiguous output to GM. Keep the original broadcast-aware scalar kernel as the fallback.

For the public `[32,64]` case this used 8 AIV blocks. The final submission ZIP, after fresh extraction and installation, produced an official median of 2.981 us, with profiler min/mean/max 2.760/3.391/6.000 us: about 35.30x faster than the archived baseline. Special values, random FP16, FP32/BF16/INT32/INT8, and broadcast regressions passed.

## SquareSumV1

The correctness baseline fused square and sum but performed scalar GM access and scalar accumulation. Its official steady-state median was about 137.565 us and profiler scalar ratio was about 99.5%.

The optimized path is selected by general properties:

- FP16 input;
- exactly one reduced axis and it is the contiguous last axis;
- padded last-axis width is at most 64 elements;
- row count is at most the vector repeat limit (255);
- all padded rows and temporary FP32 data fit UB.

Use one 2-D `DataCopyPad` to load all logical rows and zero-pad each row to a 32-byte boundary. Cast the padded tensor to FP32, square it with vector `Mul`, and call `WholeReduceSum<float, true>` with `srcRepStride = padded_cols * sizeof(float) / 32`. Cast the contiguous row sums back to FP16 and use `DataCopyPad` to write only the logical output bytes. Keep the generic multi-axis/other-dtype kernel as fallback.

For the public `[123,31]`, `axis=-1` case the final submission ZIP, after fresh extraction and installation, produced an official median of 5.641 us, with profiler min/mean/max 5.100/5.652/7.601 us: about 24.39x faster than the archived baseline. Additional FP16 fast-path geometries (`7x17`, `5x64`, `255x1`, scalar output), plus FP32/BF16 and multi-axis fallbacks, passed.

## Reusable lessons

1. A compile-successful scalar GM kernel is useful only as a correctness oracle; profiler scalar ratios near 100% are a direct signal to move the whole tile into UB.
2. Byte/half scalar writes from multiple cores can be wrong even when every logical index is unique. Alignment and physical write granularity matter.
3. Predicate-producing vector instructions may use packed masks. Always confirm the output representation before copying it as a tensor.
4. For short ragged rows, one padded 2-D DMA plus a strided whole reduction is substantially cheaper than one scalar or vector launch per row.
5. Select fast paths from dtype, contiguity, alignment, repeat limits, and UB capacity; never match a published shape literal.
