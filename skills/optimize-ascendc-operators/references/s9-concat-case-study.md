# S9 Concat case study

## Contract

- Semantics: `torch.cat(inputs, dim)` with a dynamic input list.
- Supported types: float32, float16, int32, and int8.
- Published case: nine float16 slices of a `[128,256]` tensor, concatenated on the last axis; zero-length slices are permitted by the generator.
- Official extension: `aclnnConcat(inputs, dim, result)` repeated 30 times under `msprof`.

## Invalid synchronization lesson

The first copy kernel reused one UB buffer after `DataCopyPad` but only issued same-pipeline barriers. MTE2 and MTE3 are different engines, so the kernel could consume stale UB contents and failed accuracy. Correctness required explicit `MTE2_MTE3` before the store and `MTE3_MTE2` before reusing the buffer. The corrected per-segment design measured 17.4705 us median.

## General optimization

Treat concatenation as `outer × axis × inner`. For each input, `axis_size × inner` elements form one contiguous segment per outer row. Batch several rows into one task when the compact UB payload fits:

```text
segment_bytes = axis_size * inner * element_bytes
rows_for_ub = floor(tile_bytes / max_input_segment_bytes)
chunks_per_input = ceil(vector_cores / input_count)
rows_for_parallelism = ceil(outer / chunks_per_input)
rows_per_task = clamp(min(rows_for_ub, rows_for_parallelism), 1, 65535)
```

Use a 2-D GM-to-UB copy with contiguous source rows, then a 2-D UB-to-GM copy whose destination stride skips the other inputs' output segments. If one segment exceeds the UB tile, fall back to row-wise tiled copies.

On CANN 8.5 / Ascend 910B4, the final fresh-install package measured 10.17 us median on the unchanged official case, a 41.8% reduction from the synchronization-correct V2. It passed exact comparisons for all four dtypes, first/middle/last and negative dimensions, non-aligned segments, empty slices, 64 inputs, and the over-UB fallback.

## S7-informed V4: size-aware parallelism

The public case moves only 64 KiB of input but V3 created 45 tasks to cover 40 AIVs. Following the general data-size thresholding pattern used in the `ascend-124/s7` repository, V4 estimates target parallelism at roughly 16 KiB of input per target core, then uses that estimate to limit row chunks per input. The public geometry naturally becomes nine tasks—one per input—while larger tensors still create more chunks and over-UB segments retain the tiled fallback.

Three independent official runs reported 8.70, 8.92, and 8.84 us; their median is 8.84 us, about 13.1% faster than V3. Reusing allocated MTE2/MTE3 event IDs, forcing Block Dim down to four, and a single-batch division shortcut were all measured and rejected because they did not improve the official median.

## ABI note

CANN already defines an operator named `Concat`. Registering a custom dynamic-input operator under that internal name can collide with built-in metadata. Register an internal name such as `ConcatCustom`, let opbuild generate `aclnnConcatCustom`, and expose compatibility wrappers for both `aclnnConcatGetWorkspaceSize` and `aclnnConcat` so the untouched official extension finds the standard symbols.
