# IndexAdd V3

## Status

V3 is the first submission candidate that does not match the published case's exact shape. V2 remains archived for comparison but is not valid for submission.

## General task decomposition

The kernel flattens every shape around `dim` into `outer`, `selfDim`, and `inner`. A task owns one `(outer, destination row, inner tile)` region, where the inner tile contains at most 8192 elements. Tasks are distributed over at most 40 Vector Cores.

Because only one task can write a given output-row tile, repeated indices remain ordered and cannot race across cores. Every path is selected only by dtype and UB capacity.

## Dtype handling

- float32, float16, int32: native UB vector addition.
- bfloat16: convert each source tile to float, add, then round back to bfloat16 after every update to preserve sequential rounding.
- int8: convert exactly to int16 through half, add modulo `2^16`, then write the low byte back to obtain PyTorch-compatible modulo-`2^8` behavior.

## Validation on Ascend 910B4 / CANN 8.5

- Build: passed.
- Published accuracy case: passed.
- Published case, 30 calls: median 9.2805 us; min 9.00 us; mean 9.679 us; max 20.04 us; block dim 32.
- Extended regression: 10/10 passed, including strong int8 overflow on the published geometry, `M=8000`, all five dtypes, repeated indices, non-aligned inner lengths, negative dim, empty index, multiple outer groups, and `inner=9001` multi-tile execution.

The published median is only a local microbenchmark. The competition score is the sum across hidden validation cases and can only be measured by uploading this package to the official judge.
