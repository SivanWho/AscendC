# S9 four-operator validation — 2026-07-23

## Environment

- Device: Ascend 910B4
- Driver/NPU SMI: 25.5.1
- CANN: 8.5.0
- Packaging: organizer-provided Linux `zip_op.sh`
- Packaged run filename: `custom_opp_ubuntu_aarch64.run`

## Changes

- IndexAdd
  - Preserved the official `aclnnIndexAdd(self, dim, index, source, alpha, out)`
    ABI adapter.
  - Added a safe GM-index fallback when the index count exceeds the 8000-entry
    UB buffer.
  - Avoided division by zero for empty dimensions.
- Concat
  - Limited 2-D DMA `blockCount` to 4095.
  - Calculated UB capacity with each row padded to a 32-byte boundary.
  - Added a fallback that reads dynamic input descriptors for input lists longer
    than 128, while retaining the compact 128-entry tiling layout for the normal
    fast path.
  - Guarded the 2-D DMA destination stride and retained the row-wise fallback.
- Greater
  - Added rank-0 scalar support while preserving the aligned FP16 vector path.
- SquareSumV1
  - Zero-filled padded FP16 rows before reduction.
  - Disabled the vector fast path for zero-length reduced/output dimensions.
  - Added rank-0 scalar support.
  - Added an ACLNN adapter that normalizes an empty `axis` list to all input
    dimensions.

## Correctness

- IndexAdd: 7/7 regression cases passed.
- Concat: 8/8 regression cases passed, including 8192/65536-row narrow
  segments, 129 dynamic inputs, empty dimensions, and the over-UB fallback.
- Greater: 10/10 regression cases passed, covering five dtypes, broadcasting,
  scalar inputs, and an empty output.
- SquareSumV1: 12/12 regression cases passed, covering fast and fallback paths,
  empty axis, duplicate axes, zero dimensions, BF16, and scalar input.
- All four final ZIP archives were freshly extracted and installed.
- The untouched official Case1 passed from every extracted archive.

## Official profiling

The official `run.sh`/`msprof` flow passed accuracy and performance:

| Operator | Median AICore time |
|---|---:|
| IndexAdd | 9.88 us |
| Concat | 8.68 us |
| Greater | 3.04 us |
| SquareSumV1 | 5.45 us |

The profiled Concat `.run` and the final archive's renamed `.run` have the same
SHA-256:

`a30ce02b8d277c2de54f9245b29e3ba5bb8600af65aa243dc8da493b6f394ff5`

## Final archives

| Archive | SHA-256 |
|---|---|
| `IndexAdd.zip` | `9ee5d263b0c1a0d67ff928c3ec0201334d33fddff7160b2a2a694ead184611f8` |
| `Concat.zip` | `8170620dc525c1abeb0c4e22664b1e651c6f3e9a014132c2244d8ea271daadcc` |
| `Greater.zip` | `635b550881441013d1d850d34ebddc76222e4d3b3eb5969af7266e73b548d77b` |
| `SquareSumV1.zip` | `50be7ca16e037f4f181784f9298f9d38805e2343c88de5203d0856d6c9c3ddb6` |

Each archive has Unix ZIP metadata, explicit root/`op_host`/`op_kernel`
directory entries, forward-slash paths, exactly one executable
`custom_opp_ubuntu_aarch64.run`, and the source files used for that build.
