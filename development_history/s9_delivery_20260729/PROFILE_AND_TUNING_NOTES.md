# S9 AscendC profiling and tuning notes (2026-07-29)

## Environment isolation

This ModelArts instance contains two CANN installations. Always initialize the
competition environment explicitly:

```bash
source /home/ma-user/Ascend/cann-8.5.0/set_env.sh
```

Do not use `/usr/local/Ascend/ascend-toolkit/latest`; it points to CANN
7.0.RC1 on this instance. A build or test that mixes headers, Python packages,
operator libraries, or `opc` from the two installations is not valid.

When several custom operators export similar ACLNN symbols, isolate every test:

```bash
export ASCEND_CUSTOM_OPP_PATH="$install_root/vendors/customize"
export LD_LIBRARY_PATH="$install_root/vendors/customize/op_api/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$extension_build_dir:$PYTHONPATH"
```

Verify the loaded custom library and exported symbol before trusting a result:

```python
import ctypes
library = ctypes.CDLL("libcust_opapi.so")
print(library._name)
print(library.aclnnIndexAddGetWorkspaceSize)
```

## Tools available in the current CANN 8.5 environment

- System/application profile: `$ASCEND_HOME_PATH/bin/msprof`
- Operator analysis: `$ASCEND_HOME_PATH/tools/msopt/bin/msopprof`, also exposed
  through `msprof op`
- Memory, race, initialization, and synchronization checks:
  `$ASCEND_HOME_PATH/tools/mssanitizer/bin/mssanitizer`
- On-device kernel debugger: `$ASCEND_HOME_PATH/bin/msdebug`

Minimal repeatable application profile:

```bash
msprof \
  --output=/absolute/output/path \
  --application=/absolute/path/to/profile_wrapper.sh \
  --aic-metrics=PipeUtilization
```

Use a wrapper with no command-line arguments for `--application`. This avoids
shell quoting errors and lets the wrapper fix the CANN, custom OPP,
`LD_LIBRARY_PATH`, and Python extension environment.

Useful operator-level collection:

```bash
msprof op \
  --application=/absolute/path/to/profile_wrapper.sh \
  --kernel-name=concat \
  --aic-metrics=PipeUtilization,Memory,MemoryUB,ResourceConflictRatio,Roofline \
  --warm-up=10 \
  --launch-count=30 \
  --output=/absolute/output/path
```

Run correctness tools before interpreting performance:

```bash
mssanitizer --tool=memcheck --kernel-name=concat ./test_program
mssanitizer --tool=synccheck --kernel-name=concat ./test_program
```

`msdebug` requires the platform's debug prerequisites and is intended for an
interactive crash investigation; it is not the first profiling step.

## Findings from this iteration

### SquareSumV1

The suffix vector-reduction path issued `ReduceSum` on PIPE_V and immediately
read the result with scalar `GetValue`. `PipeBarrier<PIPE_V>` does not establish
a Vector-to-Scalar dependency. The corrected build uses a `HardEvent::V_S`
`SetFlag`/`WaitFlag` pair before the scalar read.

The first diagnostic build disabled the vector reductions. That isolated the
path, but a large fp32 reduce-all case exposed unacceptable serial-accumulation
error (about 0.4%), so that diagnostic build must not be submitted. The
recommended synced V3 restores vector paths with the explicit dependency and
passes all 20 stress cases, including the three `(2024, 3000)` large cases,
rank-8, empty-axis, BF16, non-contiguous-axis, and 4095/4096/4097 boundaries.

Rule: use path disabling only to isolate a fault. Re-run both boundary and
large-value accuracy tests before treating a fallback as shippable.

### Concat

The normal public geometry is scalar/control heavy. The more important general
defect was low parallelism when one contiguous input segment exceeded UB: the
old kernel assigned one entire segment to one core.

The new tiling splits only over-UB segments into independent contiguous DMA
tasks. In the synthetic `(2, 40000) + (1, 40000)`, dim-0 fp16 test:

| version | block dim | median AICore time |
|---|---:|---:|
| previous | 2 | 6.54 us |
| segment-split V2 | 6 | 4.88 us |

That is a 25.4% reduction for the affected class. The <=UB path and all dtype,
rank, negative-axis, empty-input, and shape rules are unchanged.

### IndexAdd

The package name and ACLNN compatibility wrapper are functional. The old
output-centric kernel assigns every destination row a task and scans the whole
index in every task. Its work grows approximately with:

```text
outer * selfDim * ceil(inner / tile) * indexCount
```

This becomes pathological when the indexed dimension is large or is the last
axis. Judge timeout/abnormal termination can therefore appear as `Run failed`
even though the small public int8 shape passes.

The safe version processes source elements in index order on one core. This
removes duplicate-index races and changes the dominant work to approximately:

```text
selfCount + sourceCount
```

BF16 retains a UB vector path because direct scalar BF16 arithmetic crashes the
CANN 8.5 compiler's infer-channel stage. The build therefore uses a hybrid
implementation, not a single generic template.

## Submission discipline

1. Build with CANN 8.5.
2. Install the newly generated `.run` into a fresh directory.
3. Test with an explicitly isolated custom OPP and extension.
4. Confirm exported ACLNN symbols with `nm -D`.
5. Package the exact matching `op_host`, `op_kernel`, and generated `.run`.
6. Use the organizer-provided `zip_op.sh`.
7. List the archive and record SHA-256 before upload.
