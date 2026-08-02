# Concat Hybrid V2 submission

- Built: 2026-08-01
- Target: Ascend 910B / EulerOS aarch64
- CANN: 8.5.0
- Archive: `Concat.zip`
- Archive SHA-256: `8c81a11c9d342576cd341d4ecd9d3f29e45ac26b4607b092f492775652f9615d`
- Run package: `custom_opp_euleros_aarch64.run`
- Run package SHA-256: `42dacb9977f6d8f80b7f4c1262f42e042ac2382d3502052f74c4637b639d7a47`
- `op_host/concat.cpp` SHA-256: `50455a040cdbf33f431ccdd8ba8b7176e84ef8f2a98a7e4a197f932205ef285f`
- `op_kernel/concat.cpp` SHA-256: `d8336cb51d4b694879dd795f4a61f76e5f8d7da438e307ffd30be30920303215`

The archive was created on Linux with the organizer-provided `zip_op.sh`.
It contains exactly nine entries under `Concat_zip/`, uses forward-slash entry
names, stores Unix-origin metadata, and preserves executable mode `0750` on the
`.run` package.

Fresh-extraction validation:

- `.run` integrity check: passed
- Fresh install: passed
- Official `run.sh 1` accuracy: passed
- Official profiling result: `8.6095 us`

