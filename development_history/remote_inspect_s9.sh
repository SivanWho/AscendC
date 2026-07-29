#!/usr/bin/env bash
set -euo pipefail

echo '===== CANN'
command -v ccec || true
ccec --version 2>/dev/null | head -n 3 || true

echo '===== reduction APIs/examples'
sed -n '190,255p' /home/ma-user/Ascend/cann-8.5.0/include/ascendc/highlevel_api/lib/reduce/reduce.h
sed -n '210,290p' /home/ma-user/Ascend/cann-8.5.0/include/ascendc/basic_api/interface/kernel_operator_vec_reduce_intf.h
echo '===== ReduceSum implementation'
grep -n -A90 -B15 'inline void ReduceSum.*count' \
  /home/ma-user/Ascend/cann-8.5.0/include/ascendc/basic_api/impl/basic_api/kernel_operator_vec_reduce_intf_impl.h | head -n 160 || true

echo '===== remaining build helpers'
find /home/ma-user/work/s9_remaining_v1 -maxdepth 6 -type f \
  \( -name 'build.sh' -o -name 'run.sh' -o -name 'package*.sh' -o -name '*.run' \) \
  -print | sort

echo '===== concat build helpers'
find /home/ma-user/work/s9_concat_submit_build -maxdepth 6 -type f \
  \( -name 'build.sh' -o -name 'run.sh' -o -name 'package*.sh' -o -name '*.run' \) \
  -print | sort
