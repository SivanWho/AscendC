#!/usr/bin/env bash
set -u

test_dir="${SQUARE_TEST_DIR:-/home/ma-user/work/s9_delivery_20260723/final_profiles_v2/SquareSumV1}"
install_dir="${SQUARE_INSTALL_DIR:-/home/ma-user/work/s9_delivery_20260723/validation/install_SquareSumV1}"
api_lib="$install_dir/vendors/customize/op_api/lib"
vendor_dir="$install_dir/vendors/customize"

source /home/ma-user/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1
export LD_LIBRARY_PATH="$api_lib:${LD_LIBRARY_PATH:-}"
export ASCEND_CUSTOM_OPP_PATH="$vendor_dir"
export PYTHONPATH="$test_dir/build/lib.linux-aarch64-cpython-39:${PYTHONPATH:-}"

cd "$test_dir"
python3 - <<'PY'
import ctypes
import torch
import torch_npu

library = ctypes.CDLL("libcust_opapi.so")
print("CUSTOM_LIBRARY", library._name)
print("SQUARE_SYMBOL", library.aclnnSquareSumV1)
PY

indices=("$@")
if (( ${#indices[@]} == 0 )); then
  indices=(0 1 2 3 7 8 9 10 11 12 13 14 15 16 17 18 19)
fi

if [[ "${GROUPED:-0}" == "1" ]]; then
  timeout 300 python3 square_sum_stress.py "${indices[@]}"
  exit $?
fi

for index in "${indices[@]}"; do
  log="stress_isolated_${index}.log"
  echo "===== CASE $index ====="
  timeout 180 python3 square_sum_stress.py "$index" >"$log" 2>&1
  rc=$?
  echo "RC=$rc"
  grep -E '^(PASS|Traceback|RuntimeError|AssertionError|EZ9999|E[0-9]{5})' "$log" | tail -20 || true
done
