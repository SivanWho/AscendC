#!/usr/bin/env bash
set -euo pipefail

root=/home/ma-user/work/s9_delivery_20260722
install="${SQUARE_INSTALL:-$root/installs_v2/optimized_SquareSumV1}"
test_dir="$root/tests_v2/SquareSumV1"
cd "$test_dir"
python3 -m pip install --force-reinstall --no-deps dist/custom_ops*.whl \
  > "$root/regression_extension_install_SquareSumV1.log" 2>&1
api_lib=$(find "$install" -type f -name libcust_opapi.so -printf '%h\n' | head -n 1)
test -n "$api_lib"
vendor_dir=$(dirname "$(dirname "$api_lib")")
export LD_LIBRARY_PATH="$api_lib:${LD_LIBRARY_PATH:-}"
export ASCEND_CUSTOM_OPP_PATH="$vendor_dir"

indices=(0 1 2 3 4 5 6 7 8)
if (( $# > 0 )); then
  indices=("$@")
fi
for index in "${indices[@]}"; do
  timeout 180 python3 square_sum_regression.py "$index"
done

echo 'ALL SQUARESUM REGRESSIONS PASSED'
