#!/usr/bin/env bash
set -euo pipefail

test_dir=/home/ma-user/work/s9_delivery_20260723/final_profiles_v2/Concat
install_dir="${CONCAT_INSTALL_DIR:?set CONCAT_INSTALL_DIR}"
vendor_dir="$install_dir/vendors/customize"

export LD_LIBRARY_PATH="$vendor_dir/op_api/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_CUSTOM_OPP_PATH="$vendor_dir"
export PYTHONPATH="$test_dir/build/lib.linux-aarch64-cpython-39:${PYTHONPATH:-}"

cd "$test_dir"
exec python3 concat_profile.py large_segment --warmup 5 --repeat 30
