#!/usr/bin/env bash
set -eo pipefail

SOURCE_ROOT=/home/ma-user/work/s9_concat_db_20260730/db64
INSTALL_ROOT=/home/ma-user/work/s9_concat_db_20260730/install_1000_validation
CANN_ENV=/home/ma-user/Ascend/cann-8.5.0/set_env.sh
TEST_ROOT=/home/ma-user/work/s9_concat_db_20260730/official_input_1000

before_host=$(sha256sum "$SOURCE_ROOT/op_host/concat.cpp" | awk '{print $1}')
before_kernel=$(sha256sum "$SOURCE_ROOT/op_kernel/concat.cpp" | awk '{print $1}')

source "$CANN_ENV"
cd "$SOURCE_ROOT"
./build.sh

after_host=$(sha256sum "$SOURCE_ROOT/op_host/concat.cpp" | awk '{print $1}')
after_kernel=$(sha256sum "$SOURCE_ROOT/op_kernel/concat.cpp" | awk '{print $1}')
test "$before_host" = "$after_host"
test "$before_kernel" = "$after_kernel"

rm -rf "$INSTALL_ROOT"
RUN_PACKAGE=$(find "$SOURCE_ROOT/build_out" -maxdepth 1 -type f -name 'custom_opp_*_aarch64.run' -print -quit)
test -n "$RUN_PACKAGE"
"$RUN_PACKAGE" --quiet --install-path="$INSTALL_ROOT"

source "$INSTALL_ROOT/vendors/customize/bin/set_env.bash"
cd "$TEST_ROOT"
rm -rf build concat_direct_test_ops*.so
python3 setup.py build_ext --inplace
python3 run_1000.py --start 0 --stop 25 --results results/smoke

printf 'CORE_SHA256 op_host=%s op_kernel=%s\n' "$after_host" "$after_kernel"
