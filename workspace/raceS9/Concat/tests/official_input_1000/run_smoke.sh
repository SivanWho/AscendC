#!/usr/bin/env bash
set -eo pipefail
export CC=/home/ma-user/gcc/bin/gcc
export CXX=/home/ma-user/gcc/bin/g++
source /home/ma-user/Ascend/cann-8.5.0/set_env.sh
source /home/ma-user/work/s9_concat_db_20260730/install_1000_validation/vendors/customize/bin/set_env.bash
cd /home/ma-user/work/s9_concat_db_20260730/official_input_1000
rm -rf build concat_direct_test_ops*.so
python3 setup.py build_ext --inplace
python3 run_1000.py --start 0 --stop 25 --results results/smoke
