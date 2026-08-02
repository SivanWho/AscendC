#!/bin/bash
set -eo pipefail

version=$1
install_root=$2
shift 2

source /home/ma-user/Ascend/cann-8.5.0/set_env.sh
source "${install_root}/vendors/customize/bin/set_env.bash"
cd /home/ma-user/work/s9_concat_db_20260730/tests

profile_root=/home/ma-user/work/s9_concat_db_20260730/profiles
mkdir -p "${profile_root}"

for case_name in "$@"; do
    output_dir="${profile_root}/${version}_${case_name}"
    rm -rf "${output_dir}"
    msprof \
        --output="${output_dir}" \
        --application="python3 concat_profile_matrix.py ${case_name}" \
        >"/tmp/concat_${version}_${case_name}.log" 2>&1
    grep -E "PASS case=|ERROR|Traceback" \
        "/tmp/concat_${version}_${case_name}.log"
done
