#!/usr/bin/env bash

project_root="${1:-/home/ma-user/work/s9_concat_virtual2d_20260803}"
install_root="${2:-${project_root}/install_submit_20260803_0250}"
result_root="${3:-${project_root}/validation_submit_20260803_0250}"

source /home/ma-user/Ascend/cann-8.5.0/set_env.sh
source "${install_root}/vendors/customize/bin/set_env.bash"
set -u
mkdir -p "${result_root}"

overall=0
for case_id in 1 2 3 4 5; do
    echo "=== CASE ${case_id} ==="
    (
        cd "${project_root}/tests_official"
        bash run.sh "${case_id}"
    ) 2>&1 | tee "${result_root}/case${case_id}.log"
    case_rc=${PIPESTATUS[0]}
    echo "CASE ${case_id} RC=${case_rc}"
    if [[ ${case_rc} -ne 0 ]]; then
        overall=1
    fi
done

exit "${overall}"
