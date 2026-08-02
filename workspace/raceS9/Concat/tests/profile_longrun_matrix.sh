#!/usr/bin/env bash

set -e

if [[ $# -lt 4 ]]; then
    echo "usage: $0 <project-root> <install-root> <label> <rounds>" >&2
    exit 2
fi

project_root=$1
install_root=$2
label=$3
rounds=$4
profile_root="${project_root}/profiles/longrun_20260803"

source /home/ma-user/Ascend/cann-8.5.0/set_env.sh
source "${install_root}/vendors/customize/bin/set_env.bash"
export LD_LIBRARY_PATH="${install_root}/vendors/customize/op_api/lib:${LD_LIBRARY_PATH}"
mkdir -p "${profile_root}"
cd "${project_root}/profile_v2d"

if [[ $# -gt 4 ]]; then
    cases=("${@:5}")
else
    cases=(
        official
        many64
        many128
        tile16
        tile32
        tile64
        over64
        dim0_large
        dim0_unaligned
        dim0_fp32
        dim0_int8
        segment128k
        segment256k
        segment512k
        aligned_last_many64
        aligned_middle_many64
        virtual_uneven9
        virtual_uneven64
    )
fi

for case_name in "${cases[@]}"; do
    output_dir="${profile_root}/${label}_${case_name}"
    rm -rf "${output_dir}"
    echo "PROFILE label=${label} case=${case_name} rounds=${rounds}"
    profile_program=concat_profile_matrix.py
    if [[ ${case_name} == whole_* ]]; then
        profile_program=concat_wholerow_profile.py
    fi
    msprof \
        --application="python3 ${profile_program} ${case_name} --rounds ${rounds}" \
        --output="${output_dir}"
done
