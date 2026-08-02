#!/usr/bin/env bash

set -e

output_root=${1:-/home/ma-user/work/s9_concat_longrun_evidence_20260803}
mkdir -p "${output_root}/op_summary"

roots=(
    /home/ma-user/work/s9_concat_virtual2d_20260803/profiles/longrun_20260803
    /home/ma-user/work/s9_concat_longrun_tile16_20260803/profiles/longrun_20260803
    /home/ma-user/work/s9_concat_longrun_tile64_20260803/profiles/longrun_20260803
    /home/ma-user/work/s9_concat_longrun_bsearch_20260803/profiles/longrun_20260803
    /home/ma-user/work/s9_concat_longrun_wholerow_20260803/profiles/longrun_20260803
)

for root in "${roots[@]}"; do
    [[ -d ${root} ]] || continue
    while IFS= read -r profile_dir; do
        summary=$(find "${profile_dir}" -type f -name 'op_summary*.csv' -print -quit)
        [[ -n ${summary} ]] || continue
        label=$(basename "${profile_dir}")
        cp "${summary}" "${output_root}/op_summary/${label}_op_summary.csv"
    done < <(find "${root}" -mindepth 1 -maxdepth 1 -type d | sort)
done

cp \
    /home/ma-user/work/s9_concat_longrun_wholerow_20260803/official_input_1000/results/longrun_wholerow_20260803/manifest_1000.json \
    /home/ma-user/work/s9_concat_longrun_wholerow_20260803/official_input_1000/results/longrun_wholerow_20260803/results_0000_1000.csv \
    "${output_root}/"

find "${output_root}" -type f -printf '%P\n' | sort
