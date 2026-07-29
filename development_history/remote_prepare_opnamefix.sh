#!/usr/bin/env bash
set -euo pipefail

source_root=/home/ma-user/work/s9_delivery_20260722
target_root=/home/ma-user/work/s9_delivery_20260722_opnamefix
if [[ -e "$target_root" ]]; then
  echo "Refusing to overwrite existing $target_root" >&2
  exit 1
fi

for variant in benchmark optimized; do
  for op in Concat Greater SquareSumV1; do
    mkdir -p "$target_root/$variant"
    cp -a "$source_root/$variant/$op" "$target_root/$variant/$op"
  done
done

rm -f \
  "$target_root/benchmark/Concat/op_host/aclnn_concat_compat.cpp" \
  "$target_root/optimized/Concat/op_host/aclnn_concat_compat.cpp" \
  "$target_root/benchmark/Greater/op_host/aclnn_greater_compat.cpp" \
  "$target_root/optimized/Greater/op_host/aclnn_greater_compat.cpp"

echo "$target_root"
