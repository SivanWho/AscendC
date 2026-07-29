#!/usr/bin/env bash
set -euo pipefail

root=/home/ma-user/work/s9_delivery_20260722
if [[ -e "$root" ]]; then
  echo "Refusing to overwrite existing $root" >&2
  exit 1
fi

mkdir -p "$root/benchmark" "$root/optimized"
cp -a /home/ma-user/work/s9_concat_submit_build/project "$root/benchmark/Concat"
cp -a /home/ma-user/work/s9_concat_submit_build/project "$root/optimized/Concat"
cp -a /home/ma-user/work/s9_remaining_v1/GreaterCustomFullV1 "$root/benchmark/Greater"
cp -a /home/ma-user/work/s9_remaining_v1/GreaterCustomOptimV2 "$root/optimized/Greater"
cp -a /home/ma-user/work/s9_remaining_v1/SquareSumV1FullV1 "$root/benchmark/SquareSumV1"
cp -a /home/ma-user/work/s9_remaining_v1/SquareSumV1OptimV2 "$root/optimized/SquareSumV1"

for project in \
  "$root/benchmark/Concat" "$root/optimized/Concat" \
  "$root/benchmark/Greater" "$root/optimized/Greater" \
  "$root/benchmark/SquareSumV1" "$root/optimized/SquareSumV1"; do
  resolved=$(realpath -m "$project")
  case "$resolved" in
    "$root"/*) ;;
    *) echo "Unsafe project path: $resolved" >&2; exit 1 ;;
  esac
  rm -rf "$project/op_host" "$project/op_kernel" "$project/build" "$project/build_out"
  mkdir -p "$project/op_host" "$project/op_kernel"
done

echo "$root"
