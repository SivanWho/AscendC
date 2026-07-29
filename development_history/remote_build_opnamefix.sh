#!/usr/bin/env bash
set -euo pipefail

root=/home/ma-user/work/s9_delivery_20260722_opnamefix
projects=(
  benchmark/Concat
  optimized/Concat
  benchmark/Greater
  optimized/Greater
  benchmark/SquareSumV1
  optimized/SquareSumV1
)

for rel in "${projects[@]}"; do
  echo "===== BUILD $rel"
  cd "$root/$rel"
  bash build.sh > "$root/build_${rel//\//_}.log" 2>&1
  run="$root/$rel/build_out/custom_opp_euleros_aarch64.run"
  test -s "$run"
  stat -c '%n %s bytes' "$run"
done
