#!/usr/bin/env bash
set -euo pipefail
root=/home/ma-user/work/s9_delivery_20260722
for file in \
  "$root/benchmark/Concat/concat.cpp" "$root/optimized/Concat/concat.cpp" \
  "$root/benchmark/Greater/greater.cpp" "$root/optimized/Greater/greater.cpp"; do
  resolved=$(realpath -m "$file")
  case "$resolved" in
    "$root"/*) rm -f "$resolved" ;;
    *) echo "Unsafe path: $resolved" >&2; exit 1 ;;
  esac
done
