#!/usr/bin/env bash
set -euo pipefail

root=/home/ma-user/work/s9_delivery_20260722
mkdir -p "$root/installs" "$root/tests"

test_source() {
  case "$1" in
    Concat) echo /home/ma-user/work/s9_concat_submit_build/official_test_v12_from_zip ;;
    Greater) echo /home/ma-user/work/s9_remaining_v1/GreaterRuntimeTest ;;
    SquareSumV1) echo /home/ma-user/work/s9_remaining_v1/SquareSumV1RuntimeTest ;;
  esac
}

for variant in benchmark optimized; do
  for op in Concat Greater SquareSumV1; do
    key="${variant}_${op}"
    install="$root/installs/$key"
    test_dir="$root/tests/$key"
    if [[ -e "$install" || -e "$test_dir" ]]; then
      echo "Refusing to overwrite existing test state for $key" >&2
      exit 1
    fi
    mkdir -p "$install"
    run="$root/$variant/$op/build_out/custom_opp_euleros_aarch64.run"
    echo "===== INSTALL $key"
    "$run" --quiet --install-path="$install" > "$root/install_${key}.log" 2>&1
    api_lib=$(find "$install" -type f -name libcust_opapi.so -printf '%h\n' | head -n 1)
    test -n "$api_lib"
    vendor_dir=$(dirname "$(dirname "$api_lib")")
    cp -a "$(test_source "$op")" "$test_dir"
    cd "$test_dir"
    export LD_LIBRARY_PATH="$api_lib:${LD_LIBRARY_PATH:-}"
    export ASCEND_CUSTOM_OPP_PATH="$vendor_dir"
    echo "===== TEST $key"
    timeout 180 python3 test_op.py 1 2>&1 | tee "$root/test_${key}.log"
    grep -q 'verify result pass' "$root/test_${key}.log"
  done
done

echo '===== ALL PUBLIC CASES PASSED'
