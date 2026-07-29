#!/usr/bin/env bash
set -euo pipefail

root=/home/ma-user/work/s9_delivery_20260722_opnamefix
test_root=/home/ma-user/work/s9_delivery_20260722/tests_final
validation_root="$root/name_validation"
if [[ -e "$validation_root" ]]; then
  echo "Refusing to overwrite existing $validation_root" >&2
  exit 1
fi
mkdir -p "$validation_root"

snake_name() {
  case "$1" in
    Concat) echo concat ;;
    Greater) echo greater ;;
    SquareSumV1) echo square_sum_v1 ;;
  esac
}

for op in Concat Greater SquareSumV1; do
  test_dir="$test_root/$op"
  cd "$test_dir"
  python3 -m pip install --force-reinstall --no-deps dist/custom_ops*.whl \
    > "$validation_root/extension_install_${op}.log" 2>&1

  for variant in benchmark optimized; do
    key="${variant}_${op}"
    install="$validation_root/$key"
    mkdir -p "$install"
    run="$root/$variant/$op/build_out/custom_opp_euleros_aarch64.run"
    "$run" --quiet --install-path="$install" > "$validation_root/install_${key}.log" 2>&1

    snake=$(snake_name "$op")
    find "$install" -type f -name "aclnn_${snake}.h" | grep -q .
    find "$install" -type f -name "${op}_*.o" | grep -q .
    find "$install" -type f -name "${op}_*.json" | grep -q .
    if find "$install" -type f \( -name 'ConcatCustom_*' -o -name 'GreaterCustom_*' -o -name 'aclnn_*_custom.h' \) | grep -q .; then
      echo "Unexpected custom-name artifact in $key" >&2
      exit 1
    fi

    api_lib=$(find "$install" -type f -name libcust_opapi.so -printf '%h\n' | head -n 1)
    test -n "$api_lib"
    vendor_dir=$(dirname "$(dirname "$api_lib")")
    export LD_LIBRARY_PATH="$api_lib:${LD_LIBRARY_PATH:-}"
    export ASCEND_CUSTOM_OPP_PATH="$vendor_dir"
    timeout 180 python3 test_op.py 1 2>&1 | tee "$validation_root/test_${key}.log"
    grep -q 'verify result pass' "$validation_root/test_${key}.log"
    echo "PASS NAME AND CASE1 $key"
  done
done

echo 'ALL SIX PACKAGES HAVE EXACT OP NAMES AND PASS OFFICIAL CASE1'
