#!/usr/bin/env bash
set -euo pipefail

root=/home/ma-user/work/s9_delivery_20260722
install_root="$root/installs_v2"
test_root="$root/tests_v2"
if [[ -e "$install_root" || -e "$test_root" ]]; then
  echo 'Refusing to overwrite existing v2 smoke-test state' >&2
  exit 1
fi
mkdir -p "$install_root" "$test_root"

test_source() {
  case "$1" in
    Concat) echo /home/ma-user/work/s9_concat_submit_build/official_test_v12_from_zip ;;
    Greater) echo /home/ma-user/work/s9_remaining_v1/GreaterRuntimeTest ;;
    SquareSumV1) echo /home/ma-user/work/s9_remaining_v1/SquareSumV1RuntimeTest ;;
  esac
}

for op in Concat Greater SquareSumV1; do
  test_dir="$test_root/$op"
  cp -a "$(test_source "$op")" "$test_dir"
  resolved=$(realpath -m "$test_dir")
  case "$resolved" in
    "$test_root"/*) rm -rf "$test_dir/build" "$test_dir/dist" "$test_dir"/*.egg-info ;;
    *) echo "Unsafe test path: $resolved" >&2; exit 1 ;;
  esac
  cd "$test_dir"
  echo "===== BUILD OFFICIAL EXTENSION $op"
  python3 setup.py build bdist_wheel > "$root/extension_${op}.log" 2>&1
  python3 -m pip install --force-reinstall --no-deps dist/custom_ops*.whl \
    > "$root/extension_install_${op}.log" 2>&1

  for variant in benchmark optimized; do
    key="${variant}_${op}"
    install="$install_root/$key"
    mkdir -p "$install"
    run="$root/$variant/$op/build_out/custom_opp_euleros_aarch64.run"
    echo "===== INSTALL $key"
    "$run" --quiet --install-path="$install" > "$root/install_v2_${key}.log" 2>&1
    api_lib=$(find "$install" -type f -name libcust_opapi.so -printf '%h\n' | head -n 1)
    test -n "$api_lib"
    vendor_dir=$(dirname "$(dirname "$api_lib")")
    export LD_LIBRARY_PATH="$api_lib:${LD_LIBRARY_PATH:-}"
    export ASCEND_CUSTOM_OPP_PATH="$vendor_dir"
    echo "===== TEST $key"
    timeout 180 python3 test_op.py 1 2>&1 | tee "$root/test_v2_${key}.log"
    grep -q 'verify result pass' "$root/test_v2_${key}.log"
  done
done

echo '===== ALL SIX PUBLIC CASE RUNS PASSED'
