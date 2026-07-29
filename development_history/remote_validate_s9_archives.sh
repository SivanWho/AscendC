#!/usr/bin/env bash
set -euo pipefail

root=/home/ma-user/work/s9_delivery_20260722
package_root="$root/packages_final"
validation_root="$root/archive_validation_final"
if [[ -e "$validation_root" ]]; then
  echo "Refusing to overwrite existing $validation_root" >&2
  exit 1
fi
mkdir -p "$validation_root/extracted" "$validation_root/installs"

for op in Concat Greater SquareSumV1; do
  test_dir="$root/tests_final/$op"
  cd "$test_dir"
  python3 -m pip install --force-reinstall --no-deps dist/custom_ops*.whl \
    > "$validation_root/extension_install_${op}.log" 2>&1

  for variant in benchmark optimized; do
    key="${variant}_${op}"
    if [[ "$variant" == benchmark ]]; then
      archive="$package_root/$variant/zipfiles/${op}_benchmark.zip"
    else
      archive="$package_root/$variant/zipfiles/${op}.zip"
    fi
    extract_dir="$validation_root/extracted/$key"
    install_dir="$validation_root/installs/$key"
    mkdir -p "$extract_dir" "$install_dir"
    unzip -q "$archive" -d "$extract_dir"
    run=$(find "$extract_dir" -maxdepth 2 -type f -name 'custom_*.run')
    test -n "$run"
    test -x "$run"
    "$run" --quiet --install-path="$install_dir" > "$validation_root/install_${key}.log" 2>&1
    api_lib=$(find "$install_dir" -type f -name libcust_opapi.so -printf '%h\n' | head -n 1)
    test -n "$api_lib"
    vendor_dir=$(dirname "$(dirname "$api_lib")")
    export LD_LIBRARY_PATH="$api_lib:${LD_LIBRARY_PATH:-}"
    export ASCEND_CUSTOM_OPP_PATH="$vendor_dir"
    cd "$test_dir"
    timeout 180 python3 test_op.py 1 2>&1 | tee "$validation_root/test_${key}.log"
    grep -q 'verify result pass' "$validation_root/test_${key}.log"
    echo "PASS FRESH ARCHIVE $key"
  done
done

echo 'ALL SIX FRESH ARCHIVES INSTALLED AND PASSED OFFICIAL CASE1'
