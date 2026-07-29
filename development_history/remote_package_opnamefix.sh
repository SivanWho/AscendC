#!/usr/bin/env bash
set -euo pipefail

root=/home/ma-user/work/s9_delivery_20260722_opnamefix
package_root="$root/packages_ubuntu"
if [[ -e "$package_root" ]]; then
  echo "Refusing to overwrite existing $package_root" >&2
  exit 1
fi

for variant in benchmark optimized; do
  base="$package_root/$variant"
  mkdir -p "$base/op" "$base/zipfiles"
  cp /home/ma-user/work/zip_op.sh "$base/zipfiles/zip_op.sh"
  for op in Concat Greater SquareSumV1; do
    project="$root/$variant/$op"
    target="$base/op/$op"
    mkdir -p "$target/build_out"
    cp -a "$project/op_host" "$target/op_host"
    cp -a "$project/op_kernel" "$target/op_kernel"
    cp "$project/build_out/custom_opp_euleros_aarch64.run" \
      "$target/build_out/custom_opp_ubuntu_aarch64.run"
    chmod 755 "$target/build_out/custom_opp_ubuntu_aarch64.run"
    cd "$base/zipfiles"
    bash zip_op.sh "$op"
    if [[ "$variant" == benchmark ]]; then
      mv "$op.zip" "${op}_benchmark.zip"
    fi
  done
done

python3 /home/ma-user/work/validate_s9_packages.py "$package_root"
sha256sum "$package_root"/*/zipfiles/*.zip | sort
