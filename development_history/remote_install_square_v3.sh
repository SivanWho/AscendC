#!/usr/bin/env bash
set -euo pipefail
root=/home/ma-user/work/s9_delivery_20260722
install="$root/installs_v3/optimized_SquareSumV1"
if [[ -e "$install" ]]; then
  echo "Refusing to overwrite $install" >&2
  exit 1
fi
mkdir -p "$install"
"$root/optimized/SquareSumV1/build_out/custom_opp_euleros_aarch64.run" \
  --quiet --install-path="$install" > "$root/install_v3_optimized_SquareSumV1.log" 2>&1
SQUARE_INSTALL="$install" bash /home/ma-user/work/remote_square_regression.sh "$@"
