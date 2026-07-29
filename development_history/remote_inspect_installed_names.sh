#!/usr/bin/env bash
set -euo pipefail

root=/home/ma-user/work/s9_delivery_20260722/archive_validation_final/installs
for dir in "$root"/*; do
  echo "===== $(basename "$dir")"
  find "$dir" -type f | grep -E '(aclnn|\.json$|\.o$)' | sed 's#^.*/##' | sort | head -80
done
