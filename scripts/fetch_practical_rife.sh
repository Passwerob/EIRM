#!/usr/bin/env bash
set -euo pipefail

# Run this outside the restricted execution environment when GitHub access is
# available. The simulator never vendors or silently modifies this checkout.
target="${1:-Practical-RIFE}"
if [[ -e "$target" ]]; then
  echo "Refusing to overwrite existing path: $target" >&2
  exit 2
fi
git clone https://github.com/hzwer/Practical-RIFE.git "$target"
printf '%s\n' "Practical-RIFE checkout: $target"
printf '%s\n' "Pass --practical-rife-root $target to scripts/simulate_ami.py"
