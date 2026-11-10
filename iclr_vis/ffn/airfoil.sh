#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${CDLNO_PYTHON:-python}" -B -u "$script_dir/ffn_states.py" airfoil "$@"
