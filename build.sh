#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMD_DIR="${SCRIPT_DIR}/amd"

ARCH=${1:-gfx942}

echo "==> Compiling amd/ir_example.hip for --offload-arch=${ARCH} ..."
hipcc \
    -fPIC \
    -shared \
    -O2 \
    --offload-arch=${ARCH} \
    "${AMD_DIR}/ir_example.hip" \
    -o "${AMD_DIR}/ir_example.so"

echo "==> Built: ${AMD_DIR}/ir_example.so"
echo "==> Running: python3 amd/main.py"
python3 "${AMD_DIR}/main.py"
