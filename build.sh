#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ARCH=${1:-gfx942}

echo "==> Compiling ir_example.hip for --offload-arch=${ARCH} ..."
hipcc \
    -fPIC \
    -shared \
    -O2 \
    --offload-arch=${ARCH} \
    ir_example.hip \
    -o ir_example.so

echo "==> Built: ${SCRIPT_DIR}/ir_example.so"
echo "==> Running: python3 main.py"
python3 main.py
