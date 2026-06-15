#!/usr/bin/env bash
# build.sh — 编译 ir_example（AMD 平台，对应 NCCL IR 分层）
#
# 产物及其对应关系：
#   libscale_device.bc   ←→ libnccl_device.bc  （device bitcode）
#   libscale_host.so     ←→ libnccl.so          （host API，Python ctypes 加载）
#   libscale_kernel.so   ←→ CuTeDSL JIT 产物    （kernel launch，Python kernel.py 调用）
#
# 用法：
#   bash build.sh [gfx_arch]   默认 gfx942

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMD_DIR="${SCRIPT_DIR}/amd"
IR_DIR="${AMD_DIR}/ir"
SCALE_DIR="${AMD_DIR}/scale"

ARCH=${1:-gfx942}

# ── 步骤 1：编译 device bitcode ────────────────────────────────────────────
# 对应 NCCL 把 nccl_device_wrapper__impl.h 编成 libnccl_device.bc
# CuTeDSL 场景：运行时 llvm-link 到用户 kernel；
# 本例：hipcc 直接 inline 到 kernel .so，故 bc 仅作演示留存。
echo "==> [1/4] Compiling device bitcode: libscale_device.bc (arch=${ARCH}) ..."
/opt/rocm/bin/amdclang++ \
    -x hip \
    --offload-arch=${ARCH} \
    --cuda-device-only \
    -emit-llvm \
    -c \
    -O2 \
    -I"${IR_DIR}" \
    -o "${SCALE_DIR}/libscale_device.bc" \
    "${IR_DIR}/scale_device_wrapper__impl.cpp"
echo "    Built: ${SCALE_DIR}/libscale_device.bc"

# ── 步骤 2：编译 host 共享库 ────────────────────────────────────────────────
# 对应 libnccl.so：管理 GPU 内存生命周期（alloc/free/copy）
echo "==> [2/4] Compiling host shared lib: libscale_host.so ..."
hipcc \
    -fPIC -shared -O2 \
    --offload-arch=${ARCH} \
    -I"${IR_DIR}" \
    -o "${AMD_DIR}/libscale_host.so" \
    "${IR_DIR}/scale_host.hip"
echo "    Built: ${AMD_DIR}/libscale_host.so"

# ── 步骤 3：编译 kernel 共享库 ──────────────────────────────────────────────
# 对应 CuTeDSL JIT 产物：包含 GPU kernel + launch 函数
# Python kernel.py 通过 ctypes 调用 scale_launch()
echo "==> [3/4] Compiling kernel shared lib: libscale_kernel.so ..."
hipcc \
    -fPIC -shared -O2 \
    --offload-arch=${ARCH} \
    -I"${IR_DIR}" \
    -o "${AMD_DIR}/libscale_kernel.so" \
    "${IR_DIR}/scale_kernel.hip"
echo "    Built: ${AMD_DIR}/libscale_kernel.so"

# ── 步骤 4：运行 ────────────────────────────────────────────────────────────
echo "==> [4/4] Running: python3 amd/main.py ..."
cd "${AMD_DIR}"
SCALE_BC_PATH="${SCALE_DIR}/libscale_device.bc" \
PYTHONPATH="${AMD_DIR}:${PYTHONPATH:-}" \
    python3 main.py
