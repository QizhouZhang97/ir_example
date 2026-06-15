"""ir_example/main.py

演示用 ctypes 从 Python 调用 HIP 共享库里的 host 函数，
host 函数内部启动 GPU kernel。

三层结构：
  Python (main.py)
    → ctypes 加载 ir_example.so，调用 scale()
      → host 函数 scale()   [C++, CPU 上执行]
        → hipMalloc / hipMemcpy
        → scale_kernel<<<>>>  [GPU kernel, GPU 上执行]
        → hipMemcpy 结果回 host
    ← 返回结果给 Python
"""

import ctypes
import pathlib
import sys
import numpy as np

# ─────────────────────────────────────────────
# 第三层：Python 侧
# ─────────────────────────────────────────────

# 1. 加载共享库（等价于 ctypes.dlopen）
so_path = pathlib.Path(__file__).parent / "ir_example.so"
if not so_path.exists():
    print(f"ERROR: {so_path} not found. Run build.sh first.")
    sys.exit(1)

lib = ctypes.CDLL(str(so_path))

# 2. 声明函数签名（告诉 ctypes 参数类型和返回类型）
#    int scale(const float* host_in, float* host_out, float scalar, int n)
lib.scale.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # host_in
    ctypes.POINTER(ctypes.c_float),  # host_out
    ctypes.c_float,                  # scalar
    ctypes.c_int,                    # n
]
lib.scale.restype = ctypes.c_int

# 3. 准备数据
N = 1024
scalar = 3.0

src = np.arange(N, dtype=np.float32)          # [0, 1, 2, ..., 1023]
dst = np.zeros(N, dtype=np.float32)

# 4. 调用 host 函数（内部会启动 GPU kernel）
ret = lib.scale(
    src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    dst.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    ctypes.c_float(scalar),
    ctypes.c_int(N),
)

if ret != 0:
    print("ERROR: scale() returned", ret)
    sys.exit(1)

# 5. 验证结果
expected = src * scalar
if np.allclose(dst, expected):
    print(f"[SUCCESS] scale({N} elements, scalar={scalar})")
    print(f"  src[:4]      = {src[:4].tolist()}")
    print(f"  dst[:4]      = {dst[:4].tolist()}")
    print(f"  expected[:4] = {expected[:4].tolist()}")
else:
    mismatches = int((dst != expected).sum())
    print(f"[ERROR] {mismatches}/{N} mismatches")
    sys.exit(1)
