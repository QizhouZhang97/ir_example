"""main.py

对应 nccl4py/examples/cute/main.py 的 main() 函数。

三层调用链：
  Python (main.py)
    ├── ctypes → libscale_host.so   host API（分配 GPU 内存、上传数据）
    └── kernel.run()                CuTeDSL kernel launch
          └── scale_kernel          GPU kernel（通过 libscale_device.bc 的 FFI）

完整数据流：
  numpy src  →[H2D]→ GPU src_buf
  ScaleCtx   →[H2D]→ GPU ctx
  scale_kernel<<<>>>  →  GPU dst_buf
  GPU dst_buf →[D2H]→ numpy dst  →  验证
"""

import sys
import ctypes
import pathlib
import numpy as np

# ── 加载 host 共享库（对应 nccl4py 里 _nccl_bindings = nccl.bindings.nccl） ──
_HERE = pathlib.Path(__file__).parent
_lib  = ctypes.CDLL(str(_HERE / "libscale_host.so"))

# 声明函数签名（对应 nccl.pyx 里的 cpdef 包装）
_lib.scale_ctx_create.argtypes  = [ctypes.c_float, ctypes.c_int]
_lib.scale_ctx_create.restype   = ctypes.c_longlong

_lib.scale_ctx_destroy.argtypes = [ctypes.c_longlong]
_lib.scale_ctx_destroy.restype  = None

_lib.scale_buf_alloc.argtypes   = [ctypes.c_int]
_lib.scale_buf_alloc.restype    = ctypes.c_longlong

_lib.scale_buf_free.argtypes    = [ctypes.c_longlong]
_lib.scale_buf_free.restype     = None

_lib.scale_buf_copy_h2d.argtypes = [
    ctypes.c_longlong,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]
_lib.scale_buf_copy_h2d.restype  = None

_lib.scale_buf_copy_d2h.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_longlong,
    ctypes.c_int,
]
_lib.scale_buf_copy_d2h.restype  = None

_lib.scale_sync.argtypes = []
_lib.scale_sync.restype  = None


def main():
    N      = 1024
    SCALAR = 3.0

    # ── 准备 host 数据 ──────────────────────────────────────────────────────
    src_np = np.arange(N, dtype=np.float32)
    dst_np = np.zeros(N, dtype=np.float32)

    # ── 分配 GPU 资源（对应 create_dev_comm + register_window） ────────────
    ctx_ptr = _lib.scale_ctx_create(ctypes.c_float(SCALAR), ctypes.c_int(N))
    src_ptr = _lib.scale_buf_alloc(ctypes.c_int(N))
    dst_ptr = _lib.scale_buf_alloc(ctypes.c_int(N))
    assert ctx_ptr and src_ptr and dst_ptr, "GPU alloc failed"

    # ── host → device ──────────────────────────────────────────────────────
    _lib.scale_buf_copy_h2d(
        src_ptr,
        src_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        N,
    )

    # ── 启动 CuTeDSL kernel（对应 test_nccl_put(dev_comm.ptr, ...)） ───────
    import kernel
    kernel.run(int(ctx_ptr), int(src_ptr), int(dst_ptr), N)

    # ── 等待 + device → host ───────────────────────────────────────────────
    _lib.scale_sync()
    _lib.scale_buf_copy_d2h(
        dst_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dst_ptr,
        N,
    )

    # ── 释放 GPU 资源（对应 dev_comm.close() + win.close()） ───────────────
    _lib.scale_buf_free(dst_ptr)
    _lib.scale_buf_free(src_ptr)
    _lib.scale_ctx_destroy(ctx_ptr)

    # ── 验证 ────────────────────────────────────────────────────────────────
    expected = src_np * SCALAR
    if np.allclose(dst_np, expected):
        print(f"[SUCCESS] scale({N} elems, scalar={SCALAR})")
        print(f"  src[:4]      = {src_np[:4].tolist()}")
        print(f"  dst[:4]      = {dst_np[:4].tolist()}")
        print(f"  expected[:4] = {expected[:4].tolist()}")
        return 0
    else:
        bad = int((dst_np != expected).sum())
        print(f"[ERROR] {bad}/{N} mismatches")
        return 1


if __name__ == "__main__":
    sys.exit(main())
