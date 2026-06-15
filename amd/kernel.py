"""kernel.py

在没有 CuTeDSL 的环境里，这一层改为通过 ctypes 调用
libscale_kernel.so（包含 GPU kernel 的共享库）来 launch kernel。

对应 nccl4py 的 @cute.kernel + @cute.jit：
  - nccl4py 用 CuTeDSL trace Python 函数生成 MLIR → PTX → launch
  - 本例用 hipcc 预编译 HIP kernel 成 .so，Python 用 ctypes 调用 launch 函数

两者在"Python 不直接写 kernel 汇编，而是通过某种 FFI 层调用 GPU 函数"
这一设计思路上完全一致。区别只是 trace 时机：
  - CuTeDSL：运行时 trace → JIT 编译
  - 本例：    提前 hipcc 编译 → 运行时 dlopen
"""

import ctypes
import pathlib

_HERE = pathlib.Path(__file__).parent
_lib  = ctypes.CDLL(str(_HERE / "libscale_kernel.so"))

# launch 函数签名：
#   void scale_launch(long long ctx_ptr, long long src_ptr,
#                     long long dst_ptr, int n)
_lib.scale_launch.argtypes = [
    ctypes.c_longlong,  # ctx_ptr（GPU 上 ScaleCtx 地址）
    ctypes.c_longlong,  # src_ptr
    ctypes.c_longlong,  # dst_ptr
    ctypes.c_int,       # n
]
_lib.scale_launch.restype = None


def run(ctx_ptr: int, src_ptr: int, dst_ptr: int, n: int):
    """启动 scale kernel。

    对应 nccl4py 里的：
        test_nccl_put(dev_comm.ptr, send_win.handle, recv_win.handle)
    —— 把整数指针传给预编译好的 launch 函数，由 C 侧负责 kernel launch。
    """
    _lib.scale_launch(
        ctypes.c_longlong(ctx_ptr),
        ctypes.c_longlong(src_ptr),
        ctypes.c_longlong(dst_ptr),
        ctypes.c_int(n),
    )
