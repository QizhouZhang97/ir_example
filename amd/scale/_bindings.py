"""scale/_bindings.py

对应 nccl4py/nccl/core/device/cute/_bindings.py

每个 C device 函数对应两个 Python 对象：
  _raw_<name>  — cute.ffi prototype（MLIR 层，调用时生成 MLIR call op）
  <name>       — Python 包装函数（做参数强转、返回值包装）

这里的 C 函数定义在 ir/scale_device_wrapper__impl.h，
编译进 libscale_device.bc，运行时 llvm-link 到 kernel 里。
"""

import cutlass
from ._helpers import _ffi, _to_ptr
from ._structs import _LLVMPtrType


# === ScaleCtx accessor（对应 ncclDevComm_Rank / ncclDevComm_NRanks） ===

_raw_ScaleCtx_Scalar = _ffi(
    name="ScaleCtx_Scalar",
    params_types=[_LLVMPtrType],
    return_type=cutlass.Float32,
)

def ScaleCtx_Scalar(ctx_ptr):
    """读取 ScaleCtx.scalar（GPU 上执行）。"""
    return cutlass.Float32(_raw_ScaleCtx_Scalar(_to_ptr(ctx_ptr)))


_raw_ScaleCtx_N = _ffi(
    name="ScaleCtx_N",
    params_types=[_LLVMPtrType],
    return_type=cutlass.Int32,
)

def ScaleCtx_N(ctx_ptr):
    """读取 ScaleCtx.n（GPU 上执行）。"""
    return cutlass.Int32(_raw_ScaleCtx_N(_to_ptr(ctx_ptr)))


# === 核心运算（对应 ncclGinPut） ===

_raw_scale_elem = _ffi(
    name="scale_elem",
    params_types=[_LLVMPtrType, _LLVMPtrType, cutlass.Float32, cutlass.Int32],
)

def scale_elem(src_ptr, dst_ptr, scalar, i):
    """GPU 上执行 dst[i] = src[i] * scalar。"""
    _raw_scale_elem(
        _to_ptr(src_ptr),
        _to_ptr(dst_ptr),
        cutlass.Float32(scalar),
        cutlass.Int32(i),
    )
