"""scale/_helpers.py

对应 nccl4py/nccl/core/device/cute/_helpers.py

职责：
  1. 找到 libscale_device.bc 的路径
  2. 构造 BitCode 对象（编译时 llvm-link 的来源）
  3. 提供 _ffi 工厂（自动注入 source=_BC）
  4. 提供 _to_ptr 类型强转帮手（int / Python对象 → !llvm.ptr）
"""

import os
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass.cute import BitCode
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl._mlir_helpers.op import dsl_user_op


# === bitcode 路径解析（对应 nccl4py 的 device_bitcode_path()） ===

def _device_bitcode_path() -> str:
    """找到 libscale_device.bc。

    优先读环境变量 SCALE_BC_PATH，否则在脚本同级目录找。
    对应 nccl4py 里先查 $NCCL_HOME 再用 cuda.pathfinder 的两步逻辑。
    """
    env = os.environ.get("SCALE_BC_PATH")
    if env and Path(env).is_file():
        return env
    # 默认：bc 文件和本 _helpers.py 在同一目录
    here = Path(__file__).parent
    candidate = here / "libscale_device.bc"
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(
        f"libscale_device.bc not found. Build it first, or set SCALE_BC_PATH."
    )


_BC = BitCode(_device_bitcode_path())


def _ffi(**kw):
    """cute.ffi 的包装，自动注入 source=_BC。
    对应 nccl4py 里的 _ffi()。
    """
    return cute.ffi(source=_BC, **kw)


# === 类型强转帮手（对应 nccl4py 的 _to_ptr） ===

@dsl_user_op
def _to_ptr(x, *, loc=None, ip=None):
    """把各种形式的 x 强转成 !llvm.ptr ir.Value。

    接受：
      - 已经是 !llvm.ptr ir.Value → 直接返回
      - @cute.native_struct 带 .ptr 字段 → 取出 ptr
      - cutlass 数值类型（有 .ir_value()） → inttoptr
      - ir.Value（整数类型） → inttoptr
      - Python int → 先包成 Int64，再 inttoptr
    """
    ptr_type = ir.Type.parse("!llvm.ptr")
    if isinstance(x, ir.Value) and x.type == ptr_type:
        return x
    if hasattr(x, "ptr"):
        inner = x.ptr
        if isinstance(inner, ir.Value) and inner.type == ptr_type:
            return inner
    if hasattr(x, "ir_value"):
        int_val = x.ir_value()
    elif isinstance(x, ir.Value):
        int_val = x
    else:
        int_val = cutlass.Int64(x).ir_value()
    return llvm.inttoptr(res=ptr_type, arg=int_val, loc=loc, ip=ip)
