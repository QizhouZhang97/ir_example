"""scale/_structs.py

对应 nccl4py/nccl/core/device/cute/_structs.py

用 @cute.native_struct 把 C struct 镜像到 Python/MLIR 类型。
每个字段类型是 MLIR 类型的 Python 包装：
  - cutlass.Float32 → MLIR f32
  - cutlass.Int32   → MLIR i32
  - _LLVMPtrType    → MLIR !llvm.ptr

注意：ScaleCtx 在 GPU kernel 里通过指针访问（不在 Python 里展开），
所以这里不需要镜像 ScaleCtx 的字段，只需要一个 ptr 包装类。
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir


class _LLVMPtrType:
    """MLIR !llvm.ptr 的 Python 类型标注适配器。
    对应 nccl4py 里的同名类。
    """
    @staticmethod
    def mlir_type():
        return ir.Type.parse("!llvm.ptr")

    @staticmethod
    def __get_mlir_types__():
        return [_LLVMPtrType.mlir_type()]


@cute.native_struct
class ScaleCtxWrapper:
    """GPU 侧 ScaleCtx 的 Python 指针包装。

    对应 nccl4py 里 DevComm 的作用：
      - 持有一个 !llvm.ptr 指向 GPU 上的 ScaleCtx
      - 通过 FFI accessor 读取字段（不在 Python 里直接解引用）
    """
    ptr: _LLVMPtrType
