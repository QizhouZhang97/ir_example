/*
 * scale_kernel.hip — GPU kernel + launch 函数
 *
 * 对应 nccl4py 里由 CuTeDSL 生成的 PTX kernel：
 *   - scale_kernel_impl：GPU kernel，调用 device wrapper 里的 scale_elem
 *   - scale_launch：host 侧 launcher，Python ctypes 调用它来启动 kernel
 *
 * 与 NCCL 的对应关系：
 *   scale_elem()   ← 对应 ncclGinPut（device 函数，在 bitcode 里）
 *   scale_launch() ← 对应 CuTeDSL @cute.jit launcher（host 侧 launch 逻辑）
 *
 * 注意：scale_elem 的实现来自 scale_device_wrapper__impl.h（inline），
 * hipcc 会把它直接内联进 kernel，不需要额外 llvm-link。
 * （CuTeDSL 场景下才需要 llvm-link bitcode，因为 Python trace 在运行时发生。）
 */

#include <hip/hip_runtime.h>
#include "scale_device_wrapper__impl.h"  // 包含 inline device 函数实现

/* GPU kernel：每个线程调用 scale_elem 处理一个元素 */
__global__ void scale_kernel_impl(
    const ScaleCtx* ctx,
    const float*    src,
    float*          dst
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = ScaleCtx_N(ctx);
    if (i < n) {
        scale_elem(src, dst, ScaleCtx_Scalar(ctx), i);
    }
}

/* host launch 函数：Python ctypes 调用入口 */
extern "C" void scale_launch(
    long long ctx_ptr,
    long long src_ptr,
    long long dst_ptr,
    int       n
) {
    const ScaleCtx* ctx = (const ScaleCtx*)(uintptr_t)ctx_ptr;
    const float*    src = (const float*)   (uintptr_t)src_ptr;
    float*          dst = (float*)         (uintptr_t)dst_ptr;

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    scale_kernel_impl<<<blocks, threads>>>(ctx, src, dst);
}
