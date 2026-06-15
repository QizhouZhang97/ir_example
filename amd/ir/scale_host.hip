/*
 * scale_host.hip
 *
 * 对应 NCCL 的 init.cc / dev_runtime.cc 里的 host API：
 *   - 编译成 libscale_host.so（共享库）
 *   - extern "C" 让 Python ctypes 能用字符串找到符号
 *   - 负责分配/释放 GPU 内存，把 ScaleCtx 上传到 device
 *   - 不包含 kernel launch（kernel 由 Python CuTeDSL 层负责）
 *
 * Python 调用路径（对应 nccl4py communicator.py → resources.py）：
 *   Python ctypes
 *     → scale_ctx_create()   分配并上传 ScaleCtx 到 GPU
 *     → scale_buf_alloc()    分配 GPU float buffer
 *     → scale_buf_copy_h2d() 拷贝数据到 GPU
 *     → scale_buf_copy_d2h() 拷贝结果回 host
 *     → scale_buf_free()     释放 GPU buffer
 *     → scale_ctx_destroy()  释放 ScaleCtx
 */

#include <hip/hip_runtime.h>
#include "scale_device_wrapper.h"

/* ── ScaleCtx 生命周期（对应 ncclDevCommCreate / ncclDevCommDestroy） ── */

/*
 * 在 GPU 上分配并初始化 ScaleCtx，返回 device 指针（以 intptr_t 形式）
 * Python 把这个整数作为 kernel 参数传入，kernel 里 inttoptr 解引用
 */
extern "C" long long scale_ctx_create(float scalar, int n) {
    ScaleCtx h = { scalar, n };
    ScaleCtx* d = nullptr;
    if (hipMalloc(&d, sizeof(ScaleCtx)) != hipSuccess) return 0;
    if (hipMemcpy(d, &h, sizeof(ScaleCtx), hipMemcpyHostToDevice) != hipSuccess) {
        (void)hipFree(d);
        return 0;
    }
    return (long long)(uintptr_t)d;
}

extern "C" void scale_ctx_destroy(long long ctx_ptr) {
    (void)hipFree((void*)(uintptr_t)ctx_ptr);
}

/* ── GPU buffer 管理（对应 ncclCommWindowRegister / Deregister） ── */

extern "C" long long scale_buf_alloc(int n) {
    float* d = nullptr;
    if (hipMalloc(&d, n * sizeof(float)) != hipSuccess) return 0;
    return (long long)(uintptr_t)d;
}

extern "C" void scale_buf_free(long long ptr) {
    (void)hipFree((void*)(uintptr_t)ptr);
}

extern "C" void scale_buf_copy_h2d(long long dst, const float* src, int n) {
    (void)hipMemcpy((void*)(uintptr_t)dst, src, n * sizeof(float),
                    hipMemcpyHostToDevice);
}

extern "C" void scale_buf_copy_d2h(float* dst, long long src, int n) {
    (void)hipMemcpy(dst, (void*)(uintptr_t)src, n * sizeof(float),
                    hipMemcpyDeviceToHost);
}

extern "C" void scale_sync() {
    (void)hipDeviceSynchronize();
}
