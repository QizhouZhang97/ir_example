// ir_example.hip
//
// 演示三层结构：
//   1. __global__ kernel  — 跑在 GPU 上，每个线程把 src[i] * 2 写到 dst[i]
//   2. host 函数 (extern "C") — 分配 GPU 内存、启动 kernel、拷回结果
//   3. 由 Python ctypes 调用 host 函数
//
// 编译成共享库：
//   hipcc -fPIC -shared -O2 --offload-arch=gfx942 ir_example.hip -o ir_example.so

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ─────────────────────────────────────────────
// 第一层：GPU kernel
//   每个线程处理一个元素，执行 dst[i] = src[i] * scalar
// ─────────────────────────────────────────────
__global__ void scale_kernel(const float* src, float* dst, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i] * scalar;
    }
}

// ─────────────────────────────────────────────
// 第二层：host 函数，extern "C" 让 Python ctypes 能找到符号
//   参数：
//     host_in  — Python 传来的输入数组指针（host 内存）
//     host_out — Python 传来的输出数组指针（host 内存）
//     scalar   — 乘数
//     n        — 元素个数
//   返回：0 成功，-1 失败
// ─────────────────────────────────────────────
extern "C" int scale(const float* host_in, float* host_out, float scalar, int n) {
    float* dev_in  = nullptr;
    float* dev_out = nullptr;
    size_t bytes = n * sizeof(float);

    // 在 GPU 上分配内存
    if (hipMalloc(&dev_in,  bytes) != hipSuccess) return -1;
    if (hipMalloc(&dev_out, bytes) != hipSuccess) { (void)hipFree(dev_in); return -1; }

    // host → device
    (void)hipMemcpy(dev_in, host_in, bytes, hipMemcpyHostToDevice);

    // 启动 kernel：每个 block 256 个线程
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    scale_kernel<<<blocks, threads>>>(dev_in, dev_out, scalar, n);

    // 等 kernel 跑完
    (void)hipDeviceSynchronize();

    // device → host
    (void)hipMemcpy(host_out, dev_out, bytes, hipMemcpyDeviceToHost);

    (void)hipFree(dev_in);
    (void)hipFree(dev_out);
    return 0;
}
