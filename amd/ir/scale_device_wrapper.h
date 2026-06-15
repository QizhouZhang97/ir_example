/*
 * scale_device_wrapper.h
 *
 * 对应 NCCL 的 nccl_device_wrapper.h：
 *   - 只声明 device 函数，不含实现
 *   - 使用 extern "C" + __device__ 让符号名无 C++ mangling
 *   - Python CuTeDSL 的 cute.ffi 只需要这些声明来生成 MLIR prototype
 *   - 实现在 scale_device_wrapper__impl.h，编译成 libscale_device.bc
 */
#pragma once

#define SCALE_IR_EXTERN_C  extern "C"
#define SCALE_DEVICE_INLINE __attribute__((always_inline)) __device__

/*
 * ScaleCtx: 封装一次 scale 操作所需的上下文（对应 ncclDevComm）
 * Python 侧通过整数指针传入，在 GPU kernel 里 inttoptr 解引用
 */
struct ScaleCtx {
    float scalar;   // 乘数
    int   n;        // 元素个数
};

/* 从 ScaleCtx 指针读取字段（对应 ncclDevComm_Rank 等 accessor） */
SCALE_IR_EXTERN_C __device__ float ScaleCtx_Scalar(const ScaleCtx* ctx);
SCALE_IR_EXTERN_C __device__ int   ScaleCtx_N(const ScaleCtx* ctx);

/* 核心 scale 操作：dst[i] = src[i] * scalar（对应 ncclGinPut） */
SCALE_IR_EXTERN_C __device__ void scale_elem(
    const float* src, float* dst, float scalar, int i);
