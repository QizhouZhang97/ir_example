/*
 * scale_device_wrapper__impl.h
 *
 * 对应 NCCL 的 nccl_device_wrapper__impl.h：
 *   - 包含声明头，再给出 inline 实现
 *   - 编译目标：libscale_device.bc（LLVM bitcode）
 *   - Python cute.ffi 的 BitCode("libscale_device.bc") 会在编译时
 *     把这段 bitcode llvm-link 进用户 kernel 的 gpu.module 里
 */
#pragma once

#include "scale_device_wrapper.h"

SCALE_IR_EXTERN_C SCALE_DEVICE_INLINE
float ScaleCtx_Scalar(const ScaleCtx* ctx) { return ctx->scalar; }

SCALE_IR_EXTERN_C SCALE_DEVICE_INLINE
int   ScaleCtx_N(const ScaleCtx* ctx)      { return ctx->n; }

SCALE_IR_EXTERN_C SCALE_DEVICE_INLINE
void scale_elem(const float* src, float* dst, float scalar, int i) {
    dst[i] = src[i] * scalar;
}
