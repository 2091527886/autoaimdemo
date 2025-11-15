#pragma once
#include "letterbox.hpp"
#include "letterbox_common.cuh"

namespace trtyolo {

// 共享核心：执行单个(y, x..x+3) 的双线性采样、归一化与写回
__device__ inline void lb_vec4_kernel_core(const uint8_t* __restrict__ s, int src_cols, int src_rows, size_t src_pitch,
                                           __half* __restrict__ out_row, int dst_cols, int dst_rows,
                                           int px, int y,
                                           const LetterboxTransform params, const ProcessConfig config) {
    int plane = dst_cols * dst_rows;

    // y 方向只算一次
    float src_y = ((static_cast<float>(y) + 0.5f) - static_cast<float>(params.pad_y)) / params.scale - 0.5f;
    int   y0 = __float2int_rd(src_y);
    int   y1 = y0 + 1;
    float wy1 = src_y - y0;
    float wy0 = 1.f - wy1;

    float r[4], g[4], b[4];
    const float bv=config.border_value; const float3 bvv=make_float3(bv,bv,bv);
    #define S(xx,yy) (inb((xx),(yy))? sample_rgb_bgra(s, src_pitch, (xx), (yy), src_cols, src_rows, bv) : bvv)
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int x = px + k;
        if (x >= dst_cols) { r[k]=g[k]=b[k]=0.f; continue; }

        float src_x = ((static_cast<float>(x) + 0.5f) - static_cast<float>(params.pad_x)) / params.scale - 0.5f;
        int   x0 = __float2int_rd(src_x);
        int   x1 = x0 + 1;
        float wx1 = src_x - x0;
        float wx0 = 1.f - wx1;

        auto inb = [&](int xx, int yy){ return (xx >= 0 && xx < src_cols && yy >= 0 && yy < src_rows); };
        float3 p00=S(x0,y0);
        float3 p10=S(x1,y0);
        float3 p01=S(x0,y1);
        float3 p11=S(x1,y1);

        float3 sum;
        sum.x = fmaf(wx0*wy0, p00.x, fmaf(wx1*wy0, p10.x, fmaf(wx0*wy1, p01.x, wx1*wy1 * p11.x)));
        sum.y = fmaf(wx0*wy0, p00.y, fmaf(wx1*wy0, p10.y, fmaf(wx0*wy1, p01.y, wx1*wy1 * p11.y)));
        sum.z = fmaf(wx0*wy0, p00.z, fmaf(wx1*wy0, p10.z, fmaf(wx0*wy1, p01.z, wx1*wy1 * p11.z)));

        r[k] = fmaf(sum.x, config.alpha.x, config.beta.x);
        g[k] = fmaf(sum.y, config.alpha.y, config.beta.y);
        b[k] = fmaf(sum.z, config.alpha.z, config.beta.z);
    }

    bool full = (px + 3) < dst_cols;
    if (full) {
        __half2 r2a=__halves2half2(__float2half(r[0]),__float2half(r[1]));
        __half2 r2b=__halves2half2(__float2half(r[2]),__float2half(r[3]));
        __half2 g2a=__halves2half2(__float2half(g[0]),__float2half(g[1]));
        __half2 g2b=__halves2half2(__float2half(g[2]),__float2half(g[3]));
        __half2 b2a=__halves2half2(__float2half(b[0]),__float2half(b[1]));
        __half2 b2b=__halves2half2(__float2half(b[2]),__float2half(b[3]));
        __half2* o2=reinterpret_cast<__half2*>(out_row);
        __half2* g2=reinterpret_cast<__half2*>(out_row+plane);
        __half2* b2=reinterpret_cast<__half2*>(out_row+plane*2);
        o2[0]=r2a; o2[1]=r2b; g2[0]=g2a; g2[1]=g2b; b2[0]=b2a; b2[1]=b2b;
    } else {
        int n = dst_cols - px;
        #pragma unroll
        for (int k=0;k<4 && k<n;++k){
            out_row[k]               = __float2half(r[k]);
            out_row[plane + k]       = __float2half(g[k]);
            out_row[plane*2 + k]     = __float2half(b[k]);
        }
    }
    #undef S
}

// 单图包装：计算位置信息与基址后调用核心
__device__ inline void letterboxF16Vec4_body(const void* __restrict__ src, int src_cols, int src_rows, size_t src_pitch,
                                             void* __restrict__ dst, int dst_cols, int dst_rows,
                                             const LetterboxTransform params, const ProcessConfig config) {
    int px = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    int y  = blockDim.y * blockIdx.y + threadIdx.y;
    if (y >= dst_rows || px >= dst_cols) return;
    const uint8_t* __restrict__ s = static_cast<const uint8_t*>(src);
    __half* out_row = reinterpret_cast<__half*>(dst) + y * dst_cols + px;
    lb_vec4_kernel_core(s, src_cols, src_rows, src_pitch, out_row, dst_cols, dst_rows, px, y, params, config);
}

// 多图包装：根据 batch idx 计算基址后调用核心
__device__ inline void multiLetterboxHalfVec4_body(const void* __restrict__ src, int src_cols, int src_rows, size_t src_pitch,
                                                   void* __restrict__ dst, int dst_cols, int dst_rows,
                                                   LetterboxTransform params, ProcessConfig config,
                                                   int num_images) {
    int idx = blockIdx.z;
    if (idx >= num_images) return;
    int px = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    int y  = blockDim.y * blockIdx.y + threadIdx.y;
    if (y >= dst_rows || px >= dst_cols) return;
    const uint8_t* __restrict__ src_ptr = static_cast<const uint8_t*>(src) + idx * src_rows * src_pitch;
    __half* dst_ptr = reinterpret_cast<__half*>(dst) + idx * dst_rows * dst_cols * 3;
    __half* out_row = dst_ptr + y * dst_cols + px;
    lb_vec4_kernel_core(src_ptr, src_cols, src_rows, src_pitch, out_row, dst_cols, dst_rows, px, y, params, config);
}

} // namespace trtyolo
