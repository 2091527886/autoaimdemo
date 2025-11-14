#pragma once
#include "letterbox.hpp"
#include "letterbox_common.cuh"

namespace trtyolo {

// 4 像素/线程（Vec4）主体实现
__device__ inline void letterboxF16Vec4_body(const void* __restrict__ src, int src_cols, int src_rows, size_t src_pitch,
                                    void* __restrict__ dst, int dst_cols, int dst_rows,
                                    const LetterboxTransform params, const ProcessConfig config) {
    int px = (blockDim.x * blockIdx.x + threadIdx.x) * 4; // start x for this thread (4 pixels)
    int y  = blockDim.y * blockIdx.y + threadIdx.y;
    if (y >= dst_rows || px >= dst_cols) return;

    __half* out_row = reinterpret_cast<__half*>(dst) + y * dst_cols + px;
    int plane = dst_cols * dst_rows;

    const uint8_t* __restrict__ s = static_cast<const uint8_t*>(src);

    // y 方向只算一次
    float src_y = ((static_cast<float>(y) + 0.5f) - static_cast<float>(params.pad_y)) / params.scale - 0.5f;
    int   y0 = __float2int_rd(src_y);
    int   y1 = y0 + 1;
    float wy1 = src_y - y0;
    float wy0 = 1.f - wy1;

    float r[4], g[4], b[4];
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
        float3 p00 = inb(x0,y0) ? sample_rgb_bgra(s, src_pitch, x0, y0, src_cols, src_rows, config.border_value) : make_float3(config.border_value, config.border_value, config.border_value);
        float3 p10 = inb(x1,y0) ? sample_rgb_bgra(s, src_pitch, x1, y0, src_cols, src_rows, config.border_value) : make_float3(config.border_value, config.border_value, config.border_value);
        float3 p01 = inb(x0,y1) ? sample_rgb_bgra(s, src_pitch, x0, y1, src_cols, src_rows, config.border_value) : make_float3(config.border_value, config.border_value, config.border_value);
        float3 p11 = inb(x1,y1) ? sample_rgb_bgra(s, src_pitch, x1, y1, src_cols, src_rows, config.border_value) : make_float3(config.border_value, config.border_value, config.border_value);

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
        __half2 r2a = __halves2half2(__float2half(r[0]), __float2half(r[1]));
        __half2 r2b = __halves2half2(__float2half(r[2]), __float2half(r[3]));
        __half2 g2a = __halves2half2(__float2half(g[0]), __float2half(g[1]));
        __half2 g2b = __halves2half2(__float2half(g[2]), __float2half(g[3]));
        __half2 b2a = __halves2half2(__float2half(b[0]), __float2half(b[1]));
        __half2 b2b = __halves2half2(__float2half(b[2]), __float2half(b[3]));
        reinterpret_cast<__half2*>(out_row)[0]             = r2a;
        reinterpret_cast<__half2*>(out_row)[1]             = r2b;
        reinterpret_cast<__half2*>(out_row + plane)[0]     = g2a;
        reinterpret_cast<__half2*>(out_row + plane)[1]     = g2b;
        reinterpret_cast<__half2*>(out_row + plane * 2)[0] = b2a;
        reinterpret_cast<__half2*>(out_row + plane * 2)[1] = b2b;
    } else {
        int n = dst_cols - px;
        #pragma unroll
        for (int k=0;k<4 && k<n;++k){
            out_row[k]               = __float2half(r[k]);
            out_row[plane + k]       = __float2half(g[k]);
            out_row[plane*2 + k]     = __float2half(b[k]);
        }
    }
}

// multi-image Vec4 主体实现（batch via grid.z）
__device__ inline void multiLetterboxHalfVec4_body(const void* __restrict__ src, int src_cols, int src_rows, size_t src_pitch,
                                          void* __restrict__ dst, int dst_cols, int dst_rows,
                                          LetterboxTransform params, ProcessConfig config,
                                          int num_images) {
    int idx = blockIdx.z;
    if (idx >= num_images) return;

    int px = (blockDim.x * blockIdx.x + threadIdx.x) * 4; // four pixels per thread
    int y  = blockDim.y * blockIdx.y + threadIdx.y;
    if (y >= dst_rows || px >= dst_cols) return;

    const uint8_t* __restrict__ src_ptr = static_cast<const uint8_t*>(src) + idx * src_rows * src_pitch;
    __half* dst_ptr = reinterpret_cast<__half*>(dst) + idx * dst_rows * dst_cols * 3;

    __half* out_row = dst_ptr + y * dst_cols + px;
    int plane = dst_cols * dst_rows;

    float src_y = ((static_cast<float>(y) + 0.5f) - static_cast<float>(params.pad_y)) / params.scale - 0.5f;
    int   y0 = __float2int_rd(src_y);
    int   y1 = y0 + 1;
    float wy1 = src_y - y0;
    float wy0 = 1.f - wy1;

    float r[4], g[4], b[4];
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
        float3 p00 = inb(x0,y0) ? sample_rgb_bgra(src_ptr, src_pitch, x0, y0, src_cols, src_rows, config.border_value) : make_float3(config.border_value, config.border_value, config.border_value);
        float3 p10 = inb(x1,y0) ? sample_rgb_bgra(src_ptr, src_pitch, x1, y0, src_cols, src_rows, config.border_value) : make_float3(config.border_value, config.border_value, config.border_value);
        float3 p01 = inb(x0,y1) ? sample_rgb_bgra(src_ptr, src_pitch, x0, y1, src_cols, src_rows, config.border_value) : make_float3(config.border_value, config.border_value, config.border_value);
        float3 p11 = inb(x1,y1) ? sample_rgb_bgra(src_ptr, src_pitch, x1, y1, src_cols, src_rows, config.border_value) : make_float3(config.border_value, config.border_value, config.border_value);

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
        __half2 r2a = __halves2half2(__float2half(r[0]), __float2half(r[1]));
        __half2 r2b = __halves2half2(__float2half(r[2]), __float2half(r[3]));
        __half2 g2a = __halves2half2(__float2half(g[0]), __float2half(g[1]));
        __half2 g2b = __halves2half2(__float2half(g[2]), __float2half(g[3]));
        __half2 b2a = __halves2half2(__float2half(b[0]), __float2half(b[1]));
        __half2 b2b = __halves2half2(__float2half(b[2]), __float2half(b[3]));
        reinterpret_cast<__half2*>(out_row)[0]             = r2a;
        reinterpret_cast<__half2*>(out_row)[1]             = r2b;
        reinterpret_cast<__half2*>(out_row + plane)[0]     = g2a;
        reinterpret_cast<__half2*>(out_row + plane)[1]     = g2b;
        reinterpret_cast<__half2*>(out_row + plane * 2)[0] = b2a;
        reinterpret_cast<__half2*>(out_row + plane * 2)[1] = b2b;
    } else {
        int n = dst_cols - px;
        #pragma unroll
        for (int k=0;k<4 && k<n;++k){
            out_row[k]               = __float2half(r[k]);
            out_row[plane + k]       = __float2half(g[k]);
            out_row[plane*2 + k]     = __float2half(b[k]);
        }
    }
}

} // namespace trtyolo
