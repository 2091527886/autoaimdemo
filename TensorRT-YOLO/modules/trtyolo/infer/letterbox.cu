#include <cstdint>
#include "letterbox.hpp"

namespace trtyolo {

inline __device__ __host__ int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ inline bool in_bounds(int x, int y, int cols, int rows) {
    return (x >= 0 && x < cols && y >= 0 && y < rows);
}

#include <cuda_fp16.h>

// Vectorized helpers and kernels added by Rovo Dev
namespace {
// Sample RGB from BGRA with border handling
__device__ inline float3 sample_rgb_bgra(const uint8_t* src, size_t pitch, int x, int y, int cols, int rows, float border) {
    if (x < 0 || x >= cols || y < 0 || y >= rows) {
        return make_float3(border, border, border);
    }
    const uchar4* line = reinterpret_cast<const uchar4*>(src + y * pitch);
    uchar4 p = line[x];
    // BGRA -> RGB
    return make_float3(p.z, p.y, p.x);
}
}

// F32 vectorized: 2 pixels per thread
__global__ void gpuLetterboxF32Vec2(const void* src, int src_cols, int src_rows, size_t src_pitch,
                                    void* dst, int dst_cols, int dst_rows,
                                    LetterboxTransform params, ProcessConfig config) {
    int px = (blockDim.x * blockIdx.x + threadIdx.x) * 2; // start x for this thread (2 pixels)
    int y  = blockDim.y * blockIdx.y + threadIdx.y;
    if (y >= dst_rows || px >= dst_cols) return;

    float* out_row = static_cast<float*>(dst) + y * dst_cols + px;
    int plane = dst_cols * dst_rows;

    const uint8_t* s = static_cast<const uint8_t*>(src);

    float r[2], g[2], b[2];
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        int x = px + k;
        if (x >= dst_cols) { r[k]=g[k]=b[k]=0.f; continue; }

        float src_x = (static_cast<float>(x) - static_cast<float>(params.pad_x)) / params.scale;
        float src_y = (static_cast<float>(y) - static_cast<float>(params.pad_y)) / params.scale;

        int   x0 = __float2int_rd(src_x);
        int   y0 = __float2int_rd(src_y);
        int   x1 = x0 + 1;
        int   y1 = y0 + 1;

        float wx1 = src_x - x0;
        float wy1 = src_y - y0;
        float wx0 = 1.f - wx1;
        float wy0 = 1.f - wy1;

        float3 p00 = sample_rgb_bgra(s, src_pitch, x0, y0, src_cols, src_rows, config.border_value);
        float3 p10 = sample_rgb_bgra(s, src_pitch, x1, y0, src_cols, src_rows, config.border_value);
        float3 p01 = sample_rgb_bgra(s, src_pitch, x0, y1, src_cols, src_rows, config.border_value);
        float3 p11 = sample_rgb_bgra(s, src_pitch, x1, y1, src_cols, src_rows, config.border_value);

        float3 sum;
        sum.x = wx0 * wy0 * p00.x + wx1 * wy0 * p10.x + wx0 * wy1 * p01.x + wx1 * wy1 * p11.x;
        sum.y = wx0 * wy0 * p00.y + wx1 * wy0 * p10.y + wx0 * wy1 * p01.y + wx1 * wy1 * p11.y;
        sum.z = wx0 * wy0 * p00.z + wx1 * wy0 * p10.z + wx0 * wy1 * p01.z + wx1 * wy1 * p11.z;

        r[k] = sum.x * config.alpha.x + config.beta.x;
        g[k] = sum.y * config.alpha.y + config.beta.y;
        b[k] = sum.z * config.alpha.z + config.beta.z;
    }

    bool full = (px + 1) < dst_cols;
    if (full) {
        // vectorized stores for CHW planes
        reinterpret_cast<float2*>(out_row)[0]             = make_float2(r[0], r[1]);
        reinterpret_cast<float2*>(out_row + plane)[0]     = make_float2(g[0], g[1]);
        reinterpret_cast<float2*>(out_row + plane * 2)[0] = make_float2(b[0], b[1]);
    } else {
        out_row[0]               = r[0];
        out_row[plane]           = g[0];
        out_row[plane*2]         = b[0];
    }
}

// F16 vectorized: 2 pixels per thread using half2 stores
__global__ void gpuLetterboxF16Vec2(const void* src, int src_cols, int src_rows, size_t src_pitch,
                                    void* dst, int dst_cols, int dst_rows,
                                    LetterboxTransform params, ProcessConfig config) {
    int px = (blockDim.x * blockIdx.x + threadIdx.x) * 2; // start x for this thread (2 pixels)
    int y  = blockDim.y * blockIdx.y + threadIdx.y;
    if (y >= dst_rows || px >= dst_cols) return;

    __half* out_row = reinterpret_cast<__half*>(dst) + y * dst_cols + px;
    int plane = dst_cols * dst_rows;

    const uint8_t* s = static_cast<const uint8_t*>(src);

    float r[2], g[2], b[2];
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        int x = px + k;
        if (x >= dst_cols) { r[k]=g[k]=b[k]=0.f; continue; }

        float src_x = (static_cast<float>(x) - static_cast<float>(params.pad_x)) / params.scale;
        float src_y = (static_cast<float>(y) - static_cast<float>(params.pad_y)) / params.scale;

        int   x0 = __float2int_rd(src_x);
        int   y0 = __float2int_rd(src_y);
        int   x1 = x0 + 1;
        int   y1 = y0 + 1;

        float wx1 = src_x - x0;
        float wy1 = src_y - y0;
        float wx0 = 1.f - wx1;
        float wy0 = 1.f - wy1;

        float3 p00 = sample_rgb_bgra(s, src_pitch, x0, y0, src_cols, src_rows, config.border_value);
        float3 p10 = sample_rgb_bgra(s, src_pitch, x1, y0, src_cols, src_rows, config.border_value);
        float3 p01 = sample_rgb_bgra(s, src_pitch, x0, y1, src_cols, src_rows, config.border_value);
        float3 p11 = sample_rgb_bgra(s, src_pitch, x1, y1, src_cols, src_rows, config.border_value);

        float3 sum;
        sum.x = wx0 * wy0 * p00.x + wx1 * wy0 * p10.x + wx0 * wy1 * p01.x + wx1 * wy1 * p11.x;
        sum.y = wx0 * wy0 * p00.y + wx1 * wy0 * p10.y + wx0 * wy1 * p01.y + wx1 * wy1 * p11.y;
        sum.z = wx0 * wy0 * p00.z + wx1 * wy0 * p10.z + wx0 * wy1 * p01.z + wx1 * wy1 * p11.z;

        r[k] = sum.x * config.alpha.x + config.beta.x;
        g[k] = sum.y * config.alpha.y + config.beta.y;
        b[k] = sum.z * config.alpha.z + config.beta.z;
    }

    // pack and store when we have two pixels available; else scalar tail
    bool full = (px + 1) < dst_cols;
    if (full) {
        __half2 r2 = __halves2half2(__float2half(r[0]), __float2half(r[1]));
        __half2 g2 = __halves2half2(__float2half(g[0]), __float2half(g[1]));
        __half2 b2 = __halves2half2(__float2half(b[0]), __float2half(b[1]));
        reinterpret_cast<__half2*>(out_row)[0]             = r2;
        reinterpret_cast<__half2*>(out_row + plane)[0]     = g2;
        reinterpret_cast<__half2*>(out_row + plane * 2)[0] = b2;
    } else {
        out_row[0]               = __float2half(r[0]);
        out_row[plane]           = __float2half(g[0]);
        out_row[plane*2]         = __float2half(b[0]);
    }
}

__device__ void letterbox_bilinear_f32(const uint8_t* src, const int src_cols, const int src_rows, const size_t src_pitch,
                                       float* dst, const int dst_cols, const int dst_rows,
                                       const LetterboxTransform params, const ProcessConfig config,
                                       int element_x, int element_y) {
    if (element_x >= dst_cols || element_y >= dst_rows) return;

    // 反变换：把目标点映射回原图坐标
    float src_x = (static_cast<float>(element_x) - static_cast<float>(params.pad_x)) / params.scale;
    float src_y = (static_cast<float>(element_y) - static_cast<float>(params.pad_y)) / params.scale;

    int   x0 = __float2int_rd(src_x);
    int   y0 = __float2int_rd(src_y);
    int   x1 = x0 + 1;
    int   y1 = y0 + 1;

    float wx1 = src_x - x0;
    float wy1 = src_y - y0;
    float wx0 = 1.f - wx1;
    float wy0 = 1.f - wy1;

    float3 sum = make_float3(0, 0, 0);

    auto sample = [&](int xx, int yy) {
        if (!in_bounds(xx, yy, src_cols, src_rows)) {
            float v = config.border_value;
            return make_float3(v, v, v);
        }
        const uchar4* line = reinterpret_cast<const uchar4*>(reinterpret_cast<const uint8_t*>(src) + yy * src_pitch);
        uchar4 p = line[xx]; // BGRA
        return make_float3(p.z, p.y, p.x); // RGB
    };

    float3 p00 = sample(x0, y0);
    float3 p10 = sample(x1, y0);
    float3 p01 = sample(x0, y1);
    float3 p11 = sample(x1, y1);

    sum.x = wx0 * wy0 * p00.x + wx1 * wy0 * p10.x + wx0 * wy1 * p01.x + wx1 * wy1 * p11.x;
    sum.y = wx0 * wy0 * p00.y + wx1 * wy0 * p10.y + wx0 * wy1 * p01.y + wx1 * wy1 * p11.y;
    sum.z = wx0 * wy0 * p00.z + wx1 * wy0 * p10.z + wx0 * wy1 * p01.z + wx1 * wy1 * p11.z;

    int plane = dst_cols * dst_rows;
    float* out = dst + element_y * dst_cols + element_x;
    out[0]           = sum.x * config.alpha.x + config.beta.x;
    out[plane]       = sum.y * config.alpha.y + config.beta.y;
    out[plane * 2]   = sum.z * config.alpha.z + config.beta.z;
}

__global__ void gpuLetterbox(const void* src, int src_cols, int src_rows, size_t src_pitch,
                             void* dst, int dst_cols, int dst_rows,
                             LetterboxTransform params, ProcessConfig config) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    letterbox_bilinear_f32(static_cast<const uint8_t*>(src), src_cols, src_rows, src_pitch,
                       static_cast<float*>(dst), dst_cols, dst_rows,
                       params, config, x, y);
}

// Vectorized multi-image kernels (batch via grid.z)
__global__ void gpuMultiLetterboxF32Vec2(const void* src, int src_cols, int src_rows, size_t src_pitch,
                                         void* dst, int dst_cols, int dst_rows,
                                         LetterboxTransform params, ProcessConfig config,
                                         int num_images) {
    int idx = blockIdx.z;
    if (idx >= num_images) return;

    int px = (blockDim.x * blockIdx.x + threadIdx.x) * 2; // two pixels per thread
    int y  = blockDim.y * blockIdx.y + threadIdx.y;
    if (y >= dst_rows || px >= dst_cols) return;

    const uint8_t* src_ptr = static_cast<const uint8_t*>(src) + idx * src_rows * src_pitch;
    float* dst_ptr = static_cast<float*>(dst) + idx * dst_rows * dst_cols * 3;

    float* out_row = dst_ptr + y * dst_cols + px;
    int plane = dst_cols * dst_rows;

    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        int x = px + k;
        if (x >= dst_cols) break;

        float src_x = (static_cast<float>(x) - static_cast<float>(params.pad_x)) / params.scale;
        float src_y = (static_cast<float>(y) - static_cast<float>(params.pad_y)) / params.scale;

        int   x0 = __float2int_rd(src_x);
        int   y0 = __float2int_rd(src_y);
        int   x1 = x0 + 1;
        int   y1 = y0 + 1;

        float wx1 = src_x - x0;
        float wy1 = src_y - y0;
        float wx0 = 1.f - wx1;
        float wy0 = 1.f - wy1;

        float3 p00 = sample_rgb_bgra(src_ptr, src_pitch, x0, y0, src_cols, src_rows, config.border_value);
        float3 p10 = sample_rgb_bgra(src_ptr, src_pitch, x1, y0, src_cols, src_rows, config.border_value);
        float3 p01 = sample_rgb_bgra(src_ptr, src_pitch, x0, y1, src_cols, src_rows, config.border_value);
        float3 p11 = sample_rgb_bgra(src_ptr, src_pitch, x1, y1, src_cols, src_rows, config.border_value);

        float3 sum;
        sum.x = wx0 * wy0 * p00.x + wx1 * wy0 * p10.x + wx0 * wy1 * p01.x + wx1 * wy1 * p11.x;
        sum.y = wx0 * wy0 * p00.y + wx1 * wy0 * p10.y + wx0 * wy1 * p01.y + wx1 * wy1 * p11.y;
        sum.z = wx0 * wy0 * p00.z + wx1 * wy0 * p10.z + wx0 * wy1 * p01.z + wx1 * wy1 * p11.z;

        float r = sum.x * config.alpha.x + config.beta.x;
        float g = sum.y * config.alpha.y + config.beta.y;
        float b = sum.z * config.alpha.z + config.beta.z;

        float* out = out_row + k;
        out[0]           = r;
        out[plane]       = g;
        out[plane * 2]   = b;
    }
}

__global__ void gpuMultiLetterboxHalfVec2(const void* src, int src_cols, int src_rows, size_t src_pitch,
                                          void* dst, int dst_cols, int dst_rows,
                                          LetterboxTransform params, ProcessConfig config,
                                          int num_images) {
    int idx = blockIdx.z;
    if (idx >= num_images) return;

    int px = (blockDim.x * blockIdx.x + threadIdx.x) * 2; // two pixels per thread
    int y  = blockDim.y * blockIdx.y + threadIdx.y;
    if (y >= dst_rows || px >= dst_cols) return;

    const uint8_t* src_ptr = static_cast<const uint8_t*>(src) + idx * src_rows * src_pitch;
    __half* dst_ptr = reinterpret_cast<__half*>(dst) + idx * dst_rows * dst_cols * 3;

    __half* out_row = dst_ptr + y * dst_cols + px;
    int plane = dst_cols * dst_rows;

    float r[2], g[2], b[2];
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        int x = px + k;
        if (x >= dst_cols) { r[k]=g[k]=b[k]=0.f; continue; }

        float src_x = (static_cast<float>(x) - static_cast<float>(params.pad_x)) / params.scale;
        float src_y = (static_cast<float>(y) - static_cast<float>(params.pad_y)) / params.scale;

        int   x0 = __float2int_rd(src_x);
        int   y0 = __float2int_rd(src_y);
        int   x1 = x0 + 1;
        int   y1 = y0 + 1;

        float wx1 = src_x - x0;
        float wy1 = src_y - y0;
        float wx0 = 1.f - wx1;
        float wy0 = 1.f - wy1;

        float3 p00 = sample_rgb_bgra(src_ptr, src_pitch, x0, y0, src_cols, src_rows, config.border_value);
        float3 p10 = sample_rgb_bgra(src_ptr, src_pitch, x1, y0, src_cols, src_rows, config.border_value);
        float3 p01 = sample_rgb_bgra(src_ptr, src_pitch, x0, y1, src_cols, src_rows, config.border_value);
        float3 p11 = sample_rgb_bgra(src_ptr, src_pitch, x1, y1, src_cols, src_rows, config.border_value);

        float3 sum;
        sum.x = wx0 * wy0 * p00.x + wx1 * wy0 * p10.x + wx0 * wy1 * p01.x + wx1 * wy1 * p11.x;
        sum.y = wx0 * wy0 * p00.y + wx1 * wy0 * p10.y + wx0 * wy1 * p01.y + wx1 * wy1 * p11.y;
        sum.z = wx0 * wy0 * p00.z + wx1 * wy0 * p10.z + wx0 * wy1 * p01.z + wx1 * wy1 * p11.z;

        r[k] = sum.x * config.alpha.x + config.beta.x;
        g[k] = sum.y * config.alpha.y + config.beta.y;
        b[k] = sum.z * config.alpha.z + config.beta.z;
    }

    bool full = (px + 1) < dst_cols;
    if (full) {
        __half2 r2 = __halves2half2(__float2half(r[0]), __float2half(r[1]));
        __half2 g2 = __halves2half2(__float2half(g[0]), __float2half(g[1]));
        __half2 b2 = __halves2half2(__float2half(b[0]), __float2half(b[1]));
        reinterpret_cast<__half2*>(out_row)[0]             = r2;
        reinterpret_cast<__half2*>(out_row + plane)[0]     = g2;
        reinterpret_cast<__half2*>(out_row + plane * 2)[0] = b2;
    } else {
        out_row[0]               = __float2half(r[0]);
        out_row[plane]           = __float2half(g[0]);
        out_row[plane*2]         = __float2half(b[0]);
    }
}

__global__ void gpuMultiLetterbox(const void* src, int src_cols, int src_rows, size_t src_pitch,
                                  void* dst, int dst_cols, int dst_rows,
                                  LetterboxTransform params, ProcessConfig config,
                                  int num_images) {
    int idx = blockIdx.z;
    if (idx >= num_images) return;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    const uint8_t* src_ptr = static_cast<const uint8_t*>(src) + idx * src_rows * src_pitch;
    float* dst_ptr = static_cast<float*>(dst) + idx * dst_rows * dst_cols * 3;

    letterbox_bilinear_f32(src_ptr, src_cols, src_rows, src_pitch,
                       dst_ptr, dst_cols, dst_rows,
                       params, config, x, y);
}

void cudaLetterbox(const void* src, int src_cols, int src_rows, size_t src_pitch,
                   void* dst, int dst_cols, int dst_rows,
                   const LetterboxTransform& params, const ProcessConfig config, cudaStream_t stream) {
    // Prefer vectorized path when width is even
    if ((dst_cols & 1) == 0) {
        dim3 block(32, 8);
        dim3 grid(iDivUp(dst_cols / 2, block.x), iDivUp(dst_rows, block.y));
        gpuLetterboxF32Vec2<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        return;
    }
    dim3 block(16, 16);
    dim3 grid(iDivUp(dst_cols, block.x), iDivUp(dst_rows, block.y));
    gpuLetterbox<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
}

void cudaMultiLetterbox(const void* src, int src_cols, int src_rows, size_t src_pitch,
                        void* dst, int dst_cols, int dst_rows,
                        const LetterboxTransform& params, const ProcessConfig config,
                        int num_images, cudaStream_t stream) {
    if ((dst_cols & 1) == 0) {
        dim3 block(32, 8);
        dim3 grid(iDivUp(dst_cols / 2, block.x), iDivUp(dst_rows, block.y), num_images);
        gpuMultiLetterboxF32Vec2<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        return;
    }
    dim3 block(16, 16);
    dim3 grid(iDivUp(dst_cols, block.x), iDivUp(dst_rows, block.y), num_images);
    gpuMultiLetterbox<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
}

// Half path
__device__ void letterbox_bilinear_f16(const uint8_t* src, const int src_cols, const int src_rows, const size_t src_pitch,
                                       __half* dst, const int dst_cols, const int dst_rows,
                                       const LetterboxTransform params, const ProcessConfig config,
                                       int element_x, int element_y) {
    if (element_x >= dst_cols || element_y >= dst_rows) return;

    float src_x = (static_cast<float>(element_x) - static_cast<float>(params.pad_x)) / params.scale;
    float src_y = (static_cast<float>(element_y) - static_cast<float>(params.pad_y)) / params.scale;

    int   x0 = __float2int_rd(src_x);
    int   y0 = __float2int_rd(src_y);
    int   x1 = x0 + 1;
    int   y1 = y0 + 1;

    float wx1 = src_x - x0;
    float wy1 = src_y - y0;
    float wx0 = 1.f - wx1;
    float wy0 = 1.f - wy1;

    float3 sum = make_float3(0, 0, 0);

    auto sample = [&](int xx, int yy) {
        if (!in_bounds(xx, yy, src_cols, src_rows)) {
            float v = config.border_value;
            return make_float3(v, v, v);
        }
        const uchar4* line = reinterpret_cast<const uchar4*>(reinterpret_cast<const uint8_t*>(src) + yy * src_pitch);
        uchar4 p = line[xx];
        return make_float3(p.z, p.y, p.x);
    };

    float3 p00 = sample(x0, y0);
    float3 p10 = sample(x1, y0);
    float3 p01 = sample(x0, y1);
    float3 p11 = sample(x1, y1);

    sum.x = wx0 * wy0 * p00.x + wx1 * wy0 * p10.x + wx0 * wy1 * p01.x + wx1 * wy1 * p11.x;
    sum.y = wx0 * wy0 * p00.y + wx1 * wy0 * p10.y + wx0 * wy1 * p01.y + wx1 * wy1 * p11.y;
    sum.z = wx0 * wy0 * p00.z + wx1 * wy0 * p10.z + wx0 * wy1 * p01.z + wx1 * wy1 * p11.z;

    int plane = dst_cols * dst_rows;
    __half* out = dst + element_y * dst_cols + element_x;
    out[0]         = __float2half(sum.x * config.alpha.x + config.beta.x);
    out[plane]     = __float2half(sum.y * config.alpha.y + config.beta.y);
    out[plane * 2] = __float2half(sum.z * config.alpha.z + config.beta.z);
}

__global__ void gpuLetterboxHalf(const void* src, int src_cols, int src_rows, size_t src_pitch,
                                 void* dst, int dst_cols, int dst_rows,
                                 LetterboxTransform params, ProcessConfig config) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    letterbox_bilinear_f16(static_cast<const uint8_t*>(src), src_cols, src_rows, src_pitch,
                           reinterpret_cast<__half*>(dst), dst_cols, dst_rows,
                           params, config, x, y);
}

__global__ void gpuMultiLetterboxHalf(const void* src, int src_cols, int src_rows, size_t src_pitch,
                                      void* dst, int dst_cols, int dst_rows,
                                      LetterboxTransform params, ProcessConfig config,
                                      int num_images) {
    int idx = blockIdx.z;
    if (idx >= num_images) return;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    const uint8_t* src_ptr = static_cast<const uint8_t*>(src) + idx * src_rows * src_pitch;
    __half* dst_ptr = reinterpret_cast<__half*>(dst) + idx * dst_rows * dst_cols * 3;

    letterbox_bilinear_f16(src_ptr, src_cols, src_rows, src_pitch,
                           dst_ptr, dst_cols, dst_rows,
                           params, config, x, y);
}

void cudaLetterboxHalf(const void* src, int src_cols, int src_rows, size_t src_pitch,
                       void* dst, int dst_cols, int dst_rows,
                       const LetterboxTransform& params, const ProcessConfig config, cudaStream_t stream) {
    // Prefer half2 vectorized path when width is even
    if ((dst_cols & 1) == 0) {
        dim3 block(32, 8);
        dim3 grid(iDivUp(dst_cols / 2, block.x), iDivUp(dst_rows, block.y));
        gpuLetterboxF16Vec2<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        return;
    }
    dim3 block(16, 16);
    dim3 grid(iDivUp(dst_cols, block.x), iDivUp(dst_rows, block.y));
    gpuLetterboxHalf<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
}

void cudaMultiLetterboxHalf(const void* src, int src_cols, int src_rows, size_t src_pitch,
                            void* dst, int dst_cols, int dst_rows,
                            const LetterboxTransform& params, const ProcessConfig config,
                            int num_images, cudaStream_t stream) {
    if ((dst_cols & 1) == 0) {
        dim3 block(32, 8);
        dim3 grid(iDivUp(dst_cols / 2, block.x), iDivUp(dst_rows, block.y), num_images);
        gpuMultiLetterboxHalfVec2<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        return;
    }
    dim3 block(16, 16);
    dim3 grid(iDivUp(dst_cols, block.x), iDivUp(dst_rows, block.y), num_images);
    gpuMultiLetterboxHalf<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
}

} // namespace trtyolo
