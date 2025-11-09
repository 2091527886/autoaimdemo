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

        float src_x = ((static_cast<float>(x) + 0.5f) - static_cast<float>(params.pad_x)) / params.scale - 0.5f;
        float src_y = ((static_cast<float>(y) + 0.5f) - static_cast<float>(params.pad_y)) / params.scale - 0.5f;

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

/* } */

/* } */

// Vectorized multi-image kernels (batch via grid.z)
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

        float src_x = ((static_cast<float>(x) + 0.5f) - static_cast<float>(params.pad_x)) / params.scale - 0.5f;
        float src_y = ((static_cast<float>(y) + 0.5f) - static_cast<float>(params.pad_y)) / params.scale - 0.5f;

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

/* } */

// Half path
void cudaLetterboxHalf(const void* src, int src_cols, int src_rows, size_t src_pitch,
                       void* dst, int dst_cols, int dst_rows,
                       const LetterboxTransform& params, const ProcessConfig config, cudaStream_t stream) {
    // 强制 half2 向量化路径
    dim3 block(32, 8);
    dim3 grid(iDivUp((dst_cols + 1) / 2, block.x), iDivUp(dst_rows, block.y));
    gpuLetterboxF16Vec2<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
}

void cudaMultiLetterboxHalf(const void* src, int src_cols, int src_rows, size_t src_pitch,
                            void* dst, int dst_cols, int dst_rows,
                            const LetterboxTransform& params, const ProcessConfig config,
                            int num_images, cudaStream_t stream) {
    // 强制 half2 向量化路径
    dim3 block(32, 8);
    dim3 grid(iDivUp((dst_cols + 1) / 2, block.x), iDivUp(dst_rows, block.y), num_images);
    gpuMultiLetterboxHalfVec2<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
}

} // namespace trtyolo
