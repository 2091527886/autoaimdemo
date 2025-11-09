#include <cstdint>
#include "letterbox.hpp"

namespace trtyolo {

inline __device__ __host__ int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ inline bool in_bounds(int x, int y, int cols, int rows) {
    return (x >= 0 && x < cols && y >= 0 && y < rows);
}

#include <cuda_runtime.h>
#include <cuda.h>
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

// 核心实现体，供不同 __global__ 内核复用
__device__ inline void letterboxF16Vec2_body(const void* src, int src_cols, int src_rows, size_t src_pitch,
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

// 通用版本（无 launch_bounds），F16 vectorized: 2 pixels/thread
__global__ void gpuLetterboxF16Vec2_generic(const void* src, int src_cols, int src_rows, size_t src_pitch,
                                    void* dst, int dst_cols, int dst_rows,
                                    LetterboxTransform params, ProcessConfig config) {
    letterboxF16Vec2_body(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
}

// 选择与调优说明（3060/SM86）
// 预设矩阵（threads/block 与形状）：
//   256: 32x8, 16x16, 64x4
//   320: 40x8, 20x16   （已提供模板分派）
//   384: 48x8, 24x16   （已提供模板分派）
//   512: 32x16, 64x8, 16x32, 8x64（已支持32x16/64x8模板，另两组少用）
// 快速选择：先 32x8/MINB=2 → 再 16x16 或 64x4 → 再 32x16/64x8（需实测寄存器/并发）。
// - 两像素/线程，planar输出，half2存储，BGRA->RGB 手写双线性。
// - 线程块预设：{32x8, 16x16, 64x4}，可扩展到 {32x16, 64x8}。
//   * 32x8: 稳定默认，寄存器与占用平衡。
//   * 16x16: 二维并行更均衡，适合方形/接近方形目标输出。
//   * 64x4: X向访存合并更好，适合宽图；留意寄存器与指令吞吐。
// - MINB（min_blocks_per_sm）建议 2；若寄存器压力高，可降为 1；若资源宽裕，可试 3。
// - use_occupancy=1 时会在候选集中试算 active blocks，便于探索；正式跑请固定块。
// - 若需继续优化：
//   * 将BGRA采样做成成对预取（两像素同线，减少边界分支），
//   * 用常量内存或只读缓存加载alpha/beta，
//   * 调整寄存器限制（--maxrregcount）与 launch_bounds 协同。

// 启用 launch_bounds 的模板内核
template<int BX, int BY, int MINB>
__launch_bounds__(BX*BY, MINB)
__global__ void gpuLetterboxF16Vec2_lb(const void* src, int src_cols, int src_rows, size_t src_pitch,
                                    void* dst, int dst_cols, int dst_rows,
                                    LetterboxTransform params, ProcessConfig config) {
    letterboxF16Vec2_body(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
}

// Vectorized multi-image kernels (batch via grid.z)
__device__ inline void multiLetterboxHalfVec2_body(const void* src, int src_cols, int src_rows, size_t src_pitch,
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

// Vectorized multi-image kernels (batch via grid.z)
__global__ void gpuMultiLetterboxHalfVec2_generic(const void* src, int src_cols, int src_rows, size_t src_pitch,
                                          void* dst, int dst_cols, int dst_rows,
                                          LetterboxTransform params, ProcessConfig config,
                                          int num_images) {
    multiLetterboxHalfVec2_body(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
}

// 启用 launch_bounds 的模板多图内核
template<int BX, int BY, int MINB>
__launch_bounds__(BX*BY, MINB)
__global__ void gpuMultiLetterboxHalfVec2_lb(const void* src, int src_cols, int src_rows, size_t src_pitch,
                                          void* dst, int dst_cols, int dst_rows,
                                          LetterboxTransform params, ProcessConfig config,
                                          int num_images) {
    multiLetterboxHalfVec2_body(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
}

// 选择块大小（占用率推荐或配置）
// 选择块大小（占用推荐或配置）；若 use_occupancy=1，将在候选集中估算 active blocks 选择最优
static inline dim3 pickBlock(const ProcessConfig& cfg, int dst_cols) {
    int bx = cfg.block_x > 0 ? cfg.block_x : 32;
    int by = cfg.block_y > 0 ? cfg.block_y : 8;
    if (cfg.use_occupancy) {
        // 候选配置集合（针对3060 SM86）
        const int cand[][2] = {{32,8},{16,16},{64,4},{32,16},{64,8},{48,8},{24,16},{40,8},{20,16}};
        int best_bx = bx, best_by = by; int best_act = -1;
        for (auto& c : cand) {
            int cbx = c[0], cby = c[1];
            int threads = cbx * cby;
            int active = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active,
                (const void*)gpuLetterboxF16Vec2_generic,
                threads,
                0);
            if (active > best_act) { best_act = active; best_bx = cbx; best_by = cby; }
        }
        bx = best_bx; by = best_by;
    }
    // 对齐到2像素/线程：以像素对为单位
    return dim3(bx, by, 1);
}

// Half path
// 日志建议：可在上层记录最终选择 (block_x,block_y,MINB,use_launch_bounds,use_occupancy) 便于复现与对比。
void cudaLetterboxHalf(const void* src, int src_cols, int src_rows, size_t src_pitch,
                       void* dst, int dst_cols, int dst_rows,
                       const LetterboxTransform& params, const ProcessConfig config, cudaStream_t stream) {
    dim3 block = pickBlock(config, dst_cols);
    dim3 grid(iDivUp((dst_cols + 1) / 2, block.x), iDivUp(dst_rows, block.y));
    if (config.use_launch_bounds) {
        // 模板分派（常见配置），否则退回通用
        int MINB = config.min_blocks_per_sm; MINB = (MINB < 1 ? 1 : (MINB > 3 ? 3 : MINB));
        if (block.x == 32 && block.y == 8) {
            if (MINB<=1)      gpuLetterboxF16Vec2_lb<32,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else if (MINB==2) gpuLetterboxF16Vec2_lb<32,8,2><<<grid, block, 0, stream>>> (src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else              gpuLetterboxF16Vec2_lb<32,8,3><<<grid, block, 0, stream>>> (src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        } else if (block.x == 16 && block.y == 16) {
            if (MINB<=1)      gpuLetterboxF16Vec2_lb<16,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else if (MINB==2) gpuLetterboxF16Vec2_lb<16,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else              gpuLetterboxF16Vec2_lb<16,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        } else if (block.x == 64 && block.y == 4) {
            if (MINB<=1)      gpuLetterboxF16Vec2_lb<64,4,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else if (MINB==2) gpuLetterboxF16Vec2_lb<64,4,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else              gpuLetterboxF16Vec2_lb<64,4,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        } else if (block.x == 32 && block.y == 16) {
            if (MINB<=1)      gpuLetterboxF16Vec2_lb<32,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else if (MINB==2) gpuLetterboxF16Vec2_lb<32,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else              gpuLetterboxF16Vec2_lb<32,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        } else if (block.x == 64 && block.y == 8) {
            if (MINB<=1)      gpuLetterboxF16Vec2_lb<64,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else if (MINB==2) gpuLetterboxF16Vec2_lb<64,8,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else              gpuLetterboxF16Vec2_lb<64,8,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        } else if (block.x == 48 && block.y == 8) {
            if (MINB<=1)      gpuLetterboxF16Vec2_lb<48,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else if (MINB==2) gpuLetterboxF16Vec2_lb<48,8,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else              gpuLetterboxF16Vec2_lb<48,8,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        } else if (block.x == 24 && block.y == 16) {
            if (MINB<=1)      gpuLetterboxF16Vec2_lb<24,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else if (MINB==2) gpuLetterboxF16Vec2_lb<24,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else              gpuLetterboxF16Vec2_lb<24,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        } else if (block.x == 40 && block.y == 8) {
            if (MINB<=1)      gpuLetterboxF16Vec2_lb<40,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else if (MINB==2) gpuLetterboxF16Vec2_lb<40,8,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else              gpuLetterboxF16Vec2_lb<40,8,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        } else if (block.x == 20 && block.y == 16) {
            if (MINB<=1)      gpuLetterboxF16Vec2_lb<20,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else if (MINB==2) gpuLetterboxF16Vec2_lb<20,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
            else              gpuLetterboxF16Vec2_lb<20,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        } else {
            gpuLetterboxF16Vec2_generic<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
        }
    } else {
        gpuLetterboxF16Vec2_generic<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
    }
}

void cudaMultiLetterboxHalf(const void* src, int src_cols, int src_rows, size_t src_pitch,
                            void* dst, int dst_cols, int dst_rows,
                            const LetterboxTransform& params, const ProcessConfig config,
                            int num_images, cudaStream_t stream) {
    dim3 block = pickBlock(config, dst_cols);
    dim3 grid(iDivUp((dst_cols + 1) / 2, block.x), iDivUp(dst_rows, block.y), num_images);
    if (config.use_launch_bounds) {
        int MINB = config.min_blocks_per_sm; MINB = (MINB < 1 ? 1 : (MINB > 3 ? 3 : MINB));
        if (block.x == 32 && block.y == 8) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec2_lb<32,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec2_lb<32,8,2><<<grid, block, 0, stream>>> (src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec2_lb<32,8,3><<<grid, block, 0, stream>>> (src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 16 && block.y == 16) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec2_lb<16,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec2_lb<16,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec2_lb<16,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 64 && block.y == 4) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec2_lb<64,4,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec2_lb<64,4,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec2_lb<64,4,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 32 && block.y == 16) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec2_lb<32,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec2_lb<32,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec2_lb<32,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 64 && block.y == 8) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec2_lb<64,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec2_lb<64,8,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec2_lb<64,8,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 48 && block.y == 8) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec2_lb<48,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec2_lb<48,8,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec2_lb<48,8,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 24 && block.y == 16) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec2_lb<24,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec2_lb<24,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec2_lb<24,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 40 && block.y == 8) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec2_lb<40,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec2_lb<40,8,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec2_lb<40,8,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 20 && block.y == 16) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec2_lb<20,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec2_lb<20,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec2_lb<20,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else {
            gpuMultiLetterboxHalfVec2_generic<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        }
    } else {
        gpuMultiLetterboxHalfVec2_generic<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
    }
}

} // namespace trtyolo
