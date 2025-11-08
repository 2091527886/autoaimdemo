#include <cstdint>
#include "letterbox.hpp"

namespace trtyolo {

inline __device__ __host__ int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ inline bool in_bounds(int x, int y, int cols, int rows) {
    return (x >= 0 && x < cols && y >= 0 && y < rows);
}

__device__ void letterbox_bilinear(const uint8_t* src, const int src_cols, const int src_rows, const size_t src_pitch,
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
    letterbox_bilinear(static_cast<const uint8_t*>(src), src_cols, src_rows, src_pitch,
                       static_cast<float*>(dst), dst_cols, dst_rows,
                       params, config, x, y);
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

    letterbox_bilinear(src_ptr, src_cols, src_rows, src_pitch,
                       dst_ptr, dst_cols, dst_rows,
                       params, config, x, y);
}

void cudaLetterbox(const void* src, int src_cols, int src_rows, size_t src_pitch,
                   void* dst, int dst_cols, int dst_rows,
                   const LetterboxTransform& params, const ProcessConfig config, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid(iDivUp(dst_cols, block.x), iDivUp(dst_rows, block.y));
    gpuLetterbox<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
}

void cudaMultiLetterbox(const void* src, int src_cols, int src_rows, size_t src_pitch,
                        void* dst, int dst_cols, int dst_rows,
                        const LetterboxTransform& params, const ProcessConfig config,
                        int num_images, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid(iDivUp(dst_cols, block.x), iDivUp(dst_rows, block.y), num_images);
    gpuMultiLetterbox<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
}

} // namespace trtyolo
