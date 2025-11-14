#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

namespace trtyolo {

inline __device__ __host__ int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ inline bool in_bounds(int x, int y, int cols, int rows) {
    return (x >= 0 && x < cols && y >= 0 && y < rows);
}

// Sample RGB from BGRA with border handling
static inline __device__ float3 sample_rgb_bgra(const uint8_t* src, size_t pitch, int x, int y, int cols, int rows, float border) {
    if (x < 0 || x >= cols || y < 0 || y >= rows) {
        return make_float3(border, border, border);
    }
    const uchar4* line = reinterpret_cast<const uchar4*>(src + y * pitch);
    uchar4 p = line[x];
    // BGRA -> RGB
    return make_float3(p.z, p.y, p.x);
}

} // namespace trtyolo
