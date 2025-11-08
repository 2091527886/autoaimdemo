#pragma once

#include <cuda_runtime_api.h>
#include "utils/common.hpp"

namespace trtyolo {

struct LetterboxTransform {
    // 等比缩放参数与 padding（目标尺度下的偏移）
    float scale{1.0f};
    int   pad_x{0};
    int   pad_y{0};
    int   last_src_width_{-1};
    int   last_src_height_{-1};

    // 为兼容旧代码，提供同名字段
    int   dst_offset_x{0};
    int   dst_offset_y{0};

    void updateParams(int src_width, int src_height, int dst_width, int dst_height) {
        if (src_width == last_src_width_ && src_height == last_src_height_) return;
        last_src_width_  = src_width;
        last_src_height_ = src_height;

        float rs = std::min(static_cast<float>(dst_width) / static_cast<float>(src_width),
                            static_cast<float>(dst_height) / static_cast<float>(src_height));
        scale     = rs;
        pad_x     = static_cast<int>(dst_width * 0.5f - 0.5f * rs * src_width);
        pad_y     = static_cast<int>(dst_height * 0.5f - 0.5f * rs * src_height);
        dst_offset_x = pad_x;
        dst_offset_y = pad_y;
    }

    // 将目标坐标(dst)映射回源坐标(src)，保持与旧 affine.applyTransform 语义一致
    void applyTransform(float x, float y, float* out_x, float* out_y) const {
        *out_x = (x - static_cast<float>(pad_x)) / scale;
        *out_y = (y - static_cast<float>(pad_y)) / scale;
    }
};

// 单图 letterbox 预处理，输入 BGRA（uchar4），输出 CHW float 并应用归一化
void cudaLetterbox(const void* src, int src_cols, int src_rows, size_t src_pitch,
                   void* dst, int dst_cols, int dst_rows,
                   const LetterboxTransform& params, const ProcessConfig config, cudaStream_t stream);

// 多图批处理版本，src/dst 是连续内存（与现 backend 使用一致）
void cudaMultiLetterbox(const void* src, int src_cols, int src_rows, size_t src_pitch,
                        void* dst, int dst_cols, int dst_rows,
                        const LetterboxTransform& params, const ProcessConfig config,
                        int num_images, cudaStream_t stream);

} // namespace trtyolo
