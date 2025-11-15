#pragma once

#include "infer/backend.hpp"
#include "letterbox.hpp"
#include "core/core.hpp"
#include <vector>
#include <cstdint>
#include <cstring>

namespace trtyolo {

// 仅在 infer/ 后端内部复用的小工具，不改变原有接口与逻辑

// 固定输入尺寸时，将 host 源图逐行拷贝到连续 staging（inputs_buffer_->host）
inline void copy_fixed_to_staging(BaseBuffer* inputs_buffer,
                                  const std::vector<Image>& inputs,
                                  size_t image_bytes,
                                  size_t row_bytes,
                                  const InferConfig& infer_config){
    if (infer_config.cuda_mem) return;
    for (size_t idx = 0; idx < inputs.size(); ++idx) {
        uint8_t* dst = static_cast<uint8_t*>(inputs_buffer->host()) + idx * image_bytes;
        const uint8_t* src = static_cast<const uint8_t*>(inputs[idx].ptr);
        for (int r = 0; r < infer_config.input_shape->x; ++r) {
            std::memcpy(dst + r * row_bytes, src + r * inputs[idx].pitch, row_bytes);
        }
    }
}

// 变长输入时：计算每张输入字节数并累计，同时更新对应的 affine
inline void compute_sizes_and_affines(const std::vector<Image>& inputs,
                                      const int4& max_shape,
                                      std::vector<int>& input_sizes,
                                      int& total_size,
                                      std::vector<LetterboxTransform>& affs){
    int num = static_cast<int>(inputs.size());
    total_size = 0;
    input_sizes.assign(num, 0);
    for (int idx = 0; idx < num; ++idx) {
        input_sizes[idx] = inputs[idx].height * static_cast<int>(inputs[idx].pitch);
        total_size += input_sizes[idx];
        if (idx < static_cast<int>(affs.size()))
            affs[idx].updateParams(inputs[idx].width, inputs[idx].height, max_shape.w, max_shape.z);
    }
}

// 对某个 idx 调用 half-precision letterbox，按 dst 基址和 infer_size 偏移
inline void launch_letterbox_idx(const void* src,
                                 int src_w, int src_h, size_t src_pitch,
                                 void* dst_base,
                                 int dst_cols, int dst_rows,
                                 int infer_size,
                                 int idx,
                                 const LetterboxTransform& affine,
                                 const ProcessConfig& cfg,
                                 cudaStream_t stream){
    void* infer_device_ptr = static_cast<void*>(static_cast<uint16_t*>(dst_base) + idx * infer_size);
    cudaLetterboxHalf(src, src_w, src_h, src_pitch,
                      infer_device_ptr, dst_cols, dst_rows,
                      affine, cfg, stream);
}

// 静态图：更新单个 kernel node 的参数表
inline void update_kernel_params_single(CudaGraph& graph, int node_idx,
                                        const void* src_ptr,
                                        int src_w, int src_h, int src_pitch,
                                        void* dst_ptr,
                                        int dst_cols, int dst_rows,
                                        const LetterboxTransform* affine,
                                        const ProcessConfig* cfg){
    void* kernelParams[] = {
        (void*)&src_ptr,
        (void*)&src_w,
        (void*)&src_h,
        (void*)&src_pitch,
        (void*)&dst_ptr,
        (void*)&dst_cols,
        (void*)&dst_rows,
        (void*)affine,
        (void*)cfg
    };
    graph.updateKernelNodeParams(node_idx, kernelParams);
}

// 静态图：如需，更新 memcpy 节点参数
inline void update_memcpy_node_if_needed(CudaGraph& graph, int node_idx,
                                         void* h, void* d, size_t bytes,
                                         BufferType buffer_type){
    if (buffer_type == BufferType::Discrete)
        graph.updateMemcpyNodeParams(node_idx, h, d, bytes);
}

} // namespace trtyolo
