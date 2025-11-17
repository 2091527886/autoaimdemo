#include "infer/detail/trtyolo_impl.hpp"
#include "infer/impl_postproc_common.hpp"
#include "infer/affine_boxes_kernels.cuh"
#include <cuda_runtime.h>

namespace trtyolo {

// 返回设备侧结果视图：要求 decode + NMS + affine 回正已在 GPU 完成
// 暴露为可见的函数（在 models.cpp 中以 extern 使用）
DetectResDev postProcessDetectDevView_public(BaseModel::Impl* self, int idx){
    // 真·GPU 路径：直接读取 device 输出张量，并在 GPU 上做仿射逆变换
    auto& tins = self->backend()->tensor_infos;
    // layout 参考 impl_postproc_common.hpp::get_ptrs
    auto& num_tensor   = const_cast<TensorInfo&>(tins[1]);
    auto& box_tensor   = const_cast<TensorInfo&>(tins[2]);
    auto& score_tensor = const_cast<TensorInfo&>(tins[3]);
    auto& class_tensor = const_cast<TensorInfo&>(tins[4]);

    int num = static_cast<int*>(num_tensor.buffer->host())[idx]; // 主机侧读取数量
    if (num <= 0){ DetectResDev z; z.num=0; return z; }

    // 设备侧基指针及偏移
    float* boxes_dev  = static_cast<float*>(box_tensor.buffer->device());
    float* scores_dev = static_cast<float*>(score_tensor.buffer->device());
    int*   classes_dev= static_cast<int*>(class_tensor.buffer->device());
    int    stride_box = box_tensor.shape.d[1] * box_tensor.shape.d[2];
    int    stride_sc  = score_tensor.shape.d[1];
    int    stride_cl  = class_tensor.shape.d[1];

    float* boxes_ptr  = boxes_dev  + idx * stride_box; // 取本 batch 的起始
    float* scores_ptr = scores_dev + idx * stride_sc;
    int*   classes_ptr= classes_dev+ idx * stride_cl;

    // 申请/复用每批次输出缓冲，避免每帧 malloc/free 抖动
    auto& dev    = self->dev_boxes_per_batch_;
    auto& caps   = self->dev_boxes_capacity_bytes_;
    if ((int)dev.size() <= idx) { dev.resize(idx+1, nullptr); caps.resize(idx+1, 0); }
    size_t need_bytes = static_cast<size_t>(num) * 4 * sizeof(float);
    if (caps[idx] < need_bytes || dev[idx] == nullptr) {
        if (dev[idx]) cudaFree(dev[idx]);
        cudaMalloc(&dev[idx], need_bytes);
        caps[idx] = need_bytes;
    }
    // 复制到可变长输出缓冲（后续在其上做逆变换）
    cudaMemcpyAsync(dev[idx], boxes_ptr, need_bytes, cudaMemcpyDeviceToDevice, self->backend()->stream);

    // 在 GPU 上做仿射逆变换（letterbox 逆变换：回到原图坐标）
    const auto& aff = pick_affine(self->backend()->infer_config, self->backend()->affine_transforms, idx);
    cudaAffineBoxesInverse(dev[idx], num, aff.pad_x, aff.pad_y, aff.scale, self->backend()->stream);
    // 不在此同步；依赖流的顺序保证后续在同一流上的计算正确。

    DetectResDev out; out.xyxy=dev[idx]; out.scores=scores_ptr; out.classes=classes_ptr; out.num=num; return out;
}

} // namespace trtyolo
