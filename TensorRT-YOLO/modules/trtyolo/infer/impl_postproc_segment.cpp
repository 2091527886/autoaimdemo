#include "infer/detail/trtyolo_impl.hpp"
#include <cstring>
#include <utility>

namespace trtyolo {

SegmentRes BaseModel::Impl::postProcessSegment(int idx) {
    auto& num_tensor   = backend_->tensor_infos[1];
    auto& box_tensor   = backend_->tensor_infos[2];
    auto& score_tensor = backend_->tensor_infos[3];
    auto& class_tensor = backend_->tensor_infos[4];
    auto& mask_tensor  = backend_->tensor_infos[5];
    int   mask_height  = mask_tensor.shape.d[2];
    int   mask_width   = mask_tensor.shape.d[3];

    int      num     = static_cast<int*>(num_tensor.buffer->host())[idx];
    float*   boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
    float*   scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
    int*     classes = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];
    uint8_t* masks   = static_cast<uint8_t*>(mask_tensor.buffer->host()) + idx * mask_tensor.shape.d[1] * mask_height * mask_width;

    SegmentRes result;
    result.num   = num;
    int box_size = box_tensor.shape.d[2];

    auto& affine_transform = backend_->infer_config.input_shape.has_value()
                                 ? backend_->affine_transforms.front()
                                 : backend_->affine_transforms[idx];

    result.boxes.reserve(num);
    result.scores.reserve(num);
    result.classes.reserve(num);
    result.masks.reserve(num);

    for (int i = 0; i < num; ++i) {
        int   base_index = i * box_size;
        float left = boxes[base_index], top = boxes[base_index + 1];
        float right = boxes[base_index + 2], bottom = boxes[base_index + 3];

        affine_transform.applyTransform(left, top, &left, &top);
        affine_transform.applyTransform(right, bottom, &right, &bottom);

        result.boxes.emplace_back(Box{left, top, right, bottom});
        result.scores.push_back(scores[i]);
        result.classes.push_back(classes[i]);

        Mask mask(mask_width - 2 * affine_transform.dst_offset_x, mask_height - 2 * affine_transform.dst_offset_y);
        int start_idx = i * mask_height * mask_width;
        int src_idx   = start_idx + affine_transform.dst_offset_y * mask_width + affine_transform.dst_offset_x;
        for (int y = 0; y < mask.height; ++y) {
            std::memcpy(&mask.data[y * mask.width], masks + src_idx, mask.width);
            src_idx += mask_width;
        }
        result.masks.emplace_back(std::move(mask));
    }

    return result;
}

} // namespace trtyolo
