#include "infer/detail/trtyolo_impl.hpp"
#include "infer/impl_postproc_common.hpp"
#include "infer/pch_infer.hpp"

namespace trtyolo {

SegmentRes BaseModel::Impl::postProcessSegment(int idx) {
    auto& mask_tensor  = backend_->tensor_infos[5];
    int   mask_height  = mask_tensor.shape.d[2];
    int   mask_width   = mask_tensor.shape.d[3];

    auto p = get_ptrs(backend_->tensor_infos, idx);
    uint8_t* masks = static_cast<uint8_t*>(mask_tensor.buffer->host()) + idx * mask_tensor.shape.d[1] * mask_height * mask_width;

    SegmentRes result; reserve_common(result, p.num);

    auto& affine_transform = pick_affine(backend_->infer_config, backend_->affine_transforms, idx);

    for (int i = 0; i < p.num; ++i) {
        int base_index = i * p.box_size;
        TRTYOLO_EXTRACT_LTRB(p, base_index, left, top, right, bottom);
        TRTYOLO_AFFINE4(affine_transform, left, top, right, bottom);
        TRTYOLO_PUSH_DET(result, p, i, left, top, right, bottom);

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
