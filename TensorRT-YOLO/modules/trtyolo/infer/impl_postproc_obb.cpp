#include "infer/detail/trtyolo_impl.hpp"
#include "infer/impl_postproc_common.hpp"

namespace trtyolo {

OBBRes BaseModel::Impl::postProcessOBB(int idx) {
    auto p = get_ptrs(backend_->tensor_infos, idx);

    OBBRes result; reserve_common(result, p.num);

    auto& affine_transform = pick_affine(backend_->infer_config, backend_->affine_transforms, idx);

    for (int i = 0; i < p.num; ++i) {
        int base_index = i * p.box_size;
        TRTYOLO_EXTRACT_LTRB(p, base_index, left, top, right, bottom);
        float theta = p.boxes[base_index + 4];
        TRTYOLO_AFFINE4(affine_transform, left, top, right, bottom);
        TRTYOLO_PUSH_OBB(result, p, i, left, top, right, bottom, theta);
    }

    return result;
}

} // namespace trtyolo
