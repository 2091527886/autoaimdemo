#include "infer/detail/trtyolo_impl.hpp"
#include "infer/impl_postproc_common.hpp"

namespace trtyolo {

OBBRes BaseModel::Impl::postProcessOBB(int idx) {
    auto p = get_ptrs(backend_->tensor_infos, idx);

    OBBRes result; reserve_common(result, p.num);

    auto& affine_transform = pick_affine(backend_->infer_config, backend_->affine_transforms, idx);

    for (int i = 0; i < p.num; ++i) {
        int   base_index = i * p.box_size;
        float left = p.boxes[base_index], top = p.boxes[base_index + 1];
        float right = p.boxes[base_index + 2], bottom = p.boxes[base_index + 3];
        float theta = p.boxes[base_index + 4];
        affine_transform.applyTransform(left, top, &left, &top);
        affine_transform.applyTransform(right, bottom, &right, &bottom);
        result.boxes.emplace_back(RotatedBox{left, top, right, bottom, theta});
        result.scores.push_back(p.scores[i]);
        result.classes.push_back(p.classes[i]);
    }

    return result;
}

} // namespace trtyolo
