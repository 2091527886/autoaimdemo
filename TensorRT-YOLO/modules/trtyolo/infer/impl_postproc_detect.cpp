#include "infer/detail/trtyolo_impl.hpp"
#include "infer/impl_postproc_common.hpp"

namespace trtyolo {

DetectRes BaseModel::Impl::postProcessDetect(int idx) {
    auto p = get_ptrs(backend_->tensor_infos, idx);

    DetectRes result; reserve_common(result, p.num);

    auto& affine_transform = pick_affine(backend_->infer_config, backend_->affine_transforms, idx);

    for (int i = 0; i < p.num; ++i) {
        int   base_index = i * p.box_size;
        float left = p.boxes[base_index], top = p.boxes[base_index + 1];
        float right = p.boxes[base_index + 2], bottom = p.boxes[base_index + 3];
        affine_transform.applyTransform(left, top, &left, &top);
        affine_transform.applyTransform(right, bottom, &right, &bottom);
        result.boxes.emplace_back(Box{left, top, right, bottom});
        result.scores.push_back(p.scores[i]);
        result.classes.push_back(p.classes[i]);
    }

    return result;
}

} // namespace trtyolo
