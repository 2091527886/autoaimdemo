#include "infer/detail/trtyolo_impl.hpp"
#include "infer/impl_postproc_common.hpp"

namespace trtyolo {

DetectRes BaseModel::Impl::postProcessDetect(int idx) {
    auto p = get_ptrs(backend()->tensor_infos, idx);

    DetectRes result; reserve_common(result, p.num);

    auto& affine_transform = pick_affine(backend()->infer_config, backend()->affine_transforms, idx);

    for (int i = 0; i < p.num; ++i) {
        int base_index = i * p.box_size;
        TRTYOLO_EXTRACT_LTRB(p, base_index, left, top, right, bottom);
        TRTYOLO_AFFINE4(affine_transform, left, top, right, bottom);
        TRTYOLO_PUSH_DET(result, p, i, left, top, right, bottom);
    }

    return result;
}

} // namespace trtyolo
