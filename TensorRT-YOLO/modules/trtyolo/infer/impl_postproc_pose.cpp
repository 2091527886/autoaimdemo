#include "infer/detail/trtyolo_impl.hpp"
#include "infer/impl_postproc_common.hpp"
#include "infer/pch_infer.hpp"

namespace trtyolo {

PoseRes BaseModel::Impl::postProcessPose(int idx) {
    auto& kpt_tensor   = backend()->tensor_infos[5];
    int   nkpt         = kpt_tensor.shape.d[2];
    int   ndim         = kpt_tensor.shape.d[3];

    auto p = get_ptrs(backend()->tensor_infos, idx);
    float* kpts = static_cast<float*>(kpt_tensor.buffer->host()) + idx * kpt_tensor.shape.d[1] * nkpt * ndim;

    PoseRes result; reserve_common(result, p.num);

    auto& affine_transform = pick_affine(backend()->infer_config, backend()->affine_transforms, idx);

    for (int i = 0; i < p.num; ++i) {
        int base_index = i * p.box_size;
        TRTYOLO_EXTRACT_LTRB(p, base_index, left, top, right, bottom);
        TRTYOLO_AFFINE4(affine_transform, left, top, right, bottom);
        TRTYOLO_PUSH_DET(result, p, i, left, top, right, bottom);

        std::vector<KeyPoint> keypoints;
        for (int j = 0; j < nkpt; ++j) {
            float x = kpts[i * nkpt * ndim + j * ndim];
            float y = kpts[i * nkpt * ndim + j * ndim + 1];
            affine_transform.applyTransform(x, y, &x, &y);
            keypoints.emplace_back((ndim == 2) ? KeyPoint(x, y) : KeyPoint(x, y, kpts[i * nkpt * ndim + j * ndim + 2]));
        }
        result.kpts.emplace_back(std::move(keypoints));
    }

    return result;
}

} // namespace trtyolo
