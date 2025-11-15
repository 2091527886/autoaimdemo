#include "infer/detail/trtyolo_impl.hpp"
#include "infer/impl_postproc_common.hpp"
#include <utility>

namespace trtyolo {

PoseRes BaseModel::Impl::postProcessPose(int idx) {
    auto& kpt_tensor   = backend_->tensor_infos[5];
    int   nkpt         = kpt_tensor.shape.d[2];
    int   ndim         = kpt_tensor.shape.d[3];

    auto p = get_ptrs(backend_->tensor_infos, idx);
    float* kpts = static_cast<float*>(kpt_tensor.buffer->host()) + idx * kpt_tensor.shape.d[1] * nkpt * ndim;

    PoseRes result; reserve_common(result, p.num);

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
