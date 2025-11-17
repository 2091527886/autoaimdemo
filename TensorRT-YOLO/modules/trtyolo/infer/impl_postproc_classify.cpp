#include "infer/detail/trtyolo_impl.hpp"

namespace trtyolo {

ClassifyRes BaseModel::Impl::postProcessClassify(int idx) {
    auto&  tensor_info = backend()->tensor_infos[1];
    float* topk        = static_cast<float*>(tensor_info.buffer->host()) + idx * tensor_info.shape.d[1] * tensor_info.shape.d[2];

    ClassifyRes result;
    result.num = tensor_info.shape.d[1];
    result.scores.reserve(result.num);
    result.classes.reserve(result.num);

    for (int i = 0; i < result.num; ++i) {
        result.scores.push_back(topk[i * tensor_info.shape.d[2]]);
        result.classes.push_back(topk[i * tensor_info.shape.d[2] + 1]);
    }

    return result;
}

} // namespace trtyolo
