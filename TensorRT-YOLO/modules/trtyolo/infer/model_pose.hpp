#pragma once

#include "api.hpp"
#include "base_model.hpp"
#include "types.hpp"
#include "results.hpp"

namespace trtyolo {

class TRTYOLOAPI PoseModel : public BaseModel {
public:
    PoseModel();
    ~PoseModel();

    explicit PoseModel(const std::string& trt_engine_file, const InferOption& infer_option);

    std::unique_ptr<PoseModel> clone() const;

    PoseRes predict(const Image& image);
    std::vector<PoseRes> predict(const std::vector<Image>& images);
};

} // namespace trtyolo
