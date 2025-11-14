#pragma once

#include "api.hpp"
#include "base_model.hpp"
#include "types.hpp"
#include "results.hpp"

namespace trtyolo {

class TRTYOLOAPI DetectModel : public BaseModel {
public:
    DetectModel();
    ~DetectModel();

    explicit DetectModel(const std::string& trt_engine_file, const InferOption& infer_option);

    std::unique_ptr<DetectModel> clone() const;

    DetectRes predict(const Image& image);
    std::vector<DetectRes> predict(const std::vector<Image>& images);
};

} // namespace trtyolo
