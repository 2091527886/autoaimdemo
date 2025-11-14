#pragma once

#include "api.hpp"
#include "base_model.hpp"
#include "types.hpp"
#include "results.hpp"

namespace trtyolo {

class TRTYOLOAPI ClassifyModel : public BaseModel {
public:
    ClassifyModel();
    ~ClassifyModel();

    explicit ClassifyModel(const std::string& trt_engine_file, const InferOption& infer_option);

    std::unique_ptr<ClassifyModel> clone() const;

    ClassifyRes predict(const Image& image);
    std::vector<ClassifyRes> predict(const std::vector<Image>& images);
};

} // namespace trtyolo
