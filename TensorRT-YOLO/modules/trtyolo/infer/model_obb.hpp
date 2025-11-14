#pragma once

#include "api.hpp"
#include "base_model.hpp"
#include "types.hpp"
#include "results.hpp"

namespace trtyolo {

class TRTYOLOAPI OBBModel : public BaseModel {
public:
    OBBModel();
    ~OBBModel();

    explicit OBBModel(const std::string& trt_engine_file, const InferOption& infer_option);

    std::unique_ptr<OBBModel> clone() const;

    OBBRes predict(const Image& image);
    std::vector<OBBRes> predict(const std::vector<Image>& images);
};

} // namespace trtyolo
