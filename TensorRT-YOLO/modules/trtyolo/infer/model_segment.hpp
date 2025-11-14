#pragma once

#include "api.hpp"
#include "base_model.hpp"
#include "types.hpp"
#include "results.hpp"

namespace trtyolo {

class TRTYOLOAPI SegmentModel : public BaseModel {
public:
    SegmentModel();
    ~SegmentModel();

    explicit SegmentModel(const std::string& trt_engine_file, const InferOption& infer_option);

    std::unique_ptr<SegmentModel> clone() const;

    SegmentRes predict(const Image& image);
    std::vector<SegmentRes> predict(const std::vector<Image>& images);
};

} // namespace trtyolo
