#include "infer/trtyolo.hpp"
#include "infer/detail/trtyolo_impl.hpp"

namespace trtyolo {

// ClassifyModel
ClassifyModel::ClassifyModel()  = default;
ClassifyModel::~ClassifyModel() = default;
ClassifyModel::ClassifyModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : BaseModel(trt_engine_file, infer_option) {}
std::unique_ptr<ClassifyModel> ClassifyModel::clone() const {
    auto clone_model   = std::make_unique<ClassifyModel>();
    clone_model->impl_ = impl_->clone();
    return clone_model;
}
std::vector<ClassifyRes> ClassifyModel::predict(const std::vector<Image>& images) {
    auto processImages = [this](size_t num) -> std::vector<ClassifyRes> {
        std::vector<ClassifyRes> results(num);
        for (size_t idx = 0; idx < num; ++idx) {
            results[idx] = this->impl_->postProcessClassify(static_cast<int>(idx));
        }
        return results;
    };
    return impl_->withPerformanceReport<decltype(processImages), std::vector<ClassifyRes>>(images, processImages);
}
ClassifyRes ClassifyModel::predict(const Image& image) { return predict(std::vector<Image>{image}).front(); }

// DetectModel
DetectModel::DetectModel()  = default;
DetectModel::~DetectModel() = default;
DetectModel::DetectModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : BaseModel(trt_engine_file, infer_option) {}
std::unique_ptr<DetectModel> DetectModel::clone() const {
    auto clone_model   = std::make_unique<DetectModel>();
    clone_model->impl_ = impl_->clone();
    return clone_model;
}
std::vector<DetectRes> DetectModel::predict(const std::vector<Image>& images) {
    auto processImages = [this](size_t num) -> std::vector<DetectRes> {
        std::vector<DetectRes> results(num);
        for (size_t idx = 0; idx < num; ++idx) {
            results[idx] = this->impl_->postProcessDetect(static_cast<int>(idx));
        }
        return results;
    };
    return impl_->withPerformanceReport<decltype(processImages), std::vector<DetectRes>>(images, processImages);
}
DetectRes DetectModel::predict(const Image& image) { return predict(std::vector<Image>{image}).front(); }

// OBBModel
OBBModel::OBBModel()  = default;
OBBModel::~OBBModel() = default;
OBBModel::OBBModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : BaseModel(trt_engine_file, infer_option) {}
std::unique_ptr<OBBModel> OBBModel::clone() const {
    auto clone_model   = std::make_unique<OBBModel>();
    clone_model->impl_ = impl_->clone();
    return clone_model;
}
std::vector<OBBRes> OBBModel::predict(const std::vector<Image>& images) {
    auto processImages = [this](size_t num) -> std::vector<OBBRes> {
        std::vector<OBBRes> results(num);
        for (size_t idx = 0; idx < num; ++idx) {
            results[idx] = this->impl_->postProcessOBB(static_cast<int>(idx));
        }
        return results;
    };
    return impl_->withPerformanceReport<decltype(processImages), std::vector<OBBRes>>(images, processImages);
}
OBBRes OBBModel::predict(const Image& image) { return predict(std::vector<Image>{image}).front(); }

// SegmentModel
SegmentModel::SegmentModel()  = default;
SegmentModel::~SegmentModel() = default;
SegmentModel::SegmentModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : BaseModel(trt_engine_file, infer_option) {}
std::unique_ptr<SegmentModel> SegmentModel::clone() const {
    auto clone_model   = std::make_unique<SegmentModel>();
    clone_model->impl_ = impl_->clone();
    return clone_model;
}
std::vector<SegmentRes> SegmentModel::predict(const std::vector<Image>& images) {
    auto processImages = [this](size_t num) -> std::vector<SegmentRes> {
        std::vector<SegmentRes> results(num);
        for (size_t idx = 0; idx < num; ++idx) {
            results[idx] = this->impl_->postProcessSegment(static_cast<int>(idx));
        }
        return results;
    };
    return impl_->withPerformanceReport<decltype(processImages), std::vector<SegmentRes>>(images, processImages);
}
SegmentRes SegmentModel::predict(const Image& image) { return predict(std::vector<Image>{image}).front(); }

// PoseModel
PoseModel::PoseModel()  = default;
PoseModel::~PoseModel() = default;
PoseModel::PoseModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : BaseModel(trt_engine_file, infer_option) {}
std::unique_ptr<PoseModel> PoseModel::clone() const {
    auto clone_model   = std::make_unique<PoseModel>();
    clone_model->impl_ = impl_->clone();
    return clone_model;
}
std::vector<PoseRes> PoseModel::predict(const std::vector<Image>& images) {
    auto processImages = [this](size_t num) -> std::vector<PoseRes> {
        std::vector<PoseRes> results(num);
        for (size_t idx = 0; idx < num; ++idx) {
            results[idx] = this->impl_->postProcessPose(static_cast<int>(idx));
        }
        return results;
    };
    return impl_->withPerformanceReport<decltype(processImages), std::vector<PoseRes>>(images, processImages);
}
PoseRes PoseModel::predict(const Image& image) { return predict(std::vector<Image>{image}).front(); }

} // namespace trtyolo
