#include "infer/trtyolo.hpp"
#include "infer/detail/trtyolo_impl.hpp"

namespace trtyolo {

// 前向声明：设备侧检测结果视图（在 impl_postproc_detect_dev.cpp 中定义）
DetectResDev postProcessDetectDevView_public(BaseModel::Impl*, int);


#define DEF_MODEL(Name, ResType, Post) \
Name##Model::Name##Model()=default; \
Name##Model::~Name##Model()=default; \
Name##Model::Name##Model(const std::string& trt_engine_file, const InferOption& infer_option): BaseModel(trt_engine_file, infer_option){} \
std::unique_ptr<Name##Model> Name##Model::clone() const { auto m=std::make_unique<Name##Model>(); m->impl_=impl_->clone(); return m; } \
std::vector<ResType> Name##Model::predict(const std::vector<Image>& images){ auto f=[this](size_t n)->std::vector<ResType>{ std::vector<ResType> r(n); for(size_t i=0;i<n;++i) r[i]=this->impl_->Post(static_cast<int>(i)); return r; }; return impl_->withPerformanceReport<decltype(f), std::vector<ResType>>(images, f);} \
ResType Name##Model::predict(const Image& image){ return predict(std::vector<Image>{image}).front(); }

DEF_MODEL(Classify, ClassifyRes, postProcessClassify);
DEF_MODEL(Detect,   DetectRes,   postProcessDetect);

// 设备侧零拷贝版本：对 Detect 增加 predict_dev
std::vector<DetectResDev> DetectModel::predict_dev(const std::vector<Image>& images){
    auto f=[this](size_t n)->std::vector<DetectResDev>{
        std::vector<DetectResDev> r(n);
        for(size_t i=0;i<n;++i){
            // 调用我们在 impl_postproc_detect_dev.cpp 中定义的视图函数
            // 这里直接复用 Impl，假设其存在同名函数 postProcessDetectDevView
            r[i] = postProcessDetectDevView_public(this->impl_.get(), static_cast<int>(i));
        }
        return r;
    };
    return impl_->withPerformanceReport<decltype(f), std::vector<DetectResDev>>(images, f);
}

DetectResDev DetectModel::predict_dev(const Image& image){ return predict_dev(std::vector<Image>{image}).front(); }

DEF_MODEL(OBB,      OBBRes,      postProcessOBB);
DEF_MODEL(Segment,  SegmentRes,  postProcessSegment);
DEF_MODEL(Pose,     PoseRes,     postProcessPose);

// 暴露底层 CUDA 流
cudaStream_t DetectModel::getStream() const {
    // impl_ 持有 backend_，直接返回其 stream
    return this->impl_->backend()->stream;
}

#undef DEF_MODEL

} // namespace trtyolo
