#include "infer/trtyolo.hpp"
#include "infer/detail/trtyolo_impl.hpp"

namespace trtyolo {

#define DEF_MODEL(Name, ResType, Post) \
Name##Model::Name##Model()=default; \
Name##Model::~Name##Model()=default; \
Name##Model::Name##Model(const std::string& trt_engine_file, const InferOption& infer_option): BaseModel(trt_engine_file, infer_option){} \
std::unique_ptr<Name##Model> Name##Model::clone() const { auto m=std::make_unique<Name##Model>(); m->impl_=impl_->clone(); return m; } \
std::vector<ResType> Name##Model::predict(const std::vector<Image>& images){ auto f=[this](size_t n)->std::vector<ResType>{ std::vector<ResType> r(n); for(size_t i=0;i<n;++i) r[i]=this->impl_->Post(static_cast<int>(i)); return r; }; return impl_->withPerformanceReport<decltype(f), std::vector<ResType>>(images, f);} \
ResType Name##Model::predict(const Image& image){ return predict(std::vector<Image>{image}).front(); }

DEF_MODEL(Classify, ClassifyRes, postProcessClassify);
DEF_MODEL(Detect,   DetectRes,   postProcessDetect);
DEF_MODEL(OBB,      OBBRes,      postProcessOBB);
DEF_MODEL(Segment,  SegmentRes,  postProcessSegment);
DEF_MODEL(Pose,     PoseRes,     postProcessPose);

#undef DEF_MODEL

} // namespace trtyolo
