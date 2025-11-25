#include "infer/backend.hpp"
#include "utils/common.hpp"
#include "core/core.hpp"
#include "letterbox.hpp"
#include "infer/pch_infer.hpp"

namespace trtyolo {

TrtBackend::TrtBackend(const std::string& trt_engine_file, const InferConfig& infer_config)
    : infer_config(infer_config) {
    cudaSetDevice(infer_config.device_id);
    CHECK(cudaStreamCreate(&stream));

    // Zero Copy support
    zero_copy_ = SupportsIntegratedZeroCopy(infer_config.device_id);

    // TRTManager instance
    manager_ = std::make_unique<TRTManager>();

    // Load engine buffer
    std::string engine_buffer;
    ReadBinaryFromFile(trt_engine_file, &engine_buffer);

    // Initialize manager
    manager_->initialize(engine_buffer.data(), engine_buffer.size());

    // Tensor info
    getTensorInfo();

    // Initialize variables
    initialize();

    // Capture graph for static model
    if (!dynamic) captureCudaGraph();
}

TrtBackend::~TrtBackend() {
    std::vector<TensorInfo>().swap(tensor_infos);
    std::vector<LetterboxTransform>().swap(affine_transforms);
    if (!dynamic) cuda_graph_.destroy();
    CHECK(cudaStreamDestroy(stream));
}

void TrtBackend::getTensorInfo() {
    std::vector<TensorInfo>().swap(tensor_infos);
    buffer_type_ = infer_config.enable_managed_memory ? BufferType::Unified
                                                      : (zero_copy_ ? BufferType::Mapped : BufferType::Discrete);
    auto num_tensors = manager_->getNbIOTensors();
    for (auto i = 0; i < num_tensors; ++i) {
        std::string name  = std::string(manager_->getIOTensorName(i));
        auto        shape = manager_->getTensorShape(name.c_str());
        auto        dtype = manager_->getTensorDataType(name.c_str());
        bool        input = (manager_->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT);

        if (input) {
            dynamic = std::any_of(shape.d, shape.d + shape.nbDims, [](int val) { return val == -1; });
            if (dynamic) {
                shape     = manager_->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMIN);
                min_shape = make_int4(shape.d[0], shape.d[1], shape.d[2], shape.d[3]);
                shape     = manager_->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
                // print accept range if needed
            }
            max_shape = make_int4(shape.d[0], shape.d[1], shape.d[2], shape.d[3]);
        } else if (!input && dynamic) {
            shape.d[0] = max_shape.x;
        }
        tensor_infos.emplace_back(name, shape, dtype, input, input ? BufferType::Device : buffer_type_);
    }
}

void TrtBackend::initialize() {
    // clear transforms and buffers
    std::vector<LetterboxTransform>().swap(affine_transforms);
    inputs_buffer_ = BufferFactory::createBuffer(buffer_type_);

    infer_size_ = max_shape.y * max_shape.w * max_shape.z;

    if (infer_config.input_shape.has_value()) {
        int in_w = infer_config.input_shape->y;
        int in_h = infer_config.input_shape->x;
        input_row_bytes_   = static_cast<size_t>(in_w) * 4;  // Expect BGRA input
        input_image_bytes_ = static_cast<size_t>(in_h) * input_row_bytes_;
        input_size_        = static_cast<int>(input_image_bytes_);

        affine_transforms.emplace_back(LetterboxTransform());
        affine_transforms.front().updateParams(in_w, in_h, max_shape.w, max_shape.z);
        inputs_buffer_->allocate(max_shape.x * input_image_bytes_);
    } else {
        affine_transforms.resize(max_shape.x, LetterboxTransform());
        if (!dynamic) inputs_buffer_->allocate(max_shape.x * infer_size_);
    }
}

} // namespace trtyolo
